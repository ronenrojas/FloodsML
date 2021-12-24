from torch.utils.data import Dataset
import copy
from typing import List, Dict
import numpy as np
import torch
from preprocess_data import Preprocessor


class IMDGodavari(Dataset):
    """
    Torch Dataset for basic use of data from the data set.
    This data set provides meteorological observations and discharge
    of a given basin from the IMD Godavari data set.
    """

    def __init__(self, all_data: np.array,
                 basin_list: List,
                 preprocessor: Preprocessor,
                 catchment_dict: Dict,
                 seq_length: int,
                 period: str = None,
                 dates: List = None,
                 months: List = None,
                 min_values: np.array = None,
                 max_values: np.array = None,
                 idx: list = [True, True, True],
                 lead=0,
                 mask_list=[0, 0.5, 0.5],
                 include_static:
                 np.bool = True,
                 mean_y=None,
                 std_y=None):
        """Initialize Dataset containing the data of a single basin.
        :param basin_list: List of basins.
        :param seq_length: Length of the time window of meteorological
        input provided for one time step of prediction.
        (currently it's 30 - 30 days)
        :param period: (optional) One of ['train', 'eval']. None loads the entire time series.
        :param dates: (optional) List of the start and end date of the discharge period that is used.
        """
        self.num_attributes = 0
        self.basin_list = basin_list
        self.preprocessor = preprocessor
        self.catchment_dict = catchment_dict
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.months = months
        self.min_values = min_values
        self.max_values = max_values
        self.mean_y = mean_y
        self.std_y = std_y
        self.idx_features = idx
        self.lead = lead
        self.mask_list = mask_list
        self.num_features = None
        self.include_static = include_static
        self.indices_X = np.zeros((len(preprocessor.lat_grid), len(preprocessor.lon_grid)))
        self.indices_X, self.indices_Y, self.static_features = self._load_data(all_data)
        self.all_data = all_data
        self.num_samples = self.x.shape[0]
        self.indices_Y = np.zeros((self.num_samples,))
        self.start_end_indices_basins = {}

    def __len__(self):
        return self.num_samples

    # should return one sample
    # the data is of shape - number of samples * number of channels * image_width * image_height
    def __getitem__(self, idx: int):
        x, y = self.all_data[self.indices_X[idx]], self.indices_Y[self.indices_Y[idx]]
        x = (x - self.min_values) / (self.max_values - self.min_values)
        for bounds in self.start_end_indices_basins.keys():
            if bounds[0] <= idx <= bounds[1]:
                mu_y = self.mean_y[self.start_end_indices_basins[bounds]]
                std_y = self.std_y[self.start_end_indices_basins[bounds]]
                y = ((y - mu_y) / std_y)
        return x, y

    def _load_data(self, all_data):
        """Load input and output data from text files.    """
        start_date, end_date = self.dates
        idx_s = self.preprocessor.get_index_by_date(start_date)[0]
        idx_e = self.preprocessor.get_index_by_date(end_date)[0]
        # Data is reduced to the dates and features subspace.
        # the features are the channels (3 features - precipitation,
        # minimum temperature, maximum temperature)
        data = copy.deepcopy(all_data[idx_s:idx_e + 1, self.idx_features, :, :])
        time_span = data.shape[0]
        if self.period == 'train':
            # axis = 0 - getting the minimum / maximum of first dimension
            # ("rows", but here it's not really rows because it's a tensor and not a matrix)
            # axis = 1 - getting the minimum / maximum of second dimension
            # ("columns", but here it's not really columns because it's a tensor and not a matrix)
            # Normalizing the data. There are two typical types of normalization:
            # 1. (min - max normalization) - subtract the min and divide by (max - min)
            # 2. (std - mean normalization) - subtract the mean and divide by the std

            # specifically, here we are getting the minimum / maximum
            # of all the time stamps + H_LAT + W_LON per channel - i.e. - for all the channels
            # we are calculating the min / max over all the other dimensions.
            self.min_values = data.min(axis=0).min(axis=1).min(axis=1)
            self.max_values = data.max(axis=0).max(axis=1).max(axis=1)
        self.num_features = data.shape[2] * data.shape[3]
        num_of_samples_curr = 0
        num_of_samples_prev = 0
        for i, basin in enumerate(self.basin_list):
                indices_X, static_features = self.get_basin_indices_x_and_static_features(basin)
                if self.include_static:
                    static_features = np.concatenate([static_features, static_features_temp], axis=0)
                else:
                    self.num_attributes = 0
                indices_Y, y = self.preprocessor.get_basin_indices_y(basin, start_date, end_date)
                if self.period == 'train':
                    y = self.update_basin_dict(y, basin,
                                               starting_ind_basin=num_of_samples_prev,
                                               end_ind_basin=num_of_samples_curr)
            num_of_samples_prev = num_of_samples_curr
        # normalize data, reshape for LSTM training and remove invalid samples
        print("Data set for {0} for basins: {1}".format(self.period, self.basin_list))
        print("Number of attributes should be: {0}".format(self.num_attributes))
        print("Number of features should be: num_features + num_attributes= {0}".format(
            self.num_features + self.num_attributes))
        print("Number of sample should be: (time_span - sequence_len + 1 -lead) x num_basins= {0}".format(
            (time_span - self.seq_length + 1 - self.lead) * len(self.basin_list)))
        # convert arrays to torch tensors
        return indices_X, indices_Y, static_features

    def get_basin_indices_x_and_static_features(self, basin):
        indices_X = self.preprocessor.get_basin_indices_x(basin)
        x_static = (self.catchment_dict[basin] - self.catchment_dict['mean']) / self.catchment_dict['std']
        _, self.num_attributes = x_static.shape
        if not self.include_static:
            self.num_attributes = 0
            x_static = None
        return indices_X, x_static

    def local_rescale(self, feature: np.ndarray, mean_std=None) -> np.ndarray:
        """Rescale output features with local mean/std.
          :param mean_std:
          :param feature: Numpy array containing the feature(s) as matrix.
          param variable: Either 'inputs' or 'output' showing which feature will
          be normalized
        :return: array containing the normalized feature
        """
        if mean_std:
            mean_y, std_y = mean_std
            return feature * std_y + mean_y
        n_basins = len(self.basin_list)
        idx = int(len(feature) / n_basins)
        for i, basin_name in enumerate(self.basin_list):
            if basin_name not in self.mean_y.keys():
                raise RuntimeError(
                    f"Unknown Basin {basin_name}, the training data was trained on {list(self.mean_y.keys())}")
            if i == 0:
                y = feature[i * idx:(i + 1) * idx]
                y = y * self.std_y[basin_name] + self.mean_y[basin_name]
            else:
                y_temp = feature[i * idx:(i + 1) * idx]
                y_temp = y_temp * self.std_y[basin_name] + self.mean_y[basin_name]
                y = np.concatenate([y, y_temp])
        return y

    def update_basin_dict(self, y, basin_name, starting_ind_basin, end_ind_basin):
        if self.mean_y is None:
            self.mean_y = {}
            self.std_y = {}
        mu_y = y.mean()
        std_y = y.std()
        self.start_end_indices_basins[(starting_ind_basin, end_ind_basin)] = basin_name
        self.mean_y[basin_name] = mu_y
        self.std_y[basin_name] = std_y
        return (y - mu_y) / std_y

    def get_monthly_data(self, x, y, start_date, end_date):
        if self.months is None:
            return x, y
        else:
            # Rescaling the label
            if self.period == 'train':
                y = self.local_rescale(y)
            # getting the months for each date
            date_months = self.preprocessor.get_months_by_dates(start_date, end_date)
            # Adjusting for sequence length and lead
            date_months = date_months[(self.seq_length + self.lead - 1):]
            n_samples_per_basin = int(len(y) / len(self.basin_list))
            ind_date_months = [i for i in range(0, n_samples_per_basin) if date_months[i] in self.months]
            ind_include = []
            for j in range(len(self.basin_list)):
                idx_temp = [idx + j * n_samples_per_basin for idx in ind_date_months]
                if self.period == 'train':
                    y[idx_temp] = self.update_basin_dict(self.basin_list[j], y[idx_temp])
                ind_include += idx_temp
            x = x[ind_include, :, :]
            y = y[ind_include]
            return x, y

    def get_min(self):
        return self.min_values

    def get_max(self):
        return self.max_values

    def get_mean_y(self):
        return self.mean_y

    def get_std_y(self):
        return self.std_y

    def get_num_features(self):
        return self.num_features
