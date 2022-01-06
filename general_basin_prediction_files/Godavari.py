from torch.utils.data import Dataset
import copy
from typing import List, Dict
import numpy as np
import torch
from preprocess_data import Preprocessor
import pandas as pd
import datetime


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
        self.num_samples = 0
        self.x = None
        self.y = None
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
        self.sample_to_basin = {}
        self.start_end_indices_basins = {}
        self.basin_name_to_mean_std_y = {}
        self.load_data(all_data)

    # The number of samples is: (the original number of samples (timestamps) - sequence length) * number of basins
    # each samples is of size: (sequence length * number of features ((width * height * channels) + static features))
    def __len__(self):
        return self.num_samples

    # each call to __getitem__ should return ONE sample, which is of size (sequence length * number of features)
    def __getitem__(self, idx: int):
        x_new = None
        y_new = None
        for key in self.sample_to_basin.keys():
            start_ind = key[0]
            end_ind = key[1]
            indices_X, static_features, min_values, max_values = self.sample_to_basin[key]
            if idx in range(start_ind, end_ind):
                idx = idx - start_ind
                x_new = self.x[idx: idx + self.seq_length, :, :, :] * indices_X
                x_new = torch.from_numpy(x_new).float()
                if self.period == 'train':
                    for feature_ind in range(len([x for x in self.idx_features if x])):
                        x_new[:, feature_ind, :, :] -= min_values[feature_ind]
                        x_new[:, feature_ind, :, :] /= (max_values[feature_ind] - min_values[feature_ind])
                x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1], x_new.shape[2] * x_new.shape[3]))
                static_features = static_features[np.newaxis, np.newaxis, :]
                static_features = np.repeat(static_features, x_new.shape[0], axis=0)
                static_features = np.repeat(static_features, x_new.shape[1], axis=1)
                x_new = np.concatenate([x_new, static_features], axis=2)
                y_indices, y_new, mu_y, std_y = self.start_end_indices_basins[key]
                if self.period == 'train':
                    y_new = ((y_new - mu_y) / std_y)
                x_new = np.reshape(x_new, (x_new.shape[0], x_new.shape[1] * x_new.shape[2]))
        end_indices = [key[1] for key in self.sample_to_basin.keys()]
        start_indices = [key[0] for key in self.sample_to_basin.keys()]
        if x_new is None or y_new is None:
            print("Error - the requested to be retrieved is not in the range of start_ind, end_ind. "
                  "The requested index is: {}, the start indices are: {}, the end indices are: {}".format(idx,
                                                                                                          start_indices,
                                                                                                          end_indices))
        x_new = x_new.astype('float32')
        y_new = y_new.astype('float32')
        return x_new, y_new

    def load_data(self, all_data):
        """Load input and output data from text files"""
        start_date, end_date = self.dates
        idx_s = self.preprocessor.get_index_by_date(start_date)[0]
        idx_e = self.preprocessor.get_index_by_date(end_date)[0]
        # cropping the data to only the interested dates and the interested idx_features,
        # the idx_features are the "channels" (3 features - precipitation minimum temperature, maximum temperature)
        data = copy.deepcopy(all_data[idx_s:idx_e + 1, self.idx_features, :, :])
        indices_to_include = self.get_monthly_data(data, start_date, end_date)
        self.num_samples = len(self.basin_list) * len(indices_to_include)
        self.x = data
        # the number of samples (each sample is form specific date)
        time_span = data.shape[0]
        self.num_features = data.shape[1] * data.shape[2] * data.shape[3]
        for i, basin in enumerate(self.basin_list):
            indices_X, static_features = self.get_basin_indices_x_and_static_features(basin)
            # calculating the min / max over all the channels - i.e. -
            # over all the timestamps + H_LAT + W_LON per channel, to later normalize the data.
            min_values_over_timestamps = data.min(axis=0)
            min_values_over_timestamps_in_basin = min_values_over_timestamps[:, ...] * indices_X
            min_values = min_values_over_timestamps_in_basin.min(axis=1).min(axis=1)

            max_values_over_timestamps = data.max(axis=0)
            max_values_over_timestamps_in_basin = max_values_over_timestamps[:, ...] * indices_X
            max_values = max_values_over_timestamps_in_basin.max(axis=1).max(axis=1)

            self.sample_to_basin[(i * (time_span - self.seq_length + 1 - self.lead),
                                  (i + 1) * (time_span - self.seq_length + 1 - self.lead))] = \
                (indices_X, static_features, min_values, max_values)
            indices_Y, y = self.preprocessor.get_basin_indices_y(basin, start_date, end_date)
            mu_y = y.mean()
            std_y = y.std()
            self.start_end_indices_basins[(i * (time_span - self.seq_length + 1 - self.lead),
                                           (i + 1) * (time_span - self.seq_length + 1 - self.lead))] = \
                (indices_Y, y, mu_y, std_y)
            self.basin_name_to_mean_std_y[basin] = (mu_y, std_y)
            if not self.include_static:
                self.num_attributes = 0
        print("Data set for {0} for basins: {1}".format(self.period, self.basin_list))
        print("Number of attributes should be: {0}".format(self.num_attributes))
        print("Number of features should be: num_features + num_attributes= {0}".format(
            self.num_features + self.num_attributes))
        print("Number of sample should be: (time_span - sequence_len + 1 -lead) x num_basins= {0}".format(
            (time_span - self.seq_length + 1 - self.lead) * len(self.basin_list)))

    def get_basin_indices_x_and_static_features(self, basin):
        indices_X = self.preprocessor.get_basin_indices_x(basin)
        x_static = (self.catchment_dict[basin] - self.catchment_dict['mean']) / self.catchment_dict['std']
        self.num_attributes = len(x_static)
        if not self.include_static:
            self.num_attributes = 0
            x_static = None
        return indices_X, x_static

    def local_rescale(self, feature: np.ndarray, mean_std=None) -> np.ndarray:
        """
        Rescale output features with local mean/std.
          param mean_std:
          param feature: Numpy array containing the feature(s) as matrix.
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
            if i == 0:
                y = feature[i * idx:(i + 1) * idx]
                y = y * self.basin_name_to_mean_std_y[basin_name][1] + self.basin_name_to_mean_std_y[basin_name][0]
            else:
                y_temp = feature[i * idx:(i + 1) * idx]
                y_temp = y_temp * self.basin_name_to_mean_std_y[basin_name][1] + \
                         self.basin_name_to_mean_std_y[basin_name][0]
                y = np.concatenate([y, y_temp])
        return y

    def get_monthly_data(self, x, start_date, end_date):
        # getting the months for each date
        date_months = self.get_months_by_dates(start_date, end_date)
        # Adjusting for sequence length and lead
        date_months = date_months[(self.seq_length + self.lead - 1):]
        n_samples_per_basin = int(len(x) / len(self.basin_list))
        ind_date_months = [i for i in range(0, n_samples_per_basin) if date_months[i] in self.months]
        ind_include = []
        for j in range(len(self.basin_list)):
            idx_temp = [idx + j * n_samples_per_basin for idx in ind_date_months]
            ind_include += idx_temp
        return ind_include

    def get_months_by_dates(self, start_date, end_date):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        months = [date_range[i].month for i in range(0, len(date_range))]
        return months

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
