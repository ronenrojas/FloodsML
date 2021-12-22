from typing import Tuple
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import os
import json


class Preprocessor:

    def __init__(self, root_folder, idx_features, start_date: Tuple, end_date: Tuple,
                 lat_min=17.375, lat_max=22.625, lon_min=73.625,
                 lon_max=82.875, grid_delta=0.25, data_len=17532,
                 num_channels=3):
        if not root_folder:
            parent_dir = str(Path(os.getcwd()))
            self.root_folder = parent_dir + os.sep
        else:
            self.root_folder = root_folder
        self.path_data_file = self.root_folder + str(Path("/" + "Data/raw_data_fixed_17532_3_22_38"))
        self.dims_json_file_path = self.root_folder + "./dims_json.json"
        self.path_label = self.root_folder + "Data/CWC/"
        self.path_loc = self.root_folder + "Data/LatLon/{0}_lat_lon"
        self.path_data_clean = self.root_folder + "data/imd_lat_lon_reduced/"
        self.path_model = self.root_folder + "cnn_lstm/"
        self.dispatch_format = self.path_label + "cwc_discharge_{0}_clean"
        self.path_catchments = self.root_folder + "data/catchments.xlsx"
        self.file_format = self.root_folder + "data_{0}_{1}"
        self.idx_features = idx_features
        # lat - width, lon - height
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.grid_delta = grid_delta
        self.data_len = data_len
        self.num_channels = num_channels
        self.lat_grid = np.arange(self.lat_min, self.lat_max + self.grid_delta / 2, self.grid_delta)
        self.lon_grid = np.arange(self.lon_min, self.lon_max + self.grid_delta / 2, self.grid_delta)
        self.default_lat = len(self.lat_grid)
        self.default_lon = len(self.lon_grid)
        self.data_start_date = start_date
        self.data_end_date = end_date

    # Latitude - Width
    # Longitude - Height
    def reshape_data_by_lat_lon_file(self, data_file_path, dims_json_file_path):
        image_width = self.default_lat
        image_height = self.default_lon
        try:
            f = open(dims_json_file_path)
            dims_json_all = json.load(f)
            if data_file_path in dims_json_all.keys():
                dims_json = dims_json_all[data_file_path]
                if "width" in dims_json.keys():
                    image_width = dims_json["width"]
                if "height" in dims_json.keys():
                    image_height = dims_json["height"]
        except Exception as e:
            print("There was an exception in reading the dims file: {}".format(e))
        data_ret = np.fromfile(data_file_path).reshape((self.data_len, self.num_channels,
                                                        image_width, image_height))
        return data_ret, image_width, image_height

    @staticmethod
    def get_index(data, date_input):
        year, month, day = date_input
        return int(np.where(np.array(data[0] == year) * np.array(data[1] == month)
                            * np.array(data[2] == day))[0].squeeze())

    def get_geo_raw_data(self, lat, lon, start_date, end_date):
        # Getting data by lat lon coordinates
        data = pd.read_csv(self.path_data_clean + self.file_format.format(lat, lon), header=None, delim_whitespace=True)
        idx_start, idx_end = Preprocessor.get_index(data, start_date), Preprocessor.get_index(data, end_date)
        x = np.array(data[3][idx_start:idx_end + 1])
        x = np.concatenate(
            [[x], [np.array(data[4][idx_start:idx_end + 1])], [np.array(data[5][idx_start:idx_end + 1])]]).T
        return x

    @staticmethod
    def create_catchment_dict(sheet_path):
        df = pd.read_excel(sheet_path, index_col=0).dropna().T
        means = df.mean().values
        stds = df.std(ddof=0).values
        x = df.values
        catch_dict = {k: x[i, :] for i, k in enumerate(df.T.columns)}
        catch_dict['mean'] = means
        catch_dict['std'] = stds
        return catch_dict

    @staticmethod
    def get_date_range_and_idx(self, start_date, end_date, date_range):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        idx = np.where(np.bitwise_and(start_date_pd <= date_range, date_range <= end_date_pd))[0]
        date_range_out = pd.date_range(start_date_pd, end_date_pd)
        return date_range_out, idx

    @staticmethod
    def get_months_by_dates(start_date, end_date):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        months = [date_range[i].month for i in range(0, len(date_range))]
        return months

    @staticmethod
    def get_index_by_lat_lon(lat, lon, lat_grid, lon_grid):
        i = np.where(lat == lat_grid)[0]
        assert len(i) > 0, f"Please supply latitude between {min(lat_grid)} and {max(lat_grid)}"
        j = np.where(lon == lon_grid)[0]
        assert len(j) > 0, f"Please supply longitude between {min(lon_grid)} and {max(lon_grid)}"
        return i, j

    def get_index_all_data(self, date_in, lat, lon, lat_grid, lon_grid):
        date_i = self.get_index_by_date(date_in)
        lat_i, lon_i = Preprocessor.get_index_by_lat_lon(lat, lon, lat_grid=lat_grid, lon_grid=lon_grid)
        return date_i, lat_i, lon_i

    """
    generating the "image" basin from a larger "image" by mask:
    the lat_grid and lon_grid are two 1-d arrays that construct a matrix of points
    that depicting the bottom right corner of every "pixel" of the large area.
    from this large area, we are checking if the "pixels" of the basins smaller area
    are in this large area and creating a corresponding mask.
    The mask is in the size of the large area - 1 if this pixel in also in the basin's area,
    and 0 otherwise
    """

    def create_basin_mask(self, basin):
        # getting the grid describing the basin from file
        df = pd.read_csv(self.path_loc.format(basin), header=None, delim_whitespace=True)
        basin_lat_lot = df.values
        # lat_grid is an 1-d array that describing the horizontal lines of the rectangle
        # surrounding the basin
        # lon_grid is an 1-d array that describing the vertical lines of the rectangle
        # surrounding the basin
        width = self.lat_grid
        height = self.lon_grid
        indices_X = np.ndarray((len(basin_lat_lot), width, height))
        # initialize a matrix with all zeros
        # for every pixel that is also in the basin area, set the indices of this pixel
        # (bottom right corner of the pixel) in the large matrix to True
        for lat_lon_i in basin_lat_lot:
            i, j = self.get_index_by_lat_lon(lat_lon_i[0], lat_lon_i[1], width, height)
            indices_X[lat_lon_i, i, j] = [i, j]
        return indices_X

    def get_basin_discharge(self, basin_name, start_date, end_date):
        # Getting Discharge (how much water are running through
        # specific point / river in a given amount of time -
        # usually cubic metre per second)
        data_discharge = pd.read_csv(self.dispatch_format.format(basin_name),
                                     header=None, delim_whitespace=True)
        idx_start, idx_end = Preprocessor.get_index(data_discharge, start_date), \
                             Preprocessor.get_index(data_discharge, end_date)
        indices_Y = np.linspace(idx_start, idx_end + 1, 1)
        return indices_Y

    @staticmethod
    def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape matrix data into sample shape for LSTM training.
        :param x: Matrix containing input features column wise and time steps row wise
        :param y: Matrix containing the output feature.
        :param seq_length: Length of look back days for one day of prediction
        :return: Two np.ndarrays, the first of shape (samples, length of sequence,
          number of features), containing the input data for the LSTM. The second
          of shape (samples, 1) containing the expected output for each input
          sample.
        """
        num_samples, num_features = x.shape
        x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
        y_new = np.zeros((num_samples - seq_length + 1, 1))
        for i in range(0, x_new.shape[0]):
            x_new[i, :, :num_features] = x[i:i + seq_length, :]
            y_new[i, :] = y[i + seq_length - 1, 0]
        return x_new, y_new

    def reshape_data_basins(self, x: np.ndarray, y: np.ndarray, seq_length: int,
                            basin_list: list, lead: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Reshape matrix data into sample shape for LSTM training.
        :param lead:
        :param basin_list:
        :param x: Matrix containing input features column wise and time steps row wise
        :param y: Matrix containing the output feature.
        :param seq_length: Length of look back days for one day of prediction
        :return: Two np.ndarrays, the first of shape (samples, length of sequence,
          number of features), containing the input data for the LSTM. The second
          of shape (samples, 1) containing the expected output for each input
          sample.
        """
        n_basins = len(basin_list)
        data_size = int(x.shape[0] / n_basins)

        for i in range(n_basins):
            if i == 0:
                x_new, y_new = self.reshape_data(x[:data_size - lead, :], y[lead:data_size], seq_length)
            else:
                idx = i * data_size
                x_temp, y_temp = self.reshape_data(x[idx:idx - lead + data_size, :], y[idx + lead:idx + data_size],
                                                   seq_length)
                x_new = np.concatenate([x_new, x_temp], axis=0)
                y_new = np.concatenate([y_new, y_temp], axis=0)
        return x_new, y_new

    def get_index_by_date(self, date_in):
        start_date_pd = pd.to_datetime(datetime.datetime(self.data_start_date[0], self.data_start_date[1],
                                                         self.data_start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(
            datetime.datetime(self.data_end_date[0], self.data_end_date[1], self.data_end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        date_in_pd = pd.to_datetime(datetime.datetime(date_in[0], date_in[1], date_in[2], 0, 0))
        idx = np.where(date_in_pd == date_range)[0]
        assert len(idx) > 0, f"Please supply a date between {self.data_start_date} and {self.data_end_date}"
        return idx
