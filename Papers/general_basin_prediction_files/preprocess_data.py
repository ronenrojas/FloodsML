from typing import Tuple
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import os


class Preprocessor:

    def __init__(self, root_folder, start_date: Tuple, end_date: Tuple):
        if not root_folder:
            parent_dir = str(Path(os.getcwd()))
            self.root_folder = parent_dir + "/"
        else:
            self.root_folder = root_folder
        self.path_loc = self.root_folder + "Data/LatLon/{0}_lat_lon"
        self.path_data_clean = self.root_folder + "data/imd_lat_lon_reduced/"
        self.path_model = self.root_folder + "cnn_lstm/"
        self.dispatch_format = self.root_folder + "cwc_discharge_{0}_clean"
        self.path_catchments = self.root_folder + "data/catchments.xlsx"
        self.file_format = self.root_folder + "data_{0}_{1}"
        # lat - width, lon - height
        self.lat_min = 17.375
        self.lat_max = 22.625
        self.lon_min = 73.625
        self.lon_max = 82.875
        self.grid_delta = 0.25
        self.lat_grid = np.arange(self.lat_min, self.lat_max + self.grid_delta / 2, self.grid_delta)
        self.lon_grid = np.arange(self.lon_min, self.lon_max + self.grid_delta / 2, self.grid_delta)
        self.data_len = 17532
        self.num_channels = 3
        self.default_lat = len(self.lat_grid)
        self.default_lon = len(self.lon_grid)
        self.data_start_date = (1967, 1, 1)
        self.data_end_date = (2014, 12, 31)

    def get_index(self, data, date_input):
        year, month, day = date_input
        return int(np.where(np.array(data[0] == year) * np.array(data[1] == month)
                            * np.array(data[2] == day))[0].squeeze())

    def get_geo_raw_data(self, lat, lon, start_date, end_date):
        # Getting data by lat lon coordinates
        data = pd.read_csv(self.PATH_DATA_CLEAN + self.FILE_FORMAT.format(lat, lon), header=None, delim_whitespace=True)
        idx_start, idx_end = self.get_index(data, start_date), self.get_index(data, end_date)
        x = np.array(data[3][idx_start:idx_end + 1])
        x = np.concatenate(
            [[x], [np.array(data[4][idx_start:idx_end + 1])], [np.array(data[5][idx_start:idx_end + 1])]]).T
        return x

    def create_catchment_dict(self, sheet_path):
        df = pd.read_excel(sheet_path, index_col=0).dropna().T
        means = df.mean().values
        stds = df.std(ddof=0).values
        x = df.values
        catch_dict = {k: x[i, :] for i, k in enumerate(df.T.columns)}
        catch_dict['mean'] = means
        catch_dict['std'] = stds
        return catch_dict

    def get_date_range_and_idx(self, start_date, end_date, date_range):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        idx = np.where(np.bitwise_and(start_date_pd <= date_range, date_range <= end_date_pd))[0]
        date_range_out = pd.date_range(start_date_pd, end_date_pd)
        return date_range_out, idx

    def get_index_by_date(self, date_in, start_date, end_date):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        date_in_pd = pd.to_datetime(datetime.datetime(date_in[0], date_in[1], date_in[2], 0, 0))
        idx = np.where(date_in_pd == date_range)[0]
        assert len(idx) > 0, f"Please supply a date between {start_date} and {end_date}"
        return idx

    def get_months_by_dates(self, start_date, end_date):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        months = [date_range[i].month for i in range(0, len(date_range))]
        return months

    def get_index_by_lat_lon(self, lat, lon, lat_grid, lon_grid):
        i = np.where(lat == lat_grid)[0]
        assert len(i) > 0, f"Please supply latitude between {min(lat_grid)} and {max(lat_grid)}"
        j = np.where(lon == lon_grid)[0]
        assert len(j) > 0, f"Please supply longitude between {min(lon_grid)} and {max(lon_grid)}"
        return i, j

    def get_index_all_data(self, date_in, lat, lon, lat_grid, lon_grid, start_date, end_date):
        date_i = self.get_index_by_date(date_in, start_date=start_date, end_date=end_date)
        lat_i, lon_i = self.get_index_by_lat_lon(lat, lon, lat_grid=lat_grid, lon_grid=lon_grid)
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

    def create_basin_mask(self, basin, lat_grid, lon_grid):
        # getting the grid describing the basin from file
        df = pd.read_csv(self.PATH_LOC.format(basin), header=None, delim_whitespace=True)
        basin_lat_lot = df.values
        # lat_grid is an 1-d array that describing the horizontal lines of the rectangle
        # surrounding the basin
        h = len(lat_grid)
        # lon_grid is an 1-d array that describing the vertical lines of the rectangle
        # surrounding the basin
        w = len(lon_grid)
        # initialize a matrix with all zeros
        idx_mat = np.zeros((h, w), dtype=bool)
        # for every pixel that is also in the basin area, set the indices of this pixel
        # (bottom right corner of the pixel) in the large matrix to True
        for lat_lon_i in basin_lat_lot:
            i, j = self.get_index_by_lat_lon(lat_lon_i[0], lat_lon_i[1], lat_grid, lon_grid)
            idx_mat[i[0], j[0]] = True
        return idx_mat

    def get_basin_discharge(self, basin_name, start_date, end_date):
        # Getting Discharge (how much water are running through
        # specific point / river in a given amount of time -
        # usually cubic metre per second)
        data_discharge = pd.read_csv(self.PATH_LABEL + self.DISCH_FORMAT.format(basin_name),
                                     header=None, delim_whitespace=True)
        idx_start, idx_end = self.get_index(data_discharge, start_date), self.get_index(data_discharge, end_date)
        y = np.array(data_discharge[3][idx_start:idx_end + 1])
        return y

    def reshape_data(self, x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
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
