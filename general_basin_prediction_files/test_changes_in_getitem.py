from general_basin_prediction_files.Godavari import IMDGodavari as new_G
from cnn_lstm_v3 import IMDGodavari as old_G, INCLUDE_STATIC
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.notebook
import tqdm
import datetime
import pathlib
import pytz
import os
import json
from LSTM import CNNLSTM
from preprocess_data import Preprocessor

root_dir = str(Path(os.getcwd()).parent)
RUN_LOCALLY = True
PATH_ROOT = root_dir + "/"  # Change only here the path
PATH_DATA_FILE = root_dir + str(Path("/" + "Data/raw_data_fixed_17532_3_22_38"))
DIMS_JSON_FILE_PATH = root_dir + "./dims_json.json"
PATH_LABEL = PATH_ROOT + "Data/CWC/"
PATH_LOC = PATH_ROOT + "Data/LatLon/{0}_lat_lon"
PATH_DATA_CLEAN = PATH_ROOT + "Data/IMD_Lat_Lon_reduced/"
PATH_MODEL = PATH_ROOT + "cnn_lstm/"
DISPATCH_FORMAT = "CWC_discharge_{0}_clean"
PATH_CATCHMENTS = PATH_ROOT + "Data/catchments.xlsx"
FILE_FORMAT = "data_{0}_{1}"

# Lat - width, Lon - height
LAT_MIN = 17.375
LAT_MAX = 22.625
LON_MIN = 73.625
LON_MAX = 82.875
GRID_DELTA = 0.25
LAT_GRID = np.arange(LAT_MIN, LAT_MAX + GRID_DELTA / 2, GRID_DELTA)
LON_GRID = np.arange(LON_MIN, LON_MAX + GRID_DELTA / 2, GRID_DELTA)
DATA_LEN = 17532
NUM_CHANNELS = 3
DEFAULT_LAT = len(LAT_GRID)
DEFAULT_LON = len(LON_GRID)
DATA_START_DATE = (1967, 1, 1)
DATA_END_DATE = (2014, 12, 31)


def train_epoch(device, model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.notebook.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device), ys.to(device)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def main():
    # Whether to use CPU or GPU. Use False for CPU mode.
    use_gpu = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
    #################################
    ###### Meta parameters ##########
    #################################
    hidden_size = 128  # Number of LSTM cells
    dropout_rate = 0.01  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    # learning_rate = 2e-3 # Learning rate used to update the weights
    learning_rate = 1e-4  # Learning rate used to update the weights
    sequence_length = 30  # Length of the meteorological record provided to the network
    num_layers = 2  # Number of LSTM cells
    lead = 0  # 1
    cnn_output_size = 20
    num_hidden_layers = 3
    num_hidden_units = 128
    ### Choose features ###
    use_perc = True
    # maximum temperature in a given day
    use_t_max = False
    # minimum temperature in a given day
    use_t_min = False
    idx_features = [use_perc, use_t_max, use_t_min]
    basin_list = ['Tekra', 'Perur']
    preprocessor = Preprocessor(PATH_ROOT, idx_features, DATA_START_DATE, DATA_END_DATE, LAT_MIN,
                                LAT_MAX, LON_MIN, LON_MAX, GRID_DELTA, DATA_LEN, NUM_CHANNELS)
    # The data will always be in shape of - samples * channels * width * height
    all_data, image_width, image_height = preprocessor.reshape_data_by_lat_lon_file(
        PATH_DATA_FILE, DIMS_JSON_FILE_PATH)
    catchment_dict = preprocessor.create_catchment_dict(PATH_CATCHMENTS)
    include_static = True
    # Training data
    start_date = (2000, 1, 1)
    end_date = (2009, 12, 31)
    months_lst = [6, 7, 8, 9, 10]
    print('Train dataset\n===============================')
    ds_train_new = new_G(all_data,
                         catchment_dict=catchment_dict,
                         preprocessor=preprocessor,
                         basin_list=basin_list,
                         seq_length=sequence_length,
                         period="train",
                         dates=[start_date, end_date],
                         months=months_lst,
                         idx=idx_features,
                         lead=lead,
                         include_static=include_static)
    ds_train_old = old_G(all_data,
                         basin_list=basin_list,
                         seq_length=sequence_length,
                         period="train",
                         dates=[start_date, end_date],
                         months=months_lst,
                         idx=idx_features,
                         lead=lead,
                         include_static=INCLUDE_STATIC)
    tr_loader_new = DataLoader(ds_train_new, batch_size=64, shuffle=True)
    tr_loader_old = DataLoader(ds_train_old, batch_size=64, shuffle=True)
    #########################
    # Model, Optimizer, Loss#
    #########################
    # Here we create our model
    # attributes == static features
    num_attributes = catchment_dict['Tekra'].size
    if not include_static:
        num_attributes = 0
    # idx_features - a True / False list over the 3 features (channels) of each "image"
    input_size = (sum(idx_features) * DEFAULT_LON * DEFAULT_LAT + num_attributes) * sequence_length
    input_image_size = (sum(idx_features), image_width, image_height)
    model = CNNLSTM(lat=image_width, lon=image_height, input_size=cnn_output_size, num_layers=num_layers,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate, num_channels=sum(idx_features),
                    num_attributes=num_attributes, image_input_size=input_image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    n_epochs = 50  # Number of training epochs
    # Creating the checkpoint folders
    datetime_israel = datetime.datetime.now(pytz.timezone('Israel'))
    path_train_ckpt = PATH_MODEL + datetime_israel.strftime("%Y_%m_%d-%H-%M-%S/")
    pathlib.Path(path_train_ckpt).mkdir(parents=True, exist_ok=True)
    for i in range(n_epochs):
        train_epoch(device, model, optimizer, tr_loader_new, loss_func, i + 1)
        train_epoch(device, model, optimizer, tr_loader_old, loss_func, i + 1)


if __name__ == "__main__":
    main()
