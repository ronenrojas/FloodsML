import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

PARAM_LST = ["uztwm", "uzfwm", "uzk", "pctim", "adimp", "zperc", "rexp", "lztwm", "lzfsm", "lzfpm", "lzsk", "lzpk",
             "pfree", "kk", "n"]
INPUT_PATH = "SAC_SMA\\C\\rundir\in.txt"
OUTPUT_PATH = "SAC_SMA\\C\\rundir\\out.txt"
CMD_LINE = "SAC_SMA\C\sim.exe SAC_SMA\\C\\rundir\in.txt SAC_SMA\C\par_update.in SAC_SMA\C\\rundir\out.txt"


class SacSma:
    def __init__(self, param_file=None):
        self.params_dict = None
        if param_file:
            self.read_params(param_file)
        self.input_dict = None
        self.output_dict = None

    def read_params(self, param_file):
        params = pd.read_csv(param_file, header=None, sep=' ').values.flatten()
        params_dict = {PARAM_LST[i]: params[i] for i in range(len(PARAM_LST))}
        params_dict["file_path"] = param_file
        self.set_params(params_dict)

    def set_params(self, d):
        self.params_dict = d

    def set_input(self, d):
        self.input_dict = d

    def set_output(self, d):
        self.output_dict = d

    def read_input(self, input_file):
        input_dict = {}
        data = pd.read_csv(input_file)
        precip = data['Rain(mm)'].to_numpy()
        PET = data['PET(mm)'].to_numpy()
        precip[np.isnan(precip)] = 0
        PET[np.isnan(PET)] = 0
        input_dict['precip'] = precip
        input_dict['PET'] = PET
        self.set_input(input_dict)

    def simulate(self):
        pet = self.input_dict["PET"]
        precip = self.input_dict["precip"]/4.0
        df = pd.DataFrame(np.array([pet, precip, precip, precip, precip]).T)
        df.to_csv(INPUT_PATH, sep=' ', header=False, index=False)
        os.system(CMD_LINE)
        runoff = pd.read_csv(OUTPUT_PATH, header=None).values.flatten()
        output_dict = {}
        output_dict["runoff"] = runoff
        self.set_output(output_dict)

    def plot(self):
        precip = self.input_dict["precip"]
        runoff = self.output_dict["runoff"]
        plt.plot(precip, 'b')
        plt.plot(runoff, 'g')
        plt.legend(["precip", "Q"])
        plt.xlabel('Day')
        plt.ylabel('Precipitation / runoff (mm/day)')
        plt.show()



