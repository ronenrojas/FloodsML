import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dream:
    def __init__(self, param_file=None):
        self.params_dict = None
        if param_file:
            self.read_params(param_file)
        self.input_dict = None
        self.output_dict = None

    def read_params(self, param_file):
        params_dict = {}
        par = pd.read_csv(param_file)
        params_dict["ts"] = par['ts'].to_numpy()[0]
        params_dict["tfc"] = par['tfc'].to_numpy()[0]
        params_dict["tpwp"] = par['tpwp'].to_numpy()[0]
        params_dict["z"] = par['z'].to_numpy()[0]
        params_dict["mue"] = par['m'].to_numpy()[0]
        params_dict["beta"] = par['b'].to_numpy()[0]
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
        output_dict = {}
        par_tfc = self.params_dict['tfc']
        par_z = self.params_dict['z']
        par_ts = self.params_dict['ts']
        par_mue = self.params_dict['mue']
        par_tpwp = self.params_dict['tpwp']
        par_beta = self.params_dict['beta']

        precip = self.input_dict['precip']
        PET = self.input_dict['PET']

        theta0 = par_tfc

        runoff = np.zeros_like(precip)
        recharge = np.zeros_like(precip)
        aet = np.zeros_like(precip)

        theta = theta0
        for ind in range(len(precip)):
            p = precip[ind]
            pet = PET[ind]
            theta = theta + p / par_z
            if theta > par_ts:
                runoff[ind] = np.max([(theta - par_ts) * par_z, 0])
                theta = par_ts
            if theta > par_tfc:
                recharge[ind] = theta * par_z * par_mue
                theta = theta - recharge[ind] / par_z
            if theta > par_tpwp:
                if theta > par_tfc:
                    aet[ind] = pet * par_beta
                else:
                    aet[ind] = pet * par_beta * (theta - par_tpwp) / (par_tfc - par_tpwp)
                theta = theta - aet[ind] / par_z
        output_dict['runoff'] = runoff
        output_dict['recharge'] = recharge
        output_dict['aet'] = aet
        self.set_output(output_dict)

    def plot(self):
        precip = self.input_dict["precip"]
        runoff = self.output_dict["runoff"]
        plt.plot(precip, 'b')
        plt.plot(runoff, 'g')
        plt.xlabel('Day')
        plt.ylabel('Precipitation / runoff (mm/day)')
        plt.show()

    def to_SAC_SMA(self, file_name):
        precip = self.input_dict["precip"]
        runoff = self.output_dict["runoff"]
        df = pd.DataFrame(np.array([runoff, precip, precip, precip, precip]).T)
        df.to_csv(file_name, sep=' ', header=False, index=False)



