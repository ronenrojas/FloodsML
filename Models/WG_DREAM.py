import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import weibull_min
from scipy.stats import norm


# Parameters: yearnum, description, targetmeanannual
# Wet/dry probabilities: p_wet_afterwet,p_wet_afterdry: probability of wet day after dry or after wet day
# Weibull parameters for rain in rainy days: phat: 0- scale (mm/day), 1- shape
# PET parameters: pet_avg_par,pet_std,pet_dad_avg,pet_dad_std,pet_daw_avg,pet_daw_std

run_description = 'run01'  # short, extreme, stroms

yearnum = 1000 # number of years to simulate
targetmeanannual = 1000 # mm/year

p_wet_afterwet_par = 0.5 # Probability of a wet day after wet day
p_wet_afterdry_par = 0.2 # Probability of a wet day after dry day
weibull_par = [15,0.5] # Weibull parameters: 0- scale parameter mm/day, 1- shape parameter
pet_avg_par = 3 # Average daily PET mm/day
pet_std_par = 1 # Std daily PET mm/day



run_description = 'run02Long' # long storms, moderate

yearnum = 1000*25 # number of years to simulate
targetmeanannual = 1000 # mm/year

p_wet_afterwet_par = 0.8 # Probability of a wet day after wet day
p_wet_afterdry_par = 0.8 # Probability of a wet day after dry day
weibull_par = [15,1.5] # Weibull parameters: 0- scale parameter mm/day, 1- shape parameter
pet_avg_par = 3 # Average daily PET mm/day
pet_std_par = 1 # Std daily PET mm/day


rd = np.random.random((yearnum*365))
sim_wet = np.zeros((yearnum*365))
sim_precip = np.zeros((yearnum*365))
sim_pet = np.zeros((yearnum*365))

sim_wet[0] = (rd[0] < p_wet_afterdry_par)
for k in range(0,yearnum*365):
  if ((sim_wet[k-1]==True and rd[k] < p_wet_afterwet_par) or (sim_wet[k-1]==False and rd[k] < p_wet_afterdry_par)):
    sim_wet[k] = True

numofwetdays = sum(sim_wet)/yearnum;
# Adjust phat[0] to have the required mean annual, no change is shape parameter
meanwetday = targetmeanannual / numofwetdays;
weibull_par[0] = meanwetday / gamma(1+1/weibull_par[1]);

for k in range(0,yearnum*365):
  if sim_wet[k]==True:
    sim_precip[k] = weibull_min.rvs(weibull_par[1],scale=weibull_par[0])
  sim_pet[k] = norm.rvs(loc=pet_avg_par, scale=pet_std_par)
sim_pet[sim_pet<0] = 0

def synthetic_uh(dt, k, n, tr, L):
    # each channel segment responses with n linear reservoirs with coeffient k[s^-1] (S=kQ)
    # and a translation of tr (int) seconds, L maximal timesteps (dt)
    trdt = round(tr / dt)
    UH = np.zeros(L)
    for i in range(0, L - trdt):
        UH[i + trdt] = 1 / (k * gamma(n)) * (((i) * dt / k) ** (n - 1)) * np.exp(-((i) * dt / k))
    return UH

# DREAM parameters
par_ts = 0.46
par_tfc = 0.32
par_tpwp = 0.17
par_z = 300
par_mue = 0.002
par_beta = 0.95

# Unit hydrograph parameters
UHL = 20
dt = 3600*24
UHk = 1 * 3600 * 24 #  Bigger n bigger basin
UHn = 3  #  Bigger n bigger basin
UHtr = 0 # Offset in time - 1 -

precip = sim_precip
PET = sim_pet

theta0 = par_tfc

runoff = np.zeros_like(precip)
recharge = np.zeros_like(precip)
aet = np.zeros_like(precip)

theta = theta0
for ind in range(0,len(precip)):
  p = precip[ind]
  pet = PET[ind]
  theta = theta + p / par_z

  if theta > par_ts:
    runoff[ind] = np.max([(theta-par_ts)*par_z , 0])
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


UH = synthetic_uh(dt, UHk, UHn, UHtr, UHL)
h = np.convolve(runoff, UH * dt, mode='full')
routed_runoff = h[0:len(runoff)]


#plt.plot(precip,'b')
plt.plot(runoff,'g')
plt.plot(routed_runoff,'c')
plt.xlabel('Day')
plt.ylabel('Precipitation / runoff (mm/day)')
plt.show()


file_name = "inputs_" + run_description+'.csv'
df = pd.DataFrame(data=np.array([precip,PET]).T, columns=['Rain(mm)','PET(mm)'])
df.to_csv(file_name)

file_name = "lables_" + run_description+'.csv'
df = pd.DataFrame(data=np.array([routed_runoff]).T)
df.to_csv(file_name, sep=' ', header=False, index=False)

