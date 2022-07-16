from DREAM.dream import Dream
from SAC_SMA.sac_sma import SacSma
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dream_sim = Dream(param_file="Dream/Par.csv")
    dream_sim.read_input("Dream/Data/synth_data_newtest2.csv")
    dream_sim.simulate()
    print(np.sum(dream_sim.output_dict['runoff']) / np.sum(dream_sim.input_dict['precip']))
    dream_sim.plot()
    dream_sim.out_to_csv("Dream/rundir/output_data_newtest2.csv")

    dream_sim.read_input("Dream/Data/synth_data_newtest1.csv")
    dream_sim.simulate()
    print(np.sum(dream_sim.output_dict['runoff']) / np.sum(dream_sim.input_dict['precip']))
    dream_sim.plot()
    dream_sim.out_to_csv("Dream/rundir/output_data_newtest1.csv")

    """
    sac = SacSma(param_file="SAC_SMA\\C\\par_update.in")
    sac.read_input("Data/Data.csv")
    sac.simulate()
    #sac.plot()

    fig, ax1 = plt.subplots()
    Q = dream_sim.output_dict["runoff"]
    x = range(len(Q))
    color = 'tab:red'
    ax1.set_xlabel('days')
    ax1.set_ylabel('DREAMS', color=color)
    ax1.plot(x, Q, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    Q = sac.output_dict["runoff"]
    x = range(len(Q))
    color = 'tab:blue'
    ax2.set_ylabel('SAC-SMA', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, Q/5.0, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    """






