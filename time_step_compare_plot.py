"""Plot script to compare the different timesteps for DTMC and compare to CTMC

    """

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import pickle
import numpy as np
from savepdf_tex import savepdf_tex
import util.util_funcs as uf
from datetime import timedelta

save=True
model_log = uf.load_model_log(fname=f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')
run_name = 'compare_methods_001_v2'
models_to_plot = model_log[model_log['run name'] == run_name]
fig, ax  = plt.subplots(1,1, figsize=(12,7))

for file in models_to_plot.index:     
    print(f'extracting data from model {file}', end='\r')
       
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/results/{file}.pickle', 'rb') as pickle_in:
            model = pickle.load(pickle_in)
    if model.method == 'CTMC':
        final_risk_CTMC = model.risk.iloc[-1,:].mean()
        model.plot_risk(ax=ax,lw=2, comparison='method',
                        value='CTMC')
    elif model.consts['time_step'].total_seconds()/60 == 5.0:
        continue
    else:
        model.plot_risk(ax=ax,lw=2, comparison=r'method - DTMC: \$\Delta t\$',
                        value=f'{model.consts["time_step"].total_seconds()/60}mins')
        if model.consts['time_step'].total_seconds()/60 == 10.0:
            final_risk_DTMC = model.risk.iloc[-1,:].mean()
diff = 1 - final_risk_DTMC / final_risk_CTMC
arrow_x_pos = model.risk.index[-1]
plt.ylabel(r'Infection risk', labelpad=30)

fig.autofmt_xdate(rotation=45)

ax.legend(frameon=False,labelspacing = 2)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol=r'\%', is_latex=True))

ax.tick_params(axis="x", pad=22)
ax.grid(True)
p1 = patches.FancyArrowPatch((arrow_x_pos, final_risk_DTMC),
                             (arrow_x_pos, final_risk_CTMC),
                             arrowstyle='<->', mutation_scale=20)
ax.text(arrow_x_pos+timedelta(hours=3), np.mean([final_risk_CTMC,final_risk_DTMC]), r'{:0.0f}\% difference'.format(diff*100))
ax.add_patch(p1)
# fig.autofmt_xdate(rotation=45)
# ax.set_ylabel('Infection Risk')
# ax.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.tight_layout()
if save:
    save_loc = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/figures/'
    savepdf_tex(fig=fig, fig_loc=save_loc,
                name=f'compare_stochastic_methods_and_time_steps')
else:

    plt.show()
    plt.close()
    
    