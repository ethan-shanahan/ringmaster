# vXX
from pkg import CA_v06 as ca # vYY
from pkg import VIS_v02 as vis # vZZ
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplp

ask_defaults = input("Use default values?\n\ndimensions=\t\t(10,10)\nboundary condition=\tcliff\ninitial condition=\tmax30min10\nupdate rule=\t\tASM\nperturbation scheme=\trandom1causal\nactivity=\t\tproactionary\n\n([y]/n)\t\t\t\t") or 'y'
if ask_defaults == 'y':
    machine = ca.CellularAutomaton(dimensions=(10,10), 
                                   boundary_condition="cliff",
                                   initial_condition="max30min10",
                                   update_rule="ASM", 
                                   perturbation_scheme="random1causal",
                                   activity="proactionary")
else:
    user_dimensions = tuple(int(a) for a in (input(
        "What dimensions should be used? [10,10]\t\t\t\t| ") or "10,10").split(","))
    user_boundary_condition = input(
        "What boundary condition should be used? [cliff]\t\t\t| ") or "cliff"
    user_initial_condition = input(
        "What initial conditions should be applied? [max30min10]\t\t| ") or "max30min10"
    user_rule = input(
        "What update rule should be implemented? [ASM]\t\t\t| ") or "ASM"
    user_scheme = input(
        "What perturbation scheme should be performed? [random1causal]\t| ") or "random1causal"
    user_activity = input(
        "What type of activity should be considered? [proactionary]\t| ") or "proactionary"
    machine = ca.CellularAutomaton(dimensions=user_dimensions, 
                                   boundary_condition=user_boundary_condition,
                                   initial_condition=user_initial_condition,
                                   update_rule=user_rule, 
                                   perturbation_scheme=user_scheme,
                                   activity=user_activity)

user_states = int(input("\nHow many stable states should be found? [1]\t\t\t| ") or "1")

results = machine.run(user_states)
print(f"\n\nTotal Processing Time: {machine.comp_time['transient']+machine.comp_time['stable']}",
      f"Transient Processing Time: {machine.comp_time['transient']}",
      f"Stable Processing Time: {machine.comp_time['stable']}\n",
      sep='\n')

x = 1
while True:
    try:
        os.mkdir(output_directory:=r'output/'+f'{x:03d}'+r'/')
        print(f'\nOutputting to:\t\t\t{output_directory}\n')
        break
    except:
        x += 1

# figures = [vis.Visualiser(results['stable_1']['data'], f'{output_directory}stable_1_data.png', 'graphs'),
#            vis.Visualiser(results['stable_1']['grid'], f'{output_directory}stable_1_grid.png', 'image'),
#            vis.Visualiser(results['stable_1']['mask'], f'{output_directory}stable_1_mask.png', 'image'),
#            vis.Visualiser(results['stable_1']['grids'], f'{output_directory}stable_1_grids.gif', 'movie'),
#            vis.Visualiser(results['stable_1']['masks'], f'{output_directory}stable_1_masks.gif', 'movie')]

# figures = []
# for state in results:
#     figures.append(vis.Visualiser(results[state]['data'], f'{output_directory}{state}_data.png', 'graphs'))

sizes = []
for t, state in enumerate(results):
    if state == 'transient':
        continue
    sizes.append(results[state]['data'].iloc[-1]['Size'])

sizes.sort()
# print('Sizes: ', sizes)


fig, axes = mplp.subplots(1, 2)

hist_tup = np.histogram(sizes, bins=range(1, max(sizes)+2))
hist_tup_proper = (hist_tup[1][:-1], hist_tup[0])
hist_array = np.array(hist_tup_proper)
axes[0].plot(hist_array[0], hist_array[1])

log10_hist_array = np.ma.log10(hist_array).filled(-1)
cut = np.where(log10_hist_array[1] == -1)[0][0]
log10_hist_array = log10_hist_array[:,:cut]

axes[1].plot(log10_hist_array[0], log10_hist_array[1])

p = np.polynomial.Polynomial.fit(log10_hist_array[0], log10_hist_array[1], 1)
print('\u03B1 = ', p.convert().coef[1]*-1)

xs = np.linspace(log10_hist_array[0][0], log10_hist_array[0][-1], 2)
axes[1].plot(xs, [p(x) for x in xs], linestyle='dashed')

mplp.show()

# for fig in figures:
#     fig.artist()
print('\nFINISHED\n')
