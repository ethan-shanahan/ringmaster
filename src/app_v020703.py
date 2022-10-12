# vXX
from pkg import CA_v07 as ca # vYY
from pkg import VIS_v03 as vis # vZZ
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplp

machines = {}
ask_samples = int(input("How many samples should be gathered? [10]\t\t| ") or '10')
ask_defaults = input("Use default values?\n\ndimensions=\t\t(10,10)\nboundary condition=\tcliff\ninitial condition=\tmax30min10\nupdate rule=\t\tASM\nperturbation scheme=\trandom1causal\nactivity=\t\tproactionary\n\n([y]/n)\t\t\t\t") or 'y'
if ask_defaults == 'y':
    for i in range(ask_samples):
        machines[f'auto{i:03d}'] = ca.CellularAutomaton(dimensions=(10,10), 
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
    for i in range(ask_samples):
        machines[f'auto{i:03d}'] = ca.CellularAutomaton(dimensions=user_dimensions, 
                                                        boundary_condition=user_boundary_condition,
                                                        initial_condition=user_initial_condition,
                                                        update_rule=user_rule, 
                                                        perturbation_scheme=user_scheme,
                                                        activity=user_activity)

user_states = int(input("\nHow many stable states should be found? [10]\t\t\t| ") or "10")

results = {}
transient_processing_time = 0
stable_processing_time = 0
for i, auto in enumerate(machines):
    results[f'result{i:03d}'] = machines[auto].run(user_states) # dictionary of machines results as dictionaries of states
    transient_processing_time += machines[auto].comp_time['transient']
    stable_processing_time += machines[auto].comp_time['stable']
total_processing_time = transient_processing_time + stable_processing_time

print( "\n",
      f"Total Processing Time: {total_processing_time}",
      f"Transient Processing Time: {transient_processing_time}",
      f"Stable Processing Time: {stable_processing_time}",
       "\n",
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

sizes = {}
for i, result in enumerate(results):
    sizes[f'result{i:03d}'] = []
    for t, state in enumerate(results[result]):
        if state == 'transient':
            continue
        a_size = results[result][state]['data'].iloc[-1]['Size']
        # print(a_size)
        sizes[f'result{i:03d}'].append(a_size)
        # print(states[f'state{i}'])
        sizes[f'result{i:03d}'].sort()
# print('Sizes: ', sizes)

print(sizes)
fig, axes = mplp.subplots(1, 2)

lengths = [max(sizes[res]) for res in sizes]
max_size = max(lengths)
# print(lengths)
print(max_size)
# for i, sizes in enumerate(states):
#     if len(states[sizes]) == max_len:
#         max_index = i

histograms = {}
for i, res in enumerate(sizes):
    hist_tup = np.histogram(sizes[res], bins=range(1, max(sizes[res])+2))
    hist_tup_proper = (hist_tup[1][:-1], hist_tup[0])
    histograms[f'hist{i:03d}'] = np.array(hist_tup_proper)

    padding = max_size - max(sizes[res])
    print(padding)
    if padding > 0:
        print(histograms[f'hist{i:03d}'])
        histograms[f'hist{i:03d}'] = np.pad(histograms[f'hist{i:03d}'], ((0,0),(0,padding)))
        print(histograms[f'hist{i:03d}'])
    elif padding == 0:
        scale = histograms[f'hist{i:03d}'][0,:]

summation = 0
count = 0
for hist in histograms:
    summation += histograms[hist]
    count += 1
avg = summation/count
avg[0] = scale
print('\n\n', avg)

axes[0].plot(avg[0], avg[1])

log10_avg = np.ma.log10(avg).filled(-1)
cut = np.where(log10_avg[1] == -1)[0][0]
log10_avg = log10_avg[:,:cut]

axes[1].plot(log10_avg[0], log10_avg[1])

p = np.polynomial.Polynomial.fit(log10_avg[0], log10_avg[1], 1)
print('\u03B1 = ', p.convert().coef[1]*-1)

xs = np.linspace(log10_avg[0][0], log10_avg[0][-1], 2)
axes[1].plot(xs, [p(x) for x in xs], linestyle='dashed')

mplp.show()

# for fig in figures:
#     fig.artist()
print('\nFINISHED\n') 
# !!! Compile many temp files into fewer temp files!
# * check output/011
