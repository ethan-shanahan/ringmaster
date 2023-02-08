import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as mplp

with open(r'D:\GitHub\ringmaster\test\data\sample_hist_Ys.txt', 'r') as quick_in:
    sample_Ys = list(map(float, quick_in.read().split(r',')))


Xs = np.linspace(1,1787,1787)

pareto = lambda x, a, s: (a * (s ** a)) / (x ** (a + 1))
modified_pareto = lambda x, a, s, l, C, m: C * (np.exp(-(x + m)/l)) * (a * (s ** a)) / (x ** (a + 1))
simplified_modified_pareto = lambda x, a, l, C: C * (np.exp(-(x)/l)) * (a * (1 ** a)) / (x ** (a + 1))
mod_pow = lambda x, a, l, C: C * np.exp(-(x)/l, dtype=np.float64) * np.power((x), -a, dtype=np.float64)

Y_pareto =          pareto(Xs, 0.1, 1)
Y_pareto_extreme =  pareto(Xs, 0.1, 100)
Y_modified_pareto = modified_pareto(Xs, 0.1, 1, 10, 1, 0)
Y_modified_pareto_extreme = modified_pareto(Xs, 0.001, 1, 300, 150, -100)
params, _ = curve_fit(mod_pow, Xs, sample_Ys, p0=(1, 1, 1)); print(params)
y_fit = mod_pow(Xs, *params)
# y_notfit = mod_pow(Xs, 1, 500, 1)

fig1 = mplp.figure(tight_layout=True)
ax1 = fig1.add_subplot(111)

ax1.plot(Xs, Y_pareto, '-b', label='pareto')
ax1.plot(Xs, Y_pareto_extreme, '-g', label='pareto_extreme')
# ax.plot(Xs, Y_modified_pareto, '-y', label='modified_pareto')
ax1.plot(Xs[:-787], Y_modified_pareto_extreme[:-787], '-y', label='modified_pareto_extreme')
ax1.plot(Xs[:-787], sample_Ys[:-787], '.k')
ax1.plot(Xs[:-787], y_fit[:-787], '--m', label='fitted')
# ax1.plot(Xs[:-787], y_notfit[:-787], '--r', label='not fitted')

mplp.grid(); mplp.legend(); mplp.xscale('log'); mplp.yscale('log')

mplp.show()