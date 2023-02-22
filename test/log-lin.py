import numpy as np
from matplotlib import pyplot as mplp

start = 1; end = 215; base = 2
log = lambda x, b: np.log(x) / np.log(b)
linear = np.linspace(start, end, end)
logarithmic = np.logspace(log(start, base), log(end, base), np.floor(log(end, base)).astype(int), base=base)

short_lin = np.linspace(start, end, np.floor(log(end, base)).astype(int))

mlinear = np.asarray([linear[:-1],linear[1:]]).mean(axis=0)
mlogarithmic = np.asarray([logarithmic[:-1],logarithmic[1:]]).mean(axis=0)

fig = mplp.figure(figsize=[10,10]); ax = fig.add_axes([0.05,0.05,0.95,0.95])
ax.plot(linear, linear, '.c', label='lin-lin')
ax.plot(linear[1:], mlinear, '.b', label='lin-mlin')
ax.plot(mlinear, mlinear, '.g', label='mlin-mlin')

ax.plot(logarithmic, logarithmic, '.r', label='log-log')
ax.plot(logarithmic[1:], mlogarithmic, '.y', label='log-mlog')
ax.plot(mlogarithmic, mlogarithmic, '.m', label='mlog-mlog')

ax.plot(logarithmic, short_lin, 'sk', label='logs')
ax.plot(mlogarithmic, short_lin[:-1], 'dk', label='mlogs')

# ax.set_xscale('log'); ax.set_yscale('log')
mplp.legend()
mplp.show()