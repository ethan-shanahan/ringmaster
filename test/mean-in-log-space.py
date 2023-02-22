import numpy as np
from matplotlib import pyplot as mplp
np.set_printoptions(precision=3, linewidth=200)

start = 1; end = 64; base = 2
log = lambda x, b: np.log(x) / np.log(b)
def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())

lstart = log(start, base); lend = log(end, base); num = np.floor(log(end, base)).astype(int)
print(f'{lstart=}'); print(f'{lend=}'); print(f'{num=}'); 
logs = np.logspace(lstart, lend, num, base=base); print(f'{logs=}')
gmlogs = np.asarray(list(map(geo_mean_overflow, zip(logs[1:], logs[:-1])))); print(f'{gmlogs=}')

Ys = np.logspace(len(logs), 1, len(logs), base=2)
mYs = np.asarray(list(map(np.mean, zip(Ys[1:], Ys[:-1]))))

fig = mplp.figure(figsize=[10,10]); ax = fig.add_axes([0.05,0.05,0.95,0.95])
ax.plot(logs, Ys, 'db', label='logs')
ax.plot(gmlogs, mYs, 'dr', label='gmlogs')

ax.set_xscale('log'); ax.set_yscale('log')
mplp.show()