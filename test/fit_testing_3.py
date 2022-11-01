import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as mplp
from matplotlib.ticker import FuncFormatter, MultipleLocator

def ax_getter(i=(0,0)):
    if isinstance(i, list):
        axs = {}
        for I in i:
            try: axs[f'{I[0]},{I[1]}'] = axes[I]
            except TypeError: return axes
            except IndexError: axs[f'{I[0]},{I[1]}'] = axes[I[1]]
        return axs
    elif isinstance(i, tuple):
        try: return axes[i]
        except TypeError: return axes
        except IndexError: return axes[i[1]]
    else:
        raise TypeError

eg_hist = np.load('test\data\eg_histograms.npz')
pert_series = np.core.records.fromrecords(eg_hist['perturbation_time_series'], dtype=[('sizes','i4')])
eg_logloghist = np.core.records.fromarrays(eg_hist['log_log_histogram'], dtype=[('sizes',np.float64),('freq',np.float64)])

x = np.arange(1, 101)
logx = np.log10(x/max(x))

def mimic_log(A):
    A.xaxis.set_major_locator(MultipleLocator(1))
    A.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$'))
    A.yaxis.set_major_locator(MultipleLocator(1))
    A.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$'))

def mk_pwr_law(x, a, log=False):
    if log: return np.log10(np.power(x, -a, dtype=np.float64))
    else: return np.power(x, -a, dtype=np.float64)
def mk_exp_mx(x, L, w, g, log=False):
    if log: return np.log10(np.exp(-x/(L**w)))
    else: return np.exp((-x/(L**w))**g)
def mk_any_correlation(x, C, a, L, w, g, log=False):
    if log: return np.log10(C * mk_pwr_law(x,a) * mk_exp_mx(x,L,w,g))
    else: return C * mk_pwr_law(x,a) * mk_exp_mx(x,L,w,g)
def mk_correlation(x, C, a, l):
    return C * np.power(x, -a, dtype=np.float64) * np.exp(-x/l, dtype=np.float64)
def mk_log_correlation(x, C, a, L, w, g):
    return np.log10(C * mk_pwr_law(x,a) * mk_exp_mx(x,L,w,g))

def plot_eg_logloghist_fit(A):
    A.plot(eg_logloghist.sizes, eg_logloghist.freq, label='eg Data'); A.set_title(f'Example Data and Fitting'); mimic_log(A)
    params, pcovs = curve_fit(mk_log_correlation, eg_logloghist.sizes, eg_logloghist.freq)
    A.plot(eg_logloghist.sizes, mk_log_correlation(eg_logloghist.sizes, *params), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(params))
    A.legend()
def plot_hist_fit(A):
    f = lambda x: np.linspace(1, max(x)+1, max(x)+1)
    hist, bin_edges = np.histogram(pert_series.sizes, bins=f(pert_series.sizes))
    hist_proper = np.core.records.fromarrays([bin_edges[:-1], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    A.plot(hist_proper.bins, hist_proper.freq, label='size vs. freq'); A.set_title('Data and Fitting')
    params, pcovs = curve_fit(mk_correlation, hist_proper.bins, hist_proper.freq, p0=(1000,1,1))
    A.plot(hist_proper.bins, mk_correlation(hist_proper.bins, *params), label=f'fit: C={params[0]}, a={params[1]}, l={params[2]}')
    A.legend()
def log_plot_hist_fit(A):
    f = lambda x: np.linspace(1, max(x)+1, max(x)+1)
    hist, bin_edges = np.histogram(pert_series.sizes, bins=f(pert_series.sizes))
    hist_proper = np.core.records.fromarrays([bin_edges[:-1], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    A.plot(np.log10(hist_proper.bins), np.log10(hist_proper.freq), label='size vs. freq')
    params, pcovs = curve_fit(mk_correlation, hist_proper.bins, hist_proper.freq, p0=(1000,1,1))
    A.plot(np.log10(hist_proper.bins), np.log10(mk_correlation(hist_proper.bins, *params)), label=f'fit: C={params[0]}, a={params[1]}, l={params[2]}')
    A.legend(); A.set_title('Data and Fitting'); mimic_log(A)
def log_plot_norm_hist_fit(A):
    f = lambda x: np.linspace(0, max(x), max(x))
    hist, bin_edges = np.histogram(pert_series.sizes, bins=f(pert_series.sizes))
    hist_proper = np.core.records.fromarrays([bin_edges[1:], hist], dtype=[('bins',np.float64),('freq',np.float64)])

    x = np.log10(hist_proper.bins / max(hist_proper.bins))
    y = np.log10(hist_proper.freq / max(hist_proper.freq))
    print(y)
    A.plot(x, y, label='size vs. freq')
    
    params, pcovs = curve_fit(mk_correlation, hist_proper.bins, hist_proper.freq, p0=(1000,1,1))

    x = np.log10(hist_proper.bins / max(hist_proper.bins))
    y = np.log10(mk_correlation(hist_proper.bins, *params) / max(mk_correlation(hist_proper.bins, *params)))
    A.plot(x, y, label=f'fit: C={params[0]}, a={params[1]}, w={params[2]}')
    
    A.legend(); A.set_title('Data and Fitting'); mimic_log(A)

def set_scale_sequence(i):
    C=1; a=1; L=25; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=i[0]; A.plot(x, mk_pwr_law(x,a), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Power Law')
    A=i[1]; A.plot(x, mk_exp_mx(x,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Exponential Decay')
    A=i[2]; A.plot(x, mk_any_correlation(x,C,a,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Pareto')

    C=1; a=1; L=100; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=i[0]; A.plot(x, mk_pwr_law(x,a), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Power Law'); A.legend()
    A=i[1]; A.plot(x, mk_exp_mx(x,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Exponential Decay'); A.legend()
    A=i[2]; A.plot(x, mk_any_correlation(x,C,a,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Pareto'); A.legend()

def log_data_sequence(i):
    C=1; a=1; L=50; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=i[0]; A.plot(logx, mk_pwr_law(x,a,log=True)); A.set_title(f'{coefs} -- Power Law'); mimic_log(A)
    A=i[1]; A.plot(logx, mk_exp_mx(x,L,w,g,log=True)); A.set_title(f'{coefs} -- Exponential Decay'); mimic_log(A)
    A=i[2]; A.plot(logx, mk_any_correlation(x,C,a,L,w,g,log=True), label='pareto'); A.set_title(f'{coefs} -- Pareto'); mimic_log(A); A.legend()

def correlation(A):
    C=1; a=1; L=50; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A.plot(logx, mk_any_correlation(x,C,a,L,w,g,log=True), label='pareto'); A.set_title(f'{coefs} -- Pareto'); mimic_log(A); A.legend()

def pert_series_sequence(A):
    A.plot(pert_series.sizes); A.set_title(f'Perturbation Series')
def pert_sorted_sequence(A):
    A.plot(np.sort(pert_series, order='sizes')); A.set_title(f'Perturbation Series Sorted')
def normalise_sorted_sizes(A):
    normed = pert_series.sizes / np.max(pert_series.sizes)
    A.plot(np.sort(normed)); A.set_title('Normed Pert Sizes Sorted')
def normalise_sorted_series(A):
    norm_y = pert_series.sizes / np.max(pert_series.sizes)
    norm_x = np.linspace(0,1,len(norm_y))
    A.plot(norm_x, np.sort(norm_y)); A.set_title('Normed Pert Series Sorted')
def histogram(A):
    f = lambda x: np.linspace(1, max(x)+1, max(x)+1)
    hist, bin_edges = np.histogram(pert_series.sizes, bins=f(pert_series.sizes))
    hist_proper = np.core.records.fromarrays([bin_edges[:-1], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    print(hist_proper)
    A.plot(hist_proper.bins, hist_proper.freq); A.set_title('Frequency vs. Size')
def log_log_hist(A):
    f = lambda x: np.linspace(1, max(x)+1, max(x)+1)
    hist, bin_edges = np.histogram(pert_series.sizes, bins=f(pert_series.sizes))
    # hist_proper = np.array([bin_edges[:-1], hist])
    log_log = np.core.records.fromarrays([np.log10(bin_edges[:-1]),np.log10(hist)], dtype=[('bins',np.float64),('freq',np.float64)])
    A.plot(log_log.bins, log_log.freq); A.set_title('Log10 (Frequency vs. Size)'); mimic_log(A)
def histogram_norm_x(A):
    f = lambda x: np.linspace(0, max(x), max(x)) / max(x)
    norm_y = pert_series.sizes / np.max(pert_series.sizes)
    hist, bin_edges = np.histogram(norm_y, bins=f(pert_series.sizes))
    # hist_proper = np.core.records.fromarrays([bin_edges[:-1], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    hist_proper = np.core.records.fromarrays([bin_edges[1:], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    print(hist_proper)
    A.plot(hist_proper.bins, hist_proper.freq); A.set_title('Frequency vs. Norm Size')
def histogram_norm(A):
    f = lambda x: np.linspace(0, max(x), max(x)) / max(x)
    norm_y = pert_series.sizes / np.max(pert_series.sizes)
    hist, bin_edges = np.histogram(norm_y, bins=f(pert_series.sizes))
    hist = hist / max(hist)
    hist_proper = np.core.records.fromarrays([bin_edges[1:], hist], dtype=[('bins',np.float64),('freq',np.float64)])
    print(hist_proper)
    A.plot(hist_proper.bins, hist_proper.freq); A.set_title('Norm Frequency vs. Norm Size')
def log_log_hist_norm(A):
    f = lambda x: np.linspace(0, max(x), max(x)) / max(x)
    norm_y = pert_series.sizes / np.max(pert_series.sizes)
    hist, bin_edges = np.histogram(norm_y, bins=f(pert_series.sizes))
    hist = hist / max(hist)
    hist_proper = np.core.records.fromarrays([np.log10(bin_edges[1:]), np.log10(hist)], dtype=[('bins',np.float64),('freq',np.float64)])
    A.plot(hist_proper.bins, hist_proper.freq); A.set_title('Log10 (Norm Frequency vs. Norm Size)'); mimic_log(A)
def log_bin_hist_norm(A):
    f = lambda x: np.logspace(np.log10(min(x)/max(x)), 0, int(max(x)/10))
    norm_y = pert_series.sizes / max(pert_series.sizes)
    hist, bin_edges = np.histogram(norm_y, bins=f(pert_series.sizes))
    hist = hist / max(hist)
    print('bins: ', bin_edges[1:])
    print('freq: ', hist)
    hist_proper = np.core.records.fromarrays([np.log10(bin_edges[1:]), np.log10(hist)], dtype=[('bins',np.float64),('freq',np.float64)])
    A.plot(hist_proper.bins, hist_proper.freq); A.set_title('Log Bins (Norm Frequency vs. Norm Size)'); mimic_log(A)
    print('bins: ', hist_proper.bins)
    print('freq: ', hist_proper.freq)


# fig, axes = mplp.subplots(2,3)
# A = ax_getter([(0,0),(0,1),(0,2)])
# set_scale_sequence(list(A.values()))
# A = ax_getter([(1,0),(1,1),(1,2)])
# log_data_sequence(list(A.values()))

# fig, axes = mplp.subplots(1,2)
# A = ax_getter([(0,0),(0,1)])
# pert_series_sequence(A['0,0'])
# pert_sorted_sequence(A['0,1'])

# fig, axes = mplp.subplots(1,4)
# A = ax_getter([(0,0),(0,1),(0,2),(0,3)])
# pert_series_sequence(A['0,0'])
# pert_sorted_sequence(A['0,1'])
# normalise_sorted_sizes(A['0,2'])
# normalise_sorted_series(A['0,3'])

# fig, axes = mplp.subplots(2,2)
# A = ax_getter([(0,0),(0,1),(1,0),(1,1)])
# histogram(A['0,0'])
# log_log_hist(A['1,0'])
# histogram_norm(A['0,1'])
# log_log_hist_norm(A['1,1'])

# fig, axes = mplp.subplots(2,2)
# A = ax_getter([(0,0),(0,1),(1,0),(1,1)])
# log_log_hist(A['0,0'])
# log_log_hist_norm(A['0,1'])
# plot_hist_fit(A['1,0'])
# correlation(A['1,1'])

# fig, axes = mplp.subplots(1,2)
# A = ax_getter([(0,0),(0,1)])
# plot_hist_fit(A['0,0'])
# log_plot_norm_hist_fit(A['0,1'])

fig, axes = mplp.subplots(1,2)
A = ax_getter([(0,0),(0,1)])
log_log_hist_norm(A['0,0'])
log_bin_hist_norm(A['0,1'])

mplp.show()