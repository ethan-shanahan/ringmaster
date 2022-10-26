import numpy as np
import matplotlib.pyplot as mplp
from matplotlib.ticker import FuncFormatter, MultipleLocator

eg_hist = np.load('test\data\eg_histograms.npz')
pert_series = np.core.records.fromrecords(eg_hist['perturbation_time_series'], dtype=[('sizes','i4')])

x = np.arange(1, 101)
logx = np.log10(x)

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
def mk_correlation(x, C, a, L, w, g, log=False):
    if log: return np.log10(C * mk_pwr_law(x,a) * mk_exp_mx(x,L,w,g))
    else: return C * mk_pwr_law(x,a) * mk_exp_mx(x,L,w,g)

fig, axes = mplp.subplots()

# 1,3
def set_scale_sequence():
    C=1; a=1; L=25; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=axes[0,0]; A.plot(x, mk_pwr_law(x,a), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Power Law')
    A=axes[0,1]; A.plot(x, mk_exp_mx(x,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Exponential Decay')
    A=axes[0,2]; A.plot(x, mk_correlation(x,C,a,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Pareto')

    C=1; a=1; L=100; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=axes[0,0]; A.plot(x, mk_pwr_law(x,a), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Power Law'); A.legend()
    A=axes[0,1]; A.plot(x, mk_exp_mx(x,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Exponential Decay'); A.legend()
    A=axes[0,2]; A.plot(x, mk_correlation(x,C,a,L,w,g), label=coefs); A.set_yscale('log'); A.set_xscale('log'); A.set_title('Pareto'); A.legend()
# 2,3
def log_data_sequence():
    C=1; a=1; L=50; w=0.5; g=1
    coefs = f'C={C}; a={a}; L={L}; w={w}; g={g}'
    A=axes[1,0]; A.plot(logx, mk_pwr_law(x,a,log=True)); A.set_title(f'{coefs} -- Power Law'); mimic_log(A)
    A=axes[1,1]; A.plot(logx, mk_exp_mx(x,L,w,g,log=True)); A.set_title(f'{coefs} -- Exponential Decay'); mimic_log(A)
    A=axes[1,2]; A.plot(logx, mk_correlation(x,C,a,L,w,g,log=True), label='pareto'); A.set_title(f'{coefs} -- Pareto'); mimic_log(A); A.legend()

# 0,0
def pert_series_sequence():
    A=axes; A.plot(pert_series.sizes); A.set_title(f'Perturbation Series')
def pert_sorted_sequence():
    A=axes; A.plot(np.sort(pert_series, order='sizes')); A.set_title(f'Perturbation Series')


pert_series_sequence()
mplp.show()