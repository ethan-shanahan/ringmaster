import math
# import scipy as sp
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as mplp

eg_hist = np.load('test\data\eg_histograms.npz')
lin_hist = eg_hist['arr_0']
log_hist = eg_hist['arr_1']

def twister(data):
    return -data

def trimmer(data):
    return data[:,int(len(data[1])*0.25):]

# ll_hist = trimmer(ll_hist)

f_exp1 = lambda x, a, b, c: np.power(x, -a) * np.exp(-x/b) + c
f_exp2 = lambda x, a, b, l: a * np.power(b, -x/l)
f_exp3 = lambda x, a, b, l: np.power(x, -a) * np.power(b, -x/l)
f_exp4 = lambda x, a, l: np.power(x, -a) * (1 - np.exp(-x/l))
f_line = lambda x, m, c: m * x + c

f = f_exp1 ######## ! ########

fig, axes = mplp.subplots(1,2)

def line_minus_exp(data):
    f = lambda x, m, c, A, l, v: (m * x + c) - (A * np.exp(-l * (x + v)))
    axes.plot(data[0], data[1], 'ro', markersize=5)
    try:
        param, param_cov = curve_fit(f, data[0], data[1], p0=[-1, 3, 1, -1, -1])
        print("Model coefficients:")
        print(param)
        print("Covariance of coefficients:")
        print(param_cov)
        axes.plot(data[0], f(data[0], *param), 'g--')
    except RuntimeError as err:
        print(err.__str__())

    axes.plot(data[0], f(data[0], -1, 3, 1.3, -4, -2.4), 'c-')
    

axes[0].plot(lin_hist[0], lin_hist[1])
axes[1].plot(log_hist[0], log_hist[1])

mplp.show()
fig, axes = mplp.subplots()

def exp_guesses(data):
    f = lambda x, a, b, c: np.power(x, -a) * np.exp(-x/b) + c
    axes.plot(data[0], data[1], 'ro', markersize=5)
    try:
        param, param_cov = curve_fit(f, data[0], data[1], p0=[1, -5, -1])
        print("Model coefficients:")
        print(param)
        print("Covariance of coefficients:")
        print(param_cov)
        axes.plot(data[0], f(data[0], *param), 'g--')
    except RuntimeError as err:
        print(err.__str__())

exp_guesses(log_hist)

mplp.show()
