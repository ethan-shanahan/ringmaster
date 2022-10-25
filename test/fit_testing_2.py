import numpy as np
import matplotlib.pyplot as mplp

def average_histogram(data: list[np.ndarray]) -> np.ndarray:
    summation, count = 0, 0
    for h in data:
        summation += h
        count += 1
    avg_hist = summation/count
    return avg_hist

eg_hist = np.load('test\data\eg_histograms.npz')
pert_series = eg_hist['perturbation_time_series']
pert_sorted = eg_hist['perturbation_time_series_sorted']
lin_hist = eg_hist['linear_bins_histogram']
log_hist = eg_hist['log_log_histogram']

s_min = 1
s_max = 25*25
# temp_lin = np.linspace(1, len(pert_series), len(pert_series), dtype='i2')
# proper_pert_series = np.rec.array([z for z in zip(temp_lin,pert_series)], dtype=[('time','i2'),('sizes','i2')])
# proper_sorted = np.sort(proper_pert_series, order='sizes')
# reg_series = np.array(pert_series)
proper_pert_series = np.core.records.fromrecords(pert_series, dtype=[('sizes','i2')])
proper_sorted = np.sort(proper_pert_series, order='sizes')

# histofreq, freq_edges = np.histogram(proper_pert_series.sizes, bins=s_max-1, range=(s_min,s_max))
histofreq, freq_edges = np.histogram(proper_pert_series.sizes, bins=proper_pert_series.sizes.max()-1)

# norm_pert_series = proper_pert_series.sizes / s_max
normed_freq = histofreq / np.sum(histofreq)
hergarten_cumulative = 1 - np.cumsum(normed_freq)

norm_edges = freq_edges / s_max
# norm_edges = freq_edges / freq_edges.max()

histoprob, prob_edges = np.histogram(proper_pert_series.sizes, bins=s_max-1, range=(s_min/s_max,1))

# print(type(histogram), type(gram_edges))
# print(len(histogram), len(gram_edges))
# print(histogram[-6:], gram_edges[-6:], gram_edges[-6:])

proper_histofreq = np.core.records.fromarrays([histofreq, freq_edges[:-1]], dtype=[('freq','i2'), ('sizes','i2')])
proper_histoprob = np.core.records.fromarrays([histoprob, prob_edges[:-1]], dtype=[('prob','i2'), ('sizes','i2')])
new_normed = np.core.records.fromarrays([hergarten_cumulative, norm_edges[:-1]], dtype=[('norm','f2'), ('sizes','f2')])

mplp.figure()

mplp.subplot(231)
mplp.plot('sizes', 'freq', data=proper_histofreq)
# mplp.plot(norm_pert_series)

mplp.subplot(232)
mplp.plot('sizes', 'norm', data=new_normed)
# mplp.plot(proper_sorted)

mplp.subplot(233)

mplp.subplot(234)
mplp.plot('sizes', 'freq', data=proper_histofreq)
# mplp.plot(np.log10(proper_histofreq.sizes), np.log10(proper_histofreq.freq))
# mplp.yscale('log')
# mplp.xscale('log')

mplp.subplot(235)
mplp.plot('sizes', 'norm', 'r.', data=new_normed)
mplp.yscale('log')
mplp.xscale('log')

# mplp.axis([s_min/s_max, 1, new_normed.norm.min(), 1])
mplp.subplot(236)


mplp.show()


