# vXX
from pkg import CA_v08 as ca # vYY
from pkg import VIS_v04 as vis # vZZ
from pkg import utils
import os, sys, configparser
import numpy as np
import time

# TODO: make a class with methods various outputting procedures


class MyParser(configparser.ConfigParser):
    def __init__(self) -> None:
        super().__init__(
            converters={'tuple': lambda s: tuple(int(a) for a in s.split(','))},
            interpolation=configparser.ExtendedInterpolation()
        )
        self.read(utils.get_root() + '\config.ini')
        utils.dual_print('The following presets were found:', *self.sections(), sep='    ', end='\n\n')
        self.preset = input('Please enter the name of the preset you would like to use,'
                              ' or enter none to use the default settings.\t\t| ') or 'DEFAULT'
    
    def as_dict(self) -> dict:
        if self.preset == 'DEFAULT': d = self.defaults()
        else: d = dict.fromkeys(self.options(self.preset))
        utils.dual_print(f'\nUsing {self.preset}:')
        for o in d:
            utils.dual_print('\t', o, ' = ', self[self.preset][o])
            if ',' in self[self.preset][o]:
                d[o] = self[self.preset].gettuple(o)
            elif all(char.isnumeric() for char in self[self.preset][o]):
                d[o] = self[self.preset].getint(o)
            else:
                d[o] = self[self.preset].get(o)
        return d


class Application():
    def __init__(self, machine_count) -> None:
        self.machines = {}
        for n in range(machine_count): # each machine should have different config requirements
            c = MyParser().as_dict()
            utils.dual_print(f'{n+1}/{machine_count} configurations stored.\n')
            self.machines[f'm{n}'] = {f'config{n}': c}
            for s in range(c['samples']):
                self.machines[f'm{n}'][f'auto{s}'] = ca.CellularAutomaton((s, c['samples']), **c)
        self.total_transient_processing_time = 0
        self.total_stable_processing_time = 0
    
    def execute(self):
        '''self.machines[f'm{n}'][f'results{n}'][f'sample{s}'] == dict of state-data pairs'''
        for n, m in enumerate(self.machines.values()):
            transient_time, stable_time, predicted_duration, predicted_remaining = 0, 0, 0, 0
            m[f'results{n}'] = {}
            samples = m[f'config{n}']['samples']
            for s in range(samples):
                m[f'results{n}'][f'sample{s}'] = m[f'auto{s}'].run(predicted_remaining)
                transient_time += m[f'auto{s}'].comp_time['transient']
                stable_time    += m[f'auto{s}'].comp_time['stable']
                if predicted_duration == 0: predicted_duration = (m[f'auto{s}'].comp_time['transient'] + m[f'auto{s}'].comp_time['stable']) * samples
                predicted_remaining = round(predicted_duration-(transient_time + stable_time)) # TODO running average
                # utils.dual_print('*'*50 + f'{s+1} of {samples} samples computed', end='', flush=True)
                # if s+1 != samples: utils.dual_print(f', estimated time remaining: {round(predicted_duration-(transient_time + stable_time))}s' + '*'*50 + '\n')
                # else: utils.dual_print('*'*76 + '\n')
            predicted_duration = (transient_time + stable_time)*len(self.machines)
            utils.dual_print(f'\n{n+1}/{len(self.machines)} machines executed', end='', flush=True)
            if n+1 != len(self.machines): utils.dual_print(f', estimated time remaining: {round(predicted_duration-(transient_time + stable_time))}\n')
            else: utils.dual_print('\n')
            self.total_transient_processing_time += transient_time
            self.total_stable_processing_time += stable_time
    
    def data_plotter(self, data, output_path, art_type, save):
        plotter = vis.Visualiser(data, output_path, art_type, save)
        plotter.artist()


class Manipulator():
    def __init__(self, machines: dict, machine_id: int, attribute: str, state_type: str) -> None:
        self.machines = machines
        self.machine_id = machine_id
        self.attribute = attribute
        self.state_type = state_type

    def data_extractor(self, *args, data=None):
        if data is None: data = self.machines
        for arg in args:
            extractor = getattr(self, f'{arg}')
            data = extractor(data)
        return data

    # * Time Series Manipulations
    def event_time_series(self, data: dict) -> dict[str, np.ndarray]:
        events = {}
        m = data[f'm{self.machine_id}']
        samples = m[f'config{self.machine_id}']['samples']
        for s in range(samples):
            events[f's{s}'] = np.empty((2,0), dtype=np.int8)
            ex_time = 0
            for state in m[f'results{self.machine_id}'][f'sample{s}']:
                if state == 'transient' and self.state_type == 'transient':
                    time = m[f'results{self.machine_id}'][f'sample{s}'][state]['data']['time']
                    attr = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'][self.attribute]
                    events[f's{s}'] = np.concatenate((events[f's{s}'], np.array([time, attr])), axis=1)
                    break
                elif state == 'transient' and self.state_type != 'transient':
                    continue
                else:
                    time = m[f'results{self.machine_id}'][f'sample{s}'][state]['data']['time'] + ex_time
                    ex_time = time.iloc[-1]
                    attr = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'][self.attribute]
                    events[f's{s}'] = np.concatenate((events[f's{s}'], np.array([time, attr])), axis=1)
        return events


    def perturbation_time_series(self, data: dict) -> dict[str, list]: 
        # ? is averaging this series equivalent to averaging the many hists
        # ! NO
        '''
        return dictionary of sample-list pairs,
        each list contains a series of data points extracted from the end of each perturbation
        '''
        pert = {}
        m = data[f'm{self.machine_id}']
        samples = m[f'config{self.machine_id}']['samples']
        for s in range(samples):
            pert[f'sample{s}'] = []
            for state in m[f'results{self.machine_id}'][f'sample{s}']:
                if state == 'transient' and self.state_type == 'transient':
                    d = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'].iloc[-1][self.attribute]
                    pert[f'sample{s}'].append(d)
                    break
                elif state == 'transient' and self.state_type != 'transient':
                    continue
                else:
                    d = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'].iloc[-1][self.attribute]
                    pert[f'sample{s}'].append(d)
        # print(pert['sample0'], '\n\n')
        return pert
    
    def perturbation_time_series_averaged(self, data: dict) -> list:
        m = data[f'm{self.machine_id}']
        samples = m[f'config{self.machine_id}']['samples']
        summation = 0
        collection = []
        for s in range(samples):
            sample = []
            for state in m[f'results{self.machine_id}'][f'sample{s}']:
                if state == 'transient' and self.state_type == 'transient':
                    d = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'].iloc[-1][self.attribute]
                    sample.append(d)
                    break
                elif state == 'transient' and self.state_type != 'transient':
                    continue
                else:
                    d = m[f'results{self.machine_id}'][f'sample{s}'][state]['data'].iloc[-1][self.attribute]
                    sample.append(d)
            collection.append(sample)
        max_domain = max([max(sample) for sample in collection])
        summation = 0
        for sample in collection:
            size = len(sample)
            if size < max_domain:
                sample.append([0 for _ in range(max_domain - size)])
            summation += np.array(sample)
        perts = list(summation/len(collection))
        print(perts)
        # perts = [sum(x)/len(collection) for x in zip(*collection)]
        return perts

    # * Histogram Manipulations
    def linear_bins_histogram_averaged(self, data: dict[str, list]) -> np.ndarray:
        f = lambda x: list(range(1, max(x)+2))
        domains = [max(series) for series in data.values()]
        summation, count = 0, 0
        for series in data.values():
            series = series.copy()
            series.sort()
            hist_tuple_temp = np.histogram(series, bins=f(series))
            hist_tuple_temp = (hist_tuple_temp[1][:-1], hist_tuple_temp[0])
            hist_array_temp = np.array(hist_tuple_temp)
            if (padding:=max(domains) - max(series)) == 0:
                scale = hist_array_temp[0]
            hist_array_temp = np.pad(hist_array_temp, ((0,0),(0,padding)))
            summation += hist_array_temp
            count += 1

        linear_histogram = summation/count
        linear_histogram[0] = scale
        return linear_histogram

    def linear_bins_histograms(self, data: dict[str, list]) -> list[np.ndarray]:
        f = lambda x: list(range(1, max(x)+2))
        domains = [max(series) for series in data.values()]
        linbin_histlist = []
        for series in data.values():
            series = series.copy()
            series.sort()
            hist_tuple_temp = np.histogram(series, bins=f(series))
            hist_tuple_temp = (hist_tuple_temp[1][:-1], hist_tuple_temp[0])
            hist_array_temp = np.array(hist_tuple_temp)
            if (padding:=max(domains) - max(series)) == 0:
                scale = hist_array_temp[0]
            else:
                hist_array_temp = np.pad(hist_array_temp, ((0,0),(0,padding)))
            linbin_histlist.append(hist_array_temp)
        for hist in linbin_histlist:
            hist[0] = scale
        return linbin_histlist
    
    def log_bins_histograms(self, data: dict[str, list]) -> list[np.ndarray]:
        base = 2
        # f = lambda x: [(max(x)*((1-base)/(1-(base**N))))*(base**n) for n in range(N)]
        f = lambda x: np.logspace(start=np.emath.logn(base, x[0]), stop=np.emath.logn(base, x[-1]), num=N, base=base)
        linbin_histlist = []
        for series in data.values():
            series = series.copy()
            series.sort()
            N = int(np.emath.logn(base, 1-max(series)*(base-1)*(1-base)))
            print('0 - ', series[0], '| -1 - ', series[-1])
            print(f(series), '\n')
            hist_tuple_temp = np.histogram(series, bins=f(series))
            hist_tuple_temp = (hist_tuple_temp[1][:-1], hist_tuple_temp[0])
            hist_array_temp = np.array(hist_tuple_temp)
            linbin_histlist.append(hist_array_temp)
        domain_lengths = [len(hist[0]) for hist in linbin_histlist]
        for hist in linbin_histlist:
            if (padding:=max(domain_lengths) - len(hist[0])) == 0:
                max_domain = max(hist[0])
                break
        for n, hist in enumerate(linbin_histlist):
            if (padding:=max(domain_lengths) - len(hist[0])) != 0:
                print(max(domain_lengths), '-', len(hist[0]), '=', padding)
                filler = np.zeros((2,padding), dtype=np.int8)
                filler[0] = np.linspace(max(hist[0]), max_domain, padding)
                linbin_histlist[n] = np.concatenate(hist, filler, axis=1)
        # for hist in linbin_histlist:
        #     hist[0] = scale
        return linbin_histlist

    def log_log_histogram(self, data: dict[str, list]) -> np.ndarray:
        linear_histogram = self.linear_bins_histograms(data)
        linear_histogram = self.average_histogram(linear_histogram)
        log_log_histogram = np.ma.log10(linear_histogram).filled(-1)
        try:
            clean_domain = np.where(log_log_histogram[1] == -1)[0][0]
        except IndexError:
            clean_domain = None
        log_log_histogram = log_log_histogram[:,:clean_domain]
        return log_log_histogram

    def average_histogram(self, data: list[np.ndarray]) -> np.ndarray:
        summation, count = 0, 0
        for h in data:
            summation += h
            count += 1
        avg_hist = summation/count
        return avg_hist
    
    # * Domain Manipulations
    def linear_domains():
        pass

    def log_domains(self, data: np.ndarray) -> list[np.ndarray]: # only maps back to linear bins of width one
        full_domain = 10 ** data[0][-1]
        p = 0.5
        num_domains = 10
        domains = []
        for i in range(num_domains):
            a_domain = int(10**((1-i*((1-p)/num_domains))*np.log10(full_domain)))
            domains.append([np.log10(x) for x in range(1, a_domain+1)])
        hist_list = []
        for dom in domains:
            short_data = data[:,:len(dom)]
            hist_list.append(short_data)
        return hist_list

    # * Fitting Manipulations
    def poly1_fit_single(self, data: np.ndarray) -> np.ndarray:
        poly1 = np.polynomial.Polynomial.fit(data[0], data[1], 1)
        xs = [data[0][0], data[0][-1]]
        return np.array([xs, [poly1(x) for x in xs]])

    def poly1_fit_many(self, data: list[np.ndarray]) -> list[np.ndarray]:
        fits = []
        for d in data:
            fits.append(self.poly1_fit_single(d))
        return fits
    
    def poly1_fit_trend(self, data: list[np.ndarray]) -> np.ndarray:
        fits = np.empty((2,0), dtype=np.int8)
        max_dom = max([d[0][-1] for d in data])
        for d in data:
            poly1 = np.polynomial.Polynomial.fit(d[0], d[1], 1)
            tau = poly1.convert().coef[1]
            dom = d[0][-1]
            fits = np.concatenate((fits, [[max_dom-dom], [tau]]), axis=1) # plots domain reduction vs poly1 slope
        return fits
    
if __name__ == '__main__':
    output_dir = utils.mk_output_dir()
    log_path = output_dir+r'\log.txt'

    # utils.UtilFileLike(log_path)
    utils.u_open(log_path)

    distinct_machines = 1 # does not work for more than one currently, because of outputting?
    app = Application(machine_count=distinct_machines)
    app.execute()

    stable_size_m0_data =  Manipulator(app.machines, 0, 'size', 'stable')

    # data0 = stable_size_m0_data.data_extractor('event_time_series')
    # data1 = stable_size_m0_data.data_extractor('perturbation_time_series')
    # data2 = stable_size_m0_data.data_extractor('linear_bins_histogram', data=data1)
    # data3 = stable_size_m0_data.data_extractor('log_log_histogram', data=data1)
    # data4 = stable_size_m0_data.data_extractor('log_domains', data=data3)
    # data5 = stable_size_m0_data.data_extractor('poly1_fit_many', data=data4)
    # data6 = stable_size_m0_data.data_extractor('poly1_fit_trend', data=data4)

    # app.data_plotter(data0, output_path=output_dir+r'\size_fig0.png', art_type='graph', save=True)
    # app.data_plotter(data1, output_path=output_dir+r'\size_fig1.png', art_type='graph', save=True)
    # app.data_plotter(data2, output_path=output_dir+r'\size_fig2.png', art_type='graph', save=True)
    # app.data_plotter(data3, output_path=output_dir+r'\size_fig3.png', art_type='graph', save=True)
    # app.data_plotter(data4, output_path=output_dir+r'\size_fig4.png', art_type='graph', save=True)
    # app.data_plotter(data5, output_path=output_dir+r'\size_fig5.png', art_type='graph', save=True)
    # app.data_plotter([data3, data5], output_path=output_dir+r'\size_fig6.png', art_type='graph', save=True)
    # app.data_plotter(data6, output_path=output_dir+r'\size_fig7.png', art_type='graph', save=True)

    data1 = stable_size_m0_data.data_extractor('perturbation_time_series')
    data2 = stable_size_m0_data.data_extractor('linear_bins_histograms', 'average_histogram', data=data1)
    data3 = stable_size_m0_data.data_extractor('log_bins_histograms', data=data1)
    data4 = stable_size_m0_data.data_extractor('average_histogram', data=data3)
    data5 = stable_size_m0_data.data_extractor('log_log_histogram', data=data1)
    data0 = {}
    for k, v in data1.items():
        data0[k] = v.copy()
        data0[k].sort()
    data6 = data1['sample0']
    data7 = data0['sample0']
    data8 = stable_size_m0_data.data_extractor('linear_bins_histograms', data=data1)[0]
    
    with open(r'D:\GitHub\ringmaster\test\data\eg_histograms.npz', 'wb') as quick_out:
        np.savez(quick_out, perturbation_time_series=data6, perturbation_time_series_sorted=data7, linear_bins_histogram_avg=data2, linear_bins_histogram=data8, log_log_histogram=data5)

    app.data_plotter(data0, output_path=output_dir+r'\size_fig0.png', art_type='graph', save=True)
    app.data_plotter(data1, output_path=output_dir+r'\size_fig1.png', art_type='graph', save=True)
    app.data_plotter(data2, output_path=output_dir+r'\size_fig2.png', art_type='graph', save=True)
    app.data_plotter(data3, output_path=output_dir+r'\size_fig3.png', art_type='graph', save=True)
    app.data_plotter(data4, output_path=output_dir+r'\size_fig4.png', art_type='graph', save=True)
    app.data_plotter(data5, output_path=output_dir+r'\size_fig5.png', art_type='graph', save=True)
    app.data_plotter(data6, output_path=output_dir+r'\size_fig6.png', art_type='graph', save=True)
    app.data_plotter(data7, output_path=output_dir+r'\size_fig7.png', art_type='graph', save=True)
    app.data_plotter(data8, output_path=output_dir+r'\size_fig8.png', art_type='graph', save=True)

    utils.u_close()
