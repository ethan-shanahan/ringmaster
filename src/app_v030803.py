# vXX
from pkg import CA_v08 as ca # vYY
from pkg import VIS_v04 as vis # vZZ
import os, configparser
import numpy as np

# TODO: make a class with methods various outputting procedures

root = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))

def mk_output_dir() -> str:
    x = 1
    while True:
        try:
            os.mkdir(output_dir:=root + '\output' + f'\{x:03d}')
            print(f'\nOutputting to:\t\t\t{output_dir}\n')
            break
        except:
            x += 1
    return output_dir


class MyParser(configparser.ConfigParser):
    def __init__(self) -> None:
        super().__init__(
            converters={'tuple': lambda s: tuple(int(a) for a in s.split(','))},
            interpolation=configparser.ExtendedInterpolation()
        )
        self.read(root + '\src\config.ini')
        print('\nThe following presets were found:', *self.sections(), sep='    ', end='\n\n')
        self.preset = input('Please enter the name of the preset you would like to use,'
                              ' or enter none to use the default settings.\t\t| ') or 'DEFAULT'
    
    def as_dict(self) -> dict:
        if self.preset == 'DEFAULT': d = self.defaults()
        else: d = dict.fromkeys(self.options(self.preset))
        for o in d:
            print(o, ' = ', self[self.preset][o])
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
            print(f'{n+1}/{machine_count} configurations stored.\n')
            self.machines[f'm{n}'] = {f'config{n}': c}
            for s in range(c['samples']):
                self.machines[f'm{n}'][f'auto{s}'] = ca.CellularAutomaton(**c)
        self.total_transient_processing_time = 0
        self.total_stable_processing_time = 0
    
    def execute(self):
        '''self.machines[f'm{n}'][f'results{n}'][f'sample{s}'] == dict of state-data pairs'''
        for n, m in enumerate(self.machines.values()):
            transient_time, stable_time, predicted_duration = 0, 0, 0
            m[f'results{n}'] = {}
            samples = m[f'config{n}']['samples']
            for s in range(samples):
                m[f'results{n}'][f'sample{s}'] = m[f'auto{s}'].run()
                transient_time += m[f'auto{s}'].comp_time['transient']
                stable_time    += m[f'auto{s}'].comp_time['stable']
                if predicted_duration == 0: predicted_duration = (m[f'auto{s}'].comp_time['transient'] + m[f'auto{s}'].comp_time['stable']) * samples
                print(f'{s+1} of {samples} samples computed. ', end='', flush=True)
                if s+1 != samples: print(f'Estimated time remaining: {predicted_duration-(transient_time + stable_time)}')
                else: print()
            predicted_duration = (transient_time + stable_time)*len(self.machines)
            print(f'{n+1} of {len(self.machines)} machines executed. ', end='', flush=True)
            if n+1 != len(self.machines): print(f'Estimated time remaining: {predicted_duration-(transient_time + stable_time)}')
            else: print()
            self.total_transient_processing_time += transient_time
            self.total_stable_processing_time += stable_time
    
    def data_extractor(self, machine_n: int, attribute: str, state_type: str, style1: str, style2: str = None):
        extractor = getattr(self, f'{style1}_style')
        return extractor(machine_n, attribute, state_type, style2)

    # * Time Series Styles
    def event_time_series_style():
        pass

    def perturbation_time_series_style(self, machine_n: int, attribute: str, state_type: str):
        '''
        return dictionary of sample-list pairs
        each list contains a series of data points extracted from the end of each perturbation
        '''
        data = {}
        m = self.machines[f'm{machine_n}']
        samples = m[f'config{machine_n}']['samples']
        for s in range(samples):
            data[f'sample{s}'] = []
            for state in m[f'results{machine_n}'][f'sample{s}']:
                if state == 'transient' and state_type == 'transient':
                    d = m[f'results{machine_n}'][f'sample{s}'][state]['data'].iloc[-1][attribute]
                    data[f'sample{s}'].append(d)
                    break
                elif state == 'transient' and state_type != 'transient':
                    continue
                else:
                    d = m[f'results{machine_n}'][f'sample{s}'][state]['data'].iloc[-1][attribute]
                    data[f'sample{s}'].append(d)
        return data
    
    # * Histogram Styles
    def general_histogram_style(self, sub_style):
        sub_style = getattr(self, f'{sub_style}_histogram_style')
        pass

    def linear_histogram_style(self, machine_n: int, attribute: str, state_type: str, style2: str):
        f = lambda x: list(range(1, max(x)+2))
        extractor = getattr(self, f'{style2}_style')
        data = extractor(machine_n, attribute, state_type)
        domains = [max(series) for series in data.values()]
        summation, count = 0, 0
        for series in data.values():
            series.sort()
            hist_tuple_temp = np.histogram(series, bins=f(series))
            hist_tuple_temp = (hist_tuple_temp[1][:-1], hist_tuple_temp[0])
            hist_array_temp = np.array(hist_tuple_temp)
            if (padding:=max(domains) - max(series)) == 0:
                scale = hist_array_temp[0]
            hist_array_temp = np.pad(hist_array_temp, ((0,0),(0,padding)))
            summation += hist_array_temp
            count += 1

        histogram = summation/count
        histogram[0] = scale
        return histogram
    
    def log_log_histogram_style(self, machine_n: int, attribute: str, state_type: str, style2: str):
        linear_histogram = self.linear_histogram_style(machine_n, attribute, state_type, style2)
        log_log_histogram = np.ma.log10(linear_histogram).filled(-1)
        try:
            clean_domain = np.where(log_log_histogram[1] == -1)[0][0]
        except IndexError:
            clean_domain = None
        log_log_histogram = log_log_histogram[:,:clean_domain]
        return log_log_histogram

    def log_bins_histogram_style():
        f = lambda x: x
        bins = [f(x) for x in range(0)]
        pass

    def linear_histogram_staggered_style():
        pass

    def log_histogram_staggered_style():
        pass

    def data_plotter(self, data, output_path, art_type, save):
        plotter = vis.Visualiser(data, output_path, art_type, save)
        plotter.artist()

    
# ? include config section on output/data_manip options
if __name__ == '__main__':
    distinct_machines = 1
    app = Application(machine_count=distinct_machines)
    app.execute()
    data1 = app.data_extractor(0, 'size', 'stable', 'linear_histogram', 'perturbation_time_series')
    data2 = app.data_extractor(0, 'size', 'stable', 'log_log_histogram', 'perturbation_time_series')
    output_dir = mk_output_dir()
    app.data_plotter(data1, output_dir+r'\fig1.png', 'graph', True)
    app.data_plotter(data2, output_dir+r'\fig2.png', 'graph', True)
