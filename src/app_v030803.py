# vXX
from pkg import CA_v08 as ca # vYY
from pkg import VIS_v03 as vis # vZZ
import os, configparser

# TODO: make a class with methods various outputting procedures

root = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))

def mk_output_dir():
    x = 1
    while True:
        try:
            os.mkdir(output_dir:=root + '\output' + f'\{x:03d}')
            print(f'\nOutputting to:\t\t\t{output_dir}\n')
            break
        except:
            x += 1


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
            d[o] = self[self.preset].get(o)
            try:
                d[o] = self[self.preset].getint(o)
            except:
                pass
            try:
                d[o] = self[self.preset].gettuple(o)
            except:
                pass
        return d


class Application():
    def __init__(self, machine_count) -> None:
        self.machines = {}
        for n in range(machine_count): # each machine should have different config requirements
            c = MyParser().as_dict()
            print(f'Config {n} of {machine_count} stored...')
            self.machines[f'm{n}'] = {f'config{n}': c, f'auto{n}': ca.CellularAutomaton(**c)}
        self.total_transient_processing_time = 0
        self.total_stable_processing_time = 0
    
    def execute(self):
        '''self.machines[f'm{n}'][f'results{n}'][f'sample{s}'] == dict of state-data pairs'''
        for n, m in enumerate(self.machines.values()):
            transient_time, stable_time, predicted_duration = 0, 0, 0
            m[f'results{n}'] = {}
            samples = m[f'config{n}']['samples']
            for s in range(samples):
                m[f'results{n}'][f'sample{s}']  = m[f'auto{n}'].run()
                transient_time += m[f'auto{n}'].comp_time['transient']
                stable_time    += m[f'auto{n}'].comp_time['stable']
                if predicted_duration == 0: predicted_duration = (m[f'auto{n}'].comp_time['transient'] + m[f'auto{n}'].comp_time['stable']) * samples
                print(f'{s+1} of {samples} samples computed. ', end='', flush=True)
                if s+1 != samples: print(f'Estimated time remaining: {predicted_duration-(transient_time + stable_time)}')
                else: print()
            predicted_duration = (transient_time + stable_time)*len(self.machines)
            print(f'{n+1} of {len(self.machines)} machines executed. ', end='', flush=True)
            if n+1 != len(self.machines): print(f'Estimated time remaining: {predicted_duration-(transient_time + stable_time)}')
            else: print()
            self.total_transient_processing_time += transient_time
            self.total_stable_processing_time += stable_time
    
    def data_extractor(self, target):
        pass

    def data_grapher(self, target):
        pass
    

if __name__ == '__main__':
    distinct_machines = 1
    app = Application(machine_count=distinct_machines)
    app.execute()
    data = app.data_extractor(target='size')
    app.data_grapher(target=data)
