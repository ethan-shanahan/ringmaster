# Application - version 11
from pkg import CA_v08 as ca # Cellular Automaton - version 08
from pkg import DW_v01 as dw # Data Wrangler - version 01
from pkg import VI_v11 as vi # Visual Interface - version 11
import utilities as u # Utilities
import configparser
import matplotlib.pyplot as mplp


class MyParser(configparser.ConfigParser):
    def __init__(self) -> None:
        super().__init__(
            converters={'tuple': lambda s: tuple(int(a) for a in s.split(','))},
            interpolation=configparser.ExtendedInterpolation()
        )
        self.read(u.get_src() + '\config.ini')
        u.uprint('The following presets were found:', *self.sections(), sep='    ', end='\n\n')
        self.preset = input('Please enter the name of the preset you would like to use,'
                              ' or enter none to use the default settings.\t\t| ') or 'DEFAULT'
    
    def as_dict(self) -> dict:
        if self.preset == 'DEFAULT': d = self.defaults()
        else: d = dict.fromkeys(self.options(self.preset))
        u.uprint(f'\nUsing {self.preset}:')
        for o in d:
            u.uprint('\t', o, ' = ', self[self.preset][o])
            if ',' in self[self.preset][o]:
                d[o] = self[self.preset].gettuple(o)
            elif all(char.isnumeric() for char in self[self.preset][o]):
                d[o] = self[self.preset].getint(o)
            else:
                d[o] = self[self.preset].get(o)
        u.uprint('\n')
        return d


class Machine():
    def __init__(self, m_id:str) -> None:
        self.m_id = m_id
        self.config = MyParser().as_dict()
        self.autos = [ca.CellularAutomaton((s, self.config['samples']), **self.config) for s in range(self.config['samples'])]
        self.results = []
        self.total_transient_processing_time = 0
        self.total_stable_processing_time = 0
        u.uprint(f'Machine Activated\n')
    
    def execute(self):
        u.uprint(f'Executing Machine: {self.m_id}\n')
        self.results.extend([self.autos[i].run() for i in range(self.config['samples'])])
    
    def extract(self, attribute:str = 'size', state:str = 'stable', form:str = 'data') -> list:
        '''
        Return the perturbation time series.
        The attribute of each complete avalanche is appended
            to a sorted list.
        '''
        if state == 'stable' and form == 'data':
            series = []
            for r in self.results:
                for s in range(1, 1+self.config['desired_stable_states']):
                    series.append(r[f'stable_{s}']['data'].iloc[-1][attribute])
            return sorted(series)


if __name__ == '__main__':  # ! samples should be averaged, not appended. Appending is equivalent to running 1 simulation for extra long. The point of samples is to average the results.
    new = False; save = False

    if new == True:
        Ma = Machine('Ma')
        Ma.execute()
        series = Ma.extract(attribute='size')
    else:
        with open(r'test\data\sample_OFC_series.txt', 'r') as quick_in:
            series = list(map(float, quick_in.read().split(r',')))
            print('Loaded series from storage...\n')
    if save == True:
        with open(r'D:\GitHub\ringmaster\test\data\sample_OFC_series.txt', 'w') as quick_out:
            quick_out.write(r','.join(map(str,series)))

    Da = dw.DataWrangler(series)
    linbin_hist = Da.linbin_hist(resolution=int(max(series)))
    logbin_hist = Da.logbin_hist()
    Da.data = linbin_hist
    mod_pow = Da.fitter_nonlinear('modified_power_law')

    # Va = vi.VisualInterface(grid_dim=(2,2))
    # Va.graph(linbin_hist[0], linbin_hist[1], scale='linear', loc=(0,0))
    # Va.graph(linbin_hist[0], linbin_hist[1], scale='log',    loc=(0,1))
    # Va.graph(mod_pow[0],  mod_pow[1],  scale='linear', loc=(1,0))
    # Va.graph(mod_pow[0],  mod_pow[1],  scale='log',    loc=(1,1))

    # Va.show()

    fig, ax = mplp.subplots()
    ax.plot(linbin_hist[0], linbin_hist[1], '.k', mod_pow[0],  mod_pow[1], '--b')
    ax.plot(logbin_hist[0], logbin_hist[1], '-r')

    ax.set_xscale('log'); ax.set_xlabel('Size')
    ax.set_yscale('log'); ax.set_ylabel('Pr(Size)')
    ax.grid()
    mplp.show()