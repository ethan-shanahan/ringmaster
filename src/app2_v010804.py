from pkg import CA_v08 as ca # vYY
from pkg import VIS_v04 as vis # vZZ
from pkg import utils
import configparser
import numpy as np


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


class Machine():
    def __init__(self, config) -> None:
        print(f'Machine Activated\n') # utils.dual_print
        self.config = config
        self.autos = 1
        for s in range(config['samples']):
            self.machines[f'm'][f'auto{s}'] = ca.CellularAutomaton((s, c['samples']), **c)
        self.total_transient_processing_time = 0
        self.total_stable_processing_time = 0


if __name__ == '__main__':
    c = MyParser().as_dict()