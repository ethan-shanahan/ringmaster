# Application - version 11
import pkg.utilities as u  # Utilities
# from pkg import CA_v08 as ca  # Cellular Automaton - version 08
from pkg.CA_v11 import CellularAutomaton as CA
# from pkg import DW_v01 as dw  # Data Wrangler - version 01
# from pkg import VI_v11 as vi  # Visual Interface - version 11


class Machine():
    def __init__(self, m_id:str) -> None:
        self.m_id = m_id
        self.autos = []
        self.config = u.parse_config()
        self.samples = self.config.pop('samples')
        p = u.ProgressBar(header=f'Activated Machine {self.m_id}', footer=f'Completed Machine {self.m_id}', entity='seed', jobs=self.samples, steps=self.config['desired_stable_states'])
        for _ in range(self.samples):
            ca = CA(p, 'normal', **self.config); ca.run()
            self.autos.append(ca)


if __name__ == '__main__':
    m = Machine('A')
    print(f'{m.autos[0].series.size[:10]}')
