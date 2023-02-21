# Application - version 12
import pkg.utilities as u  # Utilities
from pkg.CA_v12 import CellularAutomaton as CA  # Cellular Automaton - version 12
from pkg.DW_v02 import DataWrangler as DW  # Data Wrangler - version 02
# from pkg import VI_v11 as vi  # Visual Interface - version 11
import statistics as stat


class Machine():
    _outputs = (('stable', ' transient'), ('perturbation', 'event'), ('series', 'array'))
    def __init__(self, m_id : str = '0', output : str = '') -> None:
        self.m_id = m_id
        for i, o in enumerate(output.split('_')):
            if o not in Machine._outputs[i]: raise NameError(name=o)
        self.extract = getattr(self, f'extract_{output}')
        self.autos = []
        self.config = u.parse_config()
        self.samples = self.config.pop('samples')
        p = u.ProgressBar(header=f'Activated Machine {self.m_id}', footer=f'Completed Machine {self.m_id}', entity='seed', jobs=self.samples, steps=self.config['desired_stable_states'])
        for _ in range(self.samples):
            ca = CA(p, output, **self.config); ca.run()
            self.autos.append(ca)

    def extract_stable_perturbation_series(self, *data : str) -> list[dict]:
        return [dict([(d, getattr(self.autos[s].series, d)[1:]) for d in data]) for s in range(self.samples)]

    def generate_results(self, data : list[dict] = None, resolution : int = 0, model : str = None):
        input('?')
        pass


if __name__ == '__main__':
    m = Machine(m_id='A', output='stable_perturbation_series')
    # data1 = m.extract('state', 'mass')
    data2 = m.extract('pert_time', 'size')
    print(data2)