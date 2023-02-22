# Application - version 12
import pkg.utilities as u  # Utilities
from pkg.CA_v12 import CellularAutomaton as CA  # Cellular Automaton - version 12
from pkg.DW_v02 import DataWrangler as DW  # Data Wrangler - version 02
# from pkg import VI_v11 as vi  # Visual Interface - version 11
from matplotlib import pyplot as mplp


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
        if data == None:
            pass
        elif model == None:
            dw = DW(data); dw.wrangle(resolution)
            return dw.results
        else:
            dw = DW(data); dw.wrangle(resolution); dw.fitter(model)
            return dw.results
    
    def grapher(self, data : dict):
        fig = mplp.figure(figsize=[8,6], layout='constrained')
        ax = fig.add_subplot()
        for k, v in data.items():
            if k == 'fit': ax.plot(v[0], v[1], '.k', label=k)
            if k == 'fit': ax.plot(v[0], v[1], 'r', label=k)
            if k == 'no control': ax.plot(v[0], v[1], '.b', label=k)
            if k == 'with control': ax.plot(v[0], v[1], '.r', label=k)
        ax.grid(); ax.legend()
        ax.set_xscale('log'); ax.set_xlabel('x')
        ax.set_yscale('log'); ax.set_ylabel('Pr(x)')



if __name__ == '__main__':
    m = Machine(m_id='A', output='stable_perturbation_series')
    data = m.extract('size')
    # print(f'{data=}')
    # results = m.generate_results(data, resolution=0, model='modified_power_law')
    results1 = m.generate_results(data, resolution=0, model=None)#; m.grapher(results1)
    results2 = m.generate_results(data, resolution=-20, model=None)#; m.grapher(results2)
    # print(f'{results=}')

    m = Machine(m_id='A', output='stable_perturbation_series')
    data = m.extract('size')
    results3 = m.generate_results(data, resolution=0, model=None)#; m.grapher(results3)
    results4 = m.generate_results(data, resolution=-20, model=None)#; m.grapher(results4)

    combi_lin = {
        'no control': next(iter(results1.values())),
        'with control': next(iter(results3.values()))
    }; m.grapher(combi_lin)
    combi_log = {
        'no control': next(iter(results2.values())),
        'with control': next(iter(results4.values()))
    }; m.grapher(combi_log)
    
    mplp.show()