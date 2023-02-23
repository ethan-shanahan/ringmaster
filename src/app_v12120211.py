# Application - version 12
import pkg.utilities as u  # Utilities
from pkg.CA_v12 import CellularAutomaton as CA  # Cellular Automaton - version 12
from pkg.DW_v02 import DataWrangler as DW  # Data Wrangler - version 02
# from pkg import VI_v11 as vi  # Visual Interface - version 11
import numpy as np
from matplotlib import pyplot as mplp
from statistics import mean, stdev


class Machine():
    # _outputs = (('stable', ' transient'), ('perturbation', 'event'), ('series', 'array'))
    _outputs = (('perturbation', 'event'), ('series', 'array'))
    def __init__(self, m_id : str = '0', output : str = '', seed : int = None) -> None:
        self.m_id = m_id
        for i, o in enumerate(output.split('_')):
            if o not in Machine._outputs[i]: raise NameError(name=o)
        self.extract = getattr(self, f'extract_{output}')
        self.autos = []
        self.config = u.parse_config()
        self.samples = self.config.pop('samples')
        p = u.ProgressBar(header=f'Activated Machine {self.m_id}', footer=f'Completed Machine {self.m_id}', entity='seed', jobs=self.samples, steps=self.config['desired_stable_states'])
        for _ in range(self.samples):
            ca = CA(p, output, seed=seed, **self.config); ca.run()
            self.autos.append(ca)

    def extract_perturbation_series(self, *data : str, begin : int = 1) -> list[dict]:
        return [dict([(d, getattr(self.autos[s].series, d)[begin:]) for d in data]) for s in range(self.samples)]

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
            if k == 'no control - linear bins': ax.plot(v[0], v[1], '.b', label=k)
            if k == 'with control - linear bins': ax.plot(v[0], v[1], '.r', label=k)
            if k == 'no control - log bins': ax.plot(v[0], v[1], 'o-b', label=k)
            if k == 'with control - log bins': ax.plot(v[0], v[1], 'o-r', label=k)
        ax.grid(); ax.legend()
        ax.set_xscale('log'); ax.set_xlabel('x')
        ax.set_yscale('log'); ax.set_ylabel('Pr(x)')



if __name__ == '__main__':
    # TODO: Allow manual seed specification
    def moving_avg(iterable : list, window_size : int = 10000) -> list[int|float]:
        return np.convolve(iterable, np.ones(window_size), 'valid') / window_size
    # skip = 50000
    skip = 1

    m = Machine(m_id='A', output='perturbation_series')
    A_masses = next(iter(m.extract('mass', begin=skip)[0].values())); print(f'{mean(A_masses)=}'); print(f'{stdev(A_masses)=}')
    A_mavg_masses = moving_avg(next(iter(m.extract('mass', begin=skip)[0].values()))); print(f'{mean(A_mavg_masses)=}'); print(f'{stdev(A_mavg_masses)=}')
    A_sizes  = next(iter(m.extract('size', begin=skip)[0].values()))
    A_states = next(iter(m.extract('state', begin=skip)[0].values()))


    data = m.extract('size', begin=skip)
    results1 = m.generate_results(data, resolution=0, model=None)#; m.grapher(results1)
    results2 = m.generate_results(data, resolution=-20, model=None)#; m.grapher(results2)

    m = Machine(m_id='B', output='perturbation_series', seed=m.autos[0].seed)
    B_masses = next(iter(m.extract('mass', begin=skip)[0].values())); print(f'{mean(B_masses)=}'); print(f'{stdev(B_masses)=}')
    B_mavg_masses = moving_avg(next(iter(m.extract('mass', begin=skip)[0].values()))); print(f'{mean(B_mavg_masses)=}'); print(f'{stdev(B_mavg_masses)=}')
    B_sizes  = next(iter(m.extract('size', begin=skip)[0].values()))
    B_states = next(iter(m.extract('state', begin=skip)[0].values()))


    data = m.extract('size', begin=skip)
    results3 = m.generate_results(data, resolution=0, model=None)#; m.grapher(results3)
    results4 = m.generate_results(data, resolution=-20, model=None)#; m.grapher(results4)

    combi_lin = {
        'no control - linear bins': next(iter(results1.values())),
        'with control - linear bins': next(iter(results3.values()))
    }; m.grapher(combi_lin)
    combi_log = {
        'no control - log bins': next(iter(results2.values())),
        'with control - log bins': next(iter(results4.values()))
    }; m.grapher(combi_log)
    
    mplp.show()

    fig = mplp.figure(figsize=[8,6], layout='constrained'); ax = fig.add_subplot()

    ax.plot(A_states, A_masses, '.-b', linewidth=1, label='masses: w/o control')
    ax.plot(A_states, [0.4+(x/10000) for x in A_sizes],  '-c', label='sizes:  w/o control')

    ax.plot(B_states, B_masses, '.-r', linewidth=1, label='masses: w/  control')
    ax.plot(B_states, [0.4+(x/10000) for x in B_sizes],  '-m', label='sizes:  w/  control')

    ax.plot([min(A_states), max(A_states)], [mean(A_masses), mean(A_masses)], ':k', label='avg mass: w/o control')
    ax.plot([min(B_states), max(B_states)], [mean(B_masses), mean(B_masses)], '--k', label='avg mass: w/  control')
    ax.plot(A_mavg_masses, ':g', linewidth=6, label='mavg_masses: w/o control')
    ax.plot(B_mavg_masses, ':y', linewidth=6, label='mavg_masses: w/o control')

    ax.grid()
    mplp.legend()
    mplp.show()
