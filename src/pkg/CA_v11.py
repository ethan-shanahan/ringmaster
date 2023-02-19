import itertools as itt
from copy import copy
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory as TD
from time import time_ns, sleep

import numpy as np

import pkg.utilities as u


@dataclass(slots=True)
class Series:
    '''Class for containing time-series data produced by CellularAutomaton.'''
    state : list[int] = field(default_factory=list)
    pert_time : list[float] = field(default_factory=list)
    event_time : list[int] = field(default_factory=list)
    energy : list[int] = field(default_factory=list)
    size : list[int] = field(default_factory=list)
    mass : list[float] = field(default_factory=list)

    def record(self, **data):
        for s in self.__slots__:
            getattr(self, s).append(data[s])


class CellularAutomaton():
    '''Does exactly what it says on the tin.'''
    __slots__ = ('progress_bar', 'output', 'stable_states', 'dim', 'ndim', 'ic', 'bc', 'pert', 'control', 'rule', 'activity', 'threshold', 'seed', 'rng', 'pg', 'fg', 'mask', 'data', 'series', 'tempdir')
    _options = {
        'desired_stable_states' : int,
        'dimensions'            : tuple,
        'initial_condition'     : str,
        'boundary_condition'    : str,
        'perturbation_scheme'   : str,
        'control_scheme'        : str,
        'update_rule'           : str,
        'activity_setting'      : str
    }
    def __init__(
        self,
        progress_bar : object,
        output : str,
        **options
    ) -> None:
        self.progress_bar = progress_bar        
        self.output = output  # ! figure out the output types

        for k, v in options.items():
            if k not in CellularAutomaton._options: raise NameError(name=k)
            try: CellularAutomaton._options[k](v)
            except TypeError as e: raise Exception(f'{k} cannot be converted to {CellularAutomaton._options[k]}') from e

        self.stable_states = options['desired_stable_states']
        self.dim, self.ndim = tuple(options['dimensions']), len(options['dimensions'])
        self.ic = getattr(self, f'initial_{options["initial_condition"]}')
        self.bc = getattr(self, f'boundary_{options["boundary_condition"]}')
        self.pert = getattr(self, f'perturbation_{options["perturbation_scheme"]}')
        self.control = getattr(self, f'control_{options["control_scheme"]}')
        self.rule = getattr(self, f'rule_{options["update_rule"]}')
        match options['activity_setting']:
            case 'proactionary' | 'reactionary' | 'total': self.activity = options['activity_setting']
            case _: raise ValueError('Activity must be set to "proactionary", "reactionary", or "total".')

        del options
        
        self.rule((0,), set_threshold=True)
        self.seed = time_ns()
        self.rng = np.random.default_rng(self.seed)
        self.ic()
        self.mask = np.zeros(shape=self.dim, dtype=np.int8)
        self.data = {
            'state': 0,
            'pert_time': 0,
            'event_time': 0,
            'energy': 0,
            'size': 0,
            'mass': 0
        }
        self.series = Series()
        if 'array' in self.output: self.tempdir = TD(dir='')

    def initial_rational_10_20(self) -> None:
        self.pg = (20 - 10) * self.rng.random(size=self.dim, dtype=np.float64) + 10
        self.fg = self.pg.copy()

    def initial_integer_10_30(self) -> None:
        self.pg = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        self.fg = self.pg.copy()

    def boundary_cliff(self, breach : tuple[int], magnitude : int | float, scope : set) -> None:
        scope.remove(breach); return scope
        
    def perturbation_global_maximise(self):
        magnitude = 1 - np.amax(self.pg)
        self.data['pert_time'] += magnitude
        self.pg += magnitude
        self.fg += magnitude
    
    def perturbation_random_1(self) -> tuple[set[tuple[int]],int]:
        magnitude = 0
        while True:
            target = tuple(self.rng.integers(self.dim[n]) for n in range(self.ndim))
            self.pg[target] += 1
            self.fg[target] += 1
            magnitude += 1
            if self.pg[target] >= self.threshold: break
        search__set = {target}
        search_area = len(search__set)
        self.data['pert_time'] += magnitude
        return search__set, search_area

    def control_none(self, scale : str) -> None:
        return

    def control_perturbation_mass_gt_threshold_random_1_causal(self, scale : str) -> None:
        if scale == 'perturbation':
            if self.series.mass[-1] >= 0.62:  # conservative critical mass
                x = 1.0
                while len(causal := np.argwhere(self.pg >= self.threshold * (x := x - 0.1))) == 0: continue
                target = tuple(causal[self.rng.integers(len(causal))])
                self.pg[target] += 0.1
                self.fg[target] += 0.1

    def control_perturbation_mass_lt_threshold_random_1_true(self, scale : str) -> None:
        if scale == 'perturbation':
            if self.series.mass[-1] <= 0.54:  # non-conservative critical mass
                target = tuple(self.rng.integers(self.dim[n]) for n in range(self.ndim))
                self.pg[target] += 1
                self.fg[target] += 1
        
    def rule_OFC(self, cell : tuple[int], set_threshold : bool = False) -> set[tuple[int]]:
        if set_threshold: self.threshold = 1; return
        if self.pg[cell] >= self.threshold:
            conservation = 0.8  # ? factor divided by 2 * ndim to determine dissipative effect
            index_cell = dict(zip([i for i in range(self.ndim)], cell))  # eg = {0: 6, 1: 9}
            proaction =  {tuple(index_cell[i] for i in index_cell)}
            reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                              else index_cell[i]-1 if n//2 == i
                              else index_cell[i] 
                                  for i in index_cell) 
                              for n in range(self.ndim*2))
            
            for c in proaction: self.fg[c] = 0
            for c in copy(reaction):
                if u.dim_check(c, self.dim, self.ndim): self.fg[c] += self.pg[cell] * (conservation/len(reaction))
                else: reaction = self.bc(breach=c, magnitude=self.pg[cell]/len(reaction), scope=reaction)

            match self.activity:
                case 'proactionary': activity = proaction
                case 'reactionary': activity = reaction
                case 'total': activity = proaction.union(reaction)
            self.data['energy'] += len(activity)
            for i in activity: self.mask[i] = 1
            return reaction
        else: return set()

    def rule_ABS(self, cell : tuple[int], set_threshold : bool = False) -> set[tuple[int]]:
        if set_threshold: self.threshold = 2 * self.ndim; return
        
        if self.pg[cell] >= self.threshold:
            index_cell = dict(zip([i for i in range(self.ndim)], cell))  # eg = {0: 6, 1: 9}
            proaction =  {tuple(index_cell[i] for i in index_cell)}
            reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                              else index_cell[i]-1 if n//2 == i
                              else index_cell[i] 
                                  for i in index_cell) 
                              for n in range(self.ndim*2))
            
            for c in proaction: self.fg[c] -= self.threshold
            for c in copy(reaction):
                if u.dim_check(c, self.dim, self.ndim): self.fg[c] += 1
                else: reaction = self.bc(breach=c, magnitude=self.pg[cell]/len(reaction), scope=reaction)

            match self.activity:
                case 'proactionary': activity = proaction
                case 'reactionary': activity = reaction
                case 'total': activity = proaction.union(reaction)
            self.data['energy'] += len(activity)
            for i in activity: self.mask[i] = 1
            return reaction
        else: return set()

    def run(self):
        self.progress_bar.make_bar(self.stable_states, prefix=self.seed)
        while self.data['state'] <= self.stable_states:  # * START of perturbation time-step, ie. one state of the sample
            if self.data['state'] == 0:
                search_pset = set(itt.product(*list(list(x for x in range(self.dim[n])) 
                                                    for n in range(self.ndim))))
                search_area = len(search_pset)
            else:
                self.pert(); self.control('perturbation')
                search_pset = set(map(tuple, np.transpose(np.nonzero(self.pg>=self.threshold))))
                search_area = len(search_pset)
                self.progress_bar.bar_step(self.data['state'])
            search_fset = set()
            iter_pset = iter(search_pset)
            while search_area != 0:  # * START of event time-step, ie. one iteration of a relaxation event
                search_fset.update(self.rule(next(iter_pset)))
                search_area -= 1
                if search_area == 0:
                    iter_pset = iter(search_fset)
                    search_area = len(search_fset)
                    search_fset = set()
                    self.pg = self.fg.copy()
                    self.data['event_time'] += 1  # * END of event time-step, ie. one iteration of a relaxation event
                    
            # ? A stable state has now been reached!
            self.data['size'] = self.mask.sum()
            self.data['mass'] = self.pg.mean()

            self.series.record(**self.data)  # !!! Data for current state ending is serialised here.

            self.data['state'] += 1  # * END of perturbation time-step, ie. one state of the sample
            self.data['event_time'], self.data['energy'], self.data['size'], self.data['mass'] = 0, 0, 0, 0
            self.mask.fill(0)



if __name__ == "__main__":
    def test_CA_1():
        print()
        d = {
            'desired_stable_states': 5000,
            'dimensions': (20,20),
            'initial_condition': 'float_0_1',
            'boundary_condition': 'cliff',
            'perturbation_scheme': 'global_maximise',
            'update_rule': 'OFC',
            'activity_setting': 'proactionary'
            }
        samples = 3
        p = u.ProgressBar(header='CA Test', footer='Test Complete', jobs=samples, steps=d['desired_stable_states'])
        for s in range(samples):
            c = CellularAutomaton(p, 'normal', **d)
            c.run()
            # print(f'\nSample {s+1}/{samples} : {c.series.pert_time=}\n')
        print()
    def test_CA_2():
        print()
        d = {
            'desired_stable_states': 5000,
            'dimensions': (20,20),
            'initial_condition': 'float_0_1',
            'boundary_condition': 'cliff',
            'perturbation_scheme': 'global_maximise',
            'update_rule': 'OFC',
            'activity_setting': 'proactionary'
            }
        samples = 3
        p = u.ProgressBar(header='CA Test', footer='Test Complete', jobs=samples, steps=d['desired_stable_states'])
        autos = [CellularAutomaton(p, 'normal', **d) for _ in range(samples)]
        for c in autos:
            c.run()
            # print(f'\nSample {s+1}/{samples} : {c.series.pert_time=}\n')
        print()
    # print(f'{timeit(lambda: test_CA(), number=50)=}')
    c = test_CA_2()
    
