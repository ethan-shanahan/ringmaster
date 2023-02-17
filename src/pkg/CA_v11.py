from tempfile import TemporaryDirectory as TD
from time import time_ns
from dataclasses import dataclass, field
import itertools as itt
import numpy as np
import utilities as u


@dataclass
class Series:
    '''Class for containing time-series data produced by CellularAutomaton.'''
    pert_time: list[int] = field(default_factory=list)
    event_time: list[int] = field(default_factory=list)
    energy: list[int] = field(default_factory=list)
    size: list[int] = field(default_factory=list)
    mass: list[int] = field(default_factory=list)

    def record(self, **data):
        pass


class CellularAutomaton():
    '''Does exactly what it says on the tin.'''
    __acceptable_options = {
        'desired_stable_states' : int,
        'dimensions'            : tuple,
        'initial_condition'     : str,
        'boundary_condition'    : str,
        'perturbation_scheme'   : str,
        'update_rule'           : str,
        'activity_setting'      : str
    }
    def __init__(
        self,
        progress_bar : object,
        output : str,
        **options
    ) -> None:
        '''
        stable_states : int,
        dimensions : tuple,
        initial_condition : str,
        boundary_condition : str,
        perturbation_scheme : str,
        update_rule : str,
        activity_setting : str,
        '''
        self.progress_bar = progress_bar        
        self.output = output  # ! figure out the output types

        for k, v in options.items(): 
            self.__setattr__(k, v)
            assert type(getattr(self, k)) == CellularAutomaton.__acceptable_options[k], f'{k} is not of type {CellularAutomaton.__acceptable_options[k]}'

        self.stable_states = self.desired_stable_states
        self.dim, self.ndim = self.dimensions, len(self.dimensions)
        self.ic = getattr(self, f"initial_{self.initial_condition}")
        self.bc = getattr(self, f"boundary_{self.boundary_condition}")
        self.pert = getattr(self, f"perturbation_{self.perturbation_scheme}")
        self.rule = getattr(self, f"rule_{self.update_rule}")
        match self.activity_setting:
            case 'proactionary' | 'reactionary' | 'total': self.activity = self.activity_setting
            case _: raise ValueError('Activity must be set to "proactionary", "reactionary", or "total".')

        for k in options.keys(): 
            self.__delattr__(k)
        
        self.progress_bar.make_bar(self.stable_states)
        self.rule((0,), set_threshold=True)
        self.seed = time_ns()
        self.rng = np.random.default_rng(self.seed)
        self.ic()
        self.mask = np.zeros(shape=self.dim, dtype=np.int8)
        self.state, self.pert_time, self.event_time, self.energy, self.size, self.mass = 0, 0, 0, 0, 0, 0
        self.comp_time = {'transient': 0, 'stable': 0}
        if output == 'all_arrays': self.tempdir = TD(dir='')
        
    
    def boundary_cliff(self, initial_flag : bool = False, final_flag : bool = False) -> None:
        if initial_flag: self.dim = tuple(x+2 for x in self.dim); return  # add padding
        if final_flag: return  #! remove padding

        indices = list(tuple(slice(0, self.dim[z]) if z != x else 0 
                             for z in range(self.ndim)) 
                       for x in range(self.ndim))
        indices.extend(list(tuple(slice(0, self.dim[z]) if z != x else -1 
                                  for z in range(self.ndim)) 
                            for x in range(self.ndim)))
        
        for i in indices: self.pg[i], self.fg[i] = 0, 0
        
    def initial_float_0_1(self) -> None:
        self.bc(initial_flag=True)
        self.pg = self.rng.random(size=self.dim, dtype=np.float64)
        self.fg = self.pg.copy()
        self.bc()

    def perturbation_global_maximise(self) -> tuple[set[tuple[int]],int]:
        maximum = np.amax(self.pg)
        self.pg += 1 - maximum
        self.fg += 1 - maximum
        search__set = set(map(tuple, np.transpose(np.nonzero(self.pg>self.threshold))))
        search_area = len(search__set)
        return search__set, search_area

    def rule_OFC(self, cell : tuple[int], set_threshold : bool = False) -> set[tuple[int]]:
        if set_threshold: self.threshold = 1; return

        if self.pg[cell] >= self.threshold:

            index_cell = dict(zip([i for i in range(self.ndim)], cell))  # eg = {0: 6, 1: 9}
            proaction =  set(tuple(index_cell[i] for i in index_cell))
            reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                              else index_cell[i]-1 if n//2 == i
                              else index_cell[i] 
                                  for i in index_cell) 
                              for n in range(self.ndim*2))
            
            for c in proaction: self.fg[c] = 0
            for c in reaction: self.fg[c] += self.pg[cell]/len(reaction)

            match self.activity:
                case 'proactionary': activity = proaction
                case 'reactionary': activity = reaction
                case 'total': activity = proaction.union(reaction)
            self.energy += len(activity)
            for i in activity: self.mask[i] = 1

            return reaction
        else: return set()

    def run(self):
        while self.state <= self.stable_states:
            if self.state == 0:
                search_pset = list(itt.product(*list(list(x for x in range(self.dim[n])) 
                                                     for n in range(self.ndim))))
                search_area = len(search_pset)
            else:
                search_pset, search_area = self.pert()
                self.progress_bar.bar_step(self.state)
            search_fset = set()
            cursor = 0
            while search_area != 0:
                search_fset.update(self.rule(search_pset[cursor]))  # execute update rule on the current search cell and add any cells to the set that needs to be searched next
                search_area -= 1; cursor += 1
                if search_area == 0:
                    search_pset = search_fset
                    search_area = len(search_pset)
                    search_fset = set()
                    cursor = 0
                    self.pg = self.fg.copy()
                    self.bc()
                    self.event_time += 1
            
            # ! Series() here

            self.state += 1







if __name__ == "__main__":
    print()
    d = {
        'desired_stable_states': 100000,
        'dimensions': (50,50),
        'initial_condition': 'float_0_1',
        'boundary_condition': 'cliff',
        'perturbation_scheme': 'global_maximise',
        'update_rule': 'OFC',
        'activity_setting': 'proactionary'
        }
    samples = 4
    p = u.ProgressBar(header='CA Test', footer='Test Complete', jobs=samples, steps=d['desired_stable_states'])
    for s in range(samples):
        c = CellularAutomaton(p, 'normal', **d)
        c.run()
    print()
