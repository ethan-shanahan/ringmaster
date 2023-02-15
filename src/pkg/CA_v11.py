from tempfile import TemporaryDirectory as TD
from time import time_ns, process_time
from dataclasses import dataclass
import numpy as np


@dataclass
class Series:
    '''Class for containing time-series data produced by CellularAutomaton.'''
    time : list = []
    energy : list = []
    size : list = []
    mass : list = []

    def record(self, **data):
        pass


class CellularAutomaton():
    '''Does exactly what it says on the tin.'''
    __acceptable_options = {
        'stable_states'         : int,
        'dimensions'            : tuple,
        'initial_condition'     : str,
        'boundary_condition'    : str,
        'perturbation_scheme'   : str,
        'update_rule'           : str,
        'activity_setting'      : str
    }
    def __init__(
        self,
        identifier : int,
        samples : int,
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
        self.identifier = identifier
        self.samples = samples
        self.output = output  # ! figure out the output types

        for k, v in options.items(): 
            self.__setattr__(k, v)
            assert type(getattr(self, k)) == CellularAutomaton.__acceptable_options[k], f'{k} is not of type {CellularAutomaton.__acceptable_options[k]}'

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
        
        self.rule(set_threshold=True)
        self.seed = time_ns()
        self.rng = np.random.default_rng(self.seed)
        self.ic()
        self.fg = self.pg.copy()
        self.mask = np.zeros(shape=self.dim, dtype=np.int8)
        self.state, self.time, self.energy, self.size, self.mass = 0, 0, 0, 0, 0
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
        
        for i in indices: self.pg[i] = 0
        
    def initial_float_0_1(self) -> None:
        self.bc(initial_flag=True)
        self.pg = self.rng.random(size=self.dim, dtype=np.float64)
        self.bc()

    def perturbation_global_maximise(self) -> tuple[set[tuple[int]],int]:
        maximum = np.amax(self.pg)
        self.pg += 1 - maximum
        self.fg += 1 - maximum
        pset = set(map(tuple, np.transpose(np.nonzero(self.pg>self.threshold))))
        area = len(pset)
        return pset, area

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
        pass




if __name__ == "__main__":
    d = {
        'stable_states': 1,
        'dimensions': (1,1),
        'initial_condition': 'i',
        'boundary_condition': 'b',
        'perturbation_scheme': 'p',
        'update_rule': 'u',
        'activity_setting': 'total'
        }
    c = CellularAutomaton(1, 2, 'bare', **d)

