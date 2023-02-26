import itertools as itt
from copy import copy
from dataclasses import dataclass, field, fields
from tempfile import TemporaryDirectory as TD
from time import time_ns

import numpy as np

import pkg.utilities as u


@dataclass(slots=True)
class EventTimeScale():
    parallel : list[int | float] = field(default_factory=list)
    energy   : list[int]         = field(default_factory=list)
    size     : list[int]         = field(default_factory=list)
    mass     : list[int | float] = field(default_factory=list)
    array    : list[np.ndarray]  = field(default_factory=list)
    
    def __record__(self, **data):
        for s in self.__slots__: getattr(self, s).append(data[s])


@dataclass(slots=True)
class PerturbationTimeScale():
    magnitude : list[int | float] = field(default_factory=list)
    parallel  : list[int | float] = field(default_factory=list)
    duration  : list[int]         = field(default_factory=list)
    energy    : list[int]         = field(default_factory=list)
    size      : list[int]         = field(default_factory=list)
    mass      : list[float]       = field(default_factory=list)
    array     : list[np.ndarray]  = field(default_factory=list)

    def __record__(self, **data):
        for s in self.__slots__: getattr(self, s).append(data[s])


@dataclass(slots=True)
class Series():
    state         : list[int]             = field(default_factory=list)
    natural_pert  : PerturbationTimeScale = field(default_factory=PerturbationTimeScale)
    control_pert  : PerturbationTimeScale = field(default_factory=PerturbationTimeScale)
    natural_event : list[EventTimeScale]  = field(default_factory=list)
    control_event : list[EventTimeScale]  = field(default_factory=list)

    def __post_init__(self):
        self.natural_event.append(EventTimeScale())
        self.control_event.append(EventTimeScale())

    def __record__(self, scale : str, **data):
        subseries = getattr(self, scale)
        if 'pert' in (scale := scale.split('_')): 
            subseries.__record__(**data)
            getattr(self, f'{scale[0]}_event').append(EventTimeScale())
        elif 'event' in scale:
            subseries[-1].__record__(**data)

    @staticmethod
    def __data__(cls):
        return {f.name: 0 for f in fields(cls)}


class CellularAutomaton():
    '''Does exactly what it says on the tin.'''
    __slots__ = ('progress_bar', 'output', 'transient_states', 'stable_states', 'dim', 'ndim', 'ic', 'bc', 'pert', 'control', 'rule', 'activity', 'threshold', 'seed', 'rng', 'pg', 'fg', 'mask', 'data', 'series', 'tempdir')
    _options = {
        'skip_transient_states' : int,
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
        seed : int = None,
        **options
    ) -> None:
        self.progress_bar = progress_bar        
        self.output = output  # ! figure out the output types

        for k, v in options.items():
            if k not in CellularAutomaton._options: raise NameError(name=k)
            try: CellularAutomaton._options[k](v)
            except TypeError as e: raise Exception(f'{k} cannot be converted to {CellularAutomaton._options[k]}') from e

        self.transient_states = options['skip_transient_states']                        #? arbitrary number of states to consider transient
        self.stable_states = options['desired_stable_states']                           #? arbitrary number of states to consider stable
        self.dim, self.ndim = tuple(options['dimensions']), len(options['dimensions'])  #? system dimension specifications
        self.ic = getattr(self, f'initial_{options["initial_condition"]}')              #? initial condition
        self.bc = getattr(self, f'boundary_{options["boundary_condition"]}')            #? boundary condition
        self.pert = getattr(self, f'perturbation_{options["perturbation_scheme"]}')     #? perturbation scheme
        self.control = getattr(self, f'control_{options["control_scheme"]}')            #? control strategy
        self.rule = getattr(self, f'rule_{options["update_rule"]}')                     #? update rule
        match options['activity_setting']:                                              #? definition of activity
            case 'proactive' | 'reactive' | 'active': self.activity = options['activity_setting']
            case _: raise ValueError('Activity must be set to "proactive", "reactive", or "active".')

        del options

        self.seed = time_ns() if seed == None else seed             #? seed
        self.rng = np.random.default_rng(self.seed)                 #? rng
        self.rule((0,), '', set_threshold=True)                     #? threshold
        self.ic()                                                   #? pg & fg
        self.mask = np.zeros(shape=self.dim, dtype=np.int8)         #? mask
        self.series = Series()                                      #? series
        self.data = {
            'state' : 0,                                            #? current state
            'pert'  : self.series.__data__(PerturbationTimeScale),  #? perturbation time-scale
            'event' : self.series.__data__(EventTimeScale),         #? event time-scale
        }
        if 'array' in self.output: pass  # TODO: self.tempdir = TD(dir='')

    #* INITIAL CONDITIONS
    # region
    def initial_rational_00_05(self) -> None:
        self.pg = 0.5 * self.rng.random(size=self.dim, dtype=np.float64)
        self.fg = self.pg.copy()

    def initial_rational_0_1(self) -> None:
        self.pg = self.rng.random(size=self.dim, dtype=np.float64)
        self.fg = self.pg.copy()

    def initial_rational_10_20(self) -> None:
        self.pg = (20 - 10) * self.rng.random(size=self.dim, dtype=np.float64) + 10
        self.fg = self.pg.copy()

    def initial_integer_10_30(self) -> None:
        self.pg = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        self.fg = self.pg.copy()
    # endregion
    #* BOUNDARY CONDITIONS
    # region
    def boundary_cliff(self, breach : tuple[int], magnitude : int | float, scope : set) -> None:
        '''Returns the scope excluding breach cells. Mimics material falling out of the edge of the system.'''
        return scope.difference([breach])
    # endregion
    #* PERTURBATION SCHEMES
    # region
    def perturbation_global_maximise(self) -> None:
        self.control('natural_perturbation_parallel')
        magnitude = 1 - np.amax(self.pg)
        self.data['pert']['magnitude'] += magnitude
        self.pg += magnitude
        self.fg += magnitude
    
    def perturbation_random_1(self) -> None:
        self.control('natural_perturbation_parallel')
        magnitude = 0
        while True:
            target = tuple(self.rng.integers(self.dim[n]) for n in range(self.ndim))
            self.pg[target] += 1
            self.fg[target] += 1
            magnitude += 1
            if self.pg[target] >= self.threshold: break
        self.data['pert']['magnitude'] += magnitude
    # endregion
    #* CONTROL STRATEGIES
    # region
    def control_none(self, scale : str) -> None:
        return

    def control_natural_perturbation_parallel_none(self, scale : str) -> None:
        if all([case in scale for case in ['natural', 'perturbation', 'parallel']]):
            return
    def control_control_perturbation_parallel_none(self, scale : str) -> None:
        if all([case in scale for case in ['control', 'perturbation', 'parallel']]):
            return
    def control_natural_perturbation_series_none(self, scale : str) -> None:
        if all([case in scale for case in ['natural', 'perturbation', 'series']]):
            return # impossible if natural and series
    def control_control_perturbation_series_none(self, scale : str) -> None:
        if all([case in scale for case in ['control', 'perturbation', 'series']]):
            return # inevitable if control or series
    def control_natural_event_parallel_none(self, scale : str) -> None:
        if all([case in scale for case in ['natural', 'event', 'parallel']]):
            return
    def control_control_event_parallel_none(self, scale : str) -> None:
        if all([case in scale for case in ['control', 'event', 'parallel']]):
            return
    def control_natural_event_series_none(self, scale : str) -> None:
        if all([case in scale for case in ['natural', 'event', 'series']]):
            return
    def control_control_event_series_none(self, scale : str) -> None:
        if all([case in scale for case in ['control', 'event', 'series']]):
            return

    def control_perturbation_mass_gt_critical_random_1_causal(self, scale : str) -> None:
        '''WIP'''
        if scale == 'perturbation':
            if self.series.natural_pert.mass[-1] >= 0.62:  # conservative critical mass
                x = 1.0
                while len(causal := np.argwhere(self.pg >= self.threshold * (x := x - 0.1))) == 0: continue
                target = tuple(causal[self.rng.integers(len(causal))])
                self.pg[target] += 0.1
                self.fg[target] += 0.1

    def control_control_perturbation_series_mass_lt_critical_random_1_true(self, scale : str) -> None:
        '''Control if the most recent mass is less than or equal to the uncontrolled critical mass. Intended Rule: OFC'''
        if all([case in scale for case in ['control', 'perturbation', 'series']]):
            if self.series.natural_pert.mass[-1] <= 0.54:  # non-conservative critical mass -> ~0.54 (empirical, conservation=0.8)
                target = tuple(self.rng.integers(self.dim[n]) for n in range(self.ndim))
                self.pg[target] += 1
                self.fg[target] += 1
                self.data['control_pert']['magnitude'] += 1

    def control_control_perturbation_series_mass_gt_critical_random_1_true(self, scale : str) -> None:
        '''Control if the most recent mass is greater than or equal to the uncontrolled critical mass. Intended Rule: OFC'''
        if all([case in scale for case in ['control', 'perturbation', 'series']]):
            if self.series.natural_pert.mass[-1] >= 0.54:  # non-conservative critical mass -> ~0.54 (empirical, conservation=0.8)
                target = tuple(self.rng.integers(self.dim[n]) for n in range(self.ndim))
                self.pg[target] += 1
                self.fg[target] += 1
                self.data['control_pert']['magnitude'] += 1
    # endregion
    #* UPDATE RULES
    # region
    def rule_OFC(self, cell : tuple[int], scale : str, set_threshold : bool = False) -> set[tuple[int]]:
        if set_threshold: self.threshold = 1; return
        if self.pg[cell] >= self.threshold:
            self.control(f'{scale}_event_parallel')  #!WARNING
            conservation = 0.8  # ? factor divided by 2 * ndim to determine dissipative effect
            magnitude = self.pg[cell] * (conservation/(self.ndim*2))#; print(); print(f'{self.pg[cell]=}'); print(f'{magnitude=}')
            index_cell = dict(zip([i for i in range(self.ndim)], cell))  # eg: cell = (6,9) => index_cell = {0: 6, 1: 9}
            proaction =  {tuple(index_cell[i] for i in index_cell)}
            reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                              else index_cell[i]-1 if n//2 == i
                              else index_cell[i] 
                                  for i in index_cell) 
                              for n in range(self.ndim*2))

            for c in proaction: self.fg[c] = 0
            for c in copy(reaction):
                if u.dim_check(c, self.dim, self.ndim):
                    #print(f'{c=}'); print(f'{self.fg[c]=}')
                    self.fg[c] += magnitude
                else: reaction = self.bc(breach=c, magnitude=magnitude, scope=reaction)

            match self.activity:
                case 'proactive': activity = proaction
                case 'reactive':  activity = reaction
                case 'active':    activity = proaction.union(reaction)
            for i in activity: self.mask[i] += 1

            return reaction
        else: return set()

    def rule_ABS(self, cell : tuple[int], scale : str, set_threshold : bool = False) -> set[tuple[int]]:
        if set_threshold: self.threshold = 2 * self.ndim; return
        if self.pg[cell] >= self.threshold:
            self.control(f'{scale}_event_parallel')  #!WARNING
            magnitude = 1
            index_cell = dict(zip([i for i in range(self.ndim)], cell))  # eg = {0: 6, 1: 9}
            proaction =  {tuple(index_cell[i] for i in index_cell)}
            reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                              else index_cell[i]-1 if n//2 == i
                              else index_cell[i] 
                                  for i in index_cell) 
                              for n in range(self.ndim*2))
            
            for c in proaction: self.fg[c] -= self.threshold
            for c in copy(reaction):
                if u.dim_check(c, self.dim, self.ndim): self.fg[c] += magnitude
                else: reaction = self.bc(breach=c, magnitude=magnitude, scope=reaction)

            match self.activity:
                case 'proactive': activity = proaction
                case 'reactive':  activity = reaction
                case 'active':    activity = proaction.union(reaction)
            for i in activity: self.mask[i] += 1

            return reaction
        else: return set()
    # endregion
    #* MAIN CODE
    # region
    def run(self):
        self.progress_bar.mk_bar(self.transient_states + self.stable_states, prefix=self.seed)
        while self.data['state'] < self.transient_states + self.stable_states:  #* START of perturbation time-step, ie. one state of the sample
            for scale in ('natural', 'control'):
                if self.data['state'] == 0 and scale == 'natural':
                    #? Search the entire system.
                    search_pset = set(itt.product(*list(list(x for x in range(self.dim[n])) for n in range(self.ndim))))
                    search_area = len(search_pset)
                else:
                    if scale == 'natural':
                        #? Perturb the system.
                        self.pert()
                    else:
                        #? Control the system in series with the perturbation.
                        self.control('control_perturbation_series')
                        #? Progress.
                        self.progress_bar.bar_step(self.data['state'])
                    #? Search excited cells.
                    search_pset = set(map(tuple, np.transpose(np.nonzero(self.pg>=self.threshold))))
                    search_area = len(search_pset)
                    
                search_fset = set()
                iter_pset = iter(search_pset)
                while search_area != 0:  #* START of event time-step, ie. one iteration of a relaxation event
                    search_fset.update(self.rule(next(iter_pset), scale))
                    search_area -= 1
                    if search_area == 0:
                        self.control(f'{scale}_event_series') #!WARNING: add to search
                        #? After exhausting one list of cells, restart with relaxed cells.
                        iter_pset = iter(search_fset)
                        search_area = len(search_fset)
                        search_fset = set()
                        self.pg = self.fg.copy()
                        self.data['event']['energy'] = self.mask.sum()
                        self.data['event']['size']   = np.count_nonzero(self.mask)
                        self.data['event']['mass']   = self.pg.mean()
                        if 'array' in self.output: self.data['event']['array'] = self.pg
                        if self.data['state'] >= self.transient_states and search_area != 0:
                            #? Serialise data from the current relaxation step.
                            self.series.__record__(f'{scale}_event', **self.data['event'])
                        #* END of event time-step, ie. one iteration of a relaxation event
                
                #? A steady state has now been reached!
                self.data['pert']['duration'] = len(self.series.natural_event[self.data['state']].energy)
                self.data['pert']['energy'] = self.mask.sum()
                self.data['pert']['size']   = np.count_nonzero(self.mask)
                self.data['pert']['mass']   = self.pg.mean()
                if 'array' in self.output: self.data['pert']['array'] = self.pg
                if self.data['state'] >= self.transient_states:
                    #? Serialise data from the current steady state.
                    self.series.__record__(f'{scale}_pert', **self.data['pert'])

                if scale == 'control': self.data['state'] += 1  #* END of perturbation time-step, ie. one state of the sample
                self.data['pert']  = self.series.__data__(PerturbationTimeScale)
                self.data['event'] = self.series.__data__(EventTimeScale)
                self.mask.fill(0)
    # endregion


if __name__ == "__main__":
    p = PerturbationTimeScale()
    print(f'{p.__slots__=}')
