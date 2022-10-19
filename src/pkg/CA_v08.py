from tempfile import TemporaryDirectory as TD
import os
import io
import sys
import time
import itertools as itt
import zipfile
import numpy as np
import pandas as pd
from pkg.utils import ProgressBar

# TODO: ?


class CellularAutomaton():
    '''foo bar'''
    __acceptable_keys = [
        'desired_stable_states',
        'dimensions',
        'boundary_condition',
        'initial_condition',
        'update_rule',
        'perturbation_scheme',
        'activity'
    ]
    def __init__(
            self,
            *,
            desired_stable_states: int = None,
            dimensions: tuple = None,
            boundary_condition: str = None,
            initial_condition: str = None,
            update_rule: str = None,
            perturbation_scheme: str = None,
            activity: str = None,
            **kwargs
        ) -> None:
        for k in kwargs.keys():
            if k in self.__acceptable_keys:
                self.__setattr__(k, kwargs[k])
        self.seed = time.time_ns() # object initialisation time in nanoseconds since epoch
        self.rng = np.random.default_rng(self.seed)
        self.desired_stable_states = desired_stable_states
        self.dim, self.ndim = dimensions, len(dimensions)
        self.bc = getattr(self, f"boundary_{boundary_condition}")
        self.ic = getattr(self, f"initial_{initial_condition}")
        self.rule = getattr(self, f"rule_{update_rule}") # bc only applies after rule? integrate
        self.pert = getattr(self, f"perturbation_{perturbation_scheme}")
        if activity in ('proactionary', 'reactionary', 'all'): self.activity = activity
        else: raise ValueError("Activity must be set to 'proactionary', 'reactionary', or 'all'.")
        self.pg = np.empty(self.dim, dtype=np.int8)
        self.ic() # * creates self.pg
        self.fg = self.pg.copy() # ? global or local
        self.ig = self.pg.copy()
        self.threshold = 0
        self.mask = np.zeros_like(self.pg)
        self.state = 0  # * 0 == transient self.state, i == ith stable self.state
        self.time = 0
        self.series = []
        self.energy, self.size, self.mass = 0, self.mask.sum(), self.pg.sum()
        self.data = np.empty((0,3), dtype=np.int8)
        self.temp_dir = TD(dir=os.path.realpath(os.path.dirname(__file__)))
        self.comp_time = {'transient':0, 'stable':0}

# * BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff_legacy(self): # !
        indices = list(tuple(slice(0, self.dim[z]) if z != x else 0 for z in range(
            self.ndim)) for x in range(self.ndim))
        indices.extend(list(tuple(slice(
            0, self.dim[z]) if z != x else -1 for z in range(self.ndim)) for x in range(self.ndim)))
        for i in indices:
            self.pg[i] = 0
            self.fg[i] = 0

    def boundary_cliff(self, initial_flag=False) -> None: # ! remember to cut padding before visualisation
        if initial_flag:
            self.dim = tuple(x+2 for x in self.dim)
            return
        else:
            indices = list(tuple(slice(0, self.dim[z]) if z != x else 0 
                                 for z in range(self.ndim)) 
                           for x in range(self.ndim))
            indices.extend(list(tuple(slice(0, self.dim[z]) if z != x else -1 
                                      for z in range(self.ndim)) 
                                for x in range(self.ndim)))
            for i in indices:
                self.pg[i] = 0

#  INITIAL CONDITIONS (all initial conditions must describe how the grid is seeded/generated and apply the boundary condition)

    def initial_max3min0(self) -> np.ndarray: # !
        self.bc(initial_flag=True)
        initial = self.rng.integers(0, 3, size=self.dim, endpoint=True)
        return initial

    def initial_max30min10(self) -> None:
        self.bc(initial_flag=True)
        self.pg = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        self.bc()
    
    def initial_max20min10(self) -> None:
        self.bc(initial_flag=True)
        self.pg = self.rng.integers(10, 20, size=self.dim, endpoint=True)
        self.bc()

    def initial_control(self) -> np.ndarray: # !
        self.bc(initial_flag=True)
        seeded = np.random.default_rng(12345)
        initial = seeded.integers(10, 30, size=self.dim, endpoint=True)
        return initial

# UPDATE RULES (all rules must manipulate self.fg, return a tuple of the cells that should be searched following an event, and increment energy if executed)

    def rule_ASM(self, cell:tuple) -> set[tuple]|list:
        icell = dict(zip([i for i in range(self.ndim)], cell))
        proactionary_cells = {tuple(icell[i] for i in icell)} # === {cell}
        reactionary_cells = set(tuple(icell[i]+1 if n/2 == i 
                                 else icell[i]-1 if n//2 == i 
                                 else icell[i] 
                                      for i in icell) 
                                for n in range(self.ndim*2))
        active_cells = proactionary_cells.union(reactionary_cells)

        self.threshold = len(reactionary_cells)

        if self.pg[cell] >= self.threshold:

            for c in proactionary_cells:
                self.fg[c] -= len(reactionary_cells)
            for c in reactionary_cells:
                self.fg[c] += 1

            if self.activity == 'proactionary':
                self.energy += len(proactionary_cells)
                for y in proactionary_cells: self.mask[y] = 1
            elif self.activity == 'reactionary':
                self.energy += len(reactionary_cells)
                for y in reactionary_cells: self.mask[y] = 1
            elif self.activity == 'all':
                self.energy += 1 + len(active_cells)
                for y in active_cells: self.mask[y] = 1

            return reactionary_cells
        else:
            return []

# PERTURBATION SCHEMES (all perturbations are applied while the system is stable and must return pset and area)

    def perturbation_random1_legacy(self): # !
        if self.rule((np.unravel_index(self.pg.argmax(), self.dim)), checking=True):
            while self.rule(target := tuple(self.rng.integers(0, self.dim[n]) for n in range(self.ndim)), checking=True):
                continue
            print(f"Random Perturbation of Cell {target}")
            self.pg[target] += 1  # the perturbation itself
            self.fg[target] += 1
            pset = [target]
            area = len(pset)
            return pset, area

    def perturbation_random1causal(self) -> tuple[list[tuple],int]:
        x = 1
        while len(causal := np.argwhere(self.pg == self.threshold - x)) == 0:
            x += 1
            continue
        del x
        target = tuple(causal[self.rng.integers(len(causal))])
        # print(f"Random Causal Perturbation of Cell {target}")
        self.pg[target] += 1  # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area

    def perturbation_control(self): # !
        target = tuple(int(self.dim[n]/2) if self.dim[n] % 2 ==
                       0 else int((self.dim[n]-1)/2) for n in range(self.ndim))
        print(f"Controlled Perturbation of Cell {target}")
        self.pg[target] += 1  # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area


# COMPUTATIONS HENCEFORTH...

    def log(self, clear=False) -> None: # TODO rename to record, use log for a saved file of ex-terminal info
        if clear:
            self.mask = np.zeros_like(self.pg)
            self.time = 0
            self.series = []
            self.energy, self.size, self.mass = 0, self.mask.sum(), self.pg.sum()
            self.data = np.empty((0,3), dtype=np.int8)
        elif not clear:
            self.series.append(self.time)
            self.data = np.concatenate((self.data, np.array([[self.energy, self.size, self.mass]])))
            # !self.append_temp('grids', self.pg)
            # !self.append_temp('masks', self.mask)

    def append_temp(self, temp_type: str, new_data: np.ndarray) -> None:
        temp_file = self.temp_dir.name + f'\\temp_{temp_type}.npz'
        bio = io.BytesIO()
        np.save(bio, new_data)
        with zipfile.ZipFile(temp_file, 'a') as temp:
            temp.writestr(f'state{self.state}_frame{self.time}.npy', data=bio.getbuffer().tobytes())
    
    def run(self) -> dict:
        pbar = ProgressBar(self.desired_stable_states-1)
        start_time = time.process_time()
        results = {}
        while self.state <= self.desired_stable_states:
            self.log()
            if self.state == 0:
                pset = list(itt.product(*list(list(x 
                                                for x in range(self.dim[n])) 
                                            for n in range(self.ndim))))
                area = len(pset)
            else:
                pset, area = self.pert()

            fset = set()
            cursor = 0

            while area != 0:
                fset.update(self.rule(pset[cursor]))
                area -= 1
                cursor += 1
                if area == 0:
                    pset = list(fset)
                    area = len(pset)
                    fset = set()
                    cursor = 0
                    self.pg = self.fg.copy()
                    self.bc()
                    self.fg = self.pg.copy()
                    self.time += 1
                    self.size, self.mass = self.mask.sum(), self.pg.sum()
                    self.log()
                    
            if self.state == 0:
                end_transient = time.process_time()
                # print( "**********************************************************************************",
                #        "",
                #       f"Seed:\n{self.seed}",
                #        "",
                #       f"Initial Transient Grid:\n{self.ig}",
                #        "",
                #       f"The duration of the transient state was {self.time}.",
                #        "",
                #       f"Initial Stable Grid:\n{self.pg}",
                #        "",
                #        "**********************************************************************************",
                #       sep='\n')
                # del self.ig
                # input('...')
            # else:
                # print( "**********************************************************************************",
                #        "",
                #       f"Found stable state {self.state}.",
                #        "",
                #       f"Duration:{self.time}",
                #       f"Energy:{self.energy}",
                #       f"Size:{self.mask.sum()}",
                #       f"Mass:{self.pg.sum()}", 
                #        "\n",
                #       f"Resultant Stable Grid:\n{self.pg}",
                #        "\n",
                #       f"Resultant Stable Mask:\n{self.mask}",
                #        "",
                #        "**********************************************************************************",
                #       sep="\n")
                # input('...')
            self.data = pd.DataFrame(self.data, index=self.series, columns=['energy', 'size', 'mass'])
            results.update({'transient' if self.state == 0 
                      else f'stable_{self.state}':{'data':self.data.copy(), 
                                                   'grid':self.pg.copy(), 
                                                   'mask':self.mask.copy(),
                                                   'grids':self.temp_dir.name + f'\\temp_grids.npz',
                                                   'masks':self.temp_dir.name + f'\\temp_masks.npz'}})
            self.log(clear=True)
            self.state += 1
            pbar.update(self.state)
        end_stable = time.process_time()
        self.comp_time = {'transient':end_transient-start_time, 'stable':end_stable-end_transient}
        return results
            

if __name__ == "__main__":  # TODO: suppress terminal outputs option
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    user_dimensions = tuple(int(a) for a in (input(
        "What dimensions should be used? [20,20]\t\t\t\t| ") or "20,20").split(","))
    user_boundary_condition = input(
        "What boundary condition should be used? [cliff]\t\t\t| ") or "cliff"
    user_initial_condition = input(
        "What initial conditions should be applied? [max30min10]\t\t| ") or "max30min10"
    user_rule = input(
        "What update rule should be implemented? [ASM]\t\t\t| ") or "ASM"
    user_scheme = input(
        "What perturbation scheme should be performed? [random1causal]\t| ") or "random1causal"
    user_states = int(input(
        "How many stable states should be found? [1]\t\t\t| ") or "1")
    user_activity = input(
        "What type of activity should be considered? [proactionary]\t| ") or "proactionary"
    while True:
        machine = CellularAutomaton(dimensions=user_dimensions, 
                                    boundary_condition=user_boundary_condition,
                                    initial_condition=user_initial_condition,
                                    update_rule=user_rule, 
                                    perturbation_scheme=user_scheme,
                                    activity=user_activity)
        results = machine.run(user_states)
        print(f"\n\nTotal Processing Time: {machine.comp_time['transient']+machine.comp_time['stable']}",
              f"Transient Processing Time: {machine.comp_time['transient']}",
              f"Stable Processing Time: {machine.comp_time['stable']}\n")
        retry = input("Retry? [y/(n)]\t\t\t\t\t\t\t| ") or "n"
        if retry == "n":
            break
