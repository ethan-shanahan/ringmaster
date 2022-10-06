import sys # temporary measure
import time # temporary measure
import itertools as itt
import numpy as np

# TODO: ?


class CellularAutomaton():
    '''foo bar'''

    def __init__(self,
                 dimensions: tuple,
                 boundary_condition: str,
                 initial_condition: str,
                 update_rule: str,
                 perturbation_scheme: str,
                 collapse_or_growth_or_both: str) -> None:
        self.seed = time.time_ns() # object initialisation time in nanoseconds since epoch
        self.rng = np.random.default_rng(self.seed)
        self.dim, self.ndim = dimensions, len(dimensions)
        self.bc = getattr(self, f"boundary_{boundary_condition}")
        self.ic = getattr(self, f"initial_{initial_condition}")
        self.rule = getattr(self, f"rule_{update_rule}") # bc only applies after rule? integrate
        self.pert = getattr(self, f"perturbation_{perturbation_scheme}")
        self.tech = 'c' if collapse_or_growth_or_both == 'collapse' else 'g' if collapse_or_growth_or_both == 'growth' else 'b'
        self.ic() # * creates self.pg
        self.fg = self.pg # ? global or local
        self.ig = self.pg.copy() # ! del after use
        self.mask = np.zeros_like(self.pg)
        self.perts = 0
        self.time, self.energy = 0, 0
        self.size, self.mass= self.mask.sum(), self.pg.sum()
        # ? self.data_structure = [('time', 'i4'), ('energy', 'i4'), ('size', 'i4'), ('mass', 'i4')]
        self.data = np.array([[self.time, self.energy, self.size, self.mass]])
        # ? self.transient_data = None
        # ? self.steady_data = None
        self.transient_grids = []
        self.transient_masks = []
        self.stable_grids = []
        self.stable_masks = []

# * BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff_legacy(self):
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
            # return np.pad(i_grid, 1)
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

    def initial_max3min0(self) -> np.ndarray:
        self.bc(initial_flag=True)
        initial = self.rng.integers(0, 3, size=self.dim, endpoint=True)
        return initial

    def initial_max30min10(self):
        self.bc(initial_flag=True)
        self.pg = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        self.bc()

    def initial_control(self) -> np.ndarray:
        self.bc(initial_flag=True)
        seeded = np.random.default_rng(12345)
        initial = seeded.integers(10, 30, size=self.dim, endpoint=True)
        return initial

# UPDATE RULES (all rules must manipulate self.fg, return a tuple of the cells that should be searched following an event, and increment energy if executed)

    def rule_ASM(self, cell, checking=False):
        icell = dict(zip([i for i in range(self.ndim)], cell))
        minus_cells = set(tuple(icell[i] for i in icell)) # === {cell}
        plus_cells = set(tuple(icell[i]+1 if n/2 == i 
                          else icell[i]-1 if n//2 == i 
                          else icell[i] 
                               for i in icell) 
                         for n in range(self.ndim*2))
        # affected_cells = minus_cells + plus_cells
        if self.pg[cell] >= len(plus_cells):
            if checking: return checking

            for c in minus_cells:
                self.fg[c] -= len(plus_cells)
            for c in plus_cells:
                self.fg[c] += 1

            if self.tech == 'c':
                self.energy += 1
                for y in minus_cells: self.mask[y] = 1
            elif self.tech == 'g':
                self.energy += len(plus_cells)
                for y in plus_cells: self.mask[y] = 1
            else:
                self.energy += 1 + len(plus_cells)
                for y in minus_cells: self.mask[y] = 1
                for y in plus_cells: self.mask[y] = 1

            return plus_cells
        else:
            return []

# PERTURBATION SCHEMES (all perturbations are applied while the system is stable and must return pset and area)

    def perturbation_random1(self):
        self.rule((np.unravel_index(self.pg.argmax(), self.dim)), checking=True)
        self.perturbations += 1
        while self.rule(target := tuple(self.rng.integers(0, self.dim[n]) for n in range(self.ndim)), checking=True):
            continue # ! infinite loop if no cells are causal
        print(f"Random Perturbation of Cell {target}")
        self.pg[target] += 1  # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area

    def perturbation_control(self):
        self.perturbations += 1
        target = tuple(int(self.dim[n]/2) if self.dim[n] % 2 ==
                       0 else int((self.dim[n]-1)/2) for n in range(self.ndim))
        print(f"Controlled Perturbation of Cell {target}")
        self.pg[target] += 1  # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area

