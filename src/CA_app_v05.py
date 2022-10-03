import sys
import time
import itertools as itt
import numpy as np
from CA_vis_v01 import Visualiser as vis

# SUMMARY:
# Create an object of the CellularAutomaton class.
# The object consists of a grid (ndarray), boundary condition, initial condition, update rule, and perturbation scheme.
# The grid's dimensions must be specified, then each cell is assigned a value depending on the
#       initial condition and boundary condition. This is saved as the seed.
# The update rule is then applied to each cell of the grid until a stable configuration is reached. This is saved as the initial grid.
# Perturbations are slowly applied until one triggers an event via the update rule.
# All cells effected by the event are queued to be checked for triggering another Event.
# The system must become stable again before another perturbation is applied. This is repeated
#       for the desired number of causal perturbations.
#
# OUTPUTS:
# number of time-steps until the system first reaches a stable configuration,
# details of each perturbation,
# Duration, Final Energy, Final Size, Final Mass,
# Seed, Initial Grid, Final Grid,
# Final Differential Matrix,
# Final Differential Sum,
# Final Mask,
# CSV containing data on energy, size, and mass at each time-step
#
#! USAGE:
# run the program in an environment with Python 3.X and NumPy installed
# the user will be prompted with multiple questions
# return blank to submit the default response, given in brackets
# observe the outputs in the terminal and specified output directory
# the user will be asked if they wish to run the simulation again with the same settings
# respond no to quit and terminate the program
#
# the default responses are intended to simulate the BTW model
# any 2-dimensional grid size can be specified

# TODO: output perturbation timescale
# TODO: consider collapses and growths


class CellularAutomaton():  # TODO: general documentation
    '''Main object class...'''

    def __init__(self,
                 perturbation_scheme, update_rule,
                 boundary_condition, initial_condition,
                 dimensions) -> None:
        self.rng = np.random.default_rng()
        self.pert = getattr(self, f"perturbation_{perturbation_scheme}")
        self.rule = getattr(self, f"rule_{update_rule}")
        self.bc = getattr(self, f"boundary_{boundary_condition}")
        self.ic = getattr(self, f"initial_{initial_condition}")
        self.dim = dimensions
        self.ndim = len(self.dim)
        self.pg = self.ic()
        self.fg = self.pg.copy()
        self.bc()
        self.seed = self.pg.copy()
        self.mask = np.zeros(self.dim, dtype=np.int8)
        self.perturbations = 0
        self.time, self.energy, self.mass, self.size = 0, 0, self.pg.sum(
        ) / np.prod(self.dim), self.mask.sum()  # ! mass includes boundary
        self.data = {'time': [], 'energy': [], 'mass': [], 'size': []}
        self.pg_complied = []
        self.mask_complied = []
        self.outputting = []
        self.figures = []

# INITIAL CONDITIONS (all initial conditions must describe how the grid is seeded/generated and apply the boundary condition)

    def initial_max3min0(self):
        initial = self.rng.integers(0, 3, size=self.dim, endpoint=True)
        return initial

    def initial_max30min10(self):
        initial = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        return initial

    def initial_control(self):
        seeded = np.random.default_rng(12345)
        initial = seeded.integers(0, 3, size=self.dim, endpoint=True)
        return initial

# BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff(self):
        indices = list(tuple(slice(0, self.dim[z]) if z != x else 0 for z in range(
            self.ndim)) for x in range(self.ndim))
        indices.extend(list(tuple(slice(
            0, self.dim[z]) if z != x else -1 for z in range(self.ndim)) for x in range(self.ndim)))
        for i in indices:
            self.pg[i] = 0
            self.fg[i] = 0

# UPDATE RULES (all rules must manipulate self.fg, return a tuple of the cells that should be searched following an event, and increment energy if executed)

    def rule_ASM(self, cell, checking=False):  # TODO: use sets where possible
        icell = dict(zip([i for i in range(self.ndim)], cell))
        minus_cells = [tuple(icell[i] for i in icell)]
        plus_cells = set(tuple(icell[i]+1 if n/2 == i else icell[i]-1 if n //
                         2 == i else icell[i] for i in icell) for n in range(self.ndim*2))
        # affected_cells = minus_cells + plus_cells
        if self.pg[cell] >= len(plus_cells):
            if checking: return checking

            for c in minus_cells:
                self.fg[c] -= len(plus_cells)
            for c in plus_cells:
                self.fg[c] += 1
            self.bc()

            # self.energy += len(plus_cells)
            # if self.perturbations != 0:
            #     for y in plus_cells: self.mask[y] = 1
            self.energy += 1
            if self.perturbations != 0:
                for y in minus_cells:
                    self.mask[y] = 1

            return plus_cells
        return []

# PERTURBATION SCHEMES (all perturbations are applied while the system is stable and must return pset and area)

    def perturbation_random1(self):
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

# COMPUTATIONS HENCEFORTH...

    def run(self, desired_perturbations, output_directory):
        pset = list(itt.product(
            *list(list(x for x in range(self.dim[n])) for n in range(self.ndim))))
        area = len(pset)
        fset = set()
        while True:
            for key in iter(self.data):
                self.data[key].append(getattr(self, key))
            self.pg_complied.append(self.pg.copy())
            self.mask_complied.append(self.mask.copy())
            cursor = 0
            while area != 0:
                queue = self.rule(pset[cursor])
                fset.update(queue)
                area -= 1
                cursor += 1
                if area == 0:
                    pset = list(dict.fromkeys(fset))
                    area = len(pset)
                    fset = set()
                    cursor = 0
                    if self.energy != 0:  # ? I think this check is redundant if a causal perturbation is guaranteed. Should there also be an event counter that resets at each time-step?
                        self.pg = self.fg.copy()
                        self.time += 1
                        self.mass = self.pg.sum() / np.prod(self.dim)  # ! mass includes boundary
                        self.size = self.mask.sum()
                        for key in iter(self.data):
                            self.data[key].append(getattr(self, key))
                        self.pg_complied.append(self.pg.copy())
                        self.mask_complied.append(self.mask.copy())
            if self.perturbations == 0:
                print(f"\nA stable configuration was reached after {self.time} time-steps...\n",
                      f"Seed:\n{self.seed}", f"Initial Grid:\n{(ig:=self.pg.copy())}",
                      sep='\n')
                if not input('Return empty to generate outputs:\t\t\t'):
                    self.export(csv_file=f'{output_directory}initial_data.csv')
                    self.outputting = self.pg.copy()
                    self.visualise(
                        png_file=f'{output_directory}initial_grid.png', i=None, exporting_final=True)
                    self.outputting = self.pg_complied.copy()
                    self.animate(
                        gif_file=f'{output_directory}initial_grid_animation.gif')
                self.time, self.energy, self.mass, self.size = 0, 0, self.pg.sum(
                ) / np.prod(self.dim), self.mask.sum()  # ! mass includes boundary
                self.data = {'time': [], 'energy': [], 'mass': [], 'size': []}
                self.pg_complied, self.mask_complied = [], []
            # ! works for a single causal perturbation only
            elif self.perturbations >= desired_perturbations and self.energy != 0:
                diff = self.pg - ig
                print(f"",
                      f"Perturbations:{self.perturbations}", f"Duration:{self.time}", f"Energy:{self.energy}",
                      f"\n",
                      f"Resultant Grid:\n{self.pg}",
                      f"\n",
                      f"Mass:{self.pg.sum() / np.prod(self.dim)}", f"Size:{self.mask.sum()}",
                      f"\n",
                      f"Differential Matrix:\n{diff}", f"Differential Sum:{diff.sum()}",
                      f"Mask:\n{self.mask}",
                      sep="\n")
                if not input('Return empty to generate outputs:\t\t\t'):
                    self.figures.extend([
                        vis(self.data,
                            f'{output_directory}resultant_data.png'),
                        vis(self.pg, f'{output_directory}resultant_grid.png'),
                        vis(self.mask,
                            f'{output_directory}resultant_mask.png'),
                        vis(self.pg_complied,
                            f'{output_directory}resultant_grid.gif'),
                        vis(self.mask_complied,
                            f'{output_directory}resultant_mask.gif')
                    ])
                    for fig in self.figures:
                        fig.artist()
                return
            pset, area = self.pert()


if __name__ == "__main__":  # TODO: suppress terminal outputs option
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    user_scheme = input(
        "What perturbation scheme should be performed? [random1]\t| ") or "random1"
    user_rule = input(
        "What update rule should be implemented? [ASM]\t\t| ") or "ASM"
    user_boundary_condition = input(
        "What boundary condition should be used? [cliff]\t\t| ") or "cliff"
    user_initial_condition = input(
        "What initial conditions should be applied? [max30min10]\t| ") or "max30min10"
    user_dimensions = tuple(int(a) for a in (
        input("What dimensions should be used? [20,20]\t\t\t| ") or "20,20").split(","))
    user_perturbations = int(
        input("How many causal perturbations should be performed? [1]\t| ") or "1")
    user_output = input(
        "Relative Output Directory: [output]\t\t\t| ") or "output"
    while True:
        st = time.process_time()
        machine = CellularAutomaton(perturbation_scheme=user_scheme, update_rule=user_rule,
                                    boundary_condition=user_boundary_condition, initial_condition=user_initial_condition,
                                    dimensions=user_dimensions)
        machine.run(user_perturbations, f'{user_output}/data_005_')
        et = time.process_time()
        print(f"Processing Time: {et-st}")
        retry = input("Retry? [y/(n)]") or "n"
        if retry == "n":
            break
