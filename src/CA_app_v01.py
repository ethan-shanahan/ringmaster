import sys
import itertools as itt
import numpy as np
# import matplotlib as mpl

# SUMMARY:
    # Create an object of the CellularAutomaton class. I call this object the Grid and it is an ndarray. 
    # The Grid's dimensions must be provided and then it is populated with numbers depending on the 
    #       specified Initial Condition and Boundary Condition. This is saved as the Seed. 
    # The Update Rule is then applied to each Cell of the Grid until the system is Static. This is saved as the Initial Grid. 
    # Perturbations are slowly applied until one triggers an Event via the Update Rule. 
    # All Cells effected by the Event are queued to be checked for triggering another Event. 
    # The system must become Static again before another Perturbations is applied. This is repeated 
    #       for the desired number of Causal Perturbations. 
# 
# OUTPUTS: 
    # number of Events until the system first became Static, 
    # Seed, the Initial Grid, the Resultant Grid, 
    # number of Events to become Static after Causal Perturbations, 
    # number of Perturbations to reach the desired number of Causal Perturbations, 
    # Differential Matrix equivalent to the Resultant Grid minus the Initial Grid, 
    # Differential Sum equivalent to the summation of each element of the Differential Matrix, 
    # Mask that shows which cells have been effected by Perturbations,
    # Size equivalent to the summation of each element of the Mask.
#
#! USAGE:
    # run the program in an environment with Python 3.X and NumPy installed
    # the user will be prompted with three questions
        # return blank to submit the default response, given in brackets
    # observe the outputs in the terminal
    # the user will be asked if they wish to run the simulation again
        # return blank to quit and terminate the program
    #
    # the default responses are intended to simulate the BTW model
        # any 2-dimensional grid size can be specified

# TODO: investigate the synchronised computation of events via time-steps
# TODO: implement matplotlib to visualise CA

class CellularAutomaton(): # TODO: general documentation
    def __init__(self, 
                 perturbation_scheme="random1",update_rule="BTW", 
                 boundary_condition="cliff", initial_condition="max3min0", 
                 dimensions=(10,10)):
        self.rng = np.random.default_rng()
        self.pert = getattr(self, f"perturbation_{perturbation_scheme}")
        self.rule = getattr(self, f"rule_{update_rule}")
        self.bc = getattr(self, f"boundary_{boundary_condition}")
        self.ic = getattr(self, f"initial_{initial_condition}")
        self.dim = dimensions
        self.grid = self.ic()
        self.bc()
        self.seed = self.grid.copy()
        self.events = 0
        self.perturbations = 0
        self.mask = np.zeros(self.dim, dtype=np.int8)

# PERTURBATION SCHEMES (all perturbations are applied when the system is still and must return search_group and search_area)

    def perturbation_random1(self): # TODO: adapt to n dimensions
        self.perturbations += 1
        i = self.rng.integers(0, self.dim[0] - 1)
        j = self.rng.integers(0, self.dim[1] - 1)
        search_group = [(i,j), (0,0)]
        search_area = len(search_group)
        self.grid[search_group[0]] += 1 # the perturbation itself
        return search_group, search_area


# INITIAL CONDITIONS (all initials must describe how the grid is seeded/generated)

    def initial_max3min0(self):
        return self.rng.integers(0, 3, size=self.dim, endpoint=True)

    def initial_max30min10(self):
        return self.rng.integers(10, 30, size=self.dim, endpoint=True)

# BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff(self): # TODO: adapt to n dimensions
        for i in range(self.dim[0]):
            self.grid[i, 0] = 0
            self.grid[i,-1] = 0
        for i in range(self.dim[1]):
            self.grid[0, i] = 0
            self.grid[-1,i] = 0

# UPDATE RULES (all rules must manipulate self.grid, return a tuple of the cells that should be searched following an update, and increment events if executed)

    def rule_BTW(self, cell, boolean_flag=False):
        if self.grid[cell] >= 4:
            if boolean_flag:
                return True
            self.events += 1
            i, j = cell
            minus_cells = ((i,j), (0,0))
            plus_cells = ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            affected_cells = ((i,j), (i+1,j),(i-1,j),(i,j+1),(i,j-1))
            
            for c in minus_cells:
                self.grid[c] -= 4
            for c in plus_cells:
                self.grid[c] += 1
            
            if self.perturbations != 0:
                for y in affected_cells:
                    self.mask[y] = 1

            return plus_cells

# COMPUTATIONS HENCEFORTH...

    def run(self, desired_perturbations, debug_flag=False):
####
        d = 0
####
        search_group = list(itt.product(range(self.dim[0]),range(self.dim[1]))) # TODO: adapt to n dimensions
        search_area = len(search_group)
        while True:
            cursor = 0
            while search_area > 0:
                cell = search_group[cursor]
                if self.rule(cell, boolean_flag=True):
                    add_to_search = self.rule(cell)
                    search_area += len(add_to_search)
                    for x in add_to_search:
                        search_group.append(x)
                    self.bc()
                search_area -= 1
                cursor += 1
####
                if debug_flag == True:
                    d += 1
                    print(f"Cycle:{d}\nEvents so far:{self.events}\nPerturbations so far:{self.perturbations}\n\n")
                    if d % 100 == 0:
                        continuation_a = input("Continue? [(y)/n]") or "y"
                        if continuation_a == "n":
                            print(f"Seed:\n{self.seed}\nStatic:\n{initial_grid}\nCurrent:\n{self.grid}")
                            differential = self.grid - initial_grid
                            print(f"Differential:\n{differential}\nDifferential Sum:{differential.sum()}\n")
                            raise Exception("manual termination")
####

            if self.perturbations == 0:
                print(f"The grid has become static after {self.events} events...\n{self.grid}")
####
                if debug_flag == True:
                    continuation_b = input("Continue? [(y)/n]") or "y"
                    if continuation_b == "n":
                        print(f"Seed:\n{self.seed}\nCurrent:\n{self.grid}")
                        differential = self.grid - initial_grid
                        print(f"Differential:\n{differential}\nSum:{differential.sum()}\n")
                        raise Exception("manual termination")
####
                self.events = 0
                initial_grid = self.grid.copy()

            elif self.perturbations >= desired_perturbations and self.events != 0: # TODO: create output functions and visualisations
                differential = self.grid - initial_grid
                print(f"Events:{self.events}", f"Perturbations:{self.perturbations}",
                      f"\n",
                      f"Seed:\n{self.seed}", f"Static:\n{initial_grid}", f"Result:\n{self.grid}",
                      f"\n",
                      f"Differential Matrix:\n{differential}", f"Differential Sum:{differential.sum()}",
                      f"Mask (affected cells equal 1):\n{self.mask}", f"Size:{self.mask.sum()}",
                      sep="\n")
                return

            search_group, search_area = self.pert()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    debug_tools = False
    user_perturbations = int(input("How many causal perturbations should be performed? [1]") or "1")
    user_initial_condition = input("What initial conditions should be applied? [max30min10]") or "max30min10"
    user_dimensions = tuple(int(a) for a in (input("What dimensions should be used? [25,25]") or "25,25").split(","))
    while True:
        machine = CellularAutomaton(initial_condition=user_initial_condition, dimensions=user_dimensions)
        machine.run(user_perturbations, debug_flag=debug_tools)
        retry = input("Retry? [y/(n)]") or "n"
        if retry == "n":
            break
