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

# TODO: implement matplotlib to visualise CA
# TODO: track computation time, try to optimise

class CellularAutomaton(): # TODO: general documentation
    def __init__(self, 
                 perturbation_scheme,update_rule, 
                 boundary_condition, initial_condition, 
                 dimensions):
        self.rng = np.random.default_rng()
        self.pert = getattr(self, f"perturbation_{perturbation_scheme}")
        self.rule = getattr(self, f"rule_{update_rule}")
        self.bc = getattr(self, f"boundary_{boundary_condition}")
        self.ic = getattr(self, f"initial_{initial_condition}")
        self.dim = dimensions
        self.present_grid = self.ic()
        self.future_grid = self.present_grid.copy()
        self.bc()
        self.seed = self.present_grid.copy()
        self.ts = -1
        self.events = 0
        self.perturbations = 0
        self.energy = 0
        self.mask = np.zeros(self.dim, dtype=np.int8)

# PERTURBATION SCHEMES (all perturbations are applied when the system is still and must return search_group and search_area)

    def perturbation_random1(self): # TODO: adapt to n dimensions
        self.perturbations += 1
        i = self.rng.integers(0, self.dim[0] - 1) # TODO: check if this covers the entire grid
        j = self.rng.integers(0, self.dim[1] - 1)
        search_group = [(i,j), (0,0)]
        search_area = len(search_group)
        print(f"Random Perturbation of Cell ({i},{j})")
        self.present_grid[search_group[0]] += 1 # the perturbation itself
        return search_group, search_area

    def perturbation_control(self): # TODO: adapt to n dimensions
        self.perturbations += 1
        if self.dim[0] % 2 == 0:
            i = int(self.dim[0] / 2)
        else:
            i = int((self.dim[0] - 1) / 2)
        if self.dim[1] % 2 == 0:
            j = int(self.dim[1] / 2)
        else:
            j = int((self.dim[1] - 1) / 2)
        search_group = [(i,j), (0,0)]
        search_area = len(search_group)
        print(f"Controlled Perturbation of Cell ({i},{j})")
        self.present_grid[search_group[0]] += 1 # the perturbation itself
        return search_group, search_area


# INITIAL CONDITIONS (all initials must describe how the grid is seeded/generated)

    def initial_max3min0(self):
        return self.rng.integers(0, 3, size=self.dim, endpoint=True)

    def initial_max30min10(self):
        return self.rng.integers(10, 30, size=self.dim, endpoint=True)
    
    def initial_control(self):
        return np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 19, 10, 14, 10, 10, 27, 27, 10, 14, 17, 24, 22, 29, 10, 30, 27, 19, 15, 20, 17, 16, 28, 21, 0],
             [0, 12, 22, 10, 20, 18, 21, 30, 19, 19, 19, 18, 25, 20, 26, 28, 17, 26, 10, 30, 21, 11, 28, 14, 0],
             [0, 17, 13, 24, 23, 28, 27, 18, 23, 15, 15, 23, 29, 24, 25, 24, 30, 28, 16, 21, 23, 26, 25, 15, 0],
             [0, 14, 12, 19, 28, 22, 29, 18, 28, 17, 21, 16, 22, 29, 25, 16, 30, 24, 28, 29, 15, 22, 30, 24, 0],
             [0, 21, 27, 18, 14, 18, 13, 19, 23, 12, 11, 27, 17, 14, 18, 26, 11, 25, 24, 25, 28, 27, 17, 13, 0],
             [0, 23, 25, 17, 21, 10, 17, 27, 27, 10, 19, 17, 19, 13, 21, 25, 28, 19, 14, 14, 18, 22, 30, 28, 0],
             [0, 10, 18, 14, 14, 20, 26, 26, 29, 24, 20, 18, 13, 21, 23, 17, 21, 10, 25, 22, 14, 18, 24, 24, 0],
             [0, 14, 28, 12, 28, 21, 25, 25, 28, 23, 18, 10, 23, 15, 13, 14, 10, 26, 14, 26, 19, 10, 23, 16, 0],
             [0, 30, 20, 19, 20, 19, 14, 25, 13, 30, 10, 26, 30, 15, 13, 19, 23, 19, 18, 16, 18, 14, 11, 21, 0],
             [0, 30, 17, 24, 16, 16, 22, 12, 14, 15, 29, 14, 21, 23, 10, 27, 22, 12, 16, 21, 20, 27, 23, 15, 0],
             [0, 17, 24, 20, 30, 11, 21, 25, 10, 30, 24, 30, 26, 19, 23, 16, 27, 15, 18, 29, 12, 25, 26, 11, 0],
             [0, 21, 26, 16, 27, 27, 22, 17, 24, 24, 14, 28, 29, 27, 10, 24, 15, 24, 18, 28, 25, 23, 30, 13, 0],
             [0, 19, 22, 18, 21, 11, 13, 11, 23, 20, 27, 22, 18, 25, 23, 30, 30, 18, 29, 18, 22, 10, 29, 14, 0],
             [0, 20, 12, 18, 14, 19, 25, 30, 28, 23, 28, 26, 17, 29, 11, 21, 16, 12, 29, 20, 24, 23, 25, 28, 0],
             [0, 21, 19, 15, 12, 22, 10, 10, 30, 14, 18, 11, 21, 18, 16, 12, 10, 11, 16, 15, 16, 25, 24, 11, 0],
             [0, 12, 20, 22, 30, 12, 17, 11, 11, 29, 11, 23, 17, 11, 25, 30, 28, 30, 11, 24, 14, 23, 28, 13, 0],
             [0, 21, 28, 25, 22, 24, 21, 15, 10, 29, 19, 27, 14, 19, 16, 26, 26, 29, 10, 26, 10, 19, 21, 27, 0],
             [0, 14, 30, 20, 28, 14, 18, 29, 29, 22, 13, 24, 15, 19, 19, 18, 14, 26, 23, 30, 26, 11, 13, 25, 0],
             [0, 28, 19, 16, 11, 26, 27, 12, 30, 30, 26, 10, 29, 16, 26, 29, 20, 19, 15, 20, 17, 27, 23, 15, 0],
             [0, 11, 11, 24, 12, 26, 15, 29, 28, 18, 10, 13, 18, 29, 28, 18, 27, 30, 23, 25, 12, 29, 23, 22, 0],
             [0, 21, 27, 28, 23, 12, 17, 29, 21, 28, 12, 27, 24, 25, 16, 11, 26, 25, 14, 12, 16, 13, 24, 12, 0],
             [0, 18, 11, 16, 27, 26, 29, 13, 26, 29, 30, 29, 15, 13, 19, 11, 21, 21, 16, 27, 13, 21, 16, 11, 0],
             [0, 27, 23, 20, 29, 27, 22, 26, 24, 26, 22, 21, 10, 17, 28, 25, 27, 24, 10, 30, 25, 17, 18, 27, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )

# BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff(self): # TODO: adapt to n dimensions
        for i in range(self.dim[0]):
            self.present_grid[i, 0] = 0
            self.present_grid[i,-1] = 0
            self.future_grid[i, 0] = 0
            self.future_grid[i,-1] = 0
        for i in range(self.dim[1]):
            self.present_grid[0, i] = 0
            self.present_grid[-1,i] = 0
            self.future_grid[0, i] = 0
            self.future_grid[-1,i] = 0

# UPDATE RULES (all rules must manipulate self.future_grid, return a tuple of the cells that should be searched following an update, and increment events&energy if executed)

    def rule_BTW(self, cell, boolean_flag=False):
        if self.present_grid[cell] >= 4:
            if boolean_flag:
                return True
            self.events += 1
            self.energy += 4
            i, j = cell
            minus_cells = ((i,j), (0,0))
            plus_cells = ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
            # affected_cells = ((i,j), (i+1,j),(i-1,j),(i,j+1),(i,j-1))
            
            for c in minus_cells:
                self.future_grid[c] -= 4
            for c in plus_cells:
                self.future_grid[c] += 1
            
            if self.perturbations != 0:
                for y in plus_cells: #? Should the perturbation site be included in the size?
                    self.mask[y] = 1

            return plus_cells

# COMPUTATIONS HENCEFORTH...

    def run(self, desired_perturbations, debug_flag=False):
####
        d = 0
####
        search_group = list(itt.product(range(self.dim[0]),range(self.dim[1]))) # TODO: adapt to n dimensions
        search_area = len(search_group)

        future_grid = self.present_grid.copy()
        future_search_group = []

        while True:
            cursor = 0
            while search_area != 0:
                cell = search_group[cursor]
                if self.rule(cell, boolean_flag=True):
                    add_to_search = self.rule(cell)
                    for x in add_to_search:
                        future_search_group.append(x)
                    self.bc()
                search_area -= 1
                cursor += 1
                if search_area == 0:
                    self.present_grid = self.future_grid
                    search_group = future_search_group
                    search_area = len(search_group)
                    future_search_group = []
                    if self.events != 0:
                        self.ts += 1
                    cursor = 0
####
                if debug_flag == True:
                    print(f"search_area:{search_area},\tcursor:{cursor}")
                    d += 1
                    print(f"Cycle:{d}\nEvents so far:{self.events}\nPerturbations so far:{self.perturbations}\n\n")
                    if d % 100 == 0:
                        continuation_a = input("Continue? [(y)/n]") or "y"
                        if continuation_a == "n":
                            print(f"Seed:\n{self.seed}\nStatic:\n{initial_grid}\nCurrent:\n{self.present_grid}")
                            differential = self.present_grid - initial_grid
                            print(f"Differential:\n{differential}\nDifferential Sum:{differential.sum()}\n")
                            raise Exception("manual termination")
####

            if self.perturbations == 0:
                print(f"\nThe grid has become static after {self.events} events and {self.ts} time-steps...\n")
####
                if debug_flag == True:
                    continuation_b = input("Continue? [(y)/n]") or "y"
                    if continuation_b == "n":
                        print(f"Seed:\n{self.seed}\nCurrent:\n{self.present_grid}")
                        differential = self.present_grid - initial_grid
                        print(f"Differential:\n{differential}\nSum:{differential.sum()}\n")
                        raise Exception("manual termination")
####
                self.events = 0
                self.energy = 0
                self.ts = -1
                initial_grid = self.present_grid.copy()

            elif self.perturbations >= desired_perturbations and self.events != 0: # TODO: create output functions and visualisations
                differential = self.present_grid - initial_grid
                print(f"\nEvents:{self.events}", f"Perturbations:{self.perturbations}", f"Duration:{self.ts}", f"Energy:{self.energy}",
                      f"\n",
                      f"Seed:\n{self.seed}", f"Static:\n{initial_grid}",
                      f"Result:\n{self.present_grid}", f"Mass:{self.present_grid.sum() / np.prod(self.dim)}",
                      f"\n",
                      f"Differential Matrix:\n{differential}", f"Differential Sum:{differential.sum()}",
                      f"Mask (affected cells equal 1):\n{self.mask}", f"Size:{self.mask.sum()}",
                      sep="\n")
                return

            search_group, search_area = self.pert()


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    debug_tools = False
    user_scheme = input("What perturbation scheme should be performed? [random1]\t") or "random1"
    user_rule = input("What update rule should be implemented? [BTW]\t\t") or "BTW"
    user_boundary_condition = input("What boundary condition should be used? [cliff]\t\t") or "cliff"
    user_initial_condition = input("What initial conditions should be applied? [max30min10]\t") or "max30min10"
    user_dimensions = tuple(int(a) for a in (input("What dimensions should be used? [25,25]\t\t\t") or "25,25").split(","))
    user_perturbations = int(input("How many causal perturbations should be performed? [1]\t") or "1")
    while True:
        machine = CellularAutomaton(perturbation_scheme=user_scheme, update_rule=user_rule,
                                    boundary_condition=user_boundary_condition, initial_condition=user_initial_condition,
                                    dimensions=user_dimensions)
        machine.run(user_perturbations, debug_flag=debug_tools)
        retry = input("Retry? [y/(n)]") or "n"
        if retry == "n":
            break
