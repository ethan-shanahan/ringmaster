import sys
import time
import csv
import itertools as itt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

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

# TODO: improve matplotlib implementation
# TODO: output perturbation timescale
# TODO: consider collapses and growths

class CellularAutomaton(): # TODO: general documentation
    '''Main object class...'''

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
        self.ndim = len(self.dim)
        self.pg = self.ic()
        self.fg = self.pg.copy()
        self.bc()
        self.seed = self.pg.copy()
        self.perturbations = 0
        self.ts = 0
        self.energy = 0
        self.mask = np.zeros(self.dim, dtype=np.int8)
        self.data = []
        self.pg_complied = [self.pg]
        self.mask_complied = [self.mask]
        self.outputting = []

# INITIAL CONDITIONS (all initial conditions must describe how the grid is seeded/generated and apply the boundary condition)

    def initial_max3min0(self):
        initial = self.rng.integers(0, 3, size=self.dim, endpoint=True)
        return initial

    def initial_max30min10(self):
        initial = self.rng.integers(10, 30, size=self.dim, endpoint=True)
        return initial
    
    def initial_control(self):
        initial = np.array(
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
        return initial

# BOUNDARY CONDITIONS (all boundaries are applied to the initial grid and again after each update)

    def boundary_cliff(self):
        indices = list(tuple(slice(0,self.dim[z]) if z != x else 0 for z in range(self.ndim)) for x in range(self.ndim))
        indices.extend(list(tuple(slice(0,self.dim[z]) if z != x else -1 for z in range(self.ndim)) for x in range(self.ndim)))
        for i in indices:
            self.pg[i] = 0
            self.fg[i] = 0

# UPDATE RULES (all rules must manipulate self.fg, return a tuple of the cells that should be searched following an event, and increment energy if executed)

    def rule_BTW(self, cell, boolean_flag=False): # TODO: extend to n dimensions, use sets, remove b_flag
        if self.pg[cell] >= 4:
            if boolean_flag:
                return True
            minus_cells = [(i:=cell[0],j:=cell[1])]
            plus_cells = {(i+1,j),(i-1,j),(i,j+1),(i,j-1)}
            # affected_cells = [(i,j), (i+1,j),(i-1,j),(i,j+1),(i,j-1)]
            
            for c in minus_cells: self.fg[c] -= 4
            for c in plus_cells: self.fg[c] += 1

            self.energy += 4
            
            if self.perturbations != 0:
                for y in plus_cells: self.mask[y] = 1 #? Should the perturbation site be included in the size? This only matters if a single event occurs in the BTW model.

            return plus_cells
        return []

# PERTURBATION SCHEMES (all perturbations are applied while the system is stable and must return pset and area)

    def perturbation_random1(self):
        self.perturbations += 1
        target = tuple(self.rng.integers(0,self.dim[n]) for n in range(self.ndim))
        print(f"Random Perturbation of Cell {target}")
        self.pg[target] += 1 # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area

    def perturbation_control(self):
        self.perturbations += 1
        target = tuple(int(self.dim[n]/2) if self.dim[n]%2 == 0 else int((self.dim[n]-1)/2) for n in range(self.ndim))
        print(f"Controlled Perturbation of Cell {target}")
        self.pg[target] += 1 # the perturbation itself
        self.fg[target] += 1
        pset = [target]
        area = len(pset)
        return pset, area

# COMPUTATIONS HENCEFORTH...

    def run(self, desired_perturbations, output_directory):
        pset = list(itt.product(*list(list(x for x in range(self.dim[n])) for n in range(self.ndim))))
        area = len(pset)
        fset = set()
        while True:
            cursor = 0
            while area != 0:
                queue = self.rule(pset[cursor])
                fset.update(queue)
                self.bc()
                area -= 1
                cursor += 1
                if area == 0:
                    pset = list(dict.fromkeys(fset))
                    area = len(pset)
                    fset = set()
                    cursor = 0
                    if self.energy != 0:
                        self.pg = self.fg.copy()
                        self.ts += 1
                        self.data.append({'time' : self.ts, 'energy' : self.energy, 'mass' : self.pg.sum() / np.prod(self.dim), 'size' : self.mask.sum()})
                        self.pg_complied.append(self.pg)
                        self.mask_complied.append(self.mask)
            if self.perturbations == 0:
                ig = self.pg.copy()
                print(f"\nA stable configuration was reached after {self.ts} time-steps...\n",
                      f"Seed:\n{self.seed}", f"Initial Grid:\n{ig}",
                      sep='\n')
                if not input('Return empty to generate outputs:\t\t\t'):
                    self.export(csv_file=f'{output_directory}initial_data.csv')
                    self.outputting = self.pg.copy()
                    self.visualise(png_file=f'{output_directory}initial_grid.png', i=None, exporting_final=True)
                    self.outputting = self.pg_complied.copy()
                    self.animate(gif_file=f'{output_directory}initial_grid_animation.gif')
                self.energy, self.ts= 0, 0
                self.pg_complied, self.mask_complied = [self.pg], [self.mask]
            elif self.perturbations >= desired_perturbations and self.energy != 0: #! works for a single causal perturbation only
                diff = self.pg - ig
                print(f"",
                      f"Perturbations:{self.perturbations}", f"Duration:{self.ts}", f"Energy:{self.energy}",
                      f"\n",
                      f"Resultant Grid:\n{self.pg}",
                      f"\n",
                      f"Mass:{self.pg.sum() / np.prod(self.dim)}", f"Size:{self.mask.sum()}",
                      f"\n",
                      f"Differential Matrix:\n{diff}", f"Differential Sum:{diff.sum()}",
                      f"Mask:\n{self.mask}",
                      sep="\n")
                if not input('Return empty to generate outputs:\t\t\t'):
                    self.export(csv_file=f'{output_directory}resultant_data.csv')
                    self.outputting = self.pg.copy()
                    self.visualise(png_file=f'{output_directory}resultant_grid.png', i=None, exporting_final=True)
                    self.outputting = self.mask.copy()
                    self.visualise(png_file=f'{output_directory}resultant_mask.png', i=None, exporting_final=True)
                    self.outputting = self.pg_complied.copy()
                    self.animate(gif_file=f'{output_directory}resultant_grid_animation.gif')
                    self.outputting = self.mask_complied.copy()
                    self.animate(gif_file=f'{output_directory}resultant_mask_animation.gif')
                return
            self.pg_complied.append(self.pg)
            self.mask_complied.append(self.mask)
            pset, area = self.pert()
        
    def export(self, csv_file):
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['time','energy','mass','size'])
            writer.writeheader()
            writer.writerows(self.data)
        self.data = []
    
    def visualise(self, i, png_file=None, exporting_final=False):
        if exporting_final: 
            fig = plt.figure(figsize=self.dim)
        ax = plt.axes()
        ax.clear()
        ax.set_axis_off()
        img = ax.imshow(self.outputting if exporting_final else self.outputting[i], interpolation='none', cmap='gray')
        if exporting_final: return plt.savefig(png_file, dpi=300, bbox_inches='tight')
        return [img]

    def animate(self, gif_file):
        fig = plt.figure(figsize=self.dim)
        ax = plt.axes()
        ax.set_axis_off()
        animator = ani.FuncAnimation(fig, self.visualise, frames=self.ts+1, interval=100, blit=True)
        animator.save(gif_file, writer='imagemagick')


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    user_scheme = input("What perturbation scheme should be performed? [random1]\t") or "random1"
    user_rule = input("What update rule should be implemented? [BTW]\t\t") or "BTW"
    user_boundary_condition = input("What boundary condition should be used? [cliff]\t\t") or "cliff"
    user_initial_condition = input("What initial conditions should be applied? [max30min10]\t") or "max30min10"
    user_dimensions = tuple(int(a) for a in (input("What dimensions should be used? [25,25]\t\t\t") or "25,25").split(","))
    user_perturbations = int(input("How many causal perturbations should be performed? [1]\t") or "1")
    user_output = input("Relative Output Directory: [output]\t\t\t") or "output"
    while True:
        st = time.process_time()
        machine = CellularAutomaton(perturbation_scheme=user_scheme, update_rule=user_rule,
                                    boundary_condition=user_boundary_condition, initial_condition=user_initial_condition,
                                    dimensions=user_dimensions)
        machine.run(user_perturbations, f'{user_output}/data_004_')
        et = time.process_time()
        print(f"Processing Time: {et-st}")
        retry = input("Retry? [y/(n)]") or "n"
        if retry == "n":
            break
