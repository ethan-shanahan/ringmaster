import numpy as np
import matplotlib as mpl

# create the grid and boundary conditions
# set the initial conditions and update origin
# implement the update rules
# determine the next state
# calculate fractal dimensions
# animate evolution

def rule_BTW(grid, cell):
    i, j = cell
    minus_cells = ((i,j), (0,0))
    plus_cells = ((i+1,j),(i-1,j),(i,j+1),(i,j-1))
    
    for c in minus_cells:
        grid[c] -= 4
    for c in plus_cells:
        grid[c] += 1
    
    return plus_cells

def boundary_cliff(grid):
    for i in range(10): # TODO: automate for any size
        grid[0, i] = 0
        grid[i, 0] = 0
        grid[-1,i] = 0
        grid[i,-1] = 0


rng = np.random.default_rng()
grid = rng.integers(4, size=(10, 10))
print("Generate random cell values...")
print(grid)
boundary_cliff(grid)
print("\nSet edges to 0...")
print(grid)
initial_grid = grid.copy()
print(f"\nRandom Integer: {rng.integers(1, 9)}\n")
perturbations = 1
events = 0
while True:
    print(f"number of perturbations = {perturbations}")
    i, j = rng.integers(1, 9, 2)
    search_cells = [(i,j), (0,0)]
    search_area = len(search_cells)
    print(f"Update cell: {i},{j}")
    grid[search_cells[0]] += 1 # the perturbation itself
    x = 0
    while search_area > 0:

        print(f"\nsearch_area:{search_area}\nx:{x}")

        cell = search_cells[x]

        print(f"found cell:{cell}\n")

        print(type(cell))
        if grid[cell] >= 4:
            events += 1
            print(f"\nStart of event {events}.")
            print("Perturbed grid...")
            print(grid)
            add_to_search = rule_BTW(grid, cell)
            search_area += len(add_to_search)
            for tup in add_to_search:
                search_cells.append(tup) 
            boundary_cliff(grid)
            print("Updated grid...")
            print(grid)
            print(f"End of event {events}.\n")
        search_area -= 1
        x += 1
    if perturbations == 10:
        print(f"\n\n10 perturbations and {events} events have occurred!\n\nInitial Grid...\n{initial_grid}\nFinal Grid...\n{grid}")
        differential = grid - initial_grid
        print(f"\nThe differential grid is...\n{differential}")
        print(f"The sum of the differential is: {differential.sum()}\n")
        break
    perturbations += 1
