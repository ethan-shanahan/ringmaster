'''Application - version 13'''
import pkg.utilities as u                       # Utilities
from pkg.CA_v14 import CellularAutomaton as CA  # Cellular Automaton - version 14
from pkg.DW_v03 import DataWrangler as DW       # Data Wrangler      - version 03
from pkg.VI_v11 import VisualInterface as VI    # Visual Interface   - version 11

#* user definitions...
# region
machines : int       = 3                      # repeats the entire process
output   : str       = 'perturbation'         # passed to CA, determines what gets recorded in Series
extract  : list[str] = ['natural_pert.size']  # list of attributes of Series to be analysed
# endregion
#* operations below...
# region
final = []
visual = VI(draft=True)
machine_id : str = 'A'
for m in range(machines):
    #! pass argument to parse_config to predetermine the preset
    config     : dict       = u.parse_config()
    seed       : int | None = config.pop('seed', None)
    samples    : int        = config.pop('samples')

    p = u.ProgressBar(header=f'Activated Machine {machine_id}', footer=f'Completed Machine {machine_id}', entity='seed', jobs=samples, steps=config['skip_transient_states'] + config['desired_stable_states'])

    autos = []
    for _ in range(samples):
        ca = CA(progress_bar=p, output=output, seed=seed, **config); ca.run()
        autos.append(ca)

    extract = (input('Please enter the dotted system attributes that you wish to analyse, separated by spaces.\t\t| ') or 'natural_pert.size').split(' ')
    u.pprint(f'Analysing... {extract}')
    data = DW([dict([(ext, u.recursive_getattr(autos[s].series, ext)) for ext in extract]) for s in range(samples)])
    hist = data.wrangle(0)

    # visual.plotter(data=hist, title=' '.join([machine_id, *extract]))
    final.append(hist)

    machine_id = chr(ord(machine_id) + 1)

visual.plotter(data=final, title=' '.join([*extract]))
visual.show()
# endregion
if __name__ == '__main__':
    pass