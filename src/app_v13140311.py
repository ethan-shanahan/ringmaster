'''Application - version 13'''
import pkg.utilities as u                       # Utilities
from pkg.CA_v14 import CellularAutomaton as CA  # Cellular Automaton - version 14
from pkg.DW_v03 import DataWrangler as DW       # Data Wrangler      - version 03
from pkg.VI_v11 import VisualInterface as VI    # Visual Interface   - version 11

#* user definitions...
# region
machines : int       = 1                      # repeats the entire process
model    : str       = 'OFC'
output   : str       = 'perturbation'         # passed to CA, determines what gets recorded in Series
extract  : list[str] = ['natural_pert.size', 'manual_pert.size']  # list of attributes of Series to be analysed
                    # ['natural_pert.size']
# endregion
#* operations below...
# region
final = []
visual = VI(draft=True)
for m in range(1, 1+machines):
    #! pass argument to parse_config to predetermine the preset
    config, preset       = u.parse_config(model)
    seed    : int | None = config.pop('seed', None)
    samples : int        = config.pop('samples')

    p = u.ProgressBar(header=f'Activated {model} {preset} ({m}/{machines})', footer=f'Completed {model} {preset} ({m}/{machines})', entity='seed', jobs=samples, steps=config['skip_transient_states'] + config['desired_stable_states'])

    autos = []
    for _ in range(samples):
        ca = CA(progress_bar=p, output=output, seed=seed, **config); ca.run()
        autos.append(ca)

    if extract == None:
        extracting = input('Please enter the dotted system attributes that you wish to analyse, separated by spaces.\t\t| ').split(' ')
    else:
        extracting = extract

    for ext in extracting:
        u.pprint(f'Analysing... {ext}')
        data = DW([dict([(ext, u.recursive_getattr(autos[s].series, ext))]) for s in range(samples)])
        hist = data.wrangle()
        final.append(hist)

    # u.pprint(f'Analysing... {extracting}')
    # data = DW([dict([(ext, u.recursive_getattr(autos[s].series, ext)) for ext in extracting]) for s in range(samples)])
    # hist = data.wrangle()
    # final.append(hist)

visual.plotter(data=final, title=' '.join(['pert.size']))
visual.show()
# endregion
if __name__ == '__main__':
    pass