'''Application - version 13'''
import pkg.utilities as u                       # Utilities
from pkg.CA_v14 import CellularAutomaton as CA  # Cellular Automaton - version 14
from pkg.DW_v03 import DataWrangler as DW       # Data Wrangler      - version 03
from pkg.VI_v11 import VisualInterface as VI    # Visual Interface   - version 11

#* user definitions...
# region
output : str = 'TBD'  # passed to CA, determines what gets recorded in Series
                      #! moved to CA
extract : list[str] = ['size']  # list of attributes of Series to be analysed (equivalent to output?)
                                #! should be moved to CA
begin : int = 1  # index of first state from Series to be extracted, useful for skipping transient state
                 #! moved to CA
# endregion
#* operations below...
# region
config : dict = u.parse_config()
seed : int | None = config.pop('seed', None)
samples : int = config.pop('samples')
machine_id = 'A'

p = u.ProgressBar(header=f'Activated Machine {machine_id}', footer=f'Completed Machine {machine_id}', entity='seed', jobs=samples, steps=config['skip_transient_states'] + config['desired_stable_states'])

autos = []
for _ in range(samples):
    ca = CA(progress_bar=p, output=output, seed=seed, **config); ca.run()
    autos.append(ca)

# [dict([(ext, getattr(autos[s].series, ext)[begin:]) for ext in extract]) for s in range(samples)]
print(autos[0].series)

machine_id = chr(ord(machine_id) + 1)
# endregion
if __name__ == '__main__':
    pass