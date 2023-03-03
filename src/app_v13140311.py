'''Application - version 13'''
import pkg.utilities as u                       # Utilities
from pkg.CA_v14 import CellularAutomaton as CA  # Cellular Automaton - version 14
from pkg.DW_v03 import DataWrangler as DW       # Data Wrangler      - version 03
from pkg.VI_v11 import VisualInterface as VI    # Visual Interface   - version 11

#* user definitions...
# region
file_type : str       = 'read'
file_dir  : str       = f'{u.get_src()[:-3]}test/data/v13140311/OFC.txt' # OFC-MAX_NatPertSize
log_data  : bool      = False;      trim_up : bool = True;      trim_down : bool = True
machines  : int       = 1                      # repeats the entire process
model     : str       = 'OFC'
output    : str       = 'perturbation'         # passed to CA, determines what gets recorded in Series
extract   : list[str] = ['natural_pert.size']  # list of attributes of Series to be analysed
                      # ['natural_pert.size', 'manual_pert.size']
# endregion
#* operations below...
# region
print(f'{log_data=}\t{trim_up=}\t{trim_down=}')
final = []
visual = VI(draft=True)
if file_type == 'read':
    with open(file_dir, 'r') as f:
        data = DW(from_file=f.read(), log_data=log_data, trim_upto_10=trim_up, trim_downto_10=trim_down)
    hist = data.wrangle()
    final.append(hist)
    if model := input('Fit data to a model? To skip, enter none.\t\t| '):
        fit = data.fitter(model)
        final.append(fit)
else:
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
            data = DW([dict([(ext, u.recursive_getattr(autos[s].series, ext))]) for s in range(samples)], log_data=log_data, trim_upto_10=trim_up, trim_downto_10=trim_down)

            hist = data.wrangle()
            final.append(hist)
            if model := input('Fit data to a model? To skip, enter none.\t\t| '):
                fit = data.fitter(model)
                final.append(fit)
            
            if file_type == 'write':
                with open(file_dir, 'w') as f:
                    f.write(data.to_file)

    # u.pprint(f'Analysing... {extracting}')
    # data = DW([dict([(ext, u.recursive_getattr(autos[s].series, ext)) for ext in extracting]) for s in range(samples)])
    # hist = data.wrangle()
    # final.append(hist)

visual.plotter(data=final, title=' '.join(['size?']), linear_scale=log_data)
visual.show()
# endregion
if __name__ == '__main__':
    pass