'''Application - version 13'''
import pkg.utilities as u                       # Utilities
from pkg.DW_v03 import DataWrangler as DW       # Data Wrangler      - version 03
import numpy as np
import matplotlib.pyplot as mplp
np.set_printoptions(linewidth=250, precision=3)
#* user definitions...
# region
file_dir  : str       = [f'{u.get_src()[:-3]}test/data/v13140311/OFC_Uncontrolled_020.txt',

                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltRand1_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_gtRand1_020.txt']

                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear2_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear4_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear6_020.txt']

                         f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear6_020.txt',
                         f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear6_020_manualsizes.txt',
                         f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltLinear6_020_NaturalManual.txt']

                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltRand1_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_ltExp_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_relabsLinear1_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Controlled_reltotLinear1_020.txt']

                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Olami_025.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Big_Olami_025.txt']

                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Olami_020.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Olami_015.txt',
                        #  f'{u.get_src()[:-3]}test/data/v13140311/OFC_Olami_010.txt']

log_data  : bool      = False;      trim_up : bool = False;      trim_down : bool = False
# endregion
#* operations below...
print(f'{log_data=}\t{trim_up=}\t{trim_down=}')
final = []
for file in file_dir:
    with open(file, 'r') as f:
        data = DW(from_file=f.read(), log_data=log_data, trim_upto_10=trim_up, trim_downto_10=trim_down)
    hist = data.wrangle()
    final.append(hist)

# print(final)

mplp.style.use('dark_background')
fig, axe = mplp.subplots(figsize=[5,4], dpi=300, layout='constrained')

labs = ['Uncontrolled', 
        'Natural Events',
        'Manual Events',
        'Combined']
# labs = ['Uncontrolled', 
#         'M < Mc',
#         'M > Mc']
# labs = ['Uncontrolled', 
#         '+ Δ',
#         '+ Δ×2',
#         '+ Δ×3']
for i, d in enumerate(final): 
    axe.plot(d[0], d[1], '.', markersize=3, label=labs[i])

axe.legend();# axe.grid()
axe.set_xscale('log'); axe.set_xlim((1,1e+04)); axe.set_xlabel('S')
axe.set_yscale('log'); axe.set_ylim((1e-07,1)); axe.set_ylabel('P(size = S)')

mplp.title('+ Δ×3 to a Random Cell')
# mplp.title('+ 1 to a Random Cell')
# mplp.title('Mass < Reference Mass')
mplp.show()