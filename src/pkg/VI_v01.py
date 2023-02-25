''' Visual Interface '''
import numpy as np
import matplotlib.pyplot as mplp
from matplotlib.gridspec import GridSpec


class VisualInterface():
    def __init__(self, draft : bool = True) -> None:
        self.figures = []

        pass

    def new_fig(self):
        self.figures.append(mplp.figure(num=len(self.figures)-1, figsize=[8,6], layout='constrained'))
    
    def add_ax(self):
        pass


if __name__ == '__main__':
    pass