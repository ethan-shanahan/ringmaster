''' Visual Interface '''
import numpy as np
import matplotlib.pyplot as mplp
from matplotlib.gridspec import GridSpec
from pkg import utils
# try: from pkg import utils
# except ModuleNotFoundError: import utils


class VisualInterface():
    def __init__(self, path: str = '', grid_dim: tuple[int,int] = (1,1)) -> None:
        self.path = path
        self.fig = mplp.figure(tight_layout=True)
        self.gs = GridSpec(grid_dim[0], grid_dim[1], figure=self.fig)
    
    def graph(self, xdata, ydata, scale: str = 'linear', loc: tuple = (0,0)) -> None:
        ax = self.fig.add_subplot(self.gs[loc[0],loc[1]])
        ax.plot(xdata,ydata,'.k')
        mplp.xscale(scale); mplp.yscale(scale)
        mplp.grid(True)
        if self.path: mplp.savefig(self.path, dpi=300)
    
    def show(self) -> None:
        mplp.show()


if __name__ == '__main__':
    x = np.linspace(0,10)
    VisualInterface(x,np.sin(x)).histogram()