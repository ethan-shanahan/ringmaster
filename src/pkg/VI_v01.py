''' Visual Interface '''
import numpy as np
import matplotlib.pyplot as mplp
from pkg import utils
# try: from pkg import utils
# except ModuleNotFoundError: import utils


class VisualInterface():
    def __init__(self, path: str = '', **spec: tuple) -> None:
        self.spec = {}
        for f_id, a in spec.items():
            self.spec[f_id] = {'fig','axs'}
            self.spec[f_id]['fig'], self.spec[f]['axs'] = mplp.subplots(*a, squeeze=False)
        # self.fig, self.axs = mplp.subplots(*spec, squeeze=False)
        # print(self.axs)
        self.path = path
    
    def histogram(self, xdata, ydata, figure_id: tuple[str, tuple]) -> None:
        self.spec[figure_id[0]]['axs'][*figure_id[1]].plot(xdata,ydata,'k')
        if self.path: mplp.savefig(self.path, dpi=300)
    
    def show(self) -> None:
        mplp.show()


if __name__ == '__main__':
    x = np.linspace(0,10)
    VisualInterface(x,np.sin(x)).histogram()