import numpy as np
from pkg import utils


class DataWrangler():
    def __init__(self, data) -> None:
        self.data = data
        self.dtype = type(data)
    
    def linbin_hist(self, resolution: int = 100):
        binner = lambda x: np.linspace(min(x),max(x),resolution+1)
        [yhist,xhist] = np.histogram(self.data, binner(self.data))
        xhist = np.asarray([xhist[:-1],xhist[1:]]).mean(axis=0)
        return np.asarray([xhist,yhist])
    
    def logbin_hist(self, resolution: int = 100):
        binner = lambda x: np.logspace(np.log10(min(x)),np.log10(max(x)),resolution+1)
        [yhist,xhist] = np.histogram(self.data, binner(self.data))
        xhist = np.asarray([xhist[:-1],xhist[1:]]).mean(axis=0)
        return np.asarray([xhist,yhist])
