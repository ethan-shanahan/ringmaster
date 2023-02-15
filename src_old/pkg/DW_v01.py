import numpy as np
from scipy.optimize import curve_fit
from pkg import utils


class DataWrangler():
    def __init__(self, data) -> None:
        self.data = data
        self.dtype = type(data)
    
    def linbin_hist(self, resolution: int = 0):
        if resolution == 0: binner = lambda x: np.linspace(min(x),max(x),max(x)+1)
        else: binner = lambda x: np.linspace(min(x),max(x),resolution+1)
        hist = self.mk_hist(self.data, binner)
        hist[1] = hist[1]/hist[1].sum()
        return hist

    def linbin_hist_legacy_simple(self, resolution: int = 1000):
        binner = lambda x: np.linspace(min(x),max(x),resolution+1)
        return self.mk_hist(self.data, binner)
    
    def logbin_hist(self, resolution: int = 100, b = 1.1):
        log = lambda x, b: np.log(x) / np.log(b)
        binner = lambda x: np.logspace(log(min(x),b),log(max(x),b),np.floor(log(max(x),b)).astype(int),base=b)
        return self.mk_hist(self.data, binner)

    def logbin_hist_legacy_base2DynamicWidth(self, resolution: int = 100):
        i = lambda x: np.log2(min(x))
        f = lambda x: np.log2(max(x))
        binner = lambda x: np.logspace(i(x),f(x),np.floor(np.log2(max(x))).astype(int),base=2)
        return self.mk_hist(self.data, binner)

    def logbin_hist_legacy_base10SimpleWidth(self, resolution: int = 100):
        binner = lambda x: np.logspace(np.log10(min(x)),np.log10(max(x)),resolution+1)
        return self.mk_hist(self.data, binner)
    
    def fitter_nonlinear(self, model: str) -> np.ndarray:
        model = getattr(self, model)
        params, _ = curve_fit(model, self.data[0], self.data[1], p0=(1,550,0.15)); print(params)
        x = self.data[0]; y = model(self.data[0], *params)
        return np.asarray([x,y])

    def fitter_linear(self, model: str) -> np.ndarray:
        pass

    @staticmethod
    def poly1():
        pass

    @staticmethod
    def pareto(x, *params):  # log(Pr(x)) = - (a + 1) log(x) + log(as**a)
        # [0] = a, [1] = s
        return (params[0] * (params[1] ** params[0])) / (x ** (params[0] + 1))

    @staticmethod
    def modified_pareto(x, *params):
        # [0] = a, [1] = l
        return np.exp(-x/params[1], dtype=np.float64) * (params[0] * (1 ** params[0])) / (x ** (params[0] + 1))

    @staticmethod
    def modified_power_law(x, *params):
        # [0] = a, [1] = l, [2] = C
        return params[2] * np.exp(-x/params[1], dtype=np.float64) * np.power(x, -params[0], dtype=np.float64)
    
    @staticmethod
    def mk_hist(data, bin_func) -> np.ndarray:
        [yhist,xhist] = np.histogram(data, bin_func(data))
        xhist = np.asarray([xhist[:-1],xhist[1:]]).mean(axis=0)
        return np.asarray([xhist,yhist])
