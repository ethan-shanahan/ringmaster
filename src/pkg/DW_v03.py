import numpy as np
from scipy.optimize import curve_fit


class DataWrangler():
    def __init__(self, data : list[dict[str,list[int|float]]] = None, from_file = None, log_data : bool = False, trim_upto_10 : bool = False, trim_downto_10 : bool = False) -> None:
        self.log_data = log_data
        self.trim_upto_10 = trim_upto_10
        self.trim_downto_10 = trim_downto_10
        if from_file == None: self.to_file = self.write_data(data)
        else: data = self.read_data(from_file)
        match (len(data[0]) > 1, len(data) > 1):
            case (False, False) : self.data = next(iter(data[0].values()));                             self.wrangle = self.hist
            case (False, True)  : self.data = [next(iter(data[i].values())) for i in range(len(data))]; self.wrangle = self.avg_hist
            case (True,  False) : self.data = data[0]; self.wrangle = self.series
            case (True,  True)  : self.data = data;    self.wrangle = self.avg_series
        self.results : dict[str,np.ndarray] = {}
    
    @staticmethod
    def write_data(from_data : list[dict[str,list[int|float]]]) -> str:
        return '|'.join([';'.join([k + ':' + ','.join(map(str,v)) for k, v in d.items()]) for d in from_data])
         
    @staticmethod
    def read_data(to_data : str) -> list[dict[str,list[float]]]:
        return [dict([list(map(lambda enu: enu[1] if enu[0] == 0 else list(map(float, enu[1].split(','))), enumerate(item.split(':')))) for item in d.split(';')]) for d in to_data.split('|')]

    @staticmethod
    def binner(data : list[int|float] | np.ndarray, spec : int, mean_bins : bool = False) -> np.ndarray[float]:
        '''
        spec > 0 -> resolution\n
        spec = 0 -> fine grain\n
        spec < 0 -> log base * 10
        '''
        if mean_bins:
            geo_mean = lambda x: np.exp(np.log(x).mean())
            if spec >= 0: return np.asarray([data[:-1],data[1:]]).mean(axis=0)
            else: return np.asarray(list(map(geo_mean, zip(data[1:], data[:-1]))))
        log = lambda x, b: np.log(x) / np.log(b)
        if   spec  > 0: return np.linspace(min(data),max(data),spec+1)
        elif spec == 0: return np.linspace(min(data)-0.5,max(data)+0.5,int((max(data)-min(data))+2))
        elif spec  < 0: return np.logspace(log(10,(-spec/10)),log(max(data)+0.5,(-spec/10)),1+np.floor(log(max(data),(-spec/10))).astype(int),base=(-spec/10))
        else: raise ValueError(f'Pardon?\t{spec=}')

    def mk_hist(self, data : list[int|float], bin_spec : int, bins : np.ndarray[float] = None) -> np.ndarray:
        if type(bins) == type(None):
            yhist, xhist = np.histogram(data, self.binner(data, bin_spec), density=True)
            xhist = self.binner(xhist, bin_spec, mean_bins=True)
        else:
            yhist, xhist = np.histogram(data, bins, density=True)
            xhist = self.binner(xhist, bin_spec, mean_bins=True)
        return np.asarray([xhist,yhist])
    
    def hist(self, bin_spec : int = 0) -> np.ndarray:
        result = self.mk_hist(self.data, bin_spec)
        result = result[:,~np.any(result == 0, axis=0)]
        if self.trim_upto_10: result = result[:,9:]
        if self.trim_downto_10: result = result[:, :np.nonzero(result[1] == 1)[0][0]]
        if self.log_data: result = np.log10(result)
        self.results['hist'] = result
        return result

    def avg_hist(self, bin_spec : int = 0) -> np.ndarray:
        samples = []
        for d in self.data: samples.extend(d)
        bins = self.binner(samples, bin_spec)
        # min_data = min(samples) / len(d)
        hists = []
        least1 = None
        for d in self.data:
            h = self.mk_hist(d, bin_spec, bins=bins)
            if least1 is None: least1 = np.nonzero(h[1] == 0)[0][0]
            elif least1 > (l := np.nonzero(h[1] == 0)[0][0]): least1 = l
            hists.append(h)
        for n in range(len(hists)):
            if self.trim_upto_10: hists[n] = hists[n][:,9:]
            if self.trim_downto_10: hists[n] = hists[n][:, :least1]
        #! use to ignore cutoff: result = np.dstack(hists)[:,np.all(np.dstack(hists) != 0, axis=(0,2)),:].mean(axis=2)
        result = np.dstack(hists).mean(axis=2)
        if result[0,0] == 0.0: result = result[:,1:]
        if self.log_data: result = np.log10(result)
        self.results['hist'] = result
        return result

    def series(self) -> np.ndarray:
        pass

    def avg_series(self) -> np.ndarray:
        pass

    def fitter(self, model : str) -> np.ndarray:
        data = next(iter(self.results.values()))
        if model == 'modified_power_law':
            sigmas = np.power(10, np.floor(np.log10(data[1]))*0.4)
        elif model == 'basic_power_law':
            orders = np.floor(np.log10(data[1]))
            t = np.percentile(orders, 95); print(t)
            orders[orders < t] = 1
            sigmas = np.power(10, orders)
        sigmas[sigmas == 0] = 10
        model = getattr(self, model)
        params, _ = curve_fit(model, data[0], data[1], sigma=sigmas, absolute_sigma=False)
        result = np.asarray([data[0], model(data[0], *params)])
        self.results['fit'] = result
        print(*params)
        return result

    @staticmethod
    def logged_power_law(x, a, C) -> np.float64:
        '''P(x) ~ -a * x + C'''
        return -a * x + C
    
    @staticmethod
    def basic_power_law(x, a, C) -> np.float64:
        '''P(x) ~ C * x^(-a)'''
        return C * np.power(x, -(1+a))
    
    @staticmethod
    def modified_power_law(x, a, l, C) -> np.float64:
        '''P(x) ~ C * e^(-x/l) * x^(-a)'''
        return C * np.exp(-x/l) * np.power(x, -(1+a))


if __name__ == '__main__':
    sample_series = [
        {
            'pert_time': [0.00048731122460043963, 0.0009001076098481553, 0.0023747467858196147, 0.0026211771940842787, 0.0026617213615195423, 0.003005795560662161, 0.003133660380390424, 0.0032298746846253223, 0.0034362276111892243, 0.004176165889000161, 0.0041792431457710055, 0.006572205166441969, 0.006580604100932419, 0.010064002985548104, 0.010489055737183617, 0.01102555564917973, 0.015710181136615486, 0.01611953221163831, 0.017158948487364434, 0.019365186779876087], 
            'size': [2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1]
        }, 
        {
            'pert_time': [0.0006821209600287315, 0.0025203373862752176, 0.002977233738191254, 0.0034535048240014188, 0.004732549844538192, 0.007111416141342808, 0.00797695951990851, 0.00864417082633484, 0.010146365713744387, 0.01050943475217192, 0.0117279666362895, 0.011783508058693881, 0.012268678101791775, 0.013522064139702916, 0.014823442647977636, 0.015017702046750125, 0.01652715910754532, 0.016994618967676844, 0.019116800216402385, 0.021143674561432735], 
            'size': [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        }
    ]
    hist = [
        {
            'size': [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        }
    ]
    avg_hist = [
        {
            'size': [2, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1]
        },
        {
            'size': [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        }
    ]
    DW = DataWrangler(avg_hist)
    data = DW.wrangle(0)
    fit = DW.fitter(data, 'modified_power_law')
    print(data)
    print(fit)
    