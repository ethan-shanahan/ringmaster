import numpy as np
from scipy.optimize import curve_fit
import pkg.utilities as u


class DataWrangler():
    def __init__(self, data : list[dict]) -> None:
        self.data = data

