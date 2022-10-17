# vXX
from pkg import CA_v07 as ca # vYY
from pkg import VIS_v03 as vis # vZZ
import os
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplp

# TODO: make a class with methods various outputting procedures

# use config?
# else: enter in terminal.

config = configparser.ConfigParser({'tuple': lambda s: tuple(int(a) for a in s.split(','))})
config.read('config.ini')

class Application():
    def __init__(self, preset) -> None:
        self.preset = preset

def query():
    config = input()