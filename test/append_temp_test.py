import io
import os
import zipfile
import numpy as np
from tempfile import TemporaryDirectory as TD


class temp_test():
    def __init__(self) -> None:
        self.temp_dir = TD(dir=os.path.realpath(os.path.dirname(__file__)))
        self.state = 0
        self.time = 0

    def append_temp(self, temp_type: str, new_data: np.ndarray) -> None:
        temp_file = self.temp_dir.name + f'\\temp_{temp_type}{self.state}.npz'
        bio = io.BytesIO()
        np.save(bio, new_data)
        with zipfile.ZipFile(temp_file, 'a') as temp_zip:
            temp_zip.writestr(f'frame{self.time}.npy', data=bio.getbuffer().tobytes())
        self.time += 1
    
    def new_state(self):
        self.state += 1

    def read_out(self):
        with np.load(self.temp_dir.name + f'\\temp_grids{self.state}.npz') as loaded_npz:
            print(type(loaded_npz))
            print(loaded_npz.files)
            print(loaded_npz[loaded_npz.files[0]])


data1 = np.arange(1, 10).reshape(3,3)
data2 = np.arange(11, 20).reshape(3,3)
data3 = np.arange(21, 30).reshape(3,3)
data4 = np.arange(31, 40).reshape(3,3)
data5 = np.arange(41, 50).reshape(3,3)

testing = temp_test()
testing.append_temp('grids', data1)
testing.read_out()
testing.append_temp('grids', data2)
testing.append_temp('grids', data3)
testing.append_temp('grids', data4)
testing.append_temp('grids', data5)
testing.read_out()
