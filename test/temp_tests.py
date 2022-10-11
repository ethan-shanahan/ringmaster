from tempfile import TemporaryDirectory as TD
import os
import time
import numpy as np

data1 = np.linspace(-5,5,100)
data2 = data1 ** 2
data3 = np.sin(data1)
data = {'data1': data1,
        'data2': data2,
        'data3': data3}
print('\ndata1:\n', data1[-5:], '\n')

my_path = os.path.realpath(os.path.dirname(__file__))
print(my_path, '\n')

temp_dir = TD(dir=my_path)
temp_file_name = temp_dir.name + '\\temp_data.npz'
print(temp_file_name, '\n')

with open(temp_file_name, 'w+b') as temp_file:
    # np.save(temp_file, data1)
    # np.save(temp_file, data2)
    np.savez(temp_file, data1=data1)
    np.savez(temp_file, data2=data2)
    np.savez(temp_file, data3=data3)

with open(temp_file_name, 'rb') as temp_file:
    saved_data = np.load(temp_file)
    print('files: ', saved_data.files)
    # saved_data1_snip = saved_data['data1'][-5:]
    # saved_data2_snip = saved_data['data2'][-5:]
    # saved_data3_snip = saved_data['data3'][-5:]

# print('saved data:\n', saved_data1_snip, '\n')
