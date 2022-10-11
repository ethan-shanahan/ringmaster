import zipfile
import io
import os
import numpy as np
import time

test_np1 = np.random.random(100)
test_np2 = np.random.random(100)

# initial file
filename = 'npfile.npz'
try:
    np.savez_compressed(filename, test_np1=test_np1)

    with np.load(filename) as loaded_npz:
        print(type(loaded_npz))
        print(loaded_npz.files)
        print(np.array_equal(test_np1, loaded_npz['test_np1']))

    time.sleep(5)

    bio = io.BytesIO()
    np.save(bio, test_np2)

    with zipfile.ZipFile(filename, 'a') as zipf:
        # careful, the file below must be .npy
        zipf.writestr('test_np2.npy', data=bio.getbuffer().tobytes())

    with np.load(filename) as loaded_npz:
        print(type(loaded_npz))
        print(loaded_npz.files)
        print(np.array_equal(test_np1, loaded_npz['test_np1']))
        print(np.array_equal(test_np2, loaded_npz['test_np2']))
    
    time.sleep(5)
    
finally:
    os.remove(filename)