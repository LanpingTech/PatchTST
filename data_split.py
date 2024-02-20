import h5py
import numpy as np
from tqdm import tqdm

# read h5 file
f = h5py.File('ori_data/data.h5', 'r')
print(f['data'].shape)
print(f['target'].shape)

data_num = f['data'].shape[0]
for i in tqdm(range(data_num)):
    np.save('data/data_{}.npy'.format(i), f['data'][i])
    np.save('data/target_{}.npy'.format(i), f['target'][i])

