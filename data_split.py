# import h5py
# import numpy as np
# from tqdm import tqdm

# # read h5 file
# f = h5py.File('ori_data/data.h5', 'r')
# print(f['data'].shape)
# print(f['target'].shape)

# data_num = f['data'].shape[0]
# for i in tqdm(range(data_num)):
#     np.save('data/data_{}.npy'.format(i), f['data'][i])
#     np.save('data/target_{}.npy'.format(i), f['target'][i])



import h5py  
import numpy as np  
from multiprocessing import Pool  
from itertools import repeat  
  
def save_chunk(chunk):  
    indices, filename = chunk  
    with h5py.File(filename, 'r') as f:  
        data_dset = f['data']  
        target_dset = f['target']  
        for idx in indices:  
            np.save('data/data_{}.npy'.format(idx), data_dset[idx])  
            np.save('data/target_{}.npy'.format(idx), target_dset[idx])  
  
if __name__ == "__main__":  
    # Define the chunk size and the number of processes  
    chunk_size = 2000  
    num_processes = 8  
  
    # Calculate the number of chunks  
    with h5py.File('ori_data/data.h5', 'r') as f:  
        data_num = f['data'].shape[0]  
        num_chunks = (data_num + chunk_size - 1) // chunk_size  
  
    # Create chunks of indices  
    chunks = [(range(i * chunk_size, min((i + 1) * chunk_size, data_num)), 'ori_data/data.h5')  
              for i in range(num_chunks)]  
  
    # Use multiprocessing to save data in parallel  
    with Pool(processes=num_processes) as pool:  
        for _ in pool.imap_unordered(save_chunk, chunks):  
            pass  # We don't actually need the return value, just for showing progress (if desired)