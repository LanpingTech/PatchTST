import os
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings


warnings.filterwarnings('ignore')

class CustomTSDataset(Dataset):
    def __init__(self, data_paths, target_path, scaler=None, target_scaler=None):
        self.data_paths = data_paths
        self.target_path = target_path
        self.scaler = scaler
        self.target_scaler = target_scaler

    def __getitem__(self, index):
        x = np.load(self.data_paths[index])
        y = np.load(self.target_path[index]).reshape(-1, 1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.target_scaler is not None:
            y = self.target_scaler.transform(y)

        return x, y
    
    def __len__(self):
        return len(self.data_paths)
    
class PredictTSDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.data = []
        self.read_data()

    def read_data(self):
        self.data = []
        f = h5py.File('ori_data/data.h5', 'r')
        data_num = f['data'].shape[0]
        for i in tqdm(range(data_num)):
            self.data.append(f['data'][i])


    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)
    
def load_data(data_dir, batch_size=32, split_ratio=0.2, random_state=42):
    print('Loading data...')
    data_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('data')])
    target_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('target')])
    assert len(data_paths) == len(target_paths), 'Number of data and target files must be the same'

    # Split data
    data_paths_train_val, data_paths_test, target_paths_train_val, target_paths_test = train_test_split(data_paths, target_paths, test_size=split_ratio, random_state=random_state)
    data_paths_train, data_paths_val, target_paths_train, target_paths_val = train_test_split(data_paths_train_val, target_paths_train_val, test_size=split_ratio, random_state=random_state)

    # scaler
    # scaler = StandardScaler()
    # target_scaler = StandardScaler()
    # pbar = tqdm(total=len(data_paths_train_val), desc='Fitting scaler')
    # pbar.set_description('Fitting scaler')
    # for i in range(len(data_paths_train_val)): # normalization
    #     x = np.load(data_paths_train_val[i])
    #     y = np.load(target_paths_train_val[i]).reshape(-1, 1)
    #     scaler.partial_fit(x)
    #     target_scaler.partial_fit(y)
    #     pbar.update(1)
    # pbar.close()
    scaler = None
    target_scaler = None

    # Create datasets
    train_dataset = CustomTSDataset(data_paths_train, target_paths_train, scaler, target_scaler)
    val_dataset = CustomTSDataset(data_paths_val, target_paths_val, scaler, target_scaler)
    test_dataset = CustomTSDataset(data_paths_test, target_paths_test, scaler, target_scaler)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader