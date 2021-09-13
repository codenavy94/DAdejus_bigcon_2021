import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np

def unpickle(file, batch_num):
    import pickle
    import os
    batch_file = 'data_batch_' + str(batch_num)
    file_path =  os.path.join(file, batch_file)
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

def make_dataset( dict):
    return dict[b'data'].reshape(len(dict[b'data']), 3, 32, 32).transpose(0, 2, 3, 1), np.array(dict[b'labels'])

def unmat(file):
    from scipy import io
    mat_file = io.loadmat(file)
    return mat_file



def load_data(path):
    import pandas as pd
    return pd.read_csv(path)

class SequenceDataset(data.Dataset):

    def __init__(self, X, y):
        super(SequenceDataset, self).__init__()
        self.X = X
        self.y  = y



    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class ImageDataset(data.Dataset):

    def __init__(self, data):
        super(ImageDataset, self).__init__()
        self.X, self.y = self.make_dataset(data)
        # self.X_len = self.X.shape[0]


    def make_dataset(self, dict):
        return dict[b'data'].reshape(len(dict[b'data']), 3, 32, 32).astype('float32'), np.array(dict[b'labels'])
        # return dict[b'data'].reshape(len(dict[b'data']), 3, 32, 32).transpose(0, 2, 3, 1), np.array(dict[b'labels'])

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class DataLoader(object):

    def __init__(self):
        pass

    def make_mini_batch(self, mini):



        return

