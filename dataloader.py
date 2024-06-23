# Imports
import torch
import os
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
SPECTROGRAMS_PATH = "datasets/fsdd/spectrograms/"


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train

class AudiosDataset(Dataset):

    def __init__(self, file_path, transform=None):

        self.file_path = file_path
        self.transform = transform
        
        self.x_train = load_fsdd(spectrograms_path=SPECTROGRAMS_PATH)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        
        return self.x_train[random.randint(0,2999)]

def createDataLoader(file_path):
    audioDataset = AudiosDataset(file_path=file_path)
    dataloader = DataLoader(audioDataset,batch_size = 4,shuffle=True,num_workers = 0)
    
    return audioDataset,dataloader





