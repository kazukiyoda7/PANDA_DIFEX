from os.path import join
from torchvision import transforms
import torch
import numpy as np
import os
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset

class CIFAR10C(torch.utils.data.Dataset):
    '''
    return a list consisting of 15 Dataloaders
    '''
    def __init__(
            self,
            corruption_name = 'gaussian_noise',
            transform = transforms.Compose([transforms.ToTensor(),]),
            root='./data',
            target_transform=None,
            severity=1,
        ):
        super().__init__()
        self.corruption_name = corruption_name

        root_corrupted = os.path.join(root, 'CIFAR-10-C')
        
        # get image data
        data_path = os.path.join(root_corrupted, corruption_name + '.npy')
        data = np.load(data_path)#.permute(0,3,1,2).float()
        data = data[10000*(severity-1):10000*(severity-1)+10000]

        # get label
        target_path = os.path.join(root_corrupted, 'labels.npy')
        targets = np.load(target_path)
        targets = targets[10000*(severity-1):10000*(severity-1)+10000]
        
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)
    
    
class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_in, dataset_out, in_class=0):
        if in_class is not None:
            self.in_data = dataset_in.data[np.array(dataset_in.targets) ==  in_class] # ndaray (10000, 32, 32, 3)
            self.out_data = dataset_out.data[np.array(dataset_out.targets) ==  in_class] # ndaray (10000, 32, 32, 3)
        else:
            self.in_data = dataset_in.data  # ndaray (10000, 32, 32, 3)
            self.out_data = dataset_out.data  # ndaray (10000, 32, 32, 3)

        
        self.data = np.concatenate((self.in_data, self.out_data), axis=0) # ndaray (20000, 32, 32, 3)
        self.targets = np.concatenate(([0]*self.in_data.shape[0], [1]*self.out_data.shape[0]), axis=0) # ndaray (20000,)
        
        self.transform = dataset_in.transform
        self.target_transform = dataset_in.target_transform
        

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 2つのデータセットのサイズが同じであることを前提として、サイズを返す
        return len(self.targets)
    

class SplitInDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id_class):
        
        # get dataset_in
        self.data = dataset.data[np.array(dataset.targets) ==  id_class]

        self.targets = np.array([0]*self.data.shape[0])
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        
        self.id_class = id_class
        
        

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 2つのデータセットのサイズが同じであることを前提として、サイズを返す
        return len(self.targets)
    
    
class SplitOutDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, in_class):
        
        # get dataset_in
        self.data = dataset.data[np.array(dataset.targets) !=  in_class]

        # self.data = self.data[:1000]
        self.targets = np.array([0]*self.data.shape[0])
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 2つのデータセットのサイズが同じであることを前提として、サイズを返す
        return len(self.targets)
    
