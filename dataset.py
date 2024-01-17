import os
import sys
import cv2
from glob import glob
from tqdm import tqdm
from typing import Literal, Sequence, Union, Any
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import torchvision.transforms as tvtrans
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.curdir))

    
class CatDogDataset(Dataset):
    def __init__(self, data_paths, size):
        self.images_dir_list = []
        self.target_size = size

        self.transforms = tvtrans.Compose([
            tvtrans.RandomResizedCrop((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR), 
            tvtrans.RandomHorizontalFlip(p=0.5)
            #tvtrans.Resize((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR, antialias=True)
        ])

        for data_path in data_paths:
            current_dir_list = sorted(glob(os.path.join(data_path, '*.jpg')))
            self.images_dir_list += current_dir_list
            print('Found {} sample images in directory "{}".'.format(len(current_dir_list), data_path))

    def __len__(self) -> int:
        return len(self.images_dir_list)
    
    def __getitem__(self, index) -> dict[Literal['image', 'label'], Any]:
        image_dir : str = self.images_dir_list[index]
        image_dir = image_dir.replace('\\', '/')
        image_type = image_dir.split('/')[-1].split('.')[0]

        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        image = image.transpose(-1, 0, 1)
        image = torch.tensor(image, dtype=torch.float32, device='cpu')

        image = self.transforms(image)
        
        label = torch.tensor([1, 0]) if image_type == 'cat' else torch.tensor([0, 1])
        label = label.to(device='cpu', dtype=torch.float32)

        sample = {'image': image, 'label': label}
        return sample


class CatDogTestDataset(Dataset):
    def __init__(self, data_paths, size):
        self.images_dir_list = []
        self.target_size = size

        self.transforms = tvtrans.Compose([
            tvtrans.Resize((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR, antialias=True)
        ])

        for data_path in data_paths:
            current_dir_list = sorted(glob(os.path.join(data_path, '*.jpg')))
            self.images_dir_list += current_dir_list
            print('Found {} sample images in directory "{}".'.format(len(current_dir_list), data_path))

    def __len__(self) -> int:
        return len(self.images_dir_list)
    
    def __getitem__(self, index) -> dict[Literal['image', 'label'], Any]:
        image_dir : str = self.images_dir_list[index]
        image_dir = image_dir.replace('\\', '/')
        image_type = image_dir.split('/')[-1].split('.')[0]

        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        image = image.transpose(-1, 0, 1)
        image = torch.tensor(image, dtype=torch.float32, device='cpu')

        image = self.transforms(image)
        
        label = torch.tensor([1, 0]) if image_type == 'cat' else torch.tensor([0, 1])
        label = label.to(device='cpu', dtype=torch.float32)

        sample = {'image': image, 'label': label}
        return sample


class CatDogSmallDataset(Dataset):
    def __init__(self, data_paths, size):
        self.images_list = []
        self.label_list = []
        self.target_size = size

        self.transforms = tvtrans.Compose([
            tvtrans.RandomResizedCrop((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR)
            #tvtrans.Resize((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR, antialias=True)
        ])

        for data_path in data_paths:
            current_dir_list = sorted(glob(os.path.join(data_path, '*.jpg')))
            print('Found {} sample images in directory "{}".'.format(len(current_dir_list), data_path))
            for single_dir in current_dir_list:
                image = cv2.imread(single_dir)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
                image = image.transpose(-1, 0, 1)

                image_dir = single_dir.replace('\\', '/')
                image_type = image_dir.split('/')[-1].split('.')[0]
                label = [1, 0] if image_type == 'cat' else [0, 1]
                
                self.images_list.append(image)
                self.label_list.append(label)

    def __len__(self) -> int:
        return len(self.images_list)
    
    def __getitem__(self, index) -> dict[Literal['image', 'label'], Any]:
        image = torch.tensor(self.images_list[index], dtype=torch.float32, device='cpu')

        image = self.transforms(image)
        
        label = torch.tensor(self.label_list[index], dtype=torch.float32, device='cpu')

        sample = {'image': image, 'label': label}
        return sample


if __name__ == '__main__':

    def _dataset_test():
        dataset = CatDogDataset(['data/train/cat', 'data/train/dog'], 224)

        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        if(len(dataloader) > 20):
            print('Will only take the first 20 samples.')
        for i, sample in enumerate(dataloader):
            sample_image = sample['image'].cpu().numpy()[0]
            sample_label = sample['label'].cpu().numpy()[0]
            
            fig = plt.figure(dpi=150)
            ax = fig.add_subplot()
            ax.imshow(sample_image.transpose(1, 2, 0))
            plt.title('cat' if np.argmax(sample_label) == 0 else 'dog')
            plt.show()
            plt.close()

            #break
            if i >= 20:
                break


    _dataset_test()
    #_image_data_test()
