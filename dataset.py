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
            tvtrans.RandomResizedCrop((size, size), interpolation=tvtrans.InterpolationMode.BILINEAR, antialias=True), 
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


class ChineseTitleTokenizer():
    def __init__(self):
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def process(self, text: str) -> Sequence[torch.Tensor]:
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=30)  
        
        tokens = encoding['input_ids']
        mask = encoding['attention_mask']

        tokens = torch.tensor(tokens, device='cpu')
        mask = torch.tensor(mask, dtype=torch.bool, device='cpu')

        return tokens, mask

class ChineseTitleDataset(Dataset):
    
    def __init__(self, data_paths):
        from transformers import BertTokenizer
        import pandas as pd
        
        self.sentence_data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        for data_path in data_paths:
            df = pd.read_excel(data_path)

            for index, row in df.iterrows():
                texts = row['数据']
                label = row['标签']
                label = [1, 0] if label == 0 else [0, 1]
                sample = {'seq': texts, 'label': label}
                encoding = self.tokenizer(texts, padding="max_length", truncation=True, max_length=30)  
                
                tokens = encoding['input_ids']
                mask = encoding['attention_mask']
                sample['tokens'] = tokens
                sample['mask'] = mask

                self.sentence_data.append(sample)

        print('Found {} samples.'.format(len(self.sentence_data)))

    def __len__(self) -> int:
        return len(self.sentence_data)
    
    def __getitem__(self, index) -> dict[Literal['seq', 'tokens', 'mask', 'label'], Any]:
        sample = self.sentence_data[index]
        sample['tokens'] = torch.tensor(sample['tokens'], device='cpu')
        sample['mask'] = torch.tensor(sample['mask'], dtype=torch.bool, device='cpu')
        sample['label'] = torch.tensor(sample['label'], dtype=torch.float32, device='cpu')
        return sample


if __name__ == '__main__':

    def _dataset_catdog_test():
        dataset = CatDogDataset(['data/catdog/train/cat', 'data/catdog/train/dog'], 224)

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
                
    def _dataset_chinese_test():
        dataset = ChineseTitleDataset(['data/chinese/train.xlsx'])

        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        if(len(dataloader) > 20):
            print('Will only take the first 20 samples.')
        for i, sample in enumerate(dataloader):
            sample_seq = sample['seq'][0]
            sample_label = sample['label'].cpu().numpy()[0]
            sample_tokens = sample['tokens'].cpu().numpy()[0]
            sample_mask = sample['mask'].cpu().numpy()[0]
            
            print(f'[{sample_label}]  seq: {sample_seq}, tokens: {sample_tokens}, mask: {sample_mask},   [{sample_tokens.shape}|{sample_mask.shape}]')

            #break
            if i >= 20:
                break


    #_dataset_catdog_test()
    _dataset_chinese_test()
