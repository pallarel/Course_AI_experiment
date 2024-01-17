import os
import sys
import cv2
from tqdm import tqdm
from typing import Literal
import math
import yaml
import argparse
from tqdm import tqdm
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.data import random_split
import torch.multiprocessing as mp

from dataset import *
from model import *

from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
except:
    pass



class ModelRunner():
    def __init__(
            self, 
            exp_name, 
            config_path, 
            data_paths=None, 
            val_data_paths=None,
            device = 'cpu', 
            mode: Literal['train', 'test'] = 'train', 
            use_wandb=False,
            wandb_project_name = 'Test Project', 
        ) -> None:

        self.exp_name = exp_name
        self.config_path = config_path
        self.data_paths = data_paths
        self.val_data_paths = val_data_paths
        self.device = device
        self.mode = mode
        self.use_wandb = use_wandb
        self.wandb_project_name = wandb_project_name

        
        self.exp_path = 'exp/' + exp_name
        if mode == 'train':
            os.makedirs(self.exp_path, exist_ok=True)
        
        print(f'Runner mode: {self.mode}')

        if config_path:
            with open(config_path) as f:
                config_yaml = yaml.load(f, Loader=yaml.FullLoader)
                self.config_details = config_yaml
        else:
            assert mode == 'test'
            with open(os.path.join(self.exp_path, 'snapshot.yaml')) as f:
                config_yaml = yaml.load(f, Loader=yaml.FullLoader)
                config_yaml = config_yaml['config_details']


        self.target_size = config_yaml['dataset']['target_size']
        try:
            self.validation_proportion = config_yaml['dataset']['validation_proportion']
        except:
            self.validation_proportion = None

        self.learning_rate = config_yaml['model']['train']['learning_rate']
        self.max_epoch = config_yaml['model']['train']['max_epoch']
        self.batch_size = config_yaml['model']['train']['batch_size']

        try:
            self.validation_freq = config_yaml['model']['train']['validation_freq']
        except:
            self.validation_freq = None

        try:
            self.model_save_freq = config_yaml['model']['train']['model_save_freq']
        except:
            self.model_save_freq = None

        self.model_builder = config_yaml['model']['model_builder']

        print('Use wandb: ', self.use_wandb)
        
        self.model = globals()[self.model_builder](input_size=self.target_size, input_channel=3, output_dim=2).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total parameters: {}'.format(total_params))
        
        self.load_checkpoint()
        

    def load_checkpoint(self):
        print('Loading checkpoint...')
        ckpt_list = sorted(glob(os.path.join(self.exp_path, 'ckpts', 'ckpt_*.pth')))
        if len(ckpt_list) > 0:
            ckpt = torch.load(ckpt_list[-1], map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.last_iter_count = int(ckpt_list[-1].split('_')[-1][:8])
            print('Checkpoint loaded: {}'.format(ckpt_list[-1]))
        else:
            self.last_iter_count = 0
            print('No checkpoint found.')

    def save_checkpoint(self, iter_step):
        tqdm.write('Saving checkpoint...')
        os.makedirs(os.path.join(self.exp_path, 'ckpts'), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.exp_path, 'ckpts', 'ckpt_{:0>8d}.pth'.format(iter_step)))
        if self.mode == 'train':
            self.save_training_states(iter_step)

    # load checkpoints of training optimizer and ...
    def load_training_states(self):
        optimizer_ckpt_list = sorted(glob(os.path.join(self.exp_path, 'ckpts', 'optimizer_*.pth')))
        if len(optimizer_ckpt_list) > 0:
            optimizer_ckptckpt = torch.load(optimizer_ckpt_list[-1], map_location=self.device)
            self.optimizer.load_state_dict(optimizer_ckptckpt)

    def save_training_states(self, iter_step):
        os.makedirs(os.path.join(self.exp_path, 'ckpts'), exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(self.exp_path, 'ckpts', 'optimizer_{:0>8d}.pth'.format(iter_step)))

    def training_setup(self):

        if self.validation_proportion is not None:
            self.dataset = CatDogDataset(self.data_paths, self.target_size)
            # split the full dataset into train and validation
            val_size = int(self.validation_proportion * len(self.dataset))
            train_size = len(self.dataset) - val_size
            # specify a fixed seed for the generator, ensure the random split is reproducible
            self.generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], self.generator)

        elif self.val_data_paths is not None:
            self.train_dataset = CatDogDataset(self.data_paths, self.target_size)
            self.val_dataset = CatDogDataset(self.val_data_paths, self.target_size)

        # No validation
        else:
            self.train_dataset = CatDogDataset(self.data_paths, self.target_size)
            self.val_dataset = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.load_training_states()
        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, )

        # read existing curves
        #curve_path = os.path.join(self.exp_path, 'np_logs')
        #if os.path.exists()
        if self.use_wandb:
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project=self.wandb_project_name, 
                name=self.exp_name, 
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.learning_rate,
                    "epochs": self.max_epoch,
                })
        else:
            self.writer = SummaryWriter(log_dir=os.path.join(self.exp_path, 'logs'))

    def testing_setup(self):
        self.dataset = CatDogDataset(self.data_paths, self.target_size)

    def save_configuration(self):
        cur_exp_config = {
            'data_paths': self.data_paths,
            'config_path': self.config_path,
            'config_details': self.config_details,
        }
        cur_data_config = {
            'data_augmentations': {
                'transforms': self.train_dataset.transforms
            }
        }
        with open(os.path.join(self.exp_path, 'snapshot.yaml'), 'w') as outfile:
            yaml.dump(cur_exp_config, outfile, default_flow_style=False)
        with open(os.path.join(self.exp_path, 'snapshot_dataset.yaml'), 'w') as outfile:
            yaml.dump(cur_data_config, outfile, default_flow_style=False)

    def train(self):
        assert(self.mode == 'train')
        if self.last_iter_count > 0:
            print('Continue training from a previous checkpoint.')
        self.training_setup()
        self.save_configuration()
        self.model.train()
        train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        print('Start training.')
        print('{} data as train, {} as validation.'.format(len(self.train_dataset), len(self.val_dataset)))
        self.iter_count = self.last_iter_count
        iter_bar = tqdm(range(self.max_epoch * len(self.train_dataset)), initial=self.iter_count, dynamic_ncols=True)
        
        for e in range(self.iter_count // len(self.train_dataset), self.max_epoch):
            self.model.train()
            train_loss = 0.0
            for i, sample in enumerate(train_dataloader):
                image_sample, label_sample = sample['image'].to(self.device), sample['label'].to(self.device)
                cur_batch = image_sample.shape[0]

                # (N, 2)
                predict = self.model(image_sample)

                # loss
                loss = nn.CrossEntropyLoss(reduction='sum')(predict, label_sample)
                train_loss += loss.item()
                # only take the average on the batch dimension
                loss /= cur_batch

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                iter_bar.update(cur_batch)
                iter_bar.set_description('[epoch:{}, lr:{}]'.format(e, self.optimizer.param_groups[0]['lr']))
                self.iter_count += cur_batch

                # log the training loss
                if not self.use_wandb:
                    self.writer.add_scalar('Loss/training_loss', loss, self.iter_count)
            
            train_loss /= len(self.train_dataset)
            # log current epoch and loss to the terminal
            tqdm.write('[e: {}] loss: {:.8f}'.format(e, loss.item()))
            if self.use_wandb:
                wandb.log({'training_loss': train_loss}, step=e+1)
            
            # save checkpoint
            if self.model_save_freq and (e + 1) % self.model_save_freq == 0:
                self.save_checkpoint((e + 1) * len(self.train_dataset))
            # validation
            if self.validation_freq and (e + 1) % self.validation_freq == 0:
                tqdm.write('Validating...')
                self.model.eval()
                val_loss = 0.0
                correct_count = 0
                with torch.no_grad():
                    for i, sample in enumerate(val_dataloader):
                        val_image_sample, val_label_sample = sample['image'].to(self.device), sample['label'].to(self.device)
                        val_predict = self.model(val_image_sample)

                        loss = nn.CrossEntropyLoss(reduction='sum')(val_predict, val_label_sample)
                        val_loss += loss.item()

                        correct_count += torch.sum(torch.argmax(val_predict, dim=1) == torch.argmax(val_label_sample, dim=1)).item()
                        
                
                val_loss /= len(self.val_dataset)
                accuracy = correct_count / len(self.val_dataset)
                tqdm.write('validation_loss: {:.8f}, accuracy: {:.2f}'.format(val_loss, accuracy))

                #wandb.log({'validation_loss': val_loss}, commit=False, step=e+1)
                if self.use_wandb:
                    wandb.log({'validation_loss': val_loss, 'accuracy': accuracy}, step=e+1)
                else:
                    self.writer.add_scalar('Loss/validation_loss', val_loss, e + 1)
                    self.writer.add_scalar('accuracy', accuracy, e + 1)

        
        self.save_checkpoint(self.max_epoch * len(self.train_dataset))
        tqdm.write('Complete training.')

    @torch.no_grad()
    def predict(self, data: np.ndarray) -> Union[int, Sequence[int]]:
        self.model.eval()

        if len(data.shape) == 3:
            data = data[None, ...]

        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        predicted : np.ndarray = self.model(data).cpu().numpy().tolist()
        predicted = np.argmax(predicted, axis=-1)
        predicted_list = predicted.tolist()
        
        return predicted_list

    @torch.no_grad()
    def test(self):
        assert(self.mode == 'test')
        self.testing_setup()
        self.model.eval()

        test_dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        print('Start testing.')
        print('{} data.'.format(len(self.dataset)))

        iter_bar = tqdm(range(len(self.dataset)), dynamic_ncols=True, desc='Test progress')
        iter_count = 0
        
        test_loss = 0.0
        correct_count = 0
        for sample in test_dataloader:
            image_sample, label_sample = sample['image'].to(self.device), sample['label'].to(self.device)
            cur_batch = image_sample.shape[0]
            predict = self.model(image_sample)

            loss = nn.CrossEntropyLoss(reduction='sum')(predict, label_sample)
            test_loss += loss.item()

            correct_count += torch.sum(torch.argmax(predict, dim=1) == torch.argmax(label_sample, dim=1)).item()
            
            iter_bar.update(cur_batch)
            iter_count += cur_batch

        test_loss /= len(self.dataset)
        accuracy = correct_count / len(self.val_dataset)
        tqdm.write('Test complete. Average loss: {:.8f}, accuracy: {:.2f}'.format(test_loss, accuracy))

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--data_paths', type=str, nargs='+', default=[])
    parser.add_argument('--val_data_paths', type=str, nargs='+', default=None)
    parser.add_argument('--exp_name', type=str, default='test_exp',
                        help='Experiment name. The experiment result will be saved at "exp/<exp_name>".')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default=None)
    
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    args.data_paths = ['../data1/train/cat', '../data1/train/dog']
    args.val_data_paths = ['../data1/test/cat', '../data1/test/dog']
    args.exp_name = 'test_exp_2'
    args.config_path = 'configs/catdog_another.yaml'
    args.device = 'cpu'

    runner = ModelRunner(
        exp_name=args.exp_name, 
        config_path=args.config_path, 
        data_paths=args.data_paths, 
        val_data_paths=args.val_data_paths,
        device=args.device, 
        mode=args.mode,
        use_wandb=args.use_wandb, 
        wandb_project_name=args.project_name
    )
    runner.run()
    
        
