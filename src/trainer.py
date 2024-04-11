import datetime
import os
import json
from utils.utils import YamlRead
from utils import tranform as tr
from utils.dataloader import CancerDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm.autonotebook import tqdm
import traceback
import numpy as np
import pandas as pd
import timm
from utils.utils import plot_data
from torchvision import transforms


class Trainer(object):
    def __init__(self, train_opt):
        self.project = train_opt.project
        self.model_name = train_opt.model
        self.dataset = train_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)

        self.save_dir = './runs/train/{}/{}/{}/{}/'.format(self.project, self.model_name, self.dataset, date_time)
        os.makedirs(self.save_dir, exist_ok=True)

        # Save train parameters
        with open('{}/train_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(train_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.root_dir = dataset_configs.root_dir
        self.train_dir = dataset_configs.train_dir
        self.val_dir = dataset_configs.val_dir
        self.cls_name = dataset_configs.cls_name
        self.num_cls = dataset_configs.num_cls

        train_df = pd.read_csv(self.train_dir, encoding='utf-8')
        val_df = pd.read_csv(self.val_dir, encoding='utf-8')

        self.batch_size = train_opt.batch_size
        self.img_size = train_opt.imgsize

        train_transforms = [
            tr.Normalizer(),
            tr.Resizer(img_size=self.img_size)
        ]

        training_set = CancerDataset(root_dir=self.root_dir, df=train_df, nc=self.num_cls,
                                     transform=transforms.Compose(train_transforms))

        train_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': train_opt.num_worker
        }

        self.training_generator = DataLoader(training_set, collate_fn=tr.collater, **train_params)

        validation_transforms = [
            tr.Normalizer(),
            tr.Resizer(img_size=self.img_size)
        ]

        val_set = CancerDataset(root_dir=self.root_dir, df=val_df, nc=self.num_cls,
                                transform=transforms.Compose(validation_transforms))

        val_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': train_opt.num_worker
        }
        self.val_generator = DataLoader(val_set, collate_fn=tr.collater, **val_params)

        self.device = train_opt.device

        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=self.num_cls).to(self.device)
        if train_opt.ckpt is not None:
            weight = torch.load(train_opt.ckpt, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)

        self.criterion = nn.CrossEntropyLoss()

        self.opti = train_opt.optimizer
        self.l_rate = train_opt.lr

        if self.opti == 'adamw':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.l_rate)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.l_rate,
                                             momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.num_iter_per_epoch = len(self.training_generator)
        self.step = 0
        self.best_train_loss = 1e5
        self.best_val_loss = 1e5
        self.current_val_loss = 0.0
        self.epochs = train_opt.epochs

    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.training_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                images, labels = data['image'], data['label']
                images = images.to(self.device)
                images = torch.permute(images, (0, 3, 1, 2))
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        self.step, epoch+1, self.epochs, iter + 1, self.num_iter_per_epoch, loss.item())
                progress_bar.set_description(descriptor)
                self.step += 1


            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        self.scheduler.step(np.mean(epoch_loss))

        mean_loss = np.mean(epoch_loss)

        if self.best_train_loss > mean_loss:
            self.best_train_loss = mean_loss
            self.save_checkpoint(self.model, self.save_dir, 'bestTrain.pt')

        train_descrip = '[Train] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch+1, mean_loss)
        print(train_descrip)

    def validation(self, epoch):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    images, labels = data['image'], data['label']
                    images = images.to(self.device)
                    images = torch.permute(images, (0, 3, 1, 2))
                    labels = labels.to(self.device)

                    output = self.model(images)
                    loss = self.criterion(output, labels)

                    epoch_loss.append(loss.item())

                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        epoch * len(progress_bar) + iter, epoch, self.epochs, iter + 1, len(progress_bar), loss.item())
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        mean_loss = np.mean(epoch_loss)
        self.current_val_loss = mean_loss
        val_descrip = '[Validation] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch + 1, mean_loss)
        print(val_descrip)

        self.save_checkpoint(self.model, self.save_dir, 'last.pt')

        if self.best_val_loss > mean_loss:
            self.best_val_loss = mean_loss
            self.save_checkpoint(self.model, self.save_dir, 'bestVal.pt')

    def start(self):
        # self.data_analysis()
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validation(epoch)

    def save_checkpoint(self, model, saved_path, name):
        torch.save(model.state_dict(), saved_path + name)

    def data_analysis(self):
        train_data_dis = np.zeros((1, self.num_cls))
        val_data_dis = np.zeros((1, self.num_cls))
        df = pd.read_csv(self.train_dir)
        for cls in range(self.num_cls):
            rslt_df = df[df['class'] == cls]
            train_data_dis[:, cls] = len(rslt_df)
        plot_data(data_dis=train_data_dis, cls_name=self.cls_name, save_dir=self.save_dir, save_name="train.png")
        df = pd.read_csv(self.val_dir)
        for cls in range(self.num_cls):
            rslt_df = df[df['class'] == cls]
            val_data_dis[:, cls] = len(rslt_df)
        plot_data(data_dis=val_data_dis, cls_name=self.cls_name, save_dir=self.save_dir, save_name="val.png")
