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


class Evaluator(object):
    def __init__(self, eval_opt):
        self.project = eval_opt.project
        self.model_name = eval_opt.model
        self.dataset = eval_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)

        self.save_dir = './runs/eval/{}/{}/{}/{}/'.format(self.project, self.model_name, self.dataset, date_time)
        os.makedirs(self.save_dir, exist_ok=True)

        # Save train parameters
        with open('{}/train_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(eval_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.root_dir = dataset_configs.root_dir
        self.test_dir = dataset_configs.test_dir
        self.cls_name = dataset_configs.cls_name
        self.num_cls = dataset_configs.num_cls

        test_df = pd.read_csv(self.test_dir, encoding='utf-8')

        self.batch_size = eval_opt.batch_size
        self.img_size = eval_opt.imgsize

        test_transforms = [
            tr.Normalizer(),
            tr.Resizer(img_size=self.img_size)
        ]

        test_set = CancerDataset(root_dir=self.root_dir, df=test_df, nc=self.num_cls,
                                 transform=transforms.Compose(test_transforms))


        test_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': eval_opt.num_worker
        }
        self.test_generator = DataLoader(test_set, collate_fn=tr.collater, **test_params)

        self.device = eval_opt.device

        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=self.num_cls).to(self.device)
        assert eval_opt.ckpt is not None
        weight = torch.load(eval_opt.ckpt, map_location=self.device)
        self.model.load_state_dict(weight, strict=True)

        self.criterion = nn.CrossEntropyLoss()
        self.confusion_matrix = np.zeros((self.num_cls, self.num_cls))

    def evaluation(self):
        self.model.eval()
        progress_bar = tqdm(self.test_generator)
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
                    predict = torch.softmax(output, dim=1)
                    predict = torch.argmax(predict, dim=1).cpu().numpy()
                    for i, lb in enumerate(labels):
                        self.confusion_matrix[int(lb), int(predict[i])] += 1

                    epoch_loss.append(loss.item())

                    descriptor = '[Eval] Iteration: {}/{}. Loss: {:.6f}.'.format(iter + 1, len(progress_bar), loss.item())
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        mean_loss = np.mean(epoch_loss)
        self.current_val_loss = mean_loss
        eval_descrip = '[Evaluation] Mean Loss: {:.6f}.'.format(mean_loss)
        print(eval_descrip)

    def start(self):
        self.evaluation()
        print("Confusion Matrix")
        print(self.confusion_matrix)
        total = 0
        for i, cls in enumerate(self.cls_name):
            total += self.confusion_matrix[i, i]
            acc = self.confusion_matrix[i, i] * 100.0 / np.sum(self.confusion_matrix[i, :])
            print("{} acc: {:.2f}%".format(cls, acc))
        mean_acc = total * 100.0 / np.sum(self.confusion_matrix)
        print("Mean acc: {:.2f}%".format(mean_acc))

    def data_analysis(self):
        eval_data_dis = np.zeros((1, self.num_cls))
        df = pd.read_csv(self.test_dir)
        for cls in range(self.num_cls):
            rslt_df = df[df['class'] == cls]
            eval_data_dis[:, cls] = len(rslt_df)
        plot_data(data_dis=eval_data_dis, cls_name=self.cls_name, save_dir=self.save_dir, save_name="eval.png")
