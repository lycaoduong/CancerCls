import numpy as np
import cv2
from torch.utils.data import Dataset
import os


class CancerDataset(Dataset):
    def __init__(self, root_dir, df, nc, transform=None, **kwargs):
        self.img_dir = root_dir
        self.dataframe = df
        self.nc = nc
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image, label = self.load_image(idx)
        data_loader = {'image': image, 'label': label, 'fn': None}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_image(self, idx):
        fn = self.dataframe.get('fname')[idx]
        label = self.dataframe.get('class')[idx]
        img_dir = os.path.join(self.img_dir, fn)
        img = cv2.imread(img_dir)
        img = img[:, :, ::-1] #BGR2RGB
        return img, np.array(label)
    