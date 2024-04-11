import cv2
import os
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


class DataProcessor(object):
    def __init__(self, root):
        self.root = root

    def merge_csv(self, list_csv, save_name='merge.csv'):
        dfs = list()
        for i, f in enumerate(list_csv):
            data = pd.read_csv(os.path.join(self.root, f))
            dfs.append(data)
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(self.root,save_name), index=False)

    def split_train_val_test(self, csv_dir, ratio=[0.8, 0.1, 0.1], save_name='lungcancer'):
        data = pd.read_csv(os.path.join(self.root, csv_dir))
        # data, _ = train_test_split(data, train_size=0.1)
        val_test_r = ratio[1] + ratio[2]
        train, val_test = train_test_split(data, test_size=val_test_r)
        val, test = train_test_split(val_test, test_size=ratio[2] / val_test_r)

        train.to_csv(os.path.join(self.root, '{}_train.csv'.format(save_name)), index=False)
        val.to_csv(os.path.join(self.root, '{}_val.csv'.format(save_name)), index=False)
        test.to_csv(os.path.join(self.root, '{}_test.csv'.format(save_name)), index=False)

    def makeCSVfromFolder(self, parent_folder, folder, label=0):
        dir = os.path.join(self.root, parent_folder, folder)
        allfiles = os.listdir(dir)
        fns = []
        lbs = []
        for file in tqdm(allfiles):
            if file.endswith('.jpg') or file.endswith('.png'):
                fns.append(os.path.join(parent_folder, folder, file))
                lbs.append(label)
        df = pd.DataFrame(data=zip(fns, lbs), columns=['fname', 'class'])
        save_name = '{}.csv'.format(folder)
        save_dir = os.path.join(self.root, save_name)
        df.to_csv(save_dir, index=False)
    
    def makeCSVfromXML(self, folder, tirads):
        dir = os.path.join(self.root, folder)
        allfiles = os.listdir(dir)
        fns = []
        lbs = []
        for file in tqdm(allfiles):
            if file.endswith('.xml'):
                filename, file_extension = os.path.splitext(file)
                tree = ET.parse(os.path.join(dir, file))
                root = tree.getroot()
                type = root.find("tirads").text
                for mark in root.findall("mark"):
                    image_idx = mark.find("image").text
                    img_fn = '{}_{}.jpg'.format(filename, image_idx)

                    fns.append(os.path.join(folder, img_fn))
                    if type is not None:
                        if 'cancer' in tirads[type]:
                            lbs.append(2)
                        else:
                            lbs.append(3)
                    else:
                        lbs.append(3)
        
                df = pd.DataFrame(data=zip(fns, lbs), columns=['fname', 'class'])
        save_name = '{}.csv'.format(folder)
        save_dir = os.path.join(self.root, save_name)
        df.to_csv(save_dir, index=False)
