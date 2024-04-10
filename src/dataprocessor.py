import cv2
import os
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET


class DataProcessor(object):
    def __init__(self, root):
        self.root = root
    
    def makeCSVfromFolder(self, folder, label=0):
        dir = os.path.join(self.root, folder)
        allfiles = os.listdir(dir)
        fns = []
        lbs = []
        for file in tqdm(allfiles):
            if file.endswith('.jpg') or file.endswith('.png'):
                fns.append(os.path.join(folder, file))
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
