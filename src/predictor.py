import cv2
import numpy as np
import timm
import torch
import os
import math


cls_name = ["breast_cancer", "breast_normal", "thyroid_cancer", "thyroid_normal", "lung_cancer", "lung_normal"]

class Predictor(object):
    def __init__(self, model, ckpt, device, nc=6, img_size=512):
        self.model = timm.create_model(model, pretrained=False, num_classes=nc).to(device)
        assert ckpt is not None
        weight = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(weight, strict=True)
        self.model.eval()
        self.device = device
        self.img_size = img_size
        self.font_scale = 2e-3
        self.thickness_scale = 1e-3
    def preprocess(self, imageBGR):
        image = imageBGR[:, :, ::-1]
        image = image / 255.0
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        blob = torch.from_numpy(image).to(torch.float32)
        blob = torch.unsqueeze(blob, dim=0)
        blob = torch.permute(blob, (0, 3, 1, 2))
        return blob.to(self.device)
    def __call__(self, img):
        height, width, c = img.shape
        font_scale = min(width, height) * self.font_scale
        thickness = math.ceil(min(width, height) * self.thickness_scale)
        blob = self.preprocess(img)
        with torch.no_grad():
            o = self.model(blob)
        rs = torch.softmax(o, dim=1)
        conf, predict = torch.max(rs, dim=1)
        cv2.putText(img, '{}: {:.2f}'.format(cls_name[int(predict.item())], conf.item()), (30, 30), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
        return img

    def predict(self, image_dir):
        filename, file_extension = os.path.splitext(image_dir)
        img = cv2.imread(image_dir)
        height, width, c = img.shape
        font_scale = min(width, height) * self.font_scale
        thickness = math.ceil(min(width, height) * self.thickness_scale)
        blob = self.preprocess(img)
        with torch.no_grad():
            o = self.model(blob)
        rs = torch.softmax(o, dim=1)
        conf, predict = torch.max(rs, dim=1)
        cv2.putText(img, '{}: {:.2f}'.format(cls_name[int(predict.item())], conf.item()), (30, 50), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
        save_dir = '{}_predicted.png'.format(filename)
        cv2.imwrite(save_dir, img)
