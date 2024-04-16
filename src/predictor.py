import cv2
import numpy as np
import timm
import torch
import os
import math
import onnxruntime
# from network.model import Model


cls_name = ["breast_cancer", "breast_normal", "thyroid_cancer", "thyroid_normal", "lung_cancer", "lung_normal"]

class Predictor(object):
    def __init__(self, model, ckpt, device, nc=6, img_size=512):
        self.model = timm.create_model(model, pretrained=False, num_classes=nc).to(device)
        # self.model = Model(num_cls=nc, model_name=model)
        assert ckpt is not None
        self.ckpt = ckpt
        weight = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(weight, strict=True)
        # self.model.model.load_state_dict(weight, strict=True)
        self.model.eval()
        self.device = device
        self.img_size = img_size
        self.font_scale = 2e-3
        self.thickness_scale = 1e-3
    def export_onnx(self):
        filename, file_extension = os.path.splitext(self.ckpt)
        save_path = '{}.onnx'.format(filename)
        generated_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        o = self.model(generated_input)
        torch.onnx.export(
            self.model,
            generated_input,
            save_path,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=11
        )
        print('Done')
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


class PredictorONNX(object):
    def __init__(self, onnx_dir, device, img_size=512):
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 0.5 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])

        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.model = session

        self.device = device
        self.img_size = img_size
        self.font_scale = 2e-3
        self.thickness_scale = 1e-3

    def preprocess(self, imageBGR):
        blob = cv2.dnn.blobFromImage(imageBGR, scalefactor=1/255.0, size=(self.img_size, self.img_size), swapRB=True, crop=False)
        return blob
    
    def __call__(self, img):
        height, width, c = img.shape
        font_scale = min(width, height) * self.font_scale
        thickness = math.ceil(min(width, height) * self.thickness_scale)
        blob = self.preprocess(img)

        rs = self.model.run([self.output_name], {self.input_name: blob})[0][0]

        conf = np.max(rs)
        predict = np.argmax(rs)
        cv2.putText(img, '{}: {:.2f}'.format(cls_name[int(predict.item())], conf.item()), (30, 30), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
        return img

    def predict(self, image_dir):
        filename, file_extension = os.path.splitext(image_dir)
        img = cv2.imread(image_dir)
        height, width, c = img.shape
        font_scale = min(width, height) * self.font_scale
        thickness = math.ceil(min(width, height) * self.thickness_scale)
        blob = self.preprocess(img)

        rs = self.model.run([self.output_name], {self.input_name: blob})[0][0]

        conf = np.max(rs)
        predict = np.argmax(rs)
        cv2.putText(img, '{}: {:.2f}'.format(cls_name[int(predict.item())], conf.item()), (30, 50), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), thickness)
        save_dir = '{}_predicted.png'.format(filename)
        cv2.imwrite(save_dir, img)
