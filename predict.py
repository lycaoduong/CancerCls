import os
from tqdm import tqdm
from src.predictor import Predictor
import torch


model = 'resnet101'
ckpt = 'bestVal.pt'
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
engine = Predictor(model=model, ckpt=ckpt, device=device)

def predict_img(img_dir):
    engine.predict(img_dir)
    print("Done")

def predict_folder(folder):
    all_files = os.listdir(folder)
    for file in tqdm(all_files):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_dir = os.path.join(folder, file)
            engine.predict(img_dir)
    print("Done, open folder to see result")


if __name__ == '__main__':
    img_dir = 'test.jpg'
    predict_img(img_dir)
