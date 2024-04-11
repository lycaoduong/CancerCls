from src.predictor import Predictor
import torch


if __name__ == '__main__':
    model = 'resnet101'
    ckpt = 'bestVal.pt'
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    engine = Predictor(model=model, ckpt=ckpt, device=device)
    img_dir = 'test.jpg'
    engine.predict(img_dir)
    print("Done")
