import argparse
from src.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser('Cancer Classification')
    parser.add_argument('-p', '--project', type=str, default='CancerBTL', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='resnet101', help='Choosing Model, i.e resnet34, resnet101, inception_v4')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted *.pt file')
    parser.add_argument('-d', '--dataset', type=str, default='cancer', help='Loading dataset configs file')
    parser.add_argument('-is', '--imgsize', type=int, default=512, help='Image size')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='Init Learning Rate')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='Init number of train epochs')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-op', '--optimizer', type=str, default='adamw', help='Choosing optimizer')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    trainer.start()
