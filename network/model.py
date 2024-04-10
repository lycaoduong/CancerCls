import torch.nn as nn
import torch
import timm


class Model(nn.Module):
    def __init__(self, in_ch=3, num_cls=6, model_name='resnet34'):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_cls)

    def forward(self, image):
        x = self.model(image)
        # Comment it if Loss Function has activation function
        # x = torch.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    device = 'cpu'
    model = Model(model_name='inception_v4').to(device)
    rs = torch.randn(1, 3, 512, 512).to(device)
    o = model(rs)
    print(o)
