import numpy as np
import cv2
import torch


class Normalizer(object):
    def __init__(self) -> None:
        pass
    def __call__(self, sample):
        image, label, fn = sample['image'], sample['label'], sample['fn']
        image = image / 255.0
        sample = {'image': image, 'label': label, 'fn': fn}
        return sample


class Resizer(object):
    def __init__(self, signal_size=512):
        self.signal_size = signal_size

    def __call__(self, sample):
        image, label, fn = sample['image'], sample['label'], sample['fn']
        signal = cv2.resize(signal, (self.signal_size, self.signal_size), interpolation=cv2.INTER_LINEAR)
        sample = {'image': torch.from_numpy(image).to(torch.float32), 'label': torch.from_numpy(label).to(torch.long), 'fn': fn}
        return sample


def collater(data):
    images = [s['image'] for s in data]
    labels = [s['tokenizer'] for s in data]
    fns = [s['fn'] for s in data]
    images = torch.from_numpy(np.stack(images, axis=0))
    labels = torch.from_numpy(np.stack(labels, axis=0))
    sample = {'image': images, 'label': labels, 'fn': fns}
    return sample
