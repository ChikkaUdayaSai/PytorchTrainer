import cv2
import numpy as np
from torchvision import datasets
import albumentations as A

from .generic import GenericDataset

class CIFAR10Albument(datasets.CIFAR10):

    def __init__(self, root, alb_transform=None, **kwargs):
        super(CIFAR10Albument, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform
    
    def __getitem__(self, index):
        img, target = super(CIFAR10Albument, self).__getitem__(index)
        if self.alb_transform is not None:
            img = np.array(img)
            img = self.alb_transform(image=img)['image']
        return img, target
    
    def __str__(self):
        return "CIFAR10 Albumentations"
    
    def __repr__(self) -> str:
        return "CIFAR10 Albumentations"
    
class CIFAR10(GenericDataset):

    name = CIFAR10Albument
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2427, 0.2434, 0.2615)

    default_albumentations = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(40, 40, p=1),
        A.RandomCrop(32, 32, p=1),
        A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=1),
        A.CenterCrop(32, 32, p=1)
    ]

    def __str__(self):
        return "CIFAR10"
    
    def __repr__(self) -> str:
        return "CIFAR10"