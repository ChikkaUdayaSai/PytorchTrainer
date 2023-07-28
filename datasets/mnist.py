import numpy as np
from torchvision import datasets
import albumentations as A

from .generic import GenericDataset

class MNISTAlbument(datasets.MNIST):

    def __init__(self, root, alb_transform=None, **kwargs):
        super(MNISTAlbument, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform
    
    def __getitem__(self, index):
        img, target = super(MNISTAlbument, self).__getitem__(index)
        if self.alb_transform is not None:
            img = np.array(img)
            img = self.alb_transform(image=img)['image']
        return img, target
    
    def __str__(self):
        return "MNIST"
    
class MNIST(GenericDataset):

    name = MNISTAlbument
    mean = (0.1307,)
    std = (0.3081,)
    default_albumentations = [
        A.Rotate(limit=7, p=1.),
        A.Perspective(scale=0.2, p=0.5, fit_output=False)
    ]
