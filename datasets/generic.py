import os
from abc import ABC

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt


class GenericDataset(ABC):

    name = None
    mean = None
    std = None
    classes = None
    num_classes = None
    default_albumentations = [ToTensorV2()]

    def __init__(self, batchsize=128, normalize=True, shuffle=True,augment=True, albumentations=None, num_workers=4, pin_memory=True, **kwargs):
        self.batchsize = batchsize
        self.normalize = normalize
        self.shuffle = shuffle
        self.augment = augment
        self.albumentations = albumentations or self.default_albumentations
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

        self._load_dataset()

    def _load_dataset(self):

        print(f"Loading {self.name} dataset...")
        self._load_train_dataset()
        self._load_test_dataset()

    def _fetch_train_transforms(self):

        total_transforms = []
        if self.normalize:
            total_transforms.append(A.Normalize(mean=self.mean, std=self.std))
        if self.augment and isinstance(self.albumentations, list):
            total_transforms.extend(self.albumentations)
        if ToTensorV2 not in total_transforms:
            total_transforms.append(ToTensorV2())
        return A.Compose(total_transforms)
    
    def _fetch_test_transforms(self):

        total_transforms = []
        if self.normalize:
            total_transforms.append(A.Normalize(mean=self.mean, std=self.std))
        if ToTensorV2 not in total_transforms:
            total_transforms.append(ToTensorV2())
        return A.Compose(total_transforms)


    def _load_train_dataset(self):
        
        train_transforms = self._fetch_train_transforms()

        train_data = self.name(
            root=os.path.join(self.kwargs['data_path'], 'data'),
            train=True,
            download=True,
            alb_transform=train_transforms
        )

        if self.classes is None:
            self.classes = {
                {i: c for i, c in enumerate(train_data.classes)}
            }
        self.num_classes = len(self.classes)

        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            shuffle=self.shuffle,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    
    def _load_test_dataset(self):
        
        test_transforms = self._fetch_test_transforms()

        test_data = self.name(
            root=os.path.join(self.kwargs['data_path'], 'data'),
            train=False,
            download=True,
            alb_transform=test_transforms
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_data,
            shuffle=False,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def denormalize(self, t):

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        if not self.normalize:
            return t

        mean = torch.tensor(self.mean).reshape(3, 1, 1)
        std = torch.tensor(self.std).reshape(3, 1, 1)

        return (t * std) + mean
    
    def get_classes(self):
            
        return self.classes
    
    def get_num_classes(self):
                
        return self.num_classes
    
    def get_transform(self, image):
            
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        if self.normalize:
            image = self.denormalize(image)
        
        if len(self.mean) == 3:
            image = image.permute(1, 2, 0).numpy()
            return image
        else:
            return image.squeeze().numpy()
        
    def display_examples(self, figure_size=(10, 10), num_images=25):

        fig = plt.figure(figsize=figure_size)
        for i in range(num_images):
            ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
            image, label = next(iter(self.train_loader))
            image = self.get_transform(image[i])
            label = label[i].item()
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"{label}: {self.classes[label]}"
            )

        fig.suptitle(f"Dataset Examples for {self.name}", fontsize=20)
        fig.tight_layout()
        plt.show()

