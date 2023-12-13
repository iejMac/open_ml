import torch
from torchvision import transforms

from typing import Optional, Tuple, Union


class ImageTransform:
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        mean: Optional[Tuple[float, ...]] = (0.1307,),
        std: Optional[Tuple[float, ...]] = (0.3081,),
    ):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=mean, std=std),
            lambda img: img.reshape(-1),
        ])

        # For synthetic dataset creation
        self.shape = (image_size**2,)
        self.dtype = torch.float32
    def __call__(self, sample) :
        return self.transform(sample)


class ClassTransform:
    def __init__(self):
        # For synthetic dataset creation
        self.shape = (1,)
        self.dtype = torch.long
    def __call__(self, sample):
        return torch.tensor(int(sample))
