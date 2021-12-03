import io
import os
import zipfile
from abc import ABC
from itertools import cycle

from PIL import ImageFile, Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ContentStyleDataset(Dataset):
    """
    PyTorch Dataset which loads content and style images
    """
    def __init__(self, content_path="../data/train2014", shape=256, style_path="../data/wikiart", length=10000):
        """
        Initialize dataset parameters
        :param content_path: path to content images (COCO dataset)
        :param shape: shape of cropped image
        :param style_path: path to style images (WikiArt)
        :param length: size of dataset
        """
        self.length = length
        self.content_path, self.style_path = content_path, style_path  # paths of folders
        self.transform = Compose([RandomResizedCrop(size=(shape, shape)), ToTensor()])  # convert to tensor and crop
        self.content_images = os.listdir(self.content_path)
        self.style_images = os.listdir(self.style_path)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        content = self.transform(Image.open(os.path.join(self.content_path, self.content_images[item])).convert('RGB'))
        style = self.transform(Image.open(os.path.join(self.style_path, self.style_images[item])).convert('RGB'))
        return content, style



