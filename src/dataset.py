import io
import zipfile
from abc import ABC
from itertools import cycle

from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor


class ContentStyleDataset(IterableDataset, ABC):
    """
    PyTorch Dataset which loads content and style images
    """
    def __init__(self, ratio_train=0.8, mode="train", content_path="../data/train2014.zip", style_path="../data/wikiart.zip"):
        """
        Initialize dataset parameters
        :param ratio_train: ratio of train in a full dataset
        :param mode: train or val
        :param content_path: path to zip with content images (COCO dataset)
        :param style_path: path to zip with style images (WikiArt)
        """
        self.mode, self.ratio = mode, ratio_train
        self.content_path, self.style_path = content_path, style_path  # paths of zip files
        self.transform = Compose([RandomResizedCrop(size=(256, 256)), ToTensor()])  # convert to tensor

    def get_data(self, path):
        """
        Get image from zip and transform it
        :param path: path to zip file
        :return: transformed image
        """
        zip_file = zipfile.ZipFile(path, 'r')  # zip archive of images
        images_name = list(filter(lambda x: x[-4:] == '.jpg' and x[:7] != "__MACOS", zip_file.namelist()))  # filter images
        number_of_train_files = int(len(images_name) * self.ratio)
        files = images_name[:number_of_train_files] if self.mode == "train" else images_name[number_of_train_files:]
        for x in files:
            bytes_file = io.BytesIO(zip_file.read(x)).read()  # read bytes from zip
            image = Image.open(io.BytesIO(bytes_file)).convert('RGB')  # read PIL image
            yield self.transform(image)  # transform (random crop + to tensor)

    def get_stream(self):
        """
        Cycle list of images' names in order to create iterable dataset
        :return: cycle of pairs
        """
        return cycle(zip(self.get_data(self.content_path), self.get_data(self.style_path)))

    def __iter__(self):
        return self.get_stream()

