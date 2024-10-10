#!/usr/bin/env python
"""Module providing functionality surrounding gaussian function.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size //
                       2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in [
               '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


class devdata(Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.imageFiles = [os.path.join(dataRoot, filename) for filename 
                           in os.listdir(dataRoot) if CheckImageFile(filename)]
        self.gtFiles = [os.path.join(gtRoot, filename) for filename 
                        in os.listdir(dataRoot) if CheckImageFile(filename)]
        self.loadSize = loadSize

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        gt = Image.open(self.gtFiles[index])
        to_scale = gt.size
        inputImage = ImageTransform(to_scale)(img.convert('RGB'))
        groundTruth = ImageTransform(to_scale)(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth, path

    def __len__(self):
        return len(self.imageFiles)
