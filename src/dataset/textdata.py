import json
import random

import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T

class WrappedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = data_config
        self.batch_size = data_config.batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train = TextDataset(self.config.train)
            self.val = TextDataset(self.config.validation)
        if stage == "test" or stage == "predict":
            self.val = TextDataset(self.config.test)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )


class TextDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.size = config.size
        self.data_dir_list = []
        for dir_name in os.listdir(config.root_dir):
                self.data_dir_list.append(os.path.join(config.root_dir, dir_name))
        self.font_paths = [os.path.join(config.font_dir, font_name) for font_name in os.listdir(config.font_dir)]

        # source image
        self.i_s = []
        # target image
        self.t_f = []

        self.text_s = []
        self.text_t = []

        for data_dir in self.data_dir_list:
            tmp_names = []
            with open(os.path.join(data_dir, 'i_s.txt'), 'r') as f_s:
                lines = f_s.readlines()
                for line in lines:
                    name, text = line.split(' ')[:]
                    self.text_s.append(text)
                    tmp_names.append(name)
            with open(os.path.join(data_dir, 'i_t.txt'), 'r') as f_t:
                lines = f_t.readlines()
                for line in lines:
                    _, text = line.split(' ')[:]
                    self.text_t.append(text)

            for tmp_name in tmp_names:
                self.i_s.append(os.path.join(data_dir, 'i_s', tmp_name))
                self.t_f.append(os.path.join(data_dir, 't_f', tmp_name))

        self.transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
        ])
        print(f"In all collected {len(self.i_s)} sample")


    def __len__(self):
        return len(self.i_s)

    def __getitem__(self, index):
        img_i_s = Image.open(self.i_s[index]).convert("RGB")
        i_s = self.transform(img_i_s)
        img_t_f = Image.open(self.t_f[index]).convert("RGB")
        t_f = self.transform(img_t_f)
        text_s = self.text_s[index].strip()
        text_t = self.text_t[index].strip()

        if random.random() < 0.1:
            cond = ""
        else:
            cond = text_t

        return dict(img=t_f, texts=text_t, cond=cond, hint=i_s)


class InferDataset(Dataset):
    def __init__(self, data_dir, size=256):
        super().__init__()
        self.size = size
        self.i_s = []
        self.text_s = []
        self.text_t = []

        tmp_names = []
        with open(os.path.join(data_dir, 'i_s.txt'), 'r') as f_s:
            lines = f_s.readlines()
            for line in lines:
                name, text = line.split(' ')[:]
                self.text_s.append(text)
                tmp_names.append(name)
        with open(os.path.join(data_dir, 'i_t.txt'), 'r') as f_t:
            lines = f_t.readlines()
            for line in lines:
                _, text = line.split(' ')[:]
                self.text_t.append(text)

        for tmp_name in tmp_names:
            self.i_s.append(os.path.join(data_dir, 'i_s', tmp_name))

        self.transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
        ])
        print(f"In all collected {len(self.i_s)} inference data")

    def __len__(self):
        return len(self.i_s)

    def __getitem__(self, index):
        name = self.i_s[index].rsplit("/", maxsplit=1)[1]

        img_i_s = Image.open(self.i_s[index]).convert("RGB")
        i_s = self.transform(img_i_s)
        text_s = self.text_s[index].strip()
        text_t = self.text_t[index].strip()
        cond = text_t
        return dict(img=i_s, texts=text_t, cond=cond, hint=i_s, names=name)
