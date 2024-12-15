import os
import random
import string
import torchvision.transforms as transforms
import torch.utils.data as data
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from random import choice, randint
from utils import *

class LabelDataset(data.Dataset):

    def __init__(self, size, length, font_path, min_len, max_len) -> None:
        super().__init__()

        # constraint
        self.length = length
        self.size = size
        self.font_dir = font_path

        self.character = string.printable[:-6]
        self.min_len = min_len
        self.max_len = max_len

        self.grayscale = transforms.Grayscale()
        self.resize = transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC, antialias=True)

        self.words = []
        with open("words.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                texts = line.strip().split(' ')
                self.words += texts

    def __len__(self):
        
        return self.length
    
    def __getitem__(self, index):

        while True:

            if random.random() < 0.5:
                text_len = randint(self.min_len, self.max_len)
                text = "".join([choice(self.character) for i in range(text_len)])
            else:
                text = random.choice(self.words)

            try:
                font = ImageFont.truetype(os.path.join(self.font_dir, choice(os.listdir(self.font_dir))), 128)
                
                std_l, std_t, std_r, std_b = font.getbbox(text)
                std_h, std_w = std_b - std_t, std_r - std_l
                if std_h == 0 or std_w == 0:
                    continue
            except:
                continue
            
            try:
                image = Image.new('RGB', (std_w, std_h), color = (0,0,0))
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), text, fill = (255,255,255), font=font, anchor="lt")
            except:
                continue


            image = transforms.ToTensor()(image)
            image = self.grayscale(image)
            image = self.resize(image)

            batch = {
                "image": image,
                "text": text
            }

            return batch





def get_dataloader(cfgs, datype="train"):

    dataset_cfgs = OmegaConf.load(cfgs.dataset_cfg_path)
    print(f"Extracting data from {dataset_cfgs.target}")
    Dataset = eval(dataset_cfgs.target)
    dataset = Dataset(dataset_cfgs.params, datype = datype)

    return data.DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle, num_workers=cfgs.num_workers, drop_last=True)

