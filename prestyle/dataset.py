import os
import pytorch_lightning as pl
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader


class WrappedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = data_config
        self.batch_size = data_config.batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train = StyleDataset(self.config.train)
            self.val = StyleDataset(self.config.validation)
        if stage == "test" or stage == "predict":
            self.val = StyleDataset(self.config.test)

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




class StyleDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.size = config.size
        self.data_dir_list = []
        self.template_font = config.template_font
        for dir_name in os.listdir(config.root_dir):
                self.data_dir_list.append(os.path.join(config.root_dir, dir_name))
        self.font_paths = [os.path.join(config.font_dir, font_name) for font_name in os.listdir(config.font_dir)]

        # source image
        self.i_s = []
        # background image
        self.t_b = []
        # segmentation image
        self.mask_s = []
        # colorize mask and image
        self.mask_t = []
        self.t_t = []


        # source text and target text and font
        self.text_s = []
        self.text_t = []
        self.class_font = []

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
            with open(os.path.join(data_dir, 'font.txt'), 'r') as f_f:
                lines = f_f.readlines()
                for line in lines:
                    _, text = line.split(' ')[:]
                    self.class_font.append(config.font_dir + text.rsplit('/', maxsplit=1)[1])
            for tmp_name in tmp_names:
                self.i_s.append(os.path.join(data_dir, 'i_s', tmp_name))
                self.t_b.append(os.path.join(data_dir, 't_b', tmp_name))
                self.mask_s.append(os.path.join(data_dir, 'mask_s', tmp_name))
                self.mask_t.append(os.path.join(data_dir, 'mask_t', tmp_name))
                self.t_t.append(os.path.join(data_dir, 't_t', tmp_name))



        self.resize = transforms.Resize((self.size, self.size), transforms.InterpolationMode.BICUBIC, antialias=True)
        self.transfer = transforms.ToTensor()
        print(f"In all collected {len(self.i_s)} sample")


    def __len__(self):
        return len(self.i_s)

    def detect_edges(self, input_array):
        edges = cv2.Canny(input_array, 100, 200)
        return edges

    def __getitem__(self, index):
        img_i_s = Image.open(self.i_s[index]).convert("RGB")
        img_i_s = self.resize(img_i_s)
        i_s = self.transfer(img_i_s)

        # Removal
        img_bg = Image.open(self.t_b[index]).convert("RGB")
        img_bg = self.resize(img_bg)
        bg = self.transfer(img_bg)

        # Segmentation
        img_mask_s = Image.open(self.mask_s[index])
        img_seg = self.resize(img_mask_s)
        seg = self.transfer(img_seg)

        # Colorized
        img_color_template = Image.open(self.mask_t[index]).convert("RGB")
        img_color_template = self.resize(img_color_template)
        color_template = self.transfer(img_color_template)

        img_color_result = Image.open(self.t_t[index]).convert("RGB")
        img_color_result = self.resize(img_color_result)
        array_color_result = np.array(img_color_result)

        color_mask = Image.open(self.mask_t[index]).convert("L")
        color_mask = self.resize(color_mask)
        color_mask = self.transfer(color_mask)
        color_mask = np.array(color_mask)

        for i in range(array_color_result.shape[0]):
            for j in range(array_color_result.shape[1]):
                array_color_result[i, j, :] = array_color_result[i, j, :] * color_mask[0, i, j]
        color_result = self.transfer(array_color_result)

        # Fontilized
        text_s = self.text_s[index].strip()
        text_t = self.text_t[index].strip()

        template_font = ImageFont.truetype(self.template_font, 64)
        std_l, std_t, std_r, std_b = template_font.getbbox(text_t)
        std_h, std_w = std_b - std_t, std_r - std_l
        img_font_template = Image.new('RGB', (std_w + 10, std_h + 10), color=(0, 0, 0))
        draw = ImageDraw.Draw(img_font_template)
        draw.text((5, 5), text_t, fill=(255, 255, 255), font=template_font, anchor="lt")
        img_font_template = self.resize(img_font_template)
        array_font_template = np.array(img_font_template)
        edge_font_template = self.detect_edges(array_font_template)
        font_template = self.transfer(edge_font_template)
        font_template = font_template.repeat(3, 1, 1)

        style_font = self.class_font[index].strip()
        result_font = ImageFont.truetype(style_font, 64)
        std_l, std_t, std_r, std_b = result_font.getbbox(text_t)
        std_h, std_w = std_b - std_t, std_r - std_l
        img_font_result = Image.new('RGB', (std_w + 10, std_h + 10), color=(0, 0, 0))
        draw = ImageDraw.Draw(img_font_result)
        draw.text((5, 5), text_t, fill=(255, 255, 255), font=result_font, anchor="lt")
        img_font_result = self.resize(img_font_result)
        array_font_result = np.array(img_font_result)
        edge_font_result = self.detect_edges(array_font_result)
        font_result = self.transfer(edge_font_result)
        font_result = font_result.repeat(3, 1, 1)

        batch = {
            "text_s": text_s,
            "text_t": text_t,
            "i_s": i_s,
            "bg": bg,
            "seg": seg,
            "c_t": color_template,
            "c_r": color_result,
            "f_t": font_template,
            "f_r": font_result,
        }
        return batch





