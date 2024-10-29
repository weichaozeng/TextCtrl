import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vision_F

from .model_vision import BaseVision, ContrastVision
from .model_language import BCNLanguage
from .model_alignment import BaseAlignment
from .module_util import *
from .losses import MultiLosses
from typing import Tuple


class ABINetIterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = config.iter_size
        self.max_length = config.max_length + 1  # additional stop token
        self.vision = BaseVision(config.vision)
        self.language = BCNLanguage(config.language)
        self.alignment = BaseAlignment(config.alignment)
        self.export = config.get("export", False)

    def forward(self, images, mode="train", *args):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res["logits"], dim=-1)
            lengths = a_res["pt_lengths"]
            lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res["feature"], v_res["feature"])
            all_a_res.append(a_res)
        if self.export:
            return F.softmax(a_res["logits"], dim=2), a_res["pt_lengths"]
        if mode == "train":
            return all_a_res, all_l_res, v_res
        elif mode == "validation":
            return all_a_res, all_l_res, v_res, (a_res, all_l_res[-1], v_res)
        else:
            return a_res, all_l_res[-1], v_res


class ABINetIterModelWrapper(nn.Module):
    # wrapper for ABINetIterModel to make loss_computation in this
    def __init__(self, config, width, height) -> None:
        super().__init__()
        # TODO: accomodate ContrastABINetIterModel
        self.abinet = ABINetIterModel(config)
        self.width = width
        self.height = height
        self.loss_fn = MultiLosses(True)

    def preprocess_char(self, char_tokenizer, labels, device):
        # convert label strings to char_input_ids, one_hot_label for loss computation
        inputs = char_tokenizer(
            labels,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        char_input_ids = inputs["input_ids"]
        abi_num_classes = len(char_tokenizer) - 1
        abi_labels = char_input_ids[:, 1:]
        gt_labels = F.one_hot(abi_labels, abi_num_classes)
        gt_lengths = torch.sum(inputs.attention_mask, dim=1)
        return char_input_ids.to(device), gt_labels.to(device), gt_lengths.to(device)

    def preprocess(self, image):
        # images: (C, H, W)
        # this method resize images to self.w self.h
        return vision_F.resize(image, (self.height, self.width))

    def forward(self, images, char_inputs: Tuple, mode="train", *args):
        char_input_ids, gt_labels, gt_lengths = char_inputs
        assert images.device == char_input_ids.device == gt_labels.device
        outputs = self.abinet(images, char_input_ids, mode)
        celoss_inputs = outputs[:3]
        # TODO: add contrast loss later
        celoss = self.loss_fn(celoss_inputs, gt_labels, gt_lengths)
        if mode == "train":
            return celoss
        elif mode == "test" or mode == "validation":
            #! TODO: not compatible with tokenizer
            text_preds = outputs[-1]
            pt_text, a, b = postprocess(
                text_preds,
            )
            return celoss, outputs[-1]
