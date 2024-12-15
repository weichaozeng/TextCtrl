from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import math
import string
from utils import instantiate_from_config
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
import numpy as np
import pytorch_lightning as pl

def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast
class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None
        self._emb_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def emb_key(self) -> str:
        return self._emb_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @emb_key.setter
    def emb_key(self, value: str):
        self._emb_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @emb_key.deleter
    def emb_key(self):
        del self._emb_key





class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.tile(self.pe[None, ...].to(x.device), (x.shape[0], 1, 1))
        return self.dropout(x)


class ViTSTREncoder(VisionTransformer):
    '''
    ViTSTREncoder is basically a ViT that uses ViTSTR weights
    '''

    def __init__(self, size=224, ckpt_path=None, freeze=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grayscale = transforms.Grayscale()
        self.resize = transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC, antialias=True)

        self.character = string.printable[:-6]
        self.reset_classifier(num_classes=len(self.character) + 2)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

        if freeze:
            self.freeze()

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        return x

    def encode(self, x):
        return self(x)


class LabelEncoder(AbstractEmbModel, pl.LightningModule):

    def __init__(self, max_len, emb_dim, n_heads=8, n_trans_layers=12, ckpt_path=None, trainable=False,
                 lr=1e-4, clip_dim=1024, visual_len=197, visual_dim=768,
                 visual_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_len = max_len
        self.emd_dim = emb_dim
        self.n_heads = n_heads
        self.n_trans_layers = n_trans_layers
        self.character = string.printable[:-6]
        self.num_cls = len(self.character) + 1

        self.label_embedding = nn.Embedding(self.num_cls, self.emd_dim)
        self.pos_embedding = PositionalEncoding(d_model=self.emd_dim, max_len=self.max_len)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.emd_dim, nhead=self.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_block, num_layers=self.n_trans_layers)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

        if trainable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.visual_encoder = instantiate_from_config(visual_config)

            self.learning_rate = lr
            self.clip_dim = clip_dim
            self.visual_len = visual_len
            self.visual_dim = visual_dim

            self.text_head = nn.Sequential(*[
                nn.InstanceNorm1d(self.max_len),
                nn.Linear(self.emd_dim, self.clip_dim, bias=False),
                nn.Conv1d(in_channels=self.max_len, out_channels=1, kernel_size=1)
            ])

            self.visual_head = nn.Sequential(*[
                nn.InstanceNorm1d(self.visual_len),
                nn.Linear(self.visual_dim, self.clip_dim, bias=False),
                nn.Conv1d(in_channels=self.visual_len, out_channels=1, kernel_size=1)
            ])


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_index(self, labels):

        indexes = []
        for label in labels:
            if len(label) > self.max_len:
                label = label[:self.max_len]
            index = [self.character.find(c) + 1 for c in label]
            index = index + [0] * (self.max_len - len(index))
            indexes.append(index)

        return torch.tensor(indexes, device=next(self.parameters()).device)

    def get_embeddings(self, x):

        emb = self.label_embedding(x)
        emb = self.pos_embedding(emb)
        out = self.encoder(emb)

        return out

    def forward(self, labels):

        idx = self.get_index(labels)
        out = self.get_embeddings(idx)

        return out

    def encode(self, text):
        return self(text)

    def get_loss(self, text_out, visual_out, clip_target):

        text_out = text_out / text_out.norm(dim=1, keepdim=True)  # b, 1024
        visual_out = visual_out / visual_out.norm(dim=1, keepdim=True)  # b, 1024

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * visual_out @ text_out.T  # b, b
        logits_per_text = logits_per_image.T  # b, b
        clip_loss_image = nn.functional.cross_entropy(logits_per_image, clip_target)
        clip_loss_text = nn.functional.cross_entropy(logits_per_text, clip_target)
        clip_loss = (clip_loss_image + clip_loss_text) / 2
        return clip_loss,  logits_per_text

    def training_step(self, batch, batch_idx):

        text = batch["text"]
        image = batch["image"]
        idx = self.get_index(text)
        text_emb = self.get_embeddings(idx)  # b, l, d
        visual_emb = self.visual_encoder(image)  # b, n, d
        text_out = self.text_head(text_emb).squeeze(1)  # b, 1024
        visual_out = self.visual_head(visual_emb).squeeze(1)  # b, 1024

        cls_target = idx  # b, c
        clip_target = torch.arange(0, idx.shape[0], 1).to(cls_target)  # b,

        clip_loss, logits_per_text = self.get_loss(text_out, visual_out, clip_target)
        loss_dict = {}
        loss_dict["loss/clip_loss"] = clip_loss
        clip_idx = torch.max(logits_per_text, dim=-1).indices  # b,
        clip_acc = (clip_idx == clip_target).to(dtype=torch.float32).mean()
        loss_dict["acc/clip_acc"] = clip_acc
        self.log_dict(loss_dict, prog_bar=True, batch_size=len(text),
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return clip_loss

    def configure_optimizers(self):

        lr = self.learning_rate
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return opt
