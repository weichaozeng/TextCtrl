# from fastai.vision import *

from .model import _default_tfmer_cfg
from .resnet import resnet45
from .transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer)
from .module_util import *

class ResTranformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet45()

        self.d_model = config.get("d_model", _default_tfmer_cfg["d_model"])
        # self.d_model = ifnone(config.model_vision_d_model, _default_tfmer_cfg['d_model'])
        nhead = config.get("nhead", _default_tfmer_cfg["nhead"])
        d_inner = config.get("d_inner", _default_tfmer_cfg["d_inner"])
        dropout = config.get("dropout", _default_tfmer_cfg["dropout"])
        activation = config.get("activation", _default_tfmer_cfg["activation"])
        num_layers = config.get("backbone_ln", 2)

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8*32)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature
