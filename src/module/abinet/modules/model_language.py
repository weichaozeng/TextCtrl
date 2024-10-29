import logging
import torch.nn as nn

from .model import _default_tfmer_cfg
from .model import Model
from .transformer import (PositionalEncoding, 
                                 TransformerDecoder,
                                 TransformerDecoderLayer)

from .module_util import *

class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = config.get("d_model", _default_tfmer_cfg['d_model'])
        nhead = config.get("nhead", _default_tfmer_cfg['nhead'])
        d_inner = config.get("d_inner", _default_tfmer_cfg['d_inner'])
        dropout = config.get("dropout", _default_tfmer_cfg['dropout'])
        activation = config.get("activation", _default_tfmer_cfg['activation'])
        num_layers = config.get("num_layers", 4)

        self.d_model = d_model
        self.detach = config.get("detach", True)
        self.use_self_attn = config.get("use_self_attn", False)
        self.loss_weight = config.get("loss_weight", 1.0)
        # self.max_length = self.max_length

        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout, 
                activation, self_attn=self.use_self_attn)
        self.model = TransformerDecoder(decoder_layer, num_layers)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

        if config.checkpoint is not None:
            logging.info(f'Read language model from {config.checkpoint}.')
            self.load(config.checkpoint, device="cpu")

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = embed.permute(1, 0, 2)  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        padding_mask = self._get_padding_mask(lengths, self.max_length)

        zeros = embed.new_zeros(*embed.shape)
        qeury = self.pos_encoder(zeros)
        location_mask = self._get_location_mask(self.max_length, tokens.device)
        output = self.model(qeury, embed,
                tgt_key_padding_mask=padding_mask,
                memory_mask=location_mask,
                memory_key_padding_mask=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':self.loss_weight, 'name': 'language'}
        return res
