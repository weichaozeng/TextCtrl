import logging
import torch.nn as nn

from .attention import *
from .backbone import ResTranformer
from .model import Model
from .resnet import resnet45

from .module_util import *

class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = config.get("loss_weight", 1.0)
        self.out_channels = config.get("d_model", 512)

        if config.backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: 
            self.backbone = resnet45()
        
        if config.attention == 'position':
            mode = config.get("attention_mode", 'nearest')
            self.attention = PositionAttention(
                max_length=config.max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.attention == 'attention':
            self.attention = Attention(
                max_length=config.max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.checkpoint is not None:
            logging.info(f'Read vision model from {config.checkpoint}.')
            self.load(config.checkpoint, device="cpu") # always cpu first and then convert

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision'}

class ContrastVision(BaseVision):
    def __init__(self, config):
        assert config.attention == 'position', "Contrastive learning only supports position attention in this model."
        super().__init__(config) # backbone is not changed
       
        # gather the information from attn_vecs provided by features 
        self.class_embedding_q = nn.Parameter(torch.randn(self.out_channels))
        self.class_encoder = nn.MultiheadAttention(
            embed_dim=self.out_channels, num_heads=config.class_num_heads, batch_first=True,
        )
    
    def forward(self, images, *args):
        # 1. Extract features
        features = self.backbone(images) # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features) # (N, T, E), (N, T, H, W)

        # 2. logits as before
        logits = self.cls(attn_vecs)
        pt_lengths = self._get_length(logits)

        # 3. Compute the class embedding for contrastive learning
        # attn_vecs already has position information(position embedding), 
        # therefore we use attention mechanism to do weighted-sum on them 

        # pt_lengths contain the length of each sequence, therefore we can use it to mask the padding part
        mask = torch.arange(attn_vecs.shape[1], device=logits.device)[None, :] < pt_lengths[:, None]
        class_embedding_q = self.class_embedding_q.expand(attn_vecs.shape[0], 1, -1) # expand to (N, 1, E)
        # attention weighted sum of attn_vecs, (N, 1, E)
        class_feature, _ = self.class_encoder(
            query=class_embedding_q, 
            key=attn_vecs,
            value=attn_vecs,
            key_padding_mask=mask,
        ) # we only want the weighted value 
        class_feature = class_feature[:, 0, :] # (N, E)
        
        return {'feature': attn_vecs, 'logits': logits, 
            'pt_lengths': pt_lengths, 'class_feature' : class_feature,
            'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision',}
