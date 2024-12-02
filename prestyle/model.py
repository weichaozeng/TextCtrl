import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Pos_emb
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask == None:
            x_vis = x.reshape(B, -1, C)
        else:
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # Backbone.
        self.vit = VisionTransformerEncoder(img_size=image_size, patch_size=patch_size,in_chans=in_chans,
                                            num_classes=0,
                                            embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0,
                                            use_learnable_pos_emb=False,)

        # Block for spatial and glyph.
        self.spatial_attn_block = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0)
            for _ in range(1)])

        self.glyph_attn_block = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0)
            for _ in range(1)])


    def forward(self, x):
        # Backbone
        x_feature = self.vit(x, mask=None)

        # Spatial
        x_spatial = x_feature
        for block in self.spatial_attn_block:
            x_spatial = block(x_spatial)

        # Glyph
        x_glyph = x_feature
        for block in self.glyph_attn_block:
            x_glyph = block(x_glyph)

        return x_spatial, x_glyph



class SpatialHead(nn.Module):

    def __init__(self, image_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # Block for spatial
        self.spatial_res_block = Residual_with_Attention(image_size=image_size, patch_size=patch_size,
                                                         in_channels=embed_dim, out_channels=embed_dim)

        # Head for removal, segmentation.
        self.removal_head = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim),
            nn.ELU(alpha=1.0),
            nn.Conv2d(in_channels=embed_dim, out_channels=3 * self.patch_size * self.patch_size, kernel_size=1,
                      stride=1, padding=0),
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=1 * self.patch_size * self.patch_size, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_spatial):
        x_out, x_seg = self.spatial_res_block(x_spatial)
        x_spatial = x_out + x_spatial

        x_removal = rearrange(x_spatial, 'b (h w) c -> b c h w', h=self.image_size // self.patch_size)
        x_seg = self.segmentation_head(x_seg)
        x_removal = self.removal_head(x_removal)

        out_removal = rearrange(x_removal, 'b (p1 p2 d) h w -> b d (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        out_seg = rearrange(x_seg, 'b (p1 p2) h w -> b 1 (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)

        return out_removal, out_seg



class GlyphHead(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_chans=3, embed_dim=768, res_dim=768,
                 color_backend='resnet34', font_backend='resnet34', pretrained=False):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # Color
        self.color_encoder = getattr(extractors, color_backend)(pretrained)
        self.align_to_color_feature = nn.Sequential(
            nn.Conv2d(embed_dim, res_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.InstanceNorm2d(res_dim),
            nn.ELU(alpha=1.0)
        )
        self.color_head = nn.Sequential(
            Upsample(res_dim, 256),
            Upsample(256, 64),
            Upsample(64, 3),
        )


        # Font
        self.font_encoder = getattr(extractors, font_backend)(pretrained)
        self.align_to_font_feature = nn.Sequential(
            Upsample(embed_dim, res_dim),
            nn.Conv2d(res_dim, res_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.InstanceNorm2d(res_dim),
            nn.ELU(alpha=1.0),

        )
        self.font_psp = PSPModule(res_dim * 2, res_dim, sizes=(1, 2, 3, 6))
        self.font_head = nn.Sequential(
            Upsample(res_dim, 256),
            Upsample(256, 64),
            Upsample(64, 3),
            nn.Sigmoid() 
        )



    def forward(self, x_glyph, c_t, f_t, alpha=1.0):
        x_glyph = rearrange(x_glyph, 'b (h w) c -> b c h w', h=self.image_size // self.patch_size)


        ct_feature, _ = self.color_encoder(c_t)
        x_c_feature = self.align_to_color_feature(x_glyph)
        out_c = adain(ct_feature, x_c_feature)
        out_c = alpha * out_c + (1-alpha) * ct_feature
        out_c = self.color_head(out_c)

        ft_feature, _ = self.font_encoder(f_t)
        x_f_feature = self.align_to_font_feature(x_glyph)
        out_f = self.font_psp(torch.cat((ft_feature, x_f_feature), dim=1))
        out_f = self.font_head(out_f)

        return out_c, out_f

class Residual_with_Attention(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=768, out_channels=768, stride=1, padding=1, dilation=1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation
        )
        self.conv_pool = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, dilation=dilation, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x): # x --> [B N C]
        x_in = rearrange(x, 'b (h w) c -> b c h w', h=self.image_size//self.patch_size)
        x_in = self.conv2(self.conv1(x_in))
        avg_pool = torch.mean(x_in, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_in, dim=1, keepdim=True)
        pool_mask = self.sig(self.conv_pool(torch.cat((avg_pool, max_pool), dim=1)))
        x_seg = x_in * pool_mask
        x_out = rearrange(x_seg, 'b c h w -> b (h w) c')

        return x_out, x_seg


import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=768, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ELU(alpha=1.0)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features
