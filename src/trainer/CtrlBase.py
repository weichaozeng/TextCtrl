import einops
import torch
import inspect
import torch as th
from typing import Optional, Any, Union, List, Dict
import torch.nn as nn
from diffusers import UNet2DConditionModel
from dataclasses import dataclass
from diffusers.utils import BaseOutput, logging
from src.trainer.Base import BaseTrainer, expand_hidden_states, log_txt_as_img
import torchvision
from .utils import (
    count_params,
    pl_on_train_tart,
    module_requires_grad,
    get_obj_from_str,
    instantiate_from_config
)
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from einops import rearrange
from diffusers.models.resnet import ResnetBlock2D


class ControlBase(BaseTrainer):

    def __init__(self, control_config, base_config):
        super().__init__(base_config)
        self.control_model = instantiate_from_config(control_config)
        self.control_scales = [1.0] * 13


    @torch.no_grad()
    def prepare_input(self, batch):
        dict_input = super().prepare_input(batch)
        hint = batch["hint"]
        dict_input["hint"] = hint
        return dict_input

    def apply_model(self, dict_input):
        t = dict_input["timestep"]
        zt = dict_input["latent"]
        c = dict_input["cond"]
        hint = dict_input["hint"]
        control = self.control_model(hint, zt, t, c)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        unet = self.unet
        noise_pred = unet(x=zt, timestep=t, encoder_hidden_states=c, control=control).sample
        return noise_pred

    @torch.no_grad()
    def sample_loop(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            hint: torch.Tensor,
            timesteps: torch.Tensor,
            do_classifier_free_guidance: bool = True,
            guidance_scale: float = 2,
            return_intermediates: Optional[bool] = False,
            extra_step_kwargs: Optional[Dict] = {},  
            **kwargs,
    ):
        intermediates = []
        res = {}

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            ).to(dtype=self.unet.dtype)


            if do_classifier_free_guidance:
                hint_input = torch.cat([hint] * 2)
            else:
                hint_input = hint
            control_input = self.control_model(hint_input, latent_model_input, t, encoder_hidden_states)

            noise_pred = self.unet(
                x=latent_model_input, timestep=t, encoder_hidden_states=encoder_hidden_states, control=control_input
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            scheduler_res = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            )
            latents = scheduler_res.prev_sample
            assert torch.isnan(latents).sum() == 0, print("scheduler_res")
            if return_intermediates:
                intermediates.append(scheduler_res.pred_original_sample)

        latents = 1 / self.NORMALIZER * latents  
        res["latents"] = latents
        if len(intermediates) != 0:
            intermediates = [1 / self.NORMALIZER * x for x in intermediates]
            res["intermediates"] = intermediates
        return res

    def convert_latent2image(self, latent, tocpu=True):
        image = self.vae.decode(latent).sample
        image = image.clamp(0, 1) 
        if tocpu:
            image = image.cpu()
        return image

    @torch.no_grad()
    def sample(
            self,
            batch: Dict[str, Union[torch.Tensor, List[str]]],
            guidance_scale: float = 2,
            num_sample_per_image: int = 1,
            num_inference_steps: int = 50,
            eta: float = 0.0,  
            generator: Optional[torch.Generator] = None,
            return_intermediates: Optional[bool] = False,
            **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.config.cond_on_text_image:
            cond_texts = batch["cond"].to(self.data_dtype).to(self.device)
            uncond_texts = torch.zeros_like(batch["cond"].to(self.data_dtype)).to(self.device)
        else:
            cond_texts = batch["cond"]
            uncond_texts = [""] * len(cond_texts)
        c = self.get_text_conditioning(cond_texts)
        uc = self.get_text_conditioning(uncond_texts)

        hint = batch["hint"].to(self.device)



        B, _, H, W = batch["img"].shape
        image_latents = torch.randn((B, 4, H // 8, W // 8), device=self.device)

        if num_sample_per_image > 1:
            image_latents = expand_hidden_states(
                image_latents, num_sample_per_image
            )
            uc = expand_hidden_states(
                uc, num_sample_per_image
            )
            c = expand_hidden_states(
                c, num_sample_per_image
            )
            hint = expand_hidden_states(
                hint, num_sample_per_image
            )
        encoder_hidden_states = torch.cat([uc, c])

        self.noise_scheduler.set_timesteps(
            num_inference_steps, device=self.vae.device)
        timesteps = self.noise_scheduler.timesteps

        image_latents = image_latents * self.noise_scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        latent_results = self.sample_loop(
            image_latents,
            encoder_hidden_states,
            hint,
            timesteps,
            do_classifier_free_guidance,
            guidance_scale,
            return_intermediates=return_intermediates,
            extra_step_kwargs=extra_step_kwargs,
            **kwargs,
        )

        image_results = {}
        images = self.convert_latent2image(latent_results["latents"])
        image_results["images"] = images
        if return_intermediates:
            intermediate_images = [
                self.convert_latent2image(x) for x in latent_results["intermediates"]
            ]
            image_results["intermediate_images"] = intermediate_images
        return image_results


    @torch.no_grad()
    def log_images(self, batch, generation_kwargs, stage="train", cat_gt=False):
        image_results = dict()
        if (
                stage == "train" or stage == "validation" or stage == "valid"
        ):  
            num_sample_per_image = generation_kwargs.get(
                "num_sample_per_image", 1)

            sample_results = self.sample(batch, **generation_kwargs)

            _, _, h, w = batch["img"].shape
            torch_resize = torchvision.transforms.Resize([h, w])
            for i, caption in enumerate(batch["texts"]):
                target_image = batch["img"][i].cpu()  
                target_image = target_image.clamp(0., 1.)
                target_image = torch_resize(target_image)

                style_image = batch["hint"][i].cpu() 
                style_image = style_image.clamp(0., 1.)
                style_image = torch_resize(style_image)


                cond_image = log_txt_as_img((w, h), batch["cond"][i], self.config.font_path, size= h // 20)
                cond_image = cond_image.clamp(0., 1.)
                cond_image = torch_resize(cond_image)


                sample_res = sample_results["images"][
                             i * num_sample_per_image: (i + 1) * num_sample_per_image
                             ]
                if cat_gt:
                    image_results[f"{i}-{caption}"] = torch.cat(
                        [
                         target_image.unsqueeze(0),
                         style_image.unsqueeze(0),
                         cond_image.unsqueeze(0),
                         sample_res], dim=0
                    )
                else:
                    image_results[f"{i}-{caption}"] = sample_res

        return (image_results,)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [{"params": self.control_model.parameters()}]
        if not self.sd_locked:
            params.append({"params": self.unet.parameters()})
        else:
            for name, parameter in self.unet.named_parameters():
                parameter.requires_grad = False
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor

class ControlUNetModel(UNet2DConditionModel):
    def forward(self, x, timestep=None, encoder_hidden_states=None, control=None, return_dict: bool = True,**kwargs):
        default_overall_up_factor = 2 ** self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in x.shape[-2:]):
            forward_upsample_size = True

        if self.config.center_input_sample:
            x = 2 * x - 1.0

        with torch.no_grad():
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(x.device)

            timesteps = timesteps.expand(x.shape[0])
            t_emb = self.time_proj(timesteps)
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)
            x = self.conv_in(x)
            xs = (x,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                    x, res_x = downsample_block(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states)
                else:
                    x, res_x = downsample_block(hidden_states=x, temb=emb)
                xs += res_x

        x = self.mid_block(x, emb, encoder_hidden_states)
        x += control.pop()


        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_xs = xs[-len(upsample_block.resnets):]
            xs = xs[: -len(upsample_block.resnets)]

            add_res = ()
            if control is not None:
                for item in res_xs[::-1]:
                    temp = item + control.pop()
                    add_res += (temp,)
                add_res = add_res[::-1]
            else:
                add_res = res_xs


            if not is_final_block and forward_upsample_size:
                upsample_size = xs[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                x = upsample_block(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=add_res,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                x = upsample_block(
                    hidden_states=x, temb=emb, res_hidden_states_tuple=add_res, upsample_size=upsample_size
                )
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        if not return_dict:
            return (x,)

        return UNet2DConditionOutput(sample=x)


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
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

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
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformerEncoder(nn.Module):
    """
    Pretrain Vision Transformer backbone.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
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
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask == None:
            x_vis = x.reshape(B, -1, C)
        else:
            x_vis = x[~mask].reshape(B, -1, C) 

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class StylePyramidNet(nn.Module):
    def __init__(
            self,
            image_size=256,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            model_channels=320,
            channel_mult=(1, 2, 4, 4),
            pyramid_sizes=(32, 16, 8, 4),
            use_checkpoint=True,
            use_new_attention_order=False,
            use_scale_shift_norm=False,
            dims=2,
            dropout=0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_channels = model_channels

        self.vit = VisionTransformerEncoder(img_size=image_size, patch_size=patch_size, in_chans=in_channels,
                                            embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0,
                                            use_learnable_pos_emb=False, )

        self.stages = nn.ModuleList([self._make_stage(embed_dim, size)
                                     for i, size in enumerate(pyramid_sizes)])

        self.zero_convs = nn.ModuleList([
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[0]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[1]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[2]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[3]),
            self.make_zero_conv(embed_dim, model_channels * channel_mult[3]),
        ])

        self.middle_block = nn.ModuleList([
            ResnetBlock2D(in_channels=embed_dim, down=True),
            ResnetBlock2D(in_channels=embed_dim, down=True),
        ])
        self.middle_block_out = self.make_zero_conv(embed_dim, model_channels*channel_mult[-1])
    def make_zero_conv(self, in_channels, out_channels):
            return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, hint, sample=None, timesteps=None, encoder_hidden_states=None):
        h = self.vit(hint, mask=None)
        h_in = rearrange(h, 'b (h w) c -> b c h w', h=self.image_size//self.patch_size)

        outs = []
        for i, stage in enumerate(self.stages):
            h_out = stage(h_in)
            for zero_conv in self.zero_convs[3*i: 3*i+3]:
                outs.append(zero_conv(h_out))

        hmid = h_in
        for block in self.middle_block:
            hmid = block(input_tensor=hmid, temb=None)
        outs.append(self.middle_block_out(hmid))

        return outs

from einops import repeat
import math

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding
