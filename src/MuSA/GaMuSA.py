import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.module.abinet.modules.model_vision import BaseVision
import torchvision.transforms as T
from omegaconf import OmegaConf
from src.module.abinet import CharsetMapper


class GaMuSA:
    def __init__(self, model, monitor_cfg):
        self.model = model
        self.scheduler = model.noise_scheduler
        self.device = model.device
        self.unet = model.unet
        self.vae = model.vae
        self.control_model = model.control_model
        self.text_encoder = model.text_encoder
        self.NORMALIZER = model.NORMALIZER
        monitor_cfg = OmegaConf.create(monitor_cfg)
        self.monitor = BaseVision(monitor_cfg).to(self.device)
        self.charset = CharsetMapper(filename=monitor_cfg.charset_path)
        self.max_length = monitor_cfg.max_length + 1

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            assert image.dim() == 4, print("input dims should be 4 !")
            latents = self.vae.encode(image.to(self.device)).latent_dist.sample()
            latents = latents * self.NORMALIZER
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / self.NORMALIZER * latents.detach()
        image = self.model.vae.decode(latents).sample
        if return_type == 'np':
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    def latent2image_grad(self, latents):
        latents = 1 / self.NORMALIZER * latents
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def inversion(
            self,
            image: torch.Tensor,
            hint: torch.Tensor,
            cond,
            num_inference_steps=50,
            guidance_scale=7.5,
            eta=0.0,
            return_intermediates=False,
            **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """

        assert image.shape[0] == len(cond), print("Unequal batch size for image and cond.")
        assert image.shape[0] == hint.shape[0], print("Unequal batch size for image and hint.")

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        latents = self.image2latent(image)
        start_latents = latents


        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(reversed(self.scheduler.timesteps)):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint

            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)

            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        if return_intermediates:
            return latents, latents_list
        return latents, start_latents

    @torch.no_grad()
    def __call__(
            self,
            hint,
            cond,
            start_step=24,
            start_layer=10,
            batch_size=1,
            height=256,
            width=256,
            num_inference_steps=50,
            guidance_scale=2,
            eta=0.0,
            latents=None,
            unconditioning=None,
            neg_prompt=None,
            ref_intermediate_latents=None,
            return_intermediates=False,
            enable_GaMuSA=True,
            **kwds):

        cond_embeddings = self.model.get_text_conditioning(cond)
        if guidance_scale > 1.:
            uncond = [""] * len(cond)
            uncond_embeddings = self.model.get_text_conditioning(uncond)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        else:
            text_embeddings = cond_embeddings

        batch_size = len(cond)
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device)
        else:
            pass

        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        gt_ids, _ = prepare_label(cond, self.charset, self.max_length, self.device)
        from src.MuSA.utils import MuSA_TextCtrl
        from src.MuSA.utils import regiter_attention_editor_diffusers_Edit
        if enable_GaMuSA:
            controller = MuSA_TextCtrl(start_step, start_layer)
            regiter_attention_editor_diffusers_Edit(self.unet, controller)
            controller.start_ctrl()

        for i, t in enumerate(self.scheduler.timesteps):
            if ref_intermediate_latents is not None:
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
                hint_input = torch.cat([hint] * 2)
            else:
                model_inputs = latents
                hint_input = hint

            control_input = self.control_model(hint_input, model_inputs, t, text_embeddings)

            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
            noise_pred = self.unet(
                x=model_inputs,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                control=control_input,
            ).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.step(noise_pred, t, latents)

            if enable_GaMuSA:
                glyph_resize = T.transforms.Resize([32, 128])
                if (i + 1) % 5 == 0:
                    glyph_inputs = self.latent2image(latents, return_type="pt")
                    glyph_inputs = glyph_resize(glyph_inputs)
                    outputs = self.monitor(glyph_inputs)  
                    cosine_score = glyph_cosine_similarity(outputs, gt_ids)
                    controller.reset_alpha(cosine_score)
        if enable_GaMuSA:
            controller.reset_ctrl()
            controller.reset()
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, latents_list, pred_x0_list
        return image



def glyph_cosine_similarity(output, gt_labels):
    pt_logits = nn.Softmax(dim=2)(output['logits'])
    assert pt_logits.shape[0] == gt_labels.shape[0]
    assert pt_logits.shape[2] == gt_labels.shape[2]
    score = nn.CosineSimilarity(dim=2, eps=1e-6)(pt_logits, gt_labels) 
    mean_score = torch.mean(score, dim=1)
    return mean_score 

def prepare_label(labels, charset, max_length, device):
    gt_ids = []
    gt_lengths = []
    for label in labels:
        length = torch.tensor(max_length, dtype=torch.long)
        label = charset.get_labels(label, length=max_length, padding=True, case_sensitive=False)
        label = torch.tensor(label, dtype=torch.long)
        label = CharsetMapper.onehot(label, charset.num_classes)
        gt_ids.append(label)
        gt_lengths.append(length)
    gt_ids = torch.stack(gt_ids).to(device)
    gt_lengths = torch.stack(gt_lengths).to(device)
    return gt_ids, gt_lengths