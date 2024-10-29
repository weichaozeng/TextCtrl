import pytorch_lightning as pl
import torch
import torchvision
import os
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel
import sys
from .utils import load_state_dict
import torch.nn as nn
from .utils import (
    count_params,
    pl_on_train_tart,
    module_requires_grad,
    get_obj_from_str,
    instantiate_from_config
)
import torch.distributed as torchdist
from typing import Optional, Any, Union, List, Dict
from src.module.abinet import (
    ABINetIterModel,
    MultiLosses,
    CharsetMapper,
    prepare_label,
    postprocess
)
import inspect
from src.module.loss.vgg_loss import (
    Vgg19,
    build_vgg_loss
)




class BaseTrainer(pl.LightningModule):
    def __init__(self, baseconfig):
        super().__init__()
        self.save_hyperparameters()
        self.data_dtype = torch.float32
        self.config = baseconfig
        if os.path.exists(self.config.vae.get("pretrained")):
            self.vae = AutoencoderKL.from_pretrained(self.config.vae.get("pretrained"))
        else:
            print("Initialize vae failed!")
            sys.exit()
        self.NORMALIZER = self.config.vae.get("normalizer", 0.18215)
        self.vae.requires_grad_(False)
        self.vae.to(self.data_dtype)
        self.noise_scheduler = get_obj_from_str(self.config.noise_scheduler).from_config(
            self.config.scheduler_config)
        self.text_encoder = instantiate_from_config(self.config.text_encoder)
        if self.config.text_encoder_optimized:
            self.text_encoder.requires_grad_(True)
        else:
            self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.data_dtype)
        self.unet = instantiate_from_config(self.config.unet)
        self.unet.load_state_dict(load_state_dict(self.config.unet_pretrained, location='cpu'), strict=True)
        if self.config.ocr_model.get("ocr_supervised", False):
            print("Initialize ocr model from pretrained...")
            self.ocr_model = ABINetIterModel(self.config.ocr_model)
            self.ocr_resize = torchvision.transforms.Resize([self.config.ocr_model.height, self.config.ocr_model.width])
            if os.path.exists(self.config.ocr_model.get("pretrained")):
                ocr_model_dict = torch.load(self.config.ocr_model.pretrained)
                self.ocr_model.load_state_dict(state_dict=ocr_model_dict)
            else:
                print("Initialize ocr_model failed!")
                sys.exit()

            self.ocr_model.to(self.data_dtype)
            if self.config.ocr_model.get("optimize", False):
                self.ocr_model.requires_grad_(True)
            else:
                print("Frozen ocr model weight...")
                self.ocr_model.requires_grad_(False)
            self.charset = CharsetMapper(filename=self.config.ocr_model.charset_path)
            print("Initialize ocr charsetmapper from {}...".format(self.config.ocr_model.charset_path.rsplit('/', 1)[1]))
        else:
            self.ocr_model = None

        self.loss_fn = nn.MSELoss()
        if self.config.get("vgg_weight", False):
            self.vgg19 = Vgg19(self.config.vgg_weight).to(self.device)
            self.vgg19.requires_grad_(False)
        else:
            self.vgg19 = None
        self.count_params()

    def count_params(self):
        count_params(self.vae)
        count_params(self.unet)
        if self.vgg19:
            count_params(self.vgg19)
        if self.text_encoder:
            count_params(self.text_encoder)
        if self.ocr_model:
            count_params(self.ocr_model)


    def get_text_conditioning(self, c):
        if hasattr(self.text_encoder, 'encode') and callable(self.text_encoder.encode):
            c = self.text_encoder.encode(c)
        else:
            c = self.text_encoder(c)
        return c

    def on_train_start(self) -> None:
        pl_on_train_tart(self)

    def on_fit_start(self) -> None:
        if torchdist.is_initialized():
            print(f"Fit synchronize on rank: {torchdist.get_rank()}")
            torchdist.barrier()
            torch.cuda.synchronize()

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [{"params": self.unet.parameters()}]
        if self.config.text_encoder.get("optimize", False):
            print("Optimize text encoder")
            params.append({"params": self.text_encoder.parameters()})
        print(
            f"Initialize optimizer with: lr: {lr}, weight_decay: {self.config.weight_decay}, eps: {self.config.adam_epsilon}"
        )
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )
        return opt

    def shared_step(self, batch, batch_idx, stage="train"):
        loss = self(batch)
        if len(loss) == 1:
            loss = loss[0]
            self.log(
                f"{stage}_traj/loss",
                loss,
                batch_size=len(batch["img"]),
                prog_bar=True,
                sync_dist=True,
            )
        elif len(loss) == 3:
            loss, mse_loss, ocr_loss = loss
            self.log(
                f"{stage}_traj/loss",
                loss,
                batch_size=len(batch["img"]),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/latent_loss",
                mse_loss,
                batch_size=len(batch["img"]),
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/ocr_loss",
                ocr_loss,
                batch_size=len(batch["img"]),
                sync_dist=True,
            )
        else:
            loss, mse_loss, ocr_loss, reconstruction_loss = loss
            self.log(
                f"{stage}_traj/loss",
                loss,
                batch_size=len(batch["img"]),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/latent_loss",
                mse_loss,
                batch_size=len(batch["img"]),
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/ocr_loss",
                ocr_loss,
                batch_size=len(batch["img"]),
                sync_dist=True,
            )
            self.log(
                f"{stage}_traj/reconstruction_loss",
                reconstruction_loss,
                batch_size=len(batch["img"]),
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="valid")
        return loss

    def prepare_input(self, batch):
        target_images = batch["img"].to(self.data_dtype)
        latents = self.vae.encode(target_images).latent_dist.sample()
        z0 = latents * self.NORMALIZER

        texts = batch["texts"]

        c = self.get_text_conditioning(batch["cond"])

        noise = torch.randn_like(z0)
        bsz = latents.shape[0]

        t = torch.randint(
            self.config.min_training_steps, self.noise_scheduler.num_train_timesteps, (bsz,), dtype=torch.long
        ).to(self.device)

        zt = self.noise_scheduler.add_noise(
            z0, noise, t)
        zt = self.noise_scheduler.scale_model_input(
            zt, t)

        # OCR Supervised
        if self.ocr_model:
            gt_ids, gt_lengths = prepare_label(texts, self.charset, self.device)
        else:
            gt_ids = None
            gt_lengths = None
        return {"noise": noise, "timestep": t, "latent": zt,
                "cond": c, "ids": gt_ids, "lengths": gt_lengths}


    def apply_model(self, dict_input):
        t = dict_input["t"]
        zt = dict_input["zt"]
        c = dict_input["cond"]
        noise_pred = self.unet(zt, t, c).sample
        return noise_pred

    def forward(self, batch):
        dict_input = self.prepare_input(batch)
        noise_pred = self.apply_model(dict_input)

        noise = dict_input["noise"]
        t = dict_input["timestep"]
        zt = dict_input["latent"]
        gt_ids = dict_input["ids"]
        gt_lengths = dict_input["lengths"]
        mse_loss = nn.MSELoss()(noise_pred, noise)
        loss = mse_loss

        if self.config.ocr_model.get("ocr_supervised", False):
            pred_x0 = list()
            for i in range(len(batch["img"])):
                self.noise_scheduler.set_timesteps(self.config.num_inference_steps, device=self.vae.device)
                pred_x0.append(self.noise_scheduler.step(
                    noise_pred[i: i + 1], t[i], zt[i: i + 1]
                ).pred_original_sample)
            pred_x0 = torch.cat(pred_x0)
            pred_x0 = 1.0 / self.NORMALIZER * pred_x0
            pred_image_x0 = self.convert_latent2image(pred_x0, tocpu=False)
            if self.config.reconstruction_loss:
                i_vgg = torch.cat((batch["img"].to(self.data_dtype), pred_image_x0), dim=0)
                out_vgg = self.vgg19(i_vgg)
                l_f_vgg_per, l_f_vgg_style = build_vgg_loss(out_vgg)
                reconstruction_loss = l_f_vgg_per * 0.01 + l_f_vgg_style * 100 + nn.MSELoss()(
                    batch["img"].to(self.data_dtype), pred_image_x0)
            pred_image_x0 = self.ocr_resize(pred_image_x0)
            outputs = self.ocr_model(pred_image_x0, mode="train")
            celoss_inputs = outputs[:3]
            ocr_loss = MultiLosses(True)(celoss_inputs, gt_ids, gt_lengths)

            if self.config.reconstruction_loss:
                loss = mse_loss + self.config.ocr_loss_alpha * ocr_loss + reconstruction_loss
                return (loss, mse_loss, ocr_loss, reconstruction_loss)
            else:
                loss = mse_loss + self.config.ocr_loss_alpha * ocr_loss
                return (loss, mse_loss, ocr_loss)
        return (loss,)

    @torch.no_grad()
    def sample_loop(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
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

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states
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


        B, C, H, W = batch["img"]
        image_latents = torch.randn((B, C, H // 8, W // 8), device=self.device)


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


                cond_image = log_txt_as_img((w, h), batch["texts"][i], self.config.font_path, size= h // 20)
                cond_image = cond_image.clamp(0., 1.)
                cond_image = torch_resize(cond_image)


                sample_res = sample_results["images"][
                             i * num_sample_per_image: (i + 1) * num_sample_per_image
                             ]
                if cat_gt:
                    image_results[f"{i}-{caption}"] = torch.cat(
                        [
                         target_image.unsqueeze(0),
                         cond_image.unsqueeze(0),
                         sample_res], dim=0
                    )
                else:
                    image_results[f"{i}-{caption}"] = sample_res

        return (image_results,)



from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
class BaseImageLogger(Callback):
    def __init__(
            self,
            train_batch_frequency,
            valid_batch_frequency,
            generation_kwargs,
            metric_callback=None,
            disable_wandb=False,
    ):
        super().__init__()
        self.batch_freq = {
            "train": train_batch_frequency,
            "valid": valid_batch_frequency,
        }
        self.generation_kwargs = OmegaConf.to_container(generation_kwargs)
        self.metric_callback = metric_callback
        self.disable_wandb = disable_wandb

    def get_log_dir(self, pl_module):
        if isinstance(pl_module.logger, WandbLogger):
            return pl_module.logger.experiment.dir
        elif isinstance(pl_module.logger, TensorBoardLogger):
            return pl_module.logger.log_dir

    def logger_log_image(self, pl_module, captions, images, global_step, split):
        if isinstance(pl_module.logger, WandbLogger) and not self.disable_wandb:
            pl_module.logger.log_image(
                key=f"{split}_img/{global_step}",
                images=images,
                caption=captions,
                step=global_step,
            )
        elif isinstance(pl_module.logger, TensorBoardLogger):
            big_grid = make_grid(
                torch.stack(images),
                padding=3,
                pad_value=50,
            )
            pl_module.logger.experiment.add_image(
                f"{split}_img",
                big_grid,
                global_step=global_step,
            )
            pl_module.logger.experiment.add_text(
                f"{split}_img_caption",
                " | ".join(captions),
                global_step=global_step,
            )

    @rank_zero_only
    def save_image(self, pl_module, images, global_step, split):
        print(f"Log images at: {split}/{global_step}")
        all_image_grids = []
        all_captions = []
        for k in images:
            grid = make_grid(images[k])
            all_captions.append(f"{k}")
            all_image_grids.append(grid)

        path = os.path.join(
            self.get_log_dir(pl_module), split + "_img", str(global_step)
        )
        os.makedirs(path, exist_ok=True)
        for caption, grid in zip(all_captions, all_image_grids):
            img = ToPILImage()(grid)
            img.save(os.path.join(path, caption + ".png"))

        self.logger_log_image(
            pl_module, all_captions,
            all_image_grids, global_step, split
        )



    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (
                self.check_frequency(batch_idx, split)
                and hasattr(pl_module, "log_images")
                and callable(pl_module.log_images)
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                generation_samples = pl_module.log_images(
                    batch,
                    generation_kwargs=self.generation_kwargs,
                    stage=split,
                    cat_gt=True,
                )

            image_results = generation_samples[0]
            self.save_image(pl_module, image_results,
                            pl_module.global_step, split)
            if is_train:
                pl_module.train()

            return generation_samples 


    def check_frequency(self, batch_idx, split="train"):
        if split == "train":
            if ((batch_idx + 1) % self.batch_freq[split]) == 0:  
                return True
        else:
            if (batch_idx % self.batch_freq[split.split("_")[0]]) == 0:
                return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.generation_kwargs["generator"] = torch.Generator(
            device=pl_module.device
        ).manual_seed(self.generation_kwargs["seed"])
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloadr_idx
    ):
        if trainer.state.stage == "sanity_check":
            return
        self.generation_kwargs["generator"] = torch.Generator(
            device=pl_module.device
        ).manual_seed(self.generation_kwargs["seed"])
        self.log_img(pl_module, batch, batch_idx, split="valid")


@torch.no_grad()
def expand_hidden_states(a: torch.Tensor, num_sample_per_image=1):
    origin_size = a.shape
    repeat_size = [1] * len(origin_size)
    repeat_size[1] = num_sample_per_image
    a = a.repeat(repeat_size)
    a = a.view(origin_size[0] * num_sample_per_image, *origin_size[1:])
    return a

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as T
def log_txt_as_img(wh, xc, font_path, size=10):

    txt = Image.new("RGB", wh, color="white")
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype(font_path, size=size)
    nc = int(40 * (wh[0] / 256))
    lines = "\n".join(xc[start:start + nc] for start in range(0, len(xc), nc))

    try:
        draw.text((0, 0), lines, fill="black", font=font)
    except UnicodeEncodeError:
        print("Cant encode string for logging. Skipping.")

    txt = T.ToTensor()(txt)
    return txt

