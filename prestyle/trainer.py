import pytorch_lightning as pl
import torch
import torch.nn as nn
from model import StyleEncoder, SpatialHead, GlyphHead
import torchvision
from pytorch_lightning import Callback
from omegaconf import OmegaConf


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")




class StyleNet(pl.LightningModule):

    def __init__(self, image_size=128, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, trainable=False,
                 lr=1e-4, alpha_removal=1.0, alpha_seg=1.0, alpha_color=1.0,
                 alpha_font=1.0, enable_spatial_head=True, enable_glyph_head=True, *args, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.enable_spatial_head = enable_spatial_head
        self.enable_glyph_head = enable_glyph_head



        self.encoder = StyleEncoder(image_size=self.image_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                     embed_dim=self.embed_dim)
        self.spatial_head = SpatialHead(image_size=self.image_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                        embed_dim=self.embed_dim)
        self.glyph_head = GlyphHead(image_size=self.image_size, patch_size=self.patch_size, in_chans=self.in_chans,
                                        embed_dim=self.embed_dim)

        if trainable:
            self.learning_rate = lr
            self.alpha_removal = alpha_removal
            self.alpha_seg = alpha_seg
            self.alpha_color = alpha_color
            self.alpha_font = alpha_font

            self.count_params()


    def count_params(self):
        count_params(self.encoder)

        count_params(self.spatial_head)

        count_params(self.glyph_head)


    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        return opt

    def dice_loss(self, predictive, target, ep = 1e-8):
        b, c, h, w = predictive.size()
        predictive = predictive.view(b, -1)
        target = target.view(b, -1)
        intersection = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_loss_full_supervised(self, batch, out_removal, out_seg, out_color, out_font):
        if self.enable_spatial_head:
            gt_removal = batch["bg"]
            gt_seg = batch["seg"]

            # 1.Text removal loss
            lambda_inmask = 2.
            lambda_outmask = 0.5
            loss_removal = lambda_inmask * (nn.functional.l1_loss(out_removal * gt_seg, gt_removal * gt_seg)) + \
                           lambda_outmask * (nn.functional.l1_loss(out_removal * (1 - gt_seg), gt_removal * (1 - gt_seg)))

            # 2.Text segmentation loss
            loss_seg = self.dice_loss(out_seg, gt_seg)
        else:
            loss_removal = 0.
            loss_seg = 0.


        if self.enable_glyph_head:
            gt_color = batch["c_r"]
            gt_font = batch["f_r"]
            # 3. Colorized loss
            loss_color = nn.functional.mse_loss(out_color, gt_color)

            # 4. Fontilized loss
            loss_font = self.dice_loss(out_font, gt_font)

        else:
            loss_color = 0.
            loss_font = 0.

        return loss_removal, loss_seg, loss_color, loss_font

    def get_loss_self_supervised(self, batch, out_removal, out_seg, out_color, out_font):
        pass


    def training_step(self, batch, batch_idx):
        input = batch["i_s"]

        img_spatial, img_glyph = self(input)
        out_removal, out_seg = self.spatial_head(img_spatial)
        out_color, out_font = self.glyph_head(img_glyph, batch["c_t"], batch["f_t"])

        loss_removal, loss_seg, loss_color, loss_font = self.get_loss_full_supervised(batch, out_removal, out_seg,
                                                                                      out_color, out_font)
        loss = self.alpha_removal * loss_removal + self.alpha_seg * loss_seg + \
               self.alpha_color * loss_color + self.alpha_font * loss_font

        loss_dict = {}
        loss_dict["train_loss/loss_removal"] = loss_removal
        loss_dict["train_loss/loss_seg"] = loss_seg
        loss_dict["train_loss/loss_color"] = loss_color
        loss_dict["train_loss/loss_font"] = loss_font
        loss_dict["train_loss/full_loss"] = loss

        self.log_dict(loss_dict, prog_bar=True, batch_size=len(batch["text_s"]),
                    logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["i_s"]

        img_spatial, img_glyph = self(input)
        out_removal, out_seg = self.spatial_head(img_spatial)
        out_color, out_font = self.glyph_head(img_glyph, batch["c_t"], batch["f_t"])

        loss_removal, loss_seg, loss_color, loss_font = self.get_loss_full_supervised(batch, out_removal, out_seg,
                                                                                      out_color, out_font)
        loss = self.alpha_removal * loss_removal + self.alpha_seg * loss_seg + \
               self.alpha_color * loss_color + self.alpha_font * loss_font

        loss_dict = {}
        loss_dict["val_loss/loss_removal"] = loss_removal
        loss_dict["val_loss/loss_seg"] = loss_seg
        loss_dict["val_loss/loss_color"] = loss_color
        loss_dict["val_loss/loss_font"] = loss_font
        loss_dict["val_loss/full_loss"] = loss

        self.log_dict(loss_dict, prog_bar=True, batch_size=len(batch["text_s"]),
                    logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def forward(self, img):
        img_spatial, img_glyph = self.encoder(img)
        return img_spatial, img_glyph

    @torch.no_grad()
    def log_images(self, batch):
        # generate result
        input = batch["i_s"]
        img_spatial, img_glyph = self(input)
        out_removal, out_seg = self.spatial_head(img_spatial)
        out_color, out_font = self.glyph_head(img_glyph, batch["c_t"], batch["f_t"])

        # save result
        _, _, h, w = batch["i_s"].shape
        torch_resize = torchvision.transforms.Resize([h, w])
        img_results = dict()
        for i, caption in enumerate(batch["text_s"]):
            source_image = batch["i_s"][i].cpu()
            source_image = source_image.clamp(0., 1.)
            source_image = torch_resize(source_image)

            removal_image = out_removal[i].cpu()
            removal_image = removal_image.clamp(0., 1.)
            removal_image = torch_resize(removal_image)

            seg_image = out_seg[i].cpu()
            seg_image = seg_image.clamp(0., 1.)
            seg_image = torch_resize(seg_image)
            seg_image = seg_image.repeat(3, 1, 1)


            color_gt = batch["c_r"][i].cpu()
            color_gt = color_gt.clamp(0., 1.)
            color_gt = torch_resize(color_gt)

            color_image = out_color[i].cpu()
            color_image = color_image.clamp(0., 1.)
            color_image = torch_resize(color_image)


            font_gt = batch["f_r"][i].cpu()
            font_gt = font_gt.clamp(0., 1.)
            font_gt = torch_resize(font_gt)

            font_image = out_font[i].cpu()
            font_image = font_image.clamp(0., 1.)
            font_image = torch_resize(font_image)


            img_results[f"{i}-{caption}"] = torch.cat(
                [
                    source_image.unsqueeze(0),
                    removal_image.unsqueeze(0),
                    seg_image.unsqueeze(0),
                    color_gt.unsqueeze(0),
                    color_image.unsqueeze(0),
                    font_gt.unsqueeze(0),
                    font_image.unsqueeze(0),
                    ], dim=0
            )
        return img_results



#########################################################################
# Image logger
#########################################################################
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only
import os
from torchvision.transforms import ToPILImage

class StyleNetImageLogger(Callback):
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
                generation_samples = pl_module.log_images(batch)

            image_results = generation_samples
            self.save_image(pl_module, image_results,
                            pl_module.global_step, split)
            if is_train:
                pl_module.train()

            return generation_samples

    def check_frequency(self, batch_idx, split="train"):
        if split == "train":
            if ((batch_idx + 1) % self.batch_freq[split]) == 0:  # avoid batch 0
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