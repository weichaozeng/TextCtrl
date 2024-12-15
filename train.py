import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset.textdata import TextDataset
from src.trainer.Base import BaseImageLogger
from utils import create_model, load_state_dict, create_data
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from src.trainer.utils import instantiate_from_config

learning_rate = 1e-5
checkpoint_freq = 10
sd_locked = False

cfg_path = "configs/train.yaml"
model = create_model(cfg_path).cpu()
vit_path = "weights/style_encoder.pth"
model.load_state_dict(load_state_dict(vit_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked

cfgs = OmegaConf.load(cfg_path)
data_config = cfgs.pop("data", OmegaConf.create())
data = create_data(data_config)
data.prepare_data()
data.setup(stage='fit')

checkpoint_callback = ModelCheckpoint(every_n_epochs=checkpoint_freq, save_top_k=-1)
imagelogger_callback = instantiate_from_config(cfgs.image_logger)
trainer = pl.Trainer(precision=32, callbacks=[imagelogger_callback, checkpoint_callback], **cfgs.lightning)
trainer.fit(model=model, datamodule=data)
