import logging
import os
import glob
import torch
import PIL
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import OmegaConf
from .utils import Config, CharsetMapper


BASE_DIR = "src/abinet/"
DEFAULT_OCR_CONFIG = {
    "conf": os.path.join(BASE_DIR, "configs/train_abinet.yaml"),
    "default_conf": os.path.join(BASE_DIR, "configs/template.yaml"),
    "ckpt": os.path.join(BASE_DIR, "checkpoints/abinet/train-abinet/best-train-abinet.pth"),
}


def create_ocr_model(device=None):
    print("Loading OCR model...")
    if device is None:
        device = torch.cuda.current_device()
    default_conf = OmegaConf.load(DEFAULT_OCR_CONFIG["default_conf"])
    conf = OmegaConf.load(DEFAULT_OCR_CONFIG["conf"])
    config = OmegaConf.merge(default_conf, conf)
    OmegaConf.resolve(config)
    charset = CharsetMapper(
        filename=config.dataset.charset_path, max_length=config.dataset.max_length + 1
    )
    config.model_eval = "alignment"
    ocr_model = get_model(config.model)
    model = load(
        ocr_model,
        DEFAULT_OCR_CONFIG["ckpt"],
        device=None,
        strict="Contrast" not in config.model.name,
    )  # always load to cpu first
    model = model.to(device)
    print("OCR Model loaded")
    return charset, ocr_model


def get_model(config, device="cpu", reload=False):
    import importlib

    module, cls = config.name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    cls = getattr(importlib.import_module(module, package=None), cls)

    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model


def preprocess(img, width=128, height=32):
    img = img.resize((width, height), PIL.Image.Resampling.BILINEAR)
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return (img - mean[..., None, None]) / std[..., None, None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                if res["name"] == model_eval:
                    output = res
        else:
            output = last_output
        return output

    def _decode(logit):
        """Greed decode"""
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(
                min(len(text) + 1, charset.max_length)
            )  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output["logits"], output["pt_lengths"]
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    return pt_text, pt_scores, pt_lengths_


def load(model, file, device=None, strict=True):
    if device is None:
        device = "cpu"
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {"model", "opt"}:
        state = state["model"]
    model.load_state_dict(state, strict=strict)
    return model
