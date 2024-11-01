import torch
import os
from src.trainer.CtrlBase import ControlBase
from PIL import Image
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from src.MuSA.GaMuSA_app import text_editing_demo, text_editing_benchmark
from src.MuSA.GaMuSA import GaMuSA
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from utils import create_model, load_state_dict

def load_image(image_path, image_height=256, image_width=256):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open(image_path)
    image = T.ToTensor()(T.Resize((image_height, image_width))(img.convert("RGB")))
    image = image.to(device)
    return image.unsqueeze(0)


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_height", default=256)
    parser.add_argument("--teaget_width", default=256)
    parser.add_argument("--style_height", default=256)
    parser.add_argument("--style_width", default=256)
    parser.add_argument("--ckpt_path", type=str, default="weights/model.pth")
    parser.add_argument("--dataset_dir", type=str, default="example/")
    parser.add_argument("--output_dir", type=str, default="example_result/")
    parser.add_argument("--starting_layer", default=10, type=int)
    parser.add_argument("--num_inference_steps", default=50)
    parser.add_argument("--num_sample_per_image", default=1, type=int)
    parser.add_argument("--guidance_scale", default=2, type=float)
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main(opt):
    cfg_path = 'configs/inference.yaml'
    model = create_model(cfg_path).cuda()
    model.load_state_dict(load_state_dict(opt.ckpt_path), strict=False)
    model.eval()

    dataset_dir = opt.dataset_dir
    style_dir = os.path.join(dataset_dir, 'i_s')
    style_images_path = {image_name: os.path.join(style_dir, image_name) for image_name in os.listdir(style_dir)}
    style_txt = os.path.join(dataset_dir, 'i_s.txt')
    style_dict = {}
    with open(style_txt, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                image_name, text = line.strip().split(' ')[:]
                style_dict[image_name] = text
    target_txt = os.path.join(dataset_dir, 'i_t.txt')
    target_dict = {}
    with open(target_txt, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                image_name, text = line.strip().split(' ')[:]
                target_dict[image_name] = text
    monitor_cfg = {
        "max_length": 25,
        "loss_weight": 1.,
        "attention": 'position',
        "backbone": 'transformer',
        "backbone_ln": 3,
        "checkpoint": "weights/vision_model.pth",
        "charset_path": "src/module/abinet/data/charset_36.txt"
    }
    pipeline = GaMuSA(model, monitor_cfg)
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    seed = opt.seed
    starting_layer = opt.starting_layer
    guidance_scale = opt.guidance_scale
    num_sample_per_image = opt.num_sample_per_image
    num_inference_steps = opt.num_inference_steps
    seed_everything(seed)
    for i in tqdm(range(len(list(style_images_path.keys())))):
        image_name = list(style_images_path.keys())[i]
        image_path = style_images_path[image_name]
        style_text = style_dict[image_name]
        target_text = target_dict[image_name]
        w,h = Image.open(image_path).size
        source_image = load_image(image_path)
        style_image = load_image(image_path)

        if opt.benchmark:
            result = text_editing_benchmark(pipeline, source_image, style_image, style_text, target_text,
                                    starting_layer=starting_layer,
                                    ddim_steps=num_inference_steps,
                                    scale=guidance_scale,
                                    seed=seed, )
            GaMuSA_image = result
            GaMuSA_image = Image.fromarray((GaMuSA_image * 255).astype(np.uint8))
            GaMuSA_image.save(os.path.join(output_dir, image_name))

        else:
            save_dir = os.path.join(output_dir, image_name.split('.')[0])
            os.makedirs(save_dir, exist_ok=True)
            result = text_editing_demo(pipeline, source_image, style_image, style_text, target_text,
                                    starting_layer=starting_layer,
                                    ddim_steps=num_inference_steps,
                                    scale=guidance_scale,
                                    seed=seed, )
            reconstruction_image, _, GaMuSA_image = result[:]

            reconstruction_image = Image.fromarray((reconstruction_image * 255).astype(np.uint8)).resize((w, h))
            GaMuSA_image = Image.fromarray((GaMuSA_image * 255).astype(np.uint8)).resize((w, h))
            reconstruction_image.save(os.path.join(save_dir, 'recons_' + style_text + '.png'))
            GaMuSA_image.save(os.path.join(save_dir, 'GaMUSA_' + target_text + '.png'))


if __name__ == "__main__":
    parser = create_parser()
    opt = parser.parse_args()
    main(opt)
