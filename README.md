# TextCtrl: Diffusion-based Scene Text Editing with Prior Guidance Control

<a href='https://arxiv.org/abs/2410.10133'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/weichaozeng/TextCtrl'><img src='https://img.shields.io/badge/Code-Github-green'></a> <a href='https://huggingface.co'><img src='https://img.shields.io/badge/Demo-HuggingFace-yellow'>

## TODOs
- [ ] Release ScenePair benchmark dataset and evaluation code;
- [ ] Release the model and inference code;
- [ ] Release training code;
- [ ] Provide publicly accessible demo link;


## Installation
```bash
$ git clone https://github.com/weichaozeng/TextCtrl.git
$ cd TextCtrl/
$ conda env create -f environment.yaml
$ conda activate textctrl
```


## Inference

## Training

## Evaluation
### 1. Data Preparation
Download the ScenePair dataset from [GoogleDrive](https://drive.google.com/file/d/1m_o2R2kFj_hDXJP5K21aC7lKs-eUky9s/view?usp=sharing) and unzip the files. The structure of each folder is as follows:  
```bash
├── ScenePair/
│   ├── i_s/                # source cropped text images
│   ├── t_f/                # target cropped text images
│   ├── i_full/             # full-size images
│   ├── i_s.txt             # filename and text label of images in i_s/
│   ├── i_t.txt             # filename and text label of images in t_f/
│   ├── i_s_full.txt        # filename, text label, corresponding full-size image name and location information of images in i_s/
│   └── i_t_full.txt        # filename, text label, corresponding full-size image name and location information of images in t_f/
```
### 2. Generate Images
Before evaluation, corresponding edited images should be generated for a certain method based on the ScenePair dataset and should be saved in a *'.../result_folder/'* with the same filename. Result of some methods on ScenePair dataset are provided [here](https://drive.google.com/file/d/1343td96X7SuE0hYsMbTHALFmr1Md7SnQ/view?usp=drive_link).

### 3. Style Fidelity
SSIM, PSNR, MSE and FID are uesd to evaluate the style fidelity of edited result, with reference to [qqqyd/MOSTEL](https://github.com/qqqyd/MOSTEL)
.
```bash
$ cd evaluation/
$ python evaluation.py --target_path .../result_folder/ --gt_path .../ScenePair/t_f/
```

### 4. Text Accuracy
ACC and NED are used to evaluate the text accuracy of edited result, with the offical code and checkpoint in [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).  


## Citation
