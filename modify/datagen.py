"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import cfg
from tqdm import tqdm
from Synthtext.gen import datagen, multiprocess_datagen

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    s_s_dir = os.path.join(cfg.data_dir, cfg.s_s_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)
    mask_s_dir = os.path.join(cfg.data_dir, cfg.mask_s_dir)
    
    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(s_s_dir)
    makedirs(mask_t_dir)
    makedirs(mask_s_dir)

    it_txt = open(os.path.join(cfg.data_dir, 'i_t.txt'), 'w')
    is_txt = open(os.path.join(cfg.data_dir, 'i_s.txt'), 'w')
    font_txt = open(os.path.join(cfg.data_dir, 'font.txt'), 'w')

    mp_gen = multiprocess_datagen(cfg.process_num, cfg.data_capacity)
    mp_gen.multiprocess_runningqueue()
    digit_num = len(str(cfg.sample_num))
    for idx in tqdm(range(cfg.sample_num)):
        data_dict = mp_gen.dequeue_data()
        i_t, i_s, t_sk, t_t, t_b, t_f, s_s, mask_t, mask_s =data_dict["data"]
        is_text = data_dict["is_text"]
        it_text = data_dict["it_text"]
        font_name = data_dict["font"]
        is_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + is_text + '\n')
        it_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + it_text + '\n')
        font_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + font_name + '\n')
        i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        s_s_path = os.path.join(s_s_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        mask_s_path = os.path.join(cfg.data_dir, cfg.mask_s_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(s_s_path, s_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_s_path, mask_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    mp_gen.terminate_pool()
    is_txt.close()
    it_txt.close()
    font_txt.close()
if __name__ == '__main__':
    main()
