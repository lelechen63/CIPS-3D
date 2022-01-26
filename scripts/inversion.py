import copy
from pathlib import Path
import math
import numpy as np
import os
from tqdm import tqdm
import streamlit as st
import torch
import sys

sys.path.insert(0, os.getcwd())

from tl2.proj.fvcore import TLCfgNode, global_cfg
from tl2.proj.fvcore import build_model
from tl2.proj.argparser import argparser_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.logger.logging_utils_v2 import get_logger
import cv2
from exp.comm import comm_utils
import json
import pickle
import torch.nn.functional as F
import dnnlib
import PIL
from torch import nn
from torchvision import transforms

class CIPS_3D_Demo(object):
  def __init__(self):

    pass

  def model_interpolation(self,
            cfg,
            outdir,
            debug=False,
            **kwargs):

    network_pkl = st_utils.selectbox('network_pkl', cfg.network_pkl)
    model_pkl_input = st_utils.text_input('model_pkl', "", sidebar=False)

    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)

    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    num_frames = st_utils.number_input('num_frames', cfg.num_frames, sidebar=True)

    num_samples_translate = st_utils.number_input('num_samples_translate', cfg.num_samples_translate, sidebar=True)
    translate_dist = st_utils.number_input('translate_dist', 0.04, sidebar=True)

    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)
    max_fov = st_utils.number_input('max_fov', cfg.max_fov, sidebar=True)
    alpha_pi_div = st_utils.number_input('alpha_pi_div', cfg.alpha_pi_div, sidebar=True)

    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    device = torch.device('cuda')

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')
    generator = build_model(cfg=cfg.G_cfg).to(device)
    Checkpointer(generator).load_state_dict_from_file(model_pkl)


    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    # Load target image.
    l1loss =  nn.L1Loss()
    l1loss   = l1loss.to(device)
    # cv2
    target_img = cv2.imread('results/model_interpolation/0.png')
    convert_tensor = transforms.ToTensor()
    target_img = convert_tensor(target_img)
    #pil
    # target_pil = PIL.Image.open('results/model_interpolation/0.png')
    # image = np.array(target_pil)
    # target_uint8 = image.astype(np.uint8)
    # target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

    target_images = target_img.unsqueeze(0).to(device).to(torch.float32)
    # if target_images.shape[2] > 256:
    #     target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    # target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    curriculum = comm_utils.get_metadata_from_json(metafile=cfg.metadata,
                                                   num_steps=num_steps,
                                                   image_size=image_size,
                                                   psi=psi)

    with open(f"{outdir}/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    
    info = info['results/model_interpolation/0.png']
    print ('++++++++++++++++++++++++++++')
    xyz = info['cur_camera_pos']
    xyz = torch.from_numpy(xyz).to(device)
    lookup = -xyz 
    yaw = info['yaw']
    pitch = info['pitch']
    fov_list = [fov] * len(xyz)
    
    zs = {
      'z_nerf': torch.from_numpy(info['z_nerf']).to(device),
      'z_inr': torch.from_numpy(info['z_inr']).to(device),
    }
    num_steps                  = 8000
    w_avg_samples              = 10000
    initial_learning_rate      = 0.1
    initial_noise_factor       = 0.05
    lr_rampdown_length         = 0.25
    lr_rampup_length           = 0.05
    noise_ramp_length          = 0.75    
    regularize_noise_weight    = 1

    # zs = {
    #   'z_nerf': torch.randn((1, 256), device=device, requires_grad=True),
    #   'z_inr': torch.randn((1, 512), device=device, requires_grad=True),
    # }
    # optimizer = torch.optim.Adam([zs['z_nerf']] + [zs['z_inr']] , betas=(0.9, 0.999), lr=initial_learning_rate)
    
    idx = 0
    curriculum['h_mean'] = 0
    curriculum['v_mean'] = 0
    curriculum['h_stddev'] = 0
    curriculum['v_stddev'] = 0

    cur_camera_pos = xyz[[idx]]
    cur_camera_lookup = lookup[[idx]]
    fov = fov_list[idx]
    curriculum['fov'] = fov
    
    # generator = copy.deepcopy(generator).requires_grad_(True).to(device)
    generator.eval
    for step in tqdm(range(num_steps)):

        synth_images, depth_map = generator.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img= False,
            forward_points= forward_points ** 2,
            camera_pos= cur_camera_pos,
            grad_points = forward_points ** 2,
            camera_lookup=cur_camera_lookup,
            **curriculum)
        print (synth_images.requires_grad,'!!!!')
        synth_images = (synth_images + 1) * (255/2)
        tmp_frm = (synth_images.squeeze().permute(1,2,0) )
        tmp_frm = tmp_frm.detach().cpu().numpy()
        img_name = Path(f'generated3.png')
        img_name = f"{outdir}/{img_name}"
        tmp_frm = cv2.cvtColor(tmp_frm, cv2.COLOR_RGB2BGR)

        cv2.imwrite(img_name, tmp_frm)
        
        
        l1 = l1loss(synth_images, target_images)
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # synth_images = (synth_images + 1) * (255/2)
        # if synth_images.shape[2] > 256:
        #     synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        # # Features for synth images.
        # synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        # dist = (target_features - synth_features).square().sum()      
        # l1 = (target_images - synth_images).square().sum()  
        # print (target_images.max(), target_images.min(),synth_images.max(), synth_images.min(),'+++++++' )
        
        # reg_loss = zs['z_nerf'].mean()**2
        # reg_loss += zs['z_inr'].mean()**2
        # loss = reg_loss * regularize_noise_weight + dist + l1
        # print ('reg_loss:', reg_loss, 'dist:', dist,  'l1:', l1)
        loss = l1
        print (l1)
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            synth_images = (synth_images + 1) * (255/2)
            tmp_frm = (synth_images.squeeze().permute(1,2,0) )
            tmp_frm = tmp_frm.detach().cpu().numpy()
            img_name = Path(f'generated2_{step}.png')
            img_name = f"{outdir}/{img_name}"
            tmp_frm = cv2.cvtColor(tmp_frm, cv2.COLOR_RGB2BGR)

            cv2.imwrite(img_name, tmp_frm)



def main(outdir,
         cfg_file,
         command,
         debug,
         **kwargs
         ):

  os.makedirs(outdir, exist_ok=True)
  global_cfg.tl_debug = False

  command_cfg = TLCfgNode.load_yaml_with_command(cfg_filename=cfg_file, command=command)

  logger = get_logger(filename=f"{outdir}/log.txt", logger_names=['st'], stream=False)
  logger.info(f"command_cfg:\n {command_cfg.dump()}")

  st_model = CIPS_3D_Demo()

  mode = st_utils.selectbox(label='mode', options=command_cfg.mode, sidebar=True)
  getattr(st_model, mode)(cfg=command_cfg.get(mode, {}),
                          outdir=outdir,
                          debug=debug)

  pass


def build_args():
  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_str(parser, name='outdir', default='results/model_interpolation')
  argparser_utils.add_argument_str(parser, name='cfg_file', default='configs/web_demo.yaml')
  argparser_utils.add_argument_str(parser, name='command', default='model_interpolation')
  argparser_utils.add_argument_bool(parser, name='debug', default=False)

  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)
  return args

if __name__ == '__main__':

  args = build_args()

  main(**vars(args))


