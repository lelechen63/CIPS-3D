import copy
from pathlib import Path
import math
import numpy as np
import os
import tqdm
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

    # fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    # num_frames = st_utils.number_input('num_frames', cfg.num_frames, sidebar=True)

    num_samples_translate = st_utils.number_input('num_samples_translate', cfg.num_samples_translate, sidebar=True)
    translate_dist = st_utils.number_input('translate_dist', 0.04, sidebar=True)

    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)
    max_fov = st_utils.number_input('max_fov', cfg.max_fov, sidebar=True)
    # alpha_pi_div = st_utils.number_input('alpha_pi_div', cfg.alpha_pi_div, sidebar=True)
  
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    # ****************************************************************************
    device = torch.device('cuda')

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')
    generator = build_model(cfg=cfg.G_cfg).to(device)
    Checkpointer(generator).load_state_dict_from_file(model_pkl)
    curriculum = comm_utils.get_metadata_from_json(metafile=cfg.metadata,
                                                   num_steps=num_steps,
                                                   image_size=image_size,
                                                   psi=psi)
    xyz_list = []
    lookup_list = []
    yaws_list = []
    pitchs_list = []
    fov_list = []

    # circle
    xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / 5,
                                                                              num_samples=10,
                                                                              periods=1)
    xyz_list = xyz
    lookup_list = lookup
    yaws_list = yaws
    pitchs_list = pitchs
      
    # yaw
    xyz, lookup, yaws, pitchs = comm_utils.get_yaw_camera_pos_and_lookup(num_samples=10, )
    xyz_list = np.concatenate((xyz_list, xyz), axis = 0)  
    lookup_list = np.concatenate((lookup_list, lookup), axis = 0)   
    yaws_list = np.concatenate((yaws_list, yaws), axis = 0)    
    pitchs_list = np.concatenate((pitchs_list, pitchs), axis = 0)    
    
    #'roll'
    xyz, lookup, yaws, pitchs = comm_utils.get_roll_camera_pos_and_lookup(num_samples=10, )
    xyz_list = np.concatenate((xyz_list, xyz), axis = 0)
    lookup_list = np.concatenate((lookup_list, lookup), axis = 0)   
    yaws_list = np.concatenate((yaws_list, yaws), axis = 0)    
    pitchs_list = np.concatenate((pitchs_list, pitchs), axis = 0)   
    
    xyz_list = torch.from_numpy(xyz_list).to(device)
    lookup_list = torch.from_numpy(lookup_list).to(device)
    fov_list = [fov] * len(xyz_list)

    st_image = st.empty()
    
    # seed
    seed = 3
    # for seed in range(0, 70000):
    print (xyz_list[1], yaws_list[1], pitchs_list[1] )
    torch.manual_seed(seed)
    zs = generator.get_zs(1)

    output_name = Path(f'seed_{seed}.mp4')
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name}", fps=5)

  
    info = {}
    with torch.no_grad():
      for idx in tqdm.tqdm(range(len(xyz_list))):
        curriculum['h_mean'] = 0
        curriculum['v_mean'] = 0
        curriculum['h_stddev'] = 0
        curriculum['v_stddev'] = 0

        cur_camera_pos = xyz_list[[idx]]
        cur_camera_lookup = lookup_list[[idx]]
        yaw = yaws_list[idx]
        pitch = pitchs_list[idx]
        fov = fov_list[idx]
        curriculum['fov'] = fov

        print ('cur_camera_pos', cur_camera_pos)
        print ('cur_camera_lookup', cur_camera_lookup)
        print ('yaw', yaw)
        print ('pitch', pitch)
      

        frame, depth_map = generator.forward_camera_pos_and_lookup(
          zs=zs,
          return_aux_img=False,
          forward_points=forward_points ** 2,
          camera_pos=cur_camera_pos,
          camera_lookup=cur_camera_lookup,
          **curriculum)
        #====
        tmp_frm = (frame.squeeze().permute(1,2,0) + 1) * 0.5 * 255
        tmp_frm = tmp_frm.detach().cpu().numpy()
        img_name = Path(f'{idx}.png')
        img_name = f"{outdir}/{img_name}"
        tmp_frm = cv2.cvtColor(tmp_frm, cv2.COLOR_RGB2BGR)

        cv2.imwrite(img_name, tmp_frm)
        info[img_name] = {'cur_camera_pos':str(cur_camera_pos.detach().cpu().numpy()), 'yaw': str(yaw),"pitch": str(pitch)}
        frame_pil = comm_utils.to_pil(frame)

        st_utils.st_image(frame_pil, caption=f"{frame_pil.size}, seed={seed}",
                          debug=debug, st_empty=st_image)
        video_f.write(frame_pil)
        
      video_f.release(st_video=True)
      with open(f"{outdir}/info.json", "w") as outfile:
        json.dump(info, outfile)
    pass



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


