import os
from argparse import ArgumentParser
from collections import OrderedDict
import torch
import torch.nn as nn
import random
import pickle
import pytorch_lightning as pl
from data import FFHQDataModule
from options.train_options import TrainOptions
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import sys
sys.path.append('./photometric_optimization')
import util
# define flame config
flame_config = {
        # FLAME
        'flame_model_path': '/home/uss00022/lelechen/basic/flame_data/data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': '/home/uss00022/lelechen/basic/flame_data/data/landmark_embedding.npy',
        'tex_space_path': '/home/uss00022/lelechen/basic/flame_data/data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'batch_size': 1,
        'image_size': 512,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': '/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }

flame_config = util.dict2obj(flame_config)

opt = TrainOptions().parse()

if opt.debug:
    opt.nThreads = 1

if  opt.name == 'Latent2Code':
    from rignet import Latent2CodeModule as module

dm = FFHQDataModule( opt)

if opt.isTrain:
    print ( opt.name)
    model = module(flame_config, opt)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath= os.path.join(opt.checkpoints_dir, opt.name),
        filename= opt.name +  '-{epoch:02d}-{train_loss:.2f}'
    )

    if len( opt.gpu_ids ) == 1:
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=1,  max_epochs= 100000, progress_bar_refresh_rate=20)
    else:
        # trainer = pl.Trainer(callbacks=[checkpoint_callback], precision=16,gpus= len( opt.gpu_ids ), accelerator='ddp', max_epochs= 100000, progress_bar_refresh_rate=20)
        trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=len( opt.gpu_ids ), accelerator='dp', max_epochs= 10000, progress_bar_refresh_rate=20)

    trainer.fit(model, dm)

else:
    print ('!!!!!!' + opt.name + '!!!!!!!!')
    if opt.name == 'Latent2Code':
        
        pass