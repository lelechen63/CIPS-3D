import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import functools
import torchvision
from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import pickle
from PIL import Image
import cv2
import sys
sys.path.append('./photometric_optimization/')
from renderer import Renderer
import util
from models.FLAME import FLAME, FLAMETex
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import tensor_util
from blocks import *
import face_alignment

class Latent2Code(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        # self.save_hyperparameters()
        self.flame_config = flame_config
    
        self.image_size = self.flame_config.image_size
        # networks
        self.nerf_latent_dim = 256
        self.gan_latent_dim = 512
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27
        self.Latent2ShapeExpCode = self.build_Latent2ShapeExpCodeFea( weight = '' if opt.isTrain else opt.Latent2ShapeExpCode_weight)
        self.latent2shape = self.build_latent2shape( weight = '' if opt.isTrain else opt.latent2shape_weight)
        self.latent2exp = self.build_latent2exp(weight = '' if opt.isTrain else opt.latent2exp_weight)
        self.Latent2AlbedoLitCode = self.build_Latent2AlbedoLitCodeFea(weight = '' if opt.isTrain else opt.Latent2AlbedoLitCode_weight)
        self.latent2albedo = self.build_latent2albedo(weight = '' if opt.isTrain else opt.latent2albedo_weight)
        self.latent2lit = self.build_latent2lit(weight = '' if opt.isTrain else opt.latent2lit_weight)
        if opt.isTrain:
            self._initialize_weights()
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)
    
    def build_Latent2ShapeExpCodeFea(self, weight = ''):
        Latent2ShapeExpCode = th.nn.Sequential(
            LinearWN( self.nerf_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for latent2ShapeExpCode feature extraction network')
            Latent2ShapeExpCode.load_state_dict(torch.load(weight))
        return Latent2ShapeExpCode
    def build_latent2shape(self,  weight = ''):
        latent2shape= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.shape_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2Shape network')
            latent2shape.load_state_dict(torch.load(weight))
        return latent2shape
    def build_latent2exp(self, weight = ''):
        latent2exp= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.exp_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2exp network')
            latent2exp.load_state_dict(torch.load(weight))
        return latent2exp
    def build_Latent2AlbedoLitCodeFea(self, weight = ''):
        Latent2AlbedoLitCode = th.nn.Sequential(
            LinearWN( self.gan_latent_dim , 512 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 512, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for Latent2AlbedoLitCode feature extraction network')
            Latent2AlbedoLitCode.load_state_dict(torch.load(weight))
        return Latent2AlbedoLitCode
    
    def build_latent2albedo(self, weight = ''):
        latent2albedo= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.albedo_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2albedo feature extraction network')
            latent2albedo.load_state_dict(torch.load(weight))
        return latent2albedo
    def build_latent2lit(self, weight = ''):
        latent2lit= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.lit_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2lit feature extraction network')
            latent2lit.load_state_dict(torch.load(weight))
        return latent2lit

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def forward(self, shape_latent, appearance_latent, cam, pose, flameshape = None, flameexp= None, flametex= None, flamelit= None ):
        

        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)
        
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        ## render
        albedos = self.flametex(albedocode, self.image_size) / 255.
        ops = self.render(vertices, trans_vertices, albedos, litcode)
        predicted_images = ops['images']

        if flameshape != None:
            flamelit = flamelit.view(-1, 9,3)        
            recons_vertices, _, recons_landmarks3d = self.flame(shape_params=flameshape, expression_params=flameexp, pose_params=pose)
            recons_trans_vertices = util.batch_orth_proj(recons_vertices, cam)
            recons_trans_vertices[..., 1:] = - recons_trans_vertices[..., 1:]

            ## render
            recons_albedos = self.flametex(flametex, self.image_size) / 255.
            recons_ops = self.render(recons_vertices, recons_trans_vertices, recons_albedos, flamelit)
            recons_images = recons_ops['images']
        else:
            recons_images = predicted_images

        
        return landmarks3d, predicted_images, recons_images
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class RigNerft(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        # self.save_hyperparameters()
        self.flame_config = flame_config
        self.image_size = self.flame_config.image_size
        
        # funtion F networks
        latent2code = Latent2Code(flame_config, opt)
        self.Latent2ShapeExpCode, self.latent2shape, self.latent2exp \
        self.Latent2AlbedoLitCode, self.latent2albedo, self.latent2lit = self.get_f(Latent2Code)
        
        self.WGanEncoder = build_WGanEncoder(weight = '' if opt.isTrain else opt.WGanEncoder_weight )

        # Flame
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)
    def get_f(self,negtwork):
        print ('loading weights for Latent2ShapeExpCode feature extraction network')
        network.Latent2ShapeExpCode.load_state_dict(torch.load(self.opt.Latent2ShapeExpCode_weight))
        print ('loading weights for latent2shape feature extraction network')
        network.latent2shape.load_state_dict(torch.load(self.opt.latent2shape_weight))
        print ('loading weights for latent2exp feature extraction network')
        network.latent2exp.load_state_dict(torch.load(self.opt.latent2exp_weight))
        print ('loading weights for Latent2AlbedoLitCode feature extraction network')
        network.Latent2AlbedoLitCode.load_state_dict(torch.load(self.opt.Latent2AlbedoLitCode_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2albedo.load_state_dict(torch.load(self.opt.latent2albedo_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2lit.load_state_dict(torch.load(self.opt.latent2lit_weight))
        
        return network.Latent2ShapeExpCode, network.latent2shape, network.latent2exp\
               network.Latent2AlbedoLitCode, network.latent2albedo, network.latent2lit
    
    def latent2params(self, shape_latent, appearance_latent):
        
        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)
        
        return shapecode, expcode, albedocode, litcode
    
    def build_WGanEncoder(self, weight = ''):
        WGanEncoder = th.nn.Sequential(
            LinearWN( self.shape_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WGanEncoder  network')
            WGanEncoder.load_state_dict(torch.load(weight))
        return WGanEncoder
    

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def forward(self, shape_latent, appearance_latent, cam, pose, flameshape = None, flameexp= None, flametex= None, flamelit= None ):
        

        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)
        
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        ## render
        albedos = self.flametex(albedocode, self.image_size) / 255.
        ops = self.render(vertices, trans_vertices, albedos, litcode)
        predicted_images = ops['images']

        if flameshape != None:
            flamelit = flamelit.view(-1, 9,3)        
            recons_vertices, _, recons_landmarks3d = self.flame(shape_params=flameshape, expression_params=flameexp, pose_params=pose)
            recons_trans_vertices = util.batch_orth_proj(recons_vertices, cam)
            recons_trans_vertices[..., 1:] = - recons_trans_vertices[..., 1:]

            ## render
            recons_albedos = self.flametex(flametex, self.image_size) / 255.
            recons_ops = self.render(recons_vertices, recons_trans_vertices, recons_albedos, flamelit)
            recons_images = recons_ops['images']
        else:
            recons_images = predicted_images

        
        return landmarks3d, predicted_images, recons_images
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
