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

def init_weight(m):
    if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

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
class RigNerfModule(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.flame_config = flame_config
        self.image_size = self.flame_config.image_size
        self.visualizer = Visualizer(opt)

        # Latent2CodeF = 
              
        # networks
        self.nerf_latent_dim = 256
        self.gan_latent_dim = 512
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27
    
        self.Latent2ShapeExpCode = th.nn.Sequential(
            LinearWN( self.nerf_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        self.latent2shape= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.shape_dim )
        )
        self.latent2exp= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.exp_dim )
        )

        self.Latent2AlbedoLitCode = th.nn.Sequential(
            LinearWN( self.gan_latent_dim , 512 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 512, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        self.latent2albedo= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.albedo_dim )
        )
        self.latent2lit= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.lit_dim)
        )

        self.Latent2ShapeExpCode = self.Latent2ShapeExpCode.apply(init_weight)
        self.latent2shape = self.latent2shape.apply(init_weight)
        self.latent2exp = self.latent2exp.apply(init_weight)
        self.Latent2AlbedoLitCode = self.Latent2AlbedoLitCode.apply(init_weight)
        self.latent2albedo = self.latent2albedo.apply(init_weight)
        self.latent2lit = self.latent2lit.apply(init_weight)

        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()
        self.GANloss = nn.BCEWithLogitsLoss()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)
    
    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
   

    def forward(self, shape_latent, appearance_latent, cam, pose ):
        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)

        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea)
        litcode = litcode.view(shape_latent.shape[0], 9,3)
        
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        ## render
        albedos = self.flametex(albedocode, self.image_size) / 255.
        ops = self.render(vertices, trans_vertices, albedos, litcode)
        predicted_images = ops['images']

        return landmarks3d, predicted_images

    

    def training_step(self, batch, batch_idx):
        self.batch = batch
        landmarks3d, predicted_images = self(batch['shape_latent'], batch['appearance_latent'], batch['cam'], batch['pose'])

        losses = {}
        losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2]) * self.flame_config.w_lmks
        losses['photometric_texture'] = (batch['img_mask'] * (predicted_images - batch['gt_image'] ).abs()).mean() * self.flame_config.w_pho

        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        
        tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'],  }
        output = OrderedDict({
            'loss': all_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
        self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
        return output


    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam( list(self.Latent2ShapeExpCode.parameters()) + \
                                  list(self.Latent2AlbedoLitCode.parameters()) + \
                                  list(self.latent2shape.parameters()) + \
                                  list(self.latent2exp.parameters()) + \
                                  list(self.latent2albedo.parameters()) + \
                                  list(self.latent2lit.parameters()) \
                                  , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        
        return [optimizer], []
       

    def on_epoch_end(self):
        step = 10
        if self.current_epoch % step == 0:
            batch = self.batch
            landmarks3d, predicted_images = self(batch['shape_latent'], batch['appearance_latent'], batch['cam'], batch['pose'])
            visind = 0
           
            gtimage = batch['gt_image'].data[0].cpu()
            gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][0])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            # torch.Size([1, 3, 512, 512]) torch.Size([1, 68, 2]) 
            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
            gtlmark = gtlmark.squeeze(0)
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][0])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[0].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = False)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][0])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            genlmark = util.batch_orth_proj(landmarks3d, batch['cam'])
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
            genlmark = genlmark.squeeze(0)
            genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = util.writeText(genlmark, batch['image_path'][0])
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = np.clip(genlmark, 0, 255)

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('genlmark', genlmark )
            ])
       
            self.visualizer.display_current_results(visuals, self.current_epoch, step) 

            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )
