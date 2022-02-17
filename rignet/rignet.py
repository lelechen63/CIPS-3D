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
from model import BiSeNet
import util
from utils import *
from models.FLAME import FLAME, FLAMETex
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
from blocks import *
import face_alignment

def init_weight(m):
    if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

class Latent2CodeModule(pl.LightningModule):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.save_hyperparameters()
        self.flame_config = flame_config
    
        self.image_size = self.flame_config.image_size
        
        self.visualizer = Visualizer(opt)
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
        # landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        # landmarks2d[..., 1:] = - landmarks2d[..., 1:]

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
        
        tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'] }
        output = OrderedDict({
            'loss': all_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()}            
        self.visualizer.print_current_errors(self.current_epoch, batch_idx, errors, 0)
        return output
                
    def configure_optimizers(self):
        lr = self.opt.lr
        opt_g = torch.optim.Adam( list(self.Latent2ShapeExpCode.parameters()) + \
                                  list(self.Latent2AlbedoLitCode.parameters()) + \
                                  list(self.latent2shape.parameters()) + \
                                  list(self.latent2exp.parameters()) + \
                                  list(self.latent2albedo.parameters()) + \
                                  list(self.latent2lit.parameters()) \
                                  , lr=lr, betas=(self.opt.beta1, 0.999))
        

        return [opt_g], []

    def on_epoch_end(self):
        if self.current_epoch % 10 == 0:
            batch = self.batch
            landmarks3d, predicted_images = self(batch['shape_latent'], batch['appearance_latent'], batch['cam'], batch['pose'])
            
            visind = 0
            # grids['images'] = torchvision.utils.make_grid(predicted_images[visind]).detach().cpu()
            # grids['landmarks'] = torchvision.utils.make_grid(
            #     util.tensor_vis_landmarks(predicted_images[visind], landmarks2d[visind]))
            # grids['gt_images'] = torchvision.utils.make_grid(batch['gt_image'][visind]).detach().cpu()
            # grids['gt_landmarks'] = torchvision.utils.make_grid(
            #     util.tensor_vis_landmarks(batch['gt_image'][visind], batch['gt_landmark'][visind]))

            # grid = torch.cat(list(grids.values()), 1)
            # grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            # grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
            # cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

            gtimage = batch['gt_image'].data[0].cpu() 
            gtimage = tensor_util.tensor2im(gtimage  , normalize = True)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][0])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind], gtlmark[visind])
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = True)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][0])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[0].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = True)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][0])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            genlmark = util.batch_orth_proj(landmarks2d, batch['cam'])
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'][visind],genlmark[visind])
            genlmark = tensor_util.tensor2im(genlossslmark  , normalize = True)
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
       
            self.visualizer.display_current_results(visuals, self.current_epoch, 1000000) 

            self.trainer.save_checkpoint( os.path.join( self.ckpt_path, 'latest.ckpt') )
