
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import util
from dataset import *

class Latent2CodeModule():
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.flame_config = flame_config
        self.visualizer = Visualizer(opt)
        if opt.cuda:
            self.device = torch.device("cuda")
        self.latent2code = Latent2Code( flame_config, opt)
        
        self.optimizer =  optim.Adam(self.latent2code.parameters(),lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        # self.optimizer = optim.Adam( list(self.latent2code.Latent2ShapeExpCode.parameters()) + \
        #                           list(self.latent2code.Latent2AlbedoLitCode.parameters()) + \
        #                           list(self.latent2code.latent2shape.parameters()) + \
        #                           list(self.latent2code.latent2exp.parameters()) + \
        #                           list(self.latent2code.latent2albedo.parameters()) + \
        #                           list(self.latent2code.latent2lit.parameters()) \
        #                           , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        self.latent2code =torch.nn.DataParallel(self.latent2code, device_ids=range(len(self.opt.gpu_ids)))
        self.latent2code = self.latent2code.to(self.device)
        self.dataset  = FFHQDataset(opt)
        self.data_loader = DataLoaderWithPrefetch(self.dataset, \
                                    batch_size=opt.batchSize,\
                                    drop_last=True,\
                                    num_workers = opt.nThreads, \
                                    prefetch_size = min(8, opt.nThreads))
        print ('========', len(self.data_loader),'========')
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

    def train(self):
        for p in self.latent2code.parameters():
            p.requires_grad = True 
        for epoch in range( 1000):
            for step, batch in enumerate(tqdm(self.data_loader)):

                landmarks3d, predicted_images = self.latent2code.forward(batch['shape_latent'].to(self.device), \
                                            batch['appearance_latent'].to(self.device), \
                                            batch['cam'].to(self.device), batch['pose'].to(self.device))
                losses = {}
                losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                loss = losses['landmark'] + losses['photometric_texture']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                self.visualizer.print_current_errors(epoch, step, errors, 0)

            if epoch % self.opt.save_step == 0:
                visind = 0
                gtimage = batch['gt_image'].data[0].cpu()
                gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
                gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
                gtimage = tensor_util.writeText(gtimage, batch['image_path'][0])
                gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
                gtimage = np.clip(gtimage, 0, 255)

                gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
                gtlmark[..., 1:] = - gtlmark[..., 1:]

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

                genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
                genlmark[..., 1:] = - genlmark[..., 1:]

                genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
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
        
                self.visualizer.display_current_results(visuals, epoch, save_step) 
                torch.save(self.latent2code, os.path.join( self.ckpt_path, 'latent2code.pth'))

                torch.save(self.latent2code.Latent2ShapeExpCode, os.path.join( self.ckpt_path, 'Latent2ShapeExpCode.pth'))
                torch.save(self.latent2code.Latent2AlbedoLitCode, os.path.join( self.ckpt_path, 'Latent2AlbedoLitCode.pth'))
                torch.save(self.latent2code.latent2shape, os.path.join( self.ckpt_path, 'latent2shape.pth'))
                torch.save(self.latent2code.latent2exp, os.path.join( self.ckpt_path, 'latent2exp.pth'))
                torch.save(self.latent2code.latent2albedo, os.path.join( self.ckpt_path, 'latent2albedo.pth'))
                torch.save(self.latent2code.latent2lit, os.path.join( self.ckpt_path, 'latent2lit.pth'))

