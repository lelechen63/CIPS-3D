import os.path
from PIL import Image, ImageChops, ImageFile
import PIL
import json
import pickle 
import cv2
import numpy as np
import random
import torch
from tqdm import tqdm
import  os, time
import torchvision.transforms as transforms




class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:
            list_path = os.path.join(opt.dataroot, "ffhq_trainlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_train.pkl' )
        else:
            list_path = os.path.join(opt.dataroot, "ffhq_testlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_test.pkl' )

        if opt.debug:
            list_path = list_path[:-4] + '_debug.pkl'
            zip_path = zip_path[:-4] + '_debug.pkl'
        
        _file = open(list_path, "rb")
        self.data_list = pickle.load(_file)
        _file.close()

        _file = open(zip_path, "rb")
        self.total_data = pickle.load(_file)
        _file.close()

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        print ('******************', len(self.data_list), len(self.total_data))
        self.total_t = []
    def __getitem__(self, index):
        t = time.time()
        name = self.data_list[index]

        data = self.total_data[self.data_list[index]]
        """
            data[name] ={'shape': shape, 
                 'exp': exp,
                 'pose': pose,
                 'cam': cam,
                 'tex': tex,
                 'lit': lit,
                 'cam_pose': camera_pose,
                 'z_nerf': z_nerf,
                 'z_gan': z_gan,
                 'gt_img': img,
                 'gt_landmark': landmark
                }
        """
        print (data.keys())
        data['gt_image'] = self.transform(data['gt_img'])
        return data

    def __len__(self):
        return len(self.total_data) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FFHQDataset'
