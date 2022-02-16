import os.path
from data.base_dataset import *
from data.image_folder import make_dataset
from data.data_utils import *
from PIL import Image, ImageChops, ImageFile
import PIL
import json
import pickle 
import cv2
import numpy as np
import random
import torch
import openmesh
from tqdm import tqdm
import  os, time
import torchvision.transforms as transforms



class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:
            list_path = os.path.join(opt.dataroot, "lists/ffhq_trainlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_train.pkl' )
        else:
            list_path = os.path.join(opt.dataroot, "lists/ffhq_testlist.pkl")
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

        print ('******************', len(self.data_list), len(self.total_tex))
        self.total_t = []
    def __getitem__(self, index):
        t = time.time()
        tmp = self.data_list[index].split('/')
        tex = self.total_tex[self.data_list[index]][0].astype(np.uint8)

        tex_tensor = self.transform(tex)
        input_dict = { 'Atex':tex_tensor,  'A_path': self.data_list[index]}
        return input_dict

    def __len__(self):
        return len(self.total_tex) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FacescapeTexDataset'
