import os
import pickle as pkl 

import numpy as np 

import cv2




def debug_single():
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')
    print (flame_p)

    for key in flame_p.keys():
        print ('=-------------')
        print (key, flame_p[key].shape )
        print (flame_p[key])

    with open("/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    for key in info.keys():
        print ('=-------------')
        print (key, info[key].shape )
        print (info[key])


debug_single()


