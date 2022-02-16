import os
import pickle 

import numpy as np 

import cv2

def debug_single():
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')

    for key in flame_p.keys():
        print ('=-------------')
        print (key, flame_p[key].shape )
        # print (flame_p[key])

    with open("/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    info = info['results/model_interpolation/0.png']
    for key in info.keys():
        print ('=-------------')
        try:
            print (key, info[key].shape )
        except:
            print (key, info[key] )


# debug_single()

def get_debug():
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')

    with open("/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    
    name = '0.png'
    info = info['results/model_interpolation/0.png']
    img_p = '/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/0.png'
    img = cv2.imread(img_p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = flame_p['shape'] #[1,100]
    exp = flame_p['exp'] #[1,50]
    pose = flame_p['pose'] #[1,6]
    cam = flame_p['cam'] #[1,3]
    tex = flame_p['tex'] #[1,50]
    lit = flame_p['lit'] #[1,9,3]
    
    landmark = flame_p['landmark2d'] #[1,68,2]
    """ 
        we normalize the landmark into 0-1.
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
    """
    camera_pose = info['cur_camera_pos'] #[1,3]
    z_nerf = info['z_nerf'] # [1,256]
    z_gan = info['z_inr'] #[1,512]

    data = {}
    ffhq_trainlist = []

    ffhq_trainlist.append(name)
    data[name] ={'shape': shape, 
                 'exp': exp,
                 'pose': pose,
                 'cam': cam,
                 'tex': tex,
                 'lit': lit,
                 'cam_pose': camera_pose,
                 'z_nerf': z_nerf,
                 'z_gan': z_gan,
                 'img': img
                }
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_train_debug.pickle", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_trainlist_debug.pickle", 'wb') as handle:
        pickle.dump(ffhq_trainlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
get_debug()