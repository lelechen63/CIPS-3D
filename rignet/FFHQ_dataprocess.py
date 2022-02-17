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
    shape = flame_p['shape'].reshape(-1) #[1,100]
    exp = flame_p['exp'].reshape(-1) #[1,50]
    pose = flame_p['pose'].reshape(-1) #[1,6]
    cam = flame_p['cam'].reshape(-1) #[1,3]
    tex = flame_p['tex'].reshape(-1) #[1,50]
    lit = flame_p['lit'].reshape(-1) #[1,9,3]
    print(flame_p['image_masks'].shape, '+++++')
    image_masks = np.squeeze(flame_p['image_masks'],axis=0)
    
    landmark = np.squeeze(flame_p['landmark3d'], axis=0) #[1,68,2]
    """ 
        we normalize the landmark into 0-1.
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
    """
    camera_pose = info['cur_camera_pos'].reshape(-1) #[1,3]
    z_nerf = info['z_nerf'].reshape(-1) # [1,256]
    z_gan = info['z_inr'].reshape(-1) #[1,512]

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
                 'shape_latent': z_nerf,
                 'appearance_latent': z_gan,
                 'gt_img': img,
                 'gt_landmark': landmark,
                 'img_mask':
                }
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_train_debug.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_trainlist_debug.pkl", 'wb') as handle:
        pickle.dump(ffhq_trainlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
get_debug()