import numpy as np
import os 
import pickle

# single_params = {
#             'shape': shape.detach().cpu().numpy(),
#             'exp': exp.detach().cpu().numpy(),
#             'pose': pose.detach().cpu().numpy(),
#             'cam': cam.detach().cpu().numpy(),
#             'verts': trans_vertices.detach().cpu().numpy(),
#             'albedos':albedos.detach().cpu().numpy(),
#             'tex': tex.detach().cpu().numpy(),
#             'lit': lights.detach().cpu().numpy()
#         }


with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
    flame_p = pickle.load(f, encoding='latin1')
print (flame_p)

for key in flame_p.keys():
    print ('=-------------')
    print (key, flame_p[key].shape )
    print (flame_p[key])