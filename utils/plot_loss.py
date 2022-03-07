import os
import numpy as np 
import matplotlib.pyplot as plt


losstxt = '/home/uss00022/lelechen/github/CIPS-3D/checkpoints_debug3/Latent2Code/loss_log.txt'
reader = open(losstxt)
l = reader.readline()
ss = 0
loss_land = []
loss_tex = []
axis =[]
while l:
    print (l)
    tmp = l[:-1].split(' ')
    l_land = tmp[7]
    l_tex =tmp[9]
    print (l_land, l_tex)

    loss_land.append(float(l_land))
    loss_tex.append(float(l_tex))
    axis.append(ss)
    ss += 1
    if ss == 1000:
        break
    l = reader.readline()
reader.close()

plt.plot(axis, loss_land, 'r--', axis, loss_tex, 'b--')
plt.show()
plt.savefig('./gg.png')