import os

losstxt = '/home/uss00022/lelechen/github/CIPS-3D/checkpoints_debug3/Latent2Code/loss_log.txt'
reader = open(losstxt)
l = reader.readline()
ss = 0
loss_land = []
loss_tex = []
while l:
    print (l)
    tmp = l[-1].split(' ')
    print (tmp)
    loss_land.append(1)
    ss += 1
    if ss == 10:
        break
    l = reader.readline()
reader.close()
