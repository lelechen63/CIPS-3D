import os

losstxt = '/home/uss00022/lelechen/github/CIPS-3D/checkpoints_debug3/Latent2Code/loss_log.txt'
reader = open(losstxt)
while line := reader.readline():
    print (line.rstrip())