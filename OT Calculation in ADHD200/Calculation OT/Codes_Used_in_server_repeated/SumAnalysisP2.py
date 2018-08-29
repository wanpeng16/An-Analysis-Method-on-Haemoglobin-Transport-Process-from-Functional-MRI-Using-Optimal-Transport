import numpy as np
import os


def FGcal(path, name):
    stasMap = np.load(path)
    deletmap = np.copy(stasMap)
    for i in range(8468):
        for j in range(8468):
            if i==j:
                deletmap[i,j] = 0
    deletmap = deletmap*10000
    Flux = np.sum(deletmap, axis=0)
    Gush = -(np.sum(deletmap, axis=1))
    print("Saving%s"%(name))
    np.save("FluxOf%s.npy" % (name,), Flux)
    np.save("GushOf%s.npy" % (name,), Gush)


rootdir = "/home/guoxiaobo96/OT/part2/"
list1 = os.listdir(rootdir)
fileDir = []
fileNames = []
for i in range(0,len(list1)):
    if list1[i].endswith(".npy"):
        path = os.path.join(rootdir, list1[i])
        fileDir.append(path)
        fileNames.append(list1[i])
for i in range(0,len(fileDir)):
    FGcal(fileDir[i], fileNames[i])