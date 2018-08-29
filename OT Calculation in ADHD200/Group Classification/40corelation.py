import numpy as np
import os

M40 = np.zeros(40* 40)
M40 = M40.reshape(40, 40)
def Mcal(path1, path2):
    t1 = np.load(path1)
    t2 = np.load(path2)
    d = (t1/t1.sum()) - (t2/t2.sum())
    c = abs(d.sum())
    return c



rootdir = "/home/guoxiaobo96/OTMsumResult/"
list1 = os.listdir(rootdir)
fileDir = []
fileNames = []
for i in range(0,len(list1)):
    if list1[i].endswith(".nii.gz"):
        path = os.path.join(rootdir, list1[i])
        fileDir.append(path)
        fileNames.append(list1[i])
for i in range(0,len(fileDir)):
    for j in range(0, fileDir):
        c = Mcal(fileDir[i], fileDir[j])
        M40[i, j] = c
np.save("CorrelationM.npy", M40)
N = np.array(fileNames)
np.save("list.npy", N)

