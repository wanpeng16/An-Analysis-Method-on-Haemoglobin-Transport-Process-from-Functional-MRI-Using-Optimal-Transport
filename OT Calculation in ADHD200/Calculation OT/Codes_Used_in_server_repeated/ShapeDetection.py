import os
import nilearn.image as image
import numpy as np


rootdir = "/home/guoxiaobo96/OT/data"
list1 = os.listdir(rootdir)
fileDir = []
fileNames = []
for i in range(0,len(list1)):
    path = os.path.join(rootdir,list1[i])
    list2 = os.listdir(path)
    for names in list2:
        if names.endswith(".nii.gz"):
            Dir = os.path.join(path,names)
            fileDir.append(Dir)
            fileNames.append(names)
for i in range(0,len(fileDir)):
    data = image.load_img(fileDir[i])
    shape = data.shape
    np.save("shapeOf%s.npy" % (fileNames[i],), shape)