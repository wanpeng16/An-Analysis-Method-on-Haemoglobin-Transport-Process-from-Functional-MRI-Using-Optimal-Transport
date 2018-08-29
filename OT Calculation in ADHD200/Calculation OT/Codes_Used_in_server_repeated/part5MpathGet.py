from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
import nilearn.image as image
from nilearn.image import smooth_img
import numpy as np
import os
import ot


mask_img = load_mni152_brain_mask()
maskimgdata = mask_img.get_fdata()
spos =  np.load("/home/guoxiaobo96/OT/CalUsing/spos2.npy")
spos2 = np.load("/home/guoxiaobo96/OT/CalUsing/spos2.npy")
spos3 = np.load("/home/guoxiaobo96/OT/CalUsing/spos3.npy")

#part3 = "/home/guoxiaobo96/OT/part3/"
#subjects3 = os.listdir(part3)
#part4 = "/home/guoxiaobo96/OT/part4/"
#subjects4 = os.listdir(part4)
part5 = "/home/guoxiaobo96/OT/part4/"
subjects5 = os.listdir(part5)

M1dir = "/home/guoxiaobo96/OTMsumResult/"
M1list = os.listdir(M1dir)
M1path = []
for i in range(len(subjects5)):
    for name in M1list:
        if name.startswith("stasMap%s"%(subjects5[i])):
            path = os.path.join(M1dir, name)
            M1path.append(path)

M2dir = "/home/guoxiaobo96/OT/MongeMap23/part5M23/"
M2list = os.listdir(M2dir)
M2path = []
M3path = []
for i in range(len(subjects5)):
    for name in M2list:
        if name.startswith("stasMap2%s"%(subjects5[i])):
            path = os.path.join(M2dir, name)
            M2path.append(path)
        if name.startswith("stasMap3%s"%(subjects5[i])):
            path = os.path.join(M2dir, name)
            M3path.append(path)

np.save("part5M1path", M1path)
np.save("part5M2path", M2path)
np.save("part5M3path", M3path)