from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
import nilearn.image as image
from nilearn.image import smooth_img
import nibabel as nib
import numpy as np
mask_img = load_mni152_brain_mask()

def FGcal(M):
    deletmap = np.copy(M)
    z,x = M.shape
    for i in range(z):
        for j in range(x):
            if i==j:
                deletmap[i,j] = 0
    deletmap = deletmap*10000
    Flux = np.sum(deletmap, axis=0)
    return Flux/Flux.sum()


def BackToMNI(target, data, spos):
    l = len(data)
    for i in range(l):
        x, y, z = spos[i, :]
        target[int(x), int(y), int(z)] = data[i]

def backtoMNI(data, map1, map2, map3):
    BackToMNI(data, map1, spos)
    BackToMNI(data, map2, spos2)
    BackToMNI(data, map3, spos3)
    new_image = nib.Nifti1Image(9*data, mask_img.affine, mask_img.header)
    return new_image


mask_img = load_mni152_brain_mask()
maskimgdata = mask_img.get_fdata()
spos =  np.load("/home/guoxiaobo96/OT/CalUsing/spos.npy")
spos2 = np.load("/home/guoxiaobo96/OT/CalUsing/spos2.npy")
spos3 = np.load("/home/guoxiaobo96/OT/CalUsing/spos3.npy")

part5M1path = "/home/guoxiaobo96/OT/CalUsing/part5path/part5M1path.npy"
part5M2path = "/home/guoxiaobo96/OT/CalUsing/part5path/part5M2path.npy"
part5M3path = "/home/guoxiaobo96/OT/CalUsing/part5path/part5M3path.npy"

subjects = np.load("/home/guoxiaobo96/OT/CalUsing/part5path/part5subjects.npy")

part5M1dirc = np.load(part5M1path)
part5M2dirc = np.load(part5M2path)
part5M3dirc = np.load(part5M3path)

#part3 = "/home/guoxiaobo96/OT/part3/"
#subjects3 = os.listdir(part3)
#part4 = "/home/guoxiaobo96/OT/part4/"
#subjects4 = os.listdir(part4)
#part5 = "/home/guoxiaobo96/OT/part4/"
#subjects5 = os.listdir(part5)
'''
for i in range(len(subjects5)):
    for name in M2list:
        if name.startswith("stasMap2%s"%(subjects5[i])):
            path = os.path.join(M2dir, name)
            M2path.append(path)
        if name.startswith("stasMap3%s"%(subjects5[i])):
            path = os.path.join(M2dir, name)
            M3path.append(path)
'''
for i in range(len(part5M1dirc)):
    M1 = np.load(part5M1dirc[i])
    M2 = np.load(part5M2dirc[i])
    M3 = np.load(part5M3dirc[i])
    map1 = FGcal(M1)
    map2 = FGcal(M2)
    map3 = FGcal(M3)
    back2MNI = np.zeros(91 * 109 * 91)
    back2MNI = back2MNI.reshape(91, 109, 91)
    newbrain = backtoMNI(back2MNI, map1, map2, map3)
    nib.save(newbrain,subjects[i])
