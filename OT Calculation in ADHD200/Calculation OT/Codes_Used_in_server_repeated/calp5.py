from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
import nilearn.image as image
from nilearn.image import smooth_img
import numpy as np
import os
import ot


mask_img = load_mni152_brain_mask()
maskimgdata = mask_img.get_fdata()
spos2 = np.load("/home/guoxiaobo96/OT/CalUsing/spos2.npy")
M2 = np.load("/home/guoxiaobo96/OT/CalUsing/M2.npy")
spos3 = np.load("/home/guoxiaobo96/OT/CalUsing/spos3.npy")
M3 = np.load("/home/guoxiaobo96/OT/CalUsing/M3.npy")


'''
MNI template mask, Used for choose voxels in Brain gray matter
Getting mask numpy data from nifti format

Fetch spos data, spos refer to Smoothed Points Positions, contains 
the data needed to calculate 's position, and do downsampling to the 1/27

Fetch M data, M refer to distance map used in OT method, Calculated 
through the spos

'''


def datamask(data,maskpos):
    '''

    Args:
        data: Used for being Masked with template
        maskpos: Template positions used to mask data from space (sampling)

    Returns: Sampled data through mask position, and normalized

    '''
    q, w = maskpos.shape
    newdata = np.zeros(q)
    for i in range(q):
        x, y, z = maskpos[i, :]
        newdata[i] = data[int(x), int(y), int(z)]
    return newdata/(newdata.sum())


def GetOTM(path, name, spos, M):
    data = image.load_img(path)
    shape = data.shape
    l = shape[3]
    dataS = smooth_img(data, fwhm=None)
    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=8.0, memory='nilearn_cache', memory_level=1)
    masker = masker.fit()
    data2d = masker.transform(dataS)
    data3d = masker.inverse_transform(data2d)
    odata = data3d.get_fdata()
    odata = odata[:, :, :, :]
    h,j = M.shape
    stasMap = np.zeros(h*j)
    stasMap = stasMap.reshape(h, j)
    for i in range(l-1):
        d1 = datamask(odata[:, :, :, i], spos)
        d2 = datamask(odata[:, :, :, i + 1], spos)
        T = ot.emd(d1, d2, M)
        stasMap = stasMap + T
        print("finish%s"%(name), (i/(l-1))*100, "%")
    np.save("stasMap%s.npy"%(name,), stasMap)
'''
Loading ADHD200 preprocessed data, manipulate with basic preprocessing method,
include dismiss infinite points, Gaussin smooth with 8x8x8 mm kernel in MNI space
Masking with MNI gray matter mask
Converting back to 3-D array data, convert to numpy format
'''

'''
Cut the first 10 series and the last 5 to minimize the influence of subjects condition
Calculate every OT matrix between 2 continues sampling data
Sum all the OT matrix as a statistic result
Save as a result
'''
rootdir = "/home/guoxiaobo96/OT/part4"
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
    GetOTM(fileDir[i], fileNames[i],spos2,M2)
for i in range(0,len(fileDir)):
    GetOTM(fileDir[i], fileNames[i],spos3,M3)