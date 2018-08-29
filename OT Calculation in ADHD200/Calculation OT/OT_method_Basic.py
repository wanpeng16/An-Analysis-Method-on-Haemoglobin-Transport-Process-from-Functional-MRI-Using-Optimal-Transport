from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_brain_mask
import nilearn.image as image
from nilearn.plotting import plot_roi, plot_stat_map, plot_glass_brain
from nilearn.image import smooth_img
from nilearn.image import index_img
from nilearn import datasets
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import numpy as np

import ot

mask_img = load_mni152_brain_mask()


def datamask(data,maskpos):
    q, w = maskpos.shape
    newdata = np.zeros(q)
    for i in range(q):
        x, y, z = maskpos[i, :]
        newdata[i] = data[int(x), int(y), int(z)]
    return newdata/(newdata.sum())
#%%


plot_roi(mask_img)
plt.show()
maskimgdata = mask_img.get_fdata()
pos = np.zeros(3*228453)
pos = pos.reshape(228453, 3)
o = 0
for i in range(91):
    for j in range(109):
        for k in range(91):
            if maskimgdata[i,j,k] == 1:
                pos[o,:] = i,j,k
                o = o+1

#%%


spos = np.zeros(3*8468)
spos = spos.reshape(8468, 3)
o = 0
for i in range(0,91,3):
    for j in range(0,109,3):
        for k in range(0,91,3):
            if maskimgdata[i,j,k] == 1:
                spos[o,:] = i,j,k
                o = o+1


#%%

np.save("spos.npy", spos)
#%%
spos2 = np.zeros(8442*3)
spos2 = spos2.reshape(8442,3)
o = 0
for i in range(1,91,3):
    for j in range(1,109,3):
        for k in range(1,91,3):
            if maskimgdata[i,j,k] == 1:
                spos2[o,:] = i,j,k
                o = o+1
np.save("spos2.npy", spos2)
#%%
M2 = np.zeros(8442*8442)
M2 = M2.reshape(8442,8442)
for i in range(8442):
    for j in range(8442):
        p1 = spos2[i]
        p2 = spos2[j]
        M2[i, j] = np.linalg.norm(p1-p2)
plt.imshow(M2)
plt.show()
#%%
np.save("M2.npy", M2)
#%%
spos3 = np.zeros(8454*3)
spos3 = spos3.reshape(8454,3)
o = 0
for i in range(2,91,3):
    for j in range(2,109,3):
        for k in range(2,91,3):
            if maskimgdata[i,j,k] == 1:
                spos3[o,:] = i,j,k
                o = o+1
np.save("spos3.npy", spos3)
#%%
M3 = np.zeros(8454*8454)
M3 = M3.reshape(8454,8454)
for i in range(8454):
    for j in range(8454):
        p1 = spos3[i]
        p2 = spos3[j]
        M3[i, j] = np.linalg.norm(p1-p2)
plt.imshow(M3)
plt.show()
#%%

np.save('M3.npy',M3)
#%%

M = np.zeros(8468*8468)
M = M.reshape(8468,8468)
for i in range(8468):
    for j in range(8468):
        p1 = spos[i]
        p2 = spos[j]
        M[i, j] = np.linalg.norm(p1-p2)
#%%
import matplotlib
matplotlib.use('pgf')
plt.figure(figsize=(7, 7))
plt.imshow(M, cmap='viridis')
plt.savefig('MongeMap.pgf')


#%%


imshow(M)
plt.show()
np.save("M.npy", M)
#%%


data = image.load_img("200.nii.gz")
dataS = smooth_img(data, fwhm=None)

img = index_img(dataS,12)
plot_stat_map(img)
plt.show()


masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=8.0,memory='nilearn_cache', memory_level=1)
masker = masker.fit()
data2d = masker.transform(dataS)

data3d = masker.inverse_transform(data2d)
print(data3d.shape)
#%%


odata = data3d.get_fdata()
'''
odata_11 = odata[:, :, :, 11]
odata_12 = odata[:, :, :, 12]
odata_13 = odata[:, :, :, 13]

newdata_11 = datamask(odata_11, spos)
newdata_12 = datamask(odata_12, spos)
newdata_13 = datamask(odata_13, spos)
print(newdata_11.shape)
'''
#%%

T1 = ot.emd(newdata_11, newdata_12, M) # exact linear program
imshow(T1)
plt.show()
#%%


T2=ot.emd(newdata_12, newdata_13, M) # exact linear program
imshow(T2)
plt.show()
#%%


T13=ot.emd(newdata_11, newdata_13, M) # exact linear program
imshow(T13)
plt.show()
#%%


odata_1030 = odata[:, :, :, 9:30]
OTMs1030 = np.zeros(20*8468*8468)
OTMs1030 = OTMs1030.reshape(8468, 8468, 20)

for i in range(20):
    d1 = datamask(odata_1030[:, :, :, i], spos)
    d2 = datamask(odata_1030[:, :, :, i+1], spos)
    T = ot.emd(d1, d2, M)
    OTMs1030[:, :, i] = T

print(OTMs1030.sum())
#%%


stasMap1 = np.zeros(8468)
for i in range(20):
    s = np.sum(OTMs1030[:, :, i], axis=0)
    stasMap = stasMap + s
print(stasMap.sum)
#%%
np.save("stasMap1.npy",stasMap)

#%%

odata_1080 = odata[:, :, :, 9:80]
stasMap1080 = np.zeros(8468*8468)
stasMap1080 = stasMap1080.reshape(8468, 8468)

for i in range(70):
    d1 = datamask(odata_1080[:, :, :, i], spos)
    d2 = datamask(odata_1080[:, :, :, i+1], spos)
    T = ot.emd(d1, d2, M)
    #s = np.sum(T, axis=0)
    stasMap1080 = stasMap1080 + T

print(stasMap1080.sum())
#%%


np.save("stasMap1080.npy", stasMap1080)
#%%


imshow(stasMap1080)
plt.show()
plt.plot(stasMap1080.ravel())
plt.show()
#%%


odata_80170 = odata[:, :, :, 80:171]
stasMap80170 = np.zeros(8468*8468)
stasMap80170 = stasMap80170.reshape(8468, 8468)

for i in range(90):
    d1 = datamask(odata_80170[:, :, :, i], spos)
    d2 = datamask(odata_80170[:, :, :, i+1], spos)
    T = ot.emd(d1, d2, M)
    #s = np.sum(T, axis=0)
    stasMap80170 = stasMap80170 + T
    print(i)

print(stasMap80170.sum())
#%%


np.save("stasMap80170.npy",stasMap80170)
imshow(stasMap80170)
plt.show()
plt.plot(stasMap80170.ravel())
plt.show()
#%%


stasMap_200 = stasMap80170+stasMap1080
np.save("stasMap_200.npy",stasMap_200)
#%%




#%%


import nibabel as nib
from nilearn import surface
from nilearn import plotting


back2MNI = np.zeros(91*109*91)
back2MNI = back2MNI.reshape(91, 109, 91)
for i in range(8468):
    x, y, z = spos[i, :]
    back2MNI[int(x), int(y), int(z)] = stasMapExp[i]

new_image = nib.Nifti1Image(back2MNI, mask_img.affine, mask_img.header)
'''fsaverage = datasets.fetch_surf_fsaverage()
texture = surface.vol_to_surf(new_image, fsaverage.pial_right)

plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere', colorbar=True,
                            threshold=0, bg_map=fsaverage.sulc_right)
'''
plotting.plot_glass_brain(new_image, threshold=400)
plt.show()
plot_stat_map(new_image, threshold=400)
plt.show()
#%%
plt.plot(np.log(stasMap*2000))
plt.show()
stasMapLog = -(np.log(stasMap*2000))

back2MNI = np.zeros(91*109*91)
back2MNI = back2MNI.reshape(91, 109, 91)

for i in range(8468):
    x, y, z = spos[i, :]
    back2MNI[int(x), int(y), int(z)] = stasMapExp[i]

new_image = nib.Nifti1Image(back2MNI, mask_img.affine, mask_img.header)

plotting.plot_glass_brain(new_image, threshold=200000)
plt.show()
plot_stat_map(new_image, threshold=200000)
plt.show()

#%%


MapSum1030 = np.zeros(8468*8468)
MapSum1030 = MapSum1030.reshape(8468, 8468)
for i in range(20):
    MapSum1030 = OTMs1030[:, :, i] + MapSum1030
#%%


np.save("OTMs1030.npy", OTMs1030)
np.save("MapSum1030.npy", MapSum1030)
#%%


from mpl_toolkits.mplot3d import Axes3D
MapSum1030 = np.load("MapSum1030.npy")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(8468):
    for j in range(8468):
        xs = i
        ys = j
        zs = MapSum1030[i, j]
        if zs !=0:
            ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
#%%


import matplotlib.cbook as cbook
stasMap_200 = np.load("stasMap_200.npy")
imshow(stasMap_200)
plt.show()
#%%


plt.plot(stasMap_200.ravel(),'ro')
plt.show()
#%%
deletmap = np.copy(stasMap_200)
for i in range(8468):
    for j in range(8468):
        if i==j:
            deletmap[i,j] = 0
#%%


plt.plot(deletmap.ravel())
plt.show()
#%%
def levelseg(data, thre):
    maxs = np.where(data>thre)
    xs = maxs[0]
    ys = maxs[1]
    shape1 = xs.shape
    length = shape1[0]
    return xs, ys, length
#%%
def backtoMNI(data):
    new_image = nib.Nifti1Image(data, mask_img.affine, mask_img.header)
    return new_image


def mark(data, xs ,ys ,l, tag):
    for i in range(l):
        x = xs[i]
        y = ys[i]
        x1, y1, z1 = spos[x]
        x2, y2, z2 = spos[y]
        data[int(x1), int(y1), int(z1)] = tag
        data[int(x2), int(y2), int(z2)] = tag
    return data

#plotting.plot_glass_brain(new_image, threshold=200000)
#plt.show()
#plot_stat_map(new_image, threshold=200000)
#plt.show()
#%%


back2MNI = np.zeros(91*109*91)
back2MNI = back2MNI.reshape(91, 109, 91)

for i in range(20):
    xs, ys, length = levelseg(deletmap, i*0.000001+0.00003)
    back2MNI = mark(back2MNI, xs, ys, length, i+20) + back2MNI


B = backtoMNI(back2MNI)
plot_stat_map(B)
plt.show()

plot_glass_brain(B)
plt.show()
#%%

