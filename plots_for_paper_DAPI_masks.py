#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:54:37 2021

@author: Markus Götz, CBS Montpellier, CNRS
"""


#%% IMPORTS

import os
import json
import glob
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astropy.table import Table
from stardist import random_label_cmap

from functions_paper import (createDir, get_sc_pwd, shuffleMatrix2,
                             get_Rg_from_PWD, fill_missing, get_mat_square)



#%% SETTINGS

plt.rcParams.update({"font.size": 20})
plt.rcParams.update({"axes.titlesize": 20})
    



#%% FUNCTIONS

def get_ROI(dictData, datasetName, numFolder, numROI):
    """
    Get pwd_sc, cellID and masks for DAPI and SND label for a single ROI.

    Parameters
    ----------
    dictData : dict
        DESCRIPTION.
    datasetName : str
        DESCRIPTION.
    numFolder : int
        Which folder from folders2Load. This is the list index.
    numROI : int
        Which ROI from the specified folder. This is the list index. Thus to
        get the first ROI in this folder (which may be "007_ROI"), use numROI=0.

    Raises
    ------
    SystemExit
        DESCRIPTION.

    Returns
    -------
    pwd_sc_ROI : numpy array, numBarcodes x numBarcodes x numNuclei
        Concatenated single cell pwd matrices. Converted to microns, shuffled.
    cellID : TYPE
        DESCRIPTION.
    mask_DAPI : TYPE
        DESCRIPTION.
    mask_SND : TYPE
        DESCRIPTION.

    """
    
    embryos2Load = dictData[datasetName]["embryos2Load"]
    
    # load json file that contains all the paths to the data
    with open(embryos2Load) as json_file:
        dictEmbryo = json.load(json_file)
    
    keyDatasetName = list(dictEmbryo.keys())[0]
    
    listDataPaths = dictEmbryo[keyDatasetName]["Folders"]
    
    # select folder as specified in function argument
    # this is the full path to the folder "buildsPWDmatrix"
    listDataPath = listDataPaths[numFolder]
    baseDir = os.path.dirname(listDataPath)
    
    # look for buildsPWDmatrix_3D_*.ecsv files (e.g. buildsPWDmatrix_3D_order:0_ROI:1.ecsv)
    searchStr = "buildsPWDmatrix_3D_order*.ecsv"
    ecsv = glob.glob(os.path.join(baseDir, "buildsPWDmatrix", searchStr))
    print("Found {} matching files for {} in\n{}".format(len(ecsv), searchStr, listDataPath))
    print(ecsv)
    
    # loop over ROIs to find their order and how many nuclei where found in each
    listOrderROI = []
    for e in ecsv:
        e_base = os.path.basename(e)
        SCdistanceTable = Table.read(e, format="ascii.ecsv")
        numNuclei = len(SCdistanceTable)  # z dimensions of SCmatrix
        strOrder = re.search(r"(?<=order:)\d+", e_base).group()
        strROI = re.search(r"(?<=ROI:)\d+", e_base).group()
        # print(strOrder, strROI)
        listOrderROI.append([int(strOrder), int(strROI), numNuclei])
    
  
    # create an array assigning order -> ROI
    arrOrderROI = np.full((len(listOrderROI),2), np.NaN)
    for i in range(len(listOrderROI)):
        order = listOrderROI[i][0]
        ROI = listOrderROI[i][1]
        numNuclei = listOrderROI[i][2]
        arrOrderROI[order,0] = ROI
        arrOrderROI[order,1] = numNuclei
    
    # load buildsPWDmatrix_3D_HiMscMatrix.npy (concatenated sc PWD maps for all ROIs in this folder)
    fn = os.path.join(baseDir, "buildsPWDmatrix", "buildsPWDmatrix_3D_HiMscMatrix.npy")
    pwd_sc_raw_folder = np.load(fn)
    
    # check that it has the same number of nuclei as the summed SCdistanceTable
    if (pwd_sc_raw_folder.shape[2] != np.sum(arrOrderROI[:,1])):
        print("Missmatch in number of nuclei")
    else:
        print("All looks good.")
    
    # get an array with the corresponding ROI for each nucleus in this folder
    listROI = []
    for i in range(len(listOrderROI)):
        listROI += [arrOrderROI[i,0]]*int(arrOrderROI[i,1])
    arrROI = np.array(listROI)
    
    # get ROI specified in function arguments, load DAPI mask, SND mask
    if (numROI > len(ecsv)-1):
        print("numROI larger then found number of ROIs for this folder.")
        raise SystemExit
    e = ecsv[numROI]
    e_base = os.path.basename(e)
    strROI = re.search(r'(?<=ROI:)\d+', e_base).group()
    strROI = ("{:03d}".format(int(strROI)))
    
    # find DAPI mask
    fn_mask_DAPI = "scan_*_DAPI_{}_ROI_converted_decon_ch00_Masks.npy".format(strROI)
    fn_mask_DAPI = os.path.join(baseDir, "segmentedObjects", fn_mask_DAPI)
    fn_mask_DAPI = glob.glob(fn_mask_DAPI)
    if (len(fn_mask_DAPI) == 1):
        fn_mask_DAPI = fn_mask_DAPI[0]
        mask_DAPI = np.load(fn_mask_DAPI)
    else:
        print("Found too many DAPI masks!")
        raise SystemExit
    
    # find SND mask
    fn_mask_SND = "scan_*_DAPI_{}_ROI_converted_decon_ch01_2d_registered_SNDmask_doc.npy".format(strROI)
    fn_mask_SND = os.path.join(baseDir, "segmentedObjects", fn_mask_SND)
    fn_mask_SND = glob.glob(fn_mask_SND)
    if (len(fn_mask_SND) == 1):
        fn_mask_SND = fn_mask_SND[0]
        mask_SND = np.load(fn_mask_SND)
        if (len(mask_SND.shape)>2):
            mask_SND = mask_SND[:,:,0] # might have 1 layer in the 3rd axis
    else:
        print("Found too many masks for segmentation SND!")
        raise SystemExit
    
    # get array with "CellID #" in the same order as the pwd_sc
    SCdistanceTable = Table.read(e, format="ascii.ecsv")
    cellID = np.array(SCdistanceTable["CellID #"])
    
    # get only sc pwd maps from specified ROI
    sel = (arrROI==int(strROI))
    pwd_sc_raw_ROI = pwd_sc_raw_folder[:,:,sel]
    
    # convert to microns and shuffle    
    convertToMicron = dictData[datasetName]["conversion_to_um"]
    try:
        shuffle = dictData[datasetName]["shuffle"]
    except KeyError:
        print("Attention: Couldn't find shuffle vector. Not going to perform any shuffling.\n")
        shuffle = np.arange(pwd_sc_raw_ROI.shape[0])
    pwd_sc_ROI = pwd_sc_raw_ROI*convertToMicron
    pwd_sc_ROI = shuffleMatrix2(pwd_sc_ROI, shuffle)
    
    
    
    return pwd_sc_ROI, cellID, mask_DAPI, mask_SND



#%% load masks and sc PWD maps for this ROI, plot

datasetName = "doc_wt_nc14_hiRes_17_3D"


numEmb, numROI = 3, 0 # these are list indices
pwd_sc_ROI, cellID_ROI, mask_DAPI, mask_SND = get_ROI(dictData, datasetName, numEmb, numROI)



np.random.seed(42)
lbl_cmap = random_label_cmap()

fig, axs = plt.subplots(nrows=1, ncols=2, dpi=72) # default 72dpi

ax = axs[0]
ax.matshow(mask_DAPI, cmap=lbl_cmap)

ax = axs[1]
ax.matshow(mask_SND)

fig.suptitle("numEmb {}, numROI {}".format(numEmb, numROI))





#%% separate masks of cells, highlight ON cells

from skimage.morphology import erosion
from skimage.morphology import square


# shrink masks so that there is background (0) between the masks
mask_DAPI_er = np.zeros(mask_DAPI.shape, dtype="int16")
np.unique(mask_DAPI)
for m in np.unique(mask_DAPI):
    if m == 0:
        continue
    mask = (mask_DAPI == m)
    mask_er = erosion(mask, square(7))
    mask_DAPI_er[mask_er] = m




# get cells that are in mask_SND
mask_DAPI_SND = np.zeros(mask_DAPI.shape, dtype="int8")

cellID_ON = np.unique(mask_DAPI * (mask_SND==1))
cellID_ON = cellID_ON[cellID_ON > 0] #remove zero

for cellID in cellID_ON:
    mask = (mask_DAPI_er == cellID)
    mask_DAPI_SND[mask] = 2


cellID_OFF = np.unique(mask_DAPI * (mask_SND==0))
cellID_OFF = cellID_OFF[cellID_OFF > 0] #remove zero

for cellID in cellID_OFF:
    mask = (mask_DAPI_er == cellID)
    mask_DAPI_SND[mask] = 1


cmap = ListedColormap([[1,1,1,1], [0.25,0.25,0.75,0.5], [1,0,0,1]]) # background, OFF, ON

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 5), dpi=300)

ax = axs
plot = ax.matshow(mask_DAPI_SND, cmap=cmap)
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, ticks=[1, 1.66])
cbar.ax.set_yticklabels(["OFF", "ON"])

ax.set_title("numEmb {}, numROI {}".format(numEmb, numROI), fontsize=12)


ax.set_xlim(100,1110)
ax.set_ylim(100,1900)

ax.set_xticks([])
ax.set_yticks([])


# save
figName = "Figure_3/Fig_3_C_left"
fn = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fn)
createDir(pathFigDir, 0o755)
fig.savefig(fn+".svg")





#%% OPTIONAL: highligh the nuclei for which a sc pwd is available

cmap = ListedColormap([[1,1,1], [0.5,0.5,0.5], [1,0,0]])


mask_DAPI_pwd = np.zeros(mask_DAPI.shape, dtype="int8")
mask_DAPI_pwd[mask_DAPI_er>0] = 1

for cellID in cellID_ROI:
    mask = (mask_DAPI_er == cellID)
    mask_DAPI_pwd[mask] = 2


fig, axs = plt.subplots(nrows=1, ncols=1, dpi=150) # default 72dpi
ax = axs
ax.matshow(mask_DAPI_pwd, cmap=cmap)
# ax.set_xlim(500,2000)
# ax.set_ylim(1000,0)

ax.set_title("numEmb {}, numROI {}".format(numEmb, numROI))




# %% colorcode cells acc to Rg (filling in NaNs)

minMiss, maxMiss = 0, 8
fillMode = "KDE"

cutoff = 1.0 # in µm
minFracNotNaN = 0.33 # half of barcodes missing -> only 1/4 of PWDs in map
minmaxRg = None # if None: don't force a min and max of the color scaling

if "_hiRes_" in datasetName:
    rangeMin = 0 # bins in pwd map to include; numpy indexing, thus rangeMax not included
    rangeMax = 18
    minmaxRg = [0.3, 0.4]
else:
    rangeMin = 13
    rangeMax = 21



# fill in missing values with population avg
pwd_sc = get_sc_pwd(dictData, datasetName)
pwd_sc_KDE = dictData[datasetName]["KDE"]

pwd_sc_ROI_fill_lin, numMissing, keep = \
    fill_missing(pwd_sc_ROI, pwd_sc_KDE, minMiss, maxMiss, fillMode)

pwd_sc_ROI_fill = get_mat_square(pwd_sc_ROI_fill_lin)



# get Rg
r_gyration = np.full(pwd_sc_ROI_fill.shape[2], np.NaN)

pwd_sc_clipped = pwd_sc_ROI_fill.copy()
pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.nan
pwd_sc_clipped = pwd_sc_clipped[rangeMin:rangeMax,rangeMin:rangeMax,:]


for i in range(pwd_sc_clipped.shape[2]):
    r_gyration[i] = get_Rg_from_PWD(pwd_sc_clipped[:,:,i], minFracNotNaN=minFracNotNaN)

print(sum(~np.isnan(r_gyration)), "out of", len(r_gyration))


cellID_ROI_keep = cellID_ROI[keep]


# normalize Rg to [0,1] for plotting
if minmaxRg is None:
    minRg = np.nanmin(r_gyration)
    maxRg = np.nanmax(r_gyration)
else:
    print("Using minmaxRg to set colorscale")
    minRg = minmaxRg[0]
    maxRg = minmaxRg[1]

sMinRg = "{:.2}".format(minRg)
sMaxRg = "{:.2}".format(maxRg)

Rg_norm = (r_gyration - minRg) / (maxRg - minRg)


# plot
imgSize = mask_DAPI.shape
mask_DAPI_Rg = np.ones((imgSize[0], imgSize[1], 3)) #rgb array

cmap = copy.copy(plt.get_cmap("coolwarm")) # copy of cmap to supress warning
# now cmap(float) with float in [0, 1] returns a color



# mask_DAPI_Rg[mask_DAPI_SND==2,:] = [1, 0, 0] # ON cells
# mask_DAPI_Rg[mask_DAPI_SND==1,:] = [0.5, 0.5, 0.5] # OFF cells
mask_DAPI_Rg[mask_DAPI_SND>0,:] = [0.5, 0.5, 0.5] # all nuclei are grey

for i, cellID in enumerate(cellID_ROI_keep):
    if np.isnan(Rg_norm[i]):
        continue
    
    mask = (mask_DAPI_er == cellID)
    
    # mask_DAPI_Rg[mask,:] = [1, 0.65, 0.0] # orange
    mask_DAPI_Rg[mask,:] = cmap(Rg_norm[i])[0:3] # cmap retruns rgba (thus including alpha)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 5), dpi=300)
ax = axs
plot = ax.matshow(mask_DAPI_Rg, cmap=cmap, vmin=0, vmax=1)
cbar = plt.colorbar(plot, ax=ax, fraction=0.046*2, pad=0.04, ticks=[0, 1])
cbar.set_label("Radius of gyration (µm)")
cbar.ax.set_yticklabels([sMinRg, sMaxRg])

ax.set_title("numEmb {}, numROI {}".format(numEmb, numROI), fontsize=12)

ax.set_xlim(100,1110)
ax.set_ylim(100,1900)

ax.set_xticks([])
ax.set_yticks([])


# save
figName = "Figure_3/Fig_3_C_right"
fn = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fn)
createDir(pathFigDir, 0o755)
fig.savefig(fn+".svg")


