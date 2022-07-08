#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:33:34 2021

@author: Markus Götz, CBS Montpellier, CNRS
"""


"""
doc data
========
pre-processing that was done:
1. pyHiM v0.5 on single ROIs (loRes data) or single embryos (hiRes data)
2. created a folder in "aggregated_analysis" and placed a folder2Load.json there,
   pointing to the analyzed embryos
3. processHiMmatrix
   processHiMmatrix.py --d3 --label doc --action all
   processHiMmatrix.py --d3 --label doc --action labeled


Bintu data
==========
downloaded the repo https://github.com/BogdanBintu/ChromatinImaging

"""

#%% IMPORTS

import os
import sys
import copy
import time
from datetime import date #timedelta, datetime
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap #, LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator #,MultipleLocator
from scipy.stats import ttest_ind, mannwhitneyu #, ks_2samp



# change cwd
cwd = os.getcwd()
script_path = os.path.dirname(__file__)

if cwd != script_path:
    print("Changing dir to", script_path)
    os.chdir(script_path)



# pyHiM (make sure pyHiM is in path)
dirs_pyHiM = ["/home/markus/git/marcnol/pyHiM/src",
             "/home/markus/git/marcnol/pyHiM/src/matrixOperations"]

for dir_pyHiM in dirs_pyHiM:
    if not dir_pyHiM in sys.path:
        sys.path.append(dir_pyHiM)
del dirs_pyHiM, dir_pyHiM

from HIMmatrixOperations import calculatesEnsemblePWDmatrix, plotDistanceHistograms, getMultiContact



from functions_paper import (createDir, load_sc_data, load_sc_data_info, 
                             load_sc_data_Bintu, load_SNDchan, get_sc_pwd,
                             fill_missing, get_mat_linear,
                             get_mat_square, get_insulation_score,
                             get_Rg_from_PWD, get_pair_corr, get_mixed_mat,
                             prepare_UMAP, run_UMAP, get_intra_inter_TAD,
                             plot_map, plot_representative_sc, plot_map_tria,
                             plot_pair_corr, plot_scatter_Rg, plot_scatter_UMAP)



#%% INIT VARIABLES AND PATHS

if "dictUMAP" not in locals():
    dictUMAP = {}

# paths
PDFpath = "./PDFs"
createDir(PDFpath, 0o755)


#%% COLORS, FONTS

# cm_map = "PiYG"
cm_map = copy.copy(plt.get_cmap("PiYG")) # copy of cmap to supress warning
cm_map.set_bad(color=[0.8, 0.8, 0.8]) # set color of NaNs and other invalid values

myColors = [
    (1.00, 0.50, 0.05, 1.0), # orange, rgb 255 128 13, hex #ff800d
    (0.09, 0.75, 0.81, 1.0), # cyan, rgb 23 191 207, hex #17bfcf
    (1.00, 0.00, 0.00, 1.0), # red
    (0.25, 0.25, 0.75, 1.0)  # blue, rgb 64, 64, 191, hex #4040bf
    ]

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "svg.fonttype": "none" # export text as text, see https://stackoverflow.com/questions/34387893
    })


#%% LOAD DATA SHORTCUT


import glob

print("WARNING: dummy load function. Replace for production!")

cwd = os.getcwd()
script_path = os.path.dirname(__file__)

if cwd != script_path:
    print("Changing dir to", script_path)
    os.chdir(script_path)

if "dictData" not in locals():
    latest = sorted(glob.glob("dictData_small_[0-9]*.npy"))[-1]
    print("Loading data from", latest)
    dictData = np.load(latest, allow_pickle=True)
    dictData = dictData[()] # saving with numpy wraps the dict in an object
else:
    print("Found a dictData. Not going to do anything.")

del cwd, script_path





#%% DEFINE AND LOAD DATA

if "dictData" not in locals():
    dictData = {}


datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D" # new pyHiM version, 3D
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["embryos2Load"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/wt_doc_loRes_20RTs_nc11nc12_perROI/folders2Load.json"
    dictData[datasetName]["shuffle"] = [0, 7, 1, 8, 2, 9, 10, 17, 11, 18, 12, 19, 13, 3, 14, 4, 15, 5, 16, 6]
    dictData[datasetName]["PWD_KDE_min"] = 0.30
    dictData[datasetName]["PWD_KDE_max"] = 0.80
    dictData[datasetName]["conversion_to_um"] = 0.105
    dictData[datasetName]["dimensions"] = 3


datasetName = "doc_wt_nc14_loRes_20_perROI_3D" # new pyHiM version, 3D
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["embryos2Load"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/wt_doc_loRes_20RTs_nc14_perROI/folders2Load.json"
    dictData[datasetName]["shuffle"] = [0, 7, 1, 8, 2, 9, 10, 17, 11, 18, 12, 19, 13, 3, 14, 4, 15, 5, 16, 6]
    dictData[datasetName]["PWD_KDE_min"] = 0.30
    dictData[datasetName]["PWD_KDE_max"] = 0.80
    dictData[datasetName]["conversion_to_um"] = 0.105
    dictData[datasetName]["dimensions"] = 3


datasetName = "doc_wt_nc11nc12_hiRes_17_3D"
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["embryos2Load"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/wt_doc_hiRes_17RTs_nc11nc12_pyHiM_v0.5/folders2Load.json"
    dictData[datasetName]["shuffle"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    dictData[datasetName]["PWD_KDE_min"] = 0.20
    dictData[datasetName]["PWD_KDE_max"] = 0.45
    dictData[datasetName]["conversion_to_um"] = 0.105
    dictData[datasetName]["dimensions"] = 3


datasetName = "doc_wt_nc14_hiRes_17_3D"
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["embryos2Load"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/wt_doc_hiRes_17RTs_nc14_pyHiM_v0.5/folders2Load.json"
    dictData[datasetName]["shuffle"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    dictData[datasetName]["PWD_KDE_min"] = 0.20
    dictData[datasetName]["PWD_KDE_max"] = 0.45
    dictData[datasetName]["conversion_to_um"] = 0.105
    dictData[datasetName]["dimensions"] = 3


# data from https://github.com/BogdanBintu/ChromatinImaging
datasetName = "HCT116_chr21-34-37Mb_untreated" #Chr21:34.6Mb-37.1Mb, 30kb res
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["datadir"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/ChromatinImaging-master/Data"
    dictData[datasetName]["conversion_to_um"] = 0.001


datasetName = "HCT116_chr21-34-37Mb_6h auxin" #Chr21:34.6Mb-37.1Mb, 30kb res
if datasetName not in dictData.keys():
    print("Adding {}".format(datasetName))
    dictData[datasetName] = {}
    dictData[datasetName]["datadir"] = "/mnt/grey/DATA/users/MarkusG/aggregated_analysis/ChromatinImaging-master/Data"
    dictData[datasetName]["conversion_to_um"] = 0.001



for datasetName in dictData.keys():
    if datasetName.startswith("doc_"):
        load_sc_data(dictData, datasetName)
    elif datasetName.startswith("HCT116_"):
        load_sc_data_Bintu(dictData, datasetName)
    else:
        load_sc_data(dictData, datasetName)





#%% LOAD SND CHANNEL INFORMATION (RNA LABEL)


datasetName = "doc_wt_nc14_hiRes_17_3D"

sc_label = "doc"; sc_action = "labeled"
label_SNDchan = load_SNDchan(dictData, datasetName, sc_label, sc_action)
if "SNDchan" not in dictData[datasetName].keys():
    dictData[datasetName]["SNDchan"] = {}
dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action] = label_SNDchan



datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

sc_label = "doc"; sc_action = "labeled"
label_SNDchan = load_SNDchan(dictData, datasetName, sc_label, sc_action)
if "SNDchan" not in dictData[datasetName].keys():
    dictData[datasetName]["SNDchan"] = {}
dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action] = label_SNDchan





#%% LOAD EXPERIMENT AND EMBRYO INFORMATION

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D",
                "doc_wt_nc11nc12_hiRes_17_3D",
                "doc_wt_nc14_hiRes_17_3D"]

for datasetName in datasetNames:
    exp_emb = load_sc_data_info(dictData, datasetName)
    dictData[datasetName]["exp_emb"] = exp_emb





#%% CALCULATE THE KDE ONCE

pixelSize = 1 # as pwd_sc is converted to µm already

datasetNames = dictData.keys()

for datasetName in datasetNames:
    if datasetName.startswith("doc_"):
        pwd_sc = get_sc_pwd(dictData, datasetName)
        cells2Plot =  [True]*pwd_sc.shape[2]
        
        print("\nCalculating KDE for {}, nNuc={}".format(datasetName, pwd_sc.shape[2]))
        time.sleep(0.5) # wait for print command to finish
        pwd_sc_KDE, _ = calculatesEnsemblePWDmatrix(pwd_sc, pixelSize, cells2Plot, mode="KDE")
        
        dictData[datasetName]["KDE"] = pwd_sc_KDE
        
    elif datasetName.startswith("HCT116_"):
        pwd_sc = get_sc_pwd(dictData, datasetName)
        print("\nCalculating median for {}, nNuc={}".format(datasetName, pwd_sc.shape[2]))
        pwd_sc_median = np.nanmedian(pwd_sc, axis=2)
        
        dictData[datasetName]["median"] = pwd_sc_median
        
    else:
        pwd_sc = get_sc_pwd(dictData, datasetName)
        cells2Plot =  [True]*pwd_sc.shape[2]
        
        print("\nCalculating KDE for {}, nNuc={}".format(datasetName, pwd_sc.shape[2]))
        time.sleep(0.5) # wait for print command to finish
        pwd_sc_KDE, _ = calculatesEnsemblePWDmatrix(pwd_sc, pixelSize, cells2Plot, mode="KDE")
        
        dictData[datasetName]["KDE"] = pwd_sc_KDE




#%% SAVE dictData to disk

fName = "dictData_small_{}.npy"
fName = fName.format(date.today().strftime("%y%m%d"))

np.save(fName, dictData) # pickles the dict

print("Saved dictData to {}".format(fName))





#%% Fig 1 B, population averages, nc11nc12 and nc14 (export to npy)

dataSets = {
    "doc_wt_nc11nc12_loRes_20_perROI_3D": {"short": "loRes_nc11nc12"},
    "doc_wt_nc14_loRes_20_perROI_3D": {"short": "loRes_nc14"},
    "doc_wt_nc11nc12_hiRes_17_3D": {"short": "hiRes_nc11nc12"},
    "doc_wt_nc14_hiRes_17_3D": {"short": "hiRes_nc14"}
    }


for datasetName in dataSets.keys():
    pwd_KDE = dictData[datasetName]["KDE"]
    
    figName = "Figure_1/Fig_1_B_" + dataSets[datasetName]["short"] + ".npy"
    fPath = os.path.join(PDFpath, figName)
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    np.save(fPath, pwd_KDE)





#%% Fig 1 C, population averages, nc11nc12 and nc14

vmin, vmax = 0.3, 0.8 # in µm
vmin_diff, vmax_diff = -0.1, 0.1 # in µm
vmin_log_r, vmax_log_r = -0.3, 0.3 # None, None


tickPos = np.linspace(vmin, vmax ,int(round((vmax-vmin)/0.1)+1))

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
pwd_sc_nc11nc12 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc11nc12 = dictData[datasetName]["KDE"]


datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
pwd_sc_nc14 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc14 = dictData[datasetName]["KDE"]


pwd_KDE_diff = pwd_KDE_nc14 - pwd_KDE_nc11nc12
pwd_KDE_log_r = np.log2(pwd_KDE_nc14 / pwd_KDE_nc11nc12)


# plot
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi

ax = axs[0,0]
plot_map(ax, pwd_KDE_nc11nc12, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc11nc12.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"


ax = axs[0,1]
plot_map(ax, pwd_KDE_nc14, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc14.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"


ax = axs[1,0]
plot_map(ax, pwd_KDE_diff, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff,
         title="KDE(nc14) - KDE(nc11nc12)",
         cbar_draw=True, cbar_label="Distance change (µm)")

ax = axs[1,1]
plot_map(ax, pwd_KDE_log_r, cmap="bwr", vmin=vmin_log_r, vmax=vmax_log_r,
         title="log2(nc14/nc11nc12)",
         cbar_draw=True, cbar_label="log2(nc14/nc11nc12)")

fig.tight_layout()


#%% Fig 1 D (top), representative sc maps, doc_wt_nc11nc12_loRes_20_perROI_3D

showAll = False # True: show all sc pwd with maxMissing, False: plot for paper

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"

# listShowSC = [1262, 1268, 2910, 3022, 3771] # version 1
listShowSC = [522, 1262, 2910, 3022, 3773] # version 2

pwd_sc = get_sc_pwd(dictData, datasetName)


if showAll:
    plot_representative_sc(datasetName, None, pwd_sc, pwd_KDE_nc11nc12, maxMissing=1,
                           showAll=True, vmin_sc=0.0, vmax_sc=1.5)
else:
    fig = plot_representative_sc(datasetName, listShowSC, pwd_sc, pwd_KDE_nc11nc12,
                                 showAll=False, vmin_sc=0.0, vmax_sc=1.5)
    
    figName = "Figure_1/Fig_1_D_top"
    fPath = os.path.join(PDFpath, figName)
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    fig.savefig(fPath+".pdf")  # matrix looks crappy with svg


#%% Fig 1 D (middle), pair correlation, doc_wt_nc11nc12_loRes_20_perROI_3D

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"


# load data
pwd_sc = get_sc_pwd(dictData, datasetName)


# parameter for pair_corr
p_pc = {}
p_pc["cutoff_dist"] = [0.0, np.inf] # in µm; [0.05, 1.5]
p_pc["inverse_PWD"] = True
p_pc["minDetections"] = 13 # 0-> all, <0 percentil, >0 ratio RTs, >1 num RTs
p_pc["step"] = 1
p_pc["minRatioSharedPWD"] = 0.5
p_pc["mode"] = "pair_corr" #L1, L2, RMSD, cos, Canberra, Canberra_mod, pair_corr
p_pc["numShuffle"] = 10


# run pair corr
pair_corr = get_pair_corr(pwd_sc, p_pc)

# plot histogram
color = myColors[0] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, yLim=4.0)


figName = "Figure_1/Fig_1_D_middle"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")

#%% find pairs with high or close to zero pair corr

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
# datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

pwd_sc_orig = get_sc_pwd(dictData, datasetName)

# get variables from dict pair_corr
sel = pair_corr["sel"]
S_data = pair_corr["S_data"]
O_data = pair_corr["O_data"]
P_data = pair_corr["P_data"]

pwd_sc_orig = pwd_sc_orig[:,:,sel]

vmin, vmax = 0.3, 1.5

mask = (S_data>0.4) & (O_data>0.7)
nice_pairs = [p for (p, m) in zip(P_data, mask) if m]
print(np.sum(mask), nice_pairs)


# plot
numPages = np.ceil(sum(mask)/8).astype("int")

for j in range(numPages):
    offset = j*8 # multiple of 8 as there is space for 8 pairs with a 4*4 subplots
    
    fig, axs = plt.subplots(nrows=4, ncols=4, dpi=72) # default 72dpi
    axs = axs.flatten()
    
    for i in range(8):
        try:
            pair = nice_pairs[i+offset]
        except:
            ax = axs[i*2]
            ax.set_axis_off()
            ax = axs[i*2+1]
            ax.set_axis_off()
            continue
        nuc1, nuc2 = pair.split(", ")
        ax = axs[i*2]
        ax.matshow(pwd_sc_orig[:,:,int(nuc1)], cmap=cm_map, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(pair, y=0.9, fontsize=8)
        ax = axs[i*2+1]
        ax.matshow(pwd_sc_orig[:,:,int(nuc2)], cmap=cm_map, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(pair, y=0.9, fontsize=8)
    
    plt.suptitle("Page {:.0f} out of {:.0f}".format(1+offset/8, np.ceil(sum(mask)/8)),
                 fontsize=8)



#%% plot pairs for "high" and zero pair corrrelation


nicest_pair = "111, 715" # for "high" pair corr (around 0.4)

# get pair corr for this pair
loc = np.where(np.array(nice_pairs)==nicest_pair)
S_nicest_pair = S_data[mask][loc]
print("Pair corr for this pair is", S_nicest_pair)

# get the position of these nuclei in the original data
nuc1 = nicest_pair.split(", ")[0]
nuc2 = nicest_pair.split(", ")[1]
nuc1_orig = np.arange(len(sel))[sel][int(nuc1)]
nuc2_orig = np.arange(len(sel))[sel][int(nuc2)]



datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
pwd_sc_orig = get_sc_pwd(dictData, datasetName)


# # cmap = "PiYG"
vmin, vmax = 0.3, 1.5

fig, axs = plt.subplots(nrows=1, ncols=2, dpi=72) # default 72dpi
ax = axs[0]
ax.matshow(pwd_sc_orig[:,:,int(nuc1_orig)], cmap=cm_map, vmin=vmin, vmax=vmax)
ax.set_xticks([]); ax.set_yticks([])
ax = axs[1]
ax.matshow(pwd_sc_orig[:,:,int(nuc2_orig)], cmap=cm_map, vmin=vmin, vmax=vmax)
ax.set_xticks([]); ax.set_yticks([])



#%% Fig 1 E (top), representative sc maps, doc_wt_nc14_loRes_20_perROI_3D

showAll = False # True: show all sc pwd with maxMissing, False: plot for paper


datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

listShowSC = [147, 1069, 5426, 6073, 10483] # version 3, 10024, 12521

pwd_sc = get_sc_pwd(dictData, datasetName)


if showAll:
    plot_representative_sc(datasetName, None, pwd_sc, pwd_KDE_nc14, maxMissing=3,
                           showAll=True, vmin_sc=0.0, vmax_sc=1.5)
else:
    fig = plot_representative_sc(datasetName, listShowSC, pwd_sc, pwd_KDE_nc14,
                           showAll=False, vmin_sc=0.0, vmax_sc=1.5)
    
    figName = "Figure_1/Fig_1_E_top"
    fPath = os.path.join(PDFpath, figName)
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    fig.savefig(fPath+".pdf")  # matrix looks crappy with svg







#%% Fig 1 E (middle), pair correlation, doc_wt_nc14_loRes_20_perROI_3D

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"


# load data
pwd_sc = get_sc_pwd(dictData, datasetName)


# parameter for pair_corr
p_pc = {}
p_pc["cutoff_dist"] = [0.0, np.inf] # in µm; [0.05, 1.5]
p_pc["inverse_PWD"] = True
p_pc["minDetections"] = 13 # 0-> all, <0 percentil, >0 ratio RTs, >1 num RTs
p_pc["step"] = 1
p_pc["minRatioSharedPWD"] = 0.5
p_pc["mode"] = "pair_corr" #L1, L2, RMSD, cos, Canberra, Canberra_mod, pair_corr
p_pc["numShuffle"] = 10


# run pair corr
pair_corr = get_pair_corr(pwd_sc, p_pc)

S_data = pair_corr["S_data"]
O_data = pair_corr["O_data"]
P_data = pair_corr["P_data"]
S_shuffle = pair_corr["S_shuffle"]
O_shuffle = pair_corr["O_shuffle"]
P_shuffle = pair_corr["P_shuffle"]
sel = pair_corr["sel"]
sSubR = pair_corr["sSubR"]


# plot histogram
color = myColors[1] #"C2"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, yLim=4.0)


# save
figName = "Figure_1/Fig_1_E_middle"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")

#%% Fig 1 G, UMAP Bintu data, run UMAP

selRange = None # None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 5), (0, 1)] # order is aux: (min, max), wt: (min, max)
steps = [1, 1] # order is aux, wt

keyDict = "Bintu"
datasetNames = ["HCT116_chr21-34-37Mb_6h auxin", "HCT116_chr21-34-37Mb_untreated"] #Chr21:34.6Mb-37.1Mb, 30kb res
listClasses = ["aux", "wt"]
sFunctEns = "median"


dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns)

# run UMAP
random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])

# plot scatter of the UMAP colorized according to treatement (auxin, untreated)
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)

# save
figName = "Figure_1/Fig_1_G"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 G, right, pwd maps Buntu

vmin, vmax = 0.2, 0.8 # in µm

tickPos = np.linspace(vmin, vmax ,int(round((vmax-vmin)/0.1)+1))

datasetName = "HCT116_chr21-34-37Mb_6h auxin"
pwd_sc_aux = get_sc_pwd(dictData, datasetName)
pwd_median_aux = dictData[datasetName]["median"]


datasetName = "HCT116_chr21-34-37Mb_untreated"
pwd_sc_wt = get_sc_pwd(dictData, datasetName)
pwd_median_wt = dictData[datasetName]["median"]


# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=300) # default (6.4, 4.8), 72dpi

ax = axs[0]
plot_map(ax, pwd_median_aux, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_aux.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"


ax = axs[1]
plot_map(ax, pwd_median_wt, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_wt.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"


fig.tight_layout()

figName = "Figure_1/Fig_1_G_right"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg

#%% Fig 1 H, UMAP loRes nc11nc12 vs nc14

selRange = None # None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 7), (0, 6)] # order is nc11nc12: (min, max), nc14: (min, max)
steps = [1, 1] # order is nc11nc12, nc14

keyDict = "loRes"
datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D", "doc_wt_nc14_loRes_20_perROI_3D"]
listClasses = ["nc11nc12", "nc14"]
sFunctEns = "KDE"


dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns)

# run UMAP
random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])

# plot scatter of the UMAP colorized according nuclear cycle
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)

# save
figName = "Figure_1/Fig_1_H"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")



#%% Fig 1 H, UMAP loRes nc11nc12 vs nc14 + "population average"

selRange = None # None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 7), (0, 6)] # order is nc11nc12: (min, max), nc14: (min, max)
steps = [1, 1] # order is nc11nc12, nc14

keyDict = "loRes"
datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D", "doc_wt_nc14_loRes_20_perROI_3D"]
listClasses = ["nc11nc12", "nc14"]
sFunctEns = "KDE"

dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns)

# run UMAP
random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])

arrays = [dictData[datasetNames[i]]["pwd_sc_raw"] for i in range(len(datasetNames))]
arrays = [np.nanmedian(arrays[i], axis=2) for i in range(len(datasetNames))]

pwd_KDE = np.stack(arrays, axis=2)

# pwd_KDE_lin = get_mat_linear(pwd_KDE[:,:,None])
pwd_KDE_lin = get_mat_linear(pwd_KDE)

trans = dictUMAP[keyDict]["trans"]
emb_KDE = trans.transform(pwd_KDE_lin)


# plot scatter of the UMAP colorized according nuclear cycle
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=False)

ax = fig.axes[0]
# ax.plot(emb_KDE[0][0], emb_KDE[0][1], "*")
ax.scatter(emb_KDE[:,0], emb_KDE[:,1], c=["r", "b"], s=30)


#%% Fig 1 H, get estimate for nc11/12 and nc14 regions from density

# run section for Fig 1H first

# import scipy.stats as st
from scipy.stats import gaussian_kde

keyDict = "loRes"

embedding = dictUMAP[keyDict]["embedding"]
classNum = dictUMAP[keyDict]["classNum"]

bw_method = 0.333 # None, "scott", "silverman", a scalar constant

cmap_scatter = ListedColormap([myColors[0], myColors[1]])

cmaps = ["Reds", "Blues"]
alphas = [1, 0.5]

xmin, xmax = np.min(embedding[:,0]), np.max(embedding[:,0])
ymin, ymax = np.min(embedding[:,1]), np.max(embedding[:,1])


# Peform the kernel density estimate
# (see https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python)
xx, yy = np.mgrid[xmin:xmax:250j, ymin:ymax:250j]
positions = np.vstack([xx.ravel(), yy.ravel()])


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi


kernel_nc11nc12 = gaussian_kde(embedding[classNum==0,:].T, bw_method=bw_method)
f_nc11nc12 = np.reshape(kernel_nc11nc12(positions).T, xx.shape)

kernel_nc14 = gaussian_kde(embedding[classNum==1,:].T, bw_method=bw_method)
f_nc14 = np.reshape(kernel_nc14(positions).T, xx.shape)

f = f_nc14.copy()

f[:] = -1
mask = (f_nc11nc12 > f_nc14)
f[mask] = 1
mask = (f_nc11nc12 < 0.02) & (f_nc14 < 0.02)
f[mask] = 0

# ax.imshow(f, cmap="bwr")
cfset = ax.contourf(xx, yy, f, cmap="bwr", alpha=0.15)
cset = ax.contour(xx, yy, f, colors="k")
ax.scatter(embedding[:,0], embedding[:,1], s=6, c=classNum,
           cmap=cmap_scatter, alpha=1.0)


#%% Fig 1 H, right, pwd maps loRes

vmin, vmax = 0.3, 0.8 # in µm

tickPos = np.linspace(vmin, vmax ,int(round((vmax-vmin)/0.1)+1))
# tickPos = [vmin, (vmin+vmax)/2, vmax]


datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
pwd_sc_nc11nc12 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc11nc12 = dictData[datasetName]["KDE"]


datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
pwd_sc_nc14 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc14 = dictData[datasetName]["KDE"]


# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=300) # default (6.4, 4.8), 72dpi

ax = axs[0]
plot_map(ax, pwd_KDE_nc11nc12, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc11nc12.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"
# print("N={}".format(pwd_sc_nc11nc12.shape[2]))


ax = axs[1]
plot_map(ax, pwd_KDE_nc14, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc14.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos) # cbar_orient="vertical"
# print("N={}".format(pwd_sc_nc14.shape[2]))


fig.tight_layout()

figName = "Figure_1/Fig_1_H_right"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg



#%% Fig Supp 1 B, image for localization of barcode

#TODO remove for production

baseDir = "/mnt/grey/DATA/users/MarkusG/rawData_2018/Experiment_45/Embryo_003_ROIs/001_ROI"
folder = "zProject"
fName = "scan_009_RT2_001_ROI_converted_decon_ch01_2d.npy"

fPath = os.path.join(baseDir, folder, fName)

img_barcode = np.load(fPath)

plt.imshow(img_barcode)

from PIL import Image
im = Image.fromarray(img_barcode)
im.save("Figure_1/Supp_Fig_1_B_embryo.tif")

#%% Fig Supp 1 C, detection efficiency loRes nc11nc12 and nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=150) # default (6.4, 4.8), 72dpi

for i, datasetName in enumerate(datasetNames):
    ax = axs[i]
    
    # load data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    numBarcodes = pwd_sc.shape[0]
    numNuclei = pwd_sc.shape[2]
    
    # detection when not all entries are NaN
    numDetections = np.sum(np.sum(np.isnan(pwd_sc), axis=1) < numBarcodes, axis=1)
    eff = numDetections/numNuclei
    
    ax.bar(np.arange(numBarcodes)+1, eff)
    
    ax.set_xlabel("Barcode")
    step = 5
    ax.set_xticks([1] + list(range(step, numBarcodes+1, step)))
    ax.set_xlim(0.5, numBarcodes+0.5)
    ax.set_ylabel("Detection efficiency")
    
    if "_nc11nc12_" in datasetName:
        ax.set_title("nc11/12")
    elif "_nc14_" in datasetName:
        ax.set_title("nc14")
    else:
        ax.set_title("Unknown.")
    

fig.tight_layout()


# save
figName = "Figure_1/Supp_Fig_1_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 1 C, barcode per cell, loRes nc11nc12 and nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

density = True


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=150) # default (6.4, 4.8), 72dpi

for i, datasetName in enumerate(datasetNames):
    ax = axs[i]
    
    # load data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    numBarcodes = pwd_sc.shape[0]
    numNuclei = pwd_sc.shape[2]
    
    # detection when not all entries are NaN; possible values: 0, 2, 3, ... (1 barcode would not give a PWD)
    bc_per_cell = np.sum(np.sum(np.isnan(pwd_sc), axis=1) < numBarcodes, axis=0)
    
    ax.hist(bc_per_cell, bins=np.arange(1.5, numBarcodes+1.5, 1),
            density=True, rwidth=0.8)
    
    ax.set_xlabel("Barcodes per cell")
    ax.set_xlim(1.5, numBarcodes+0.5)
    step = 5
    ax.set_xticks([2] + list(range(step, numBarcodes+1, step)))
    
    if density:
        ax.set_ylabel("Probability density")
        ax.set_ylim(0, 0.105)
    else:
        ax.set_ylabel("Number")
    
    if "_nc11nc12_" in datasetName:
        ax.set_title("nc11/12")
    elif "_nc14_" in datasetName:
        ax.set_title("nc14")
    else:
        ax.set_title("Unknown.")
    

fig.tight_layout()


# save
figName = "Figure_1/Supp_Fig_1_C_right"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")




#%% Fig Supp 1 D, E, plotDistanceHistograms

pixelSize = 1.0 # as PWD maps are converted to µm already, set this to 1
mode = "KDE" # hist, KDE
kernelWidth = 0.25 # in µm; 0.25 is default in pyHiM
maxDistance = 4.0 # in µm; 4.0 is default in pyHiM; removes larger distances

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"

figName = "Figure_1/Supp_Fig_1_D_" + datasetName + "_" + mode #"test"
fPath = os.path.join(PDFpath, figName)
createDir(os.path.dirname(fPath), 0o755)
logNameMD = fPath + "_log.md"

pwd_sc = get_sc_pwd(dictData, datasetName)

plotDistanceHistograms(
    pwd_sc, pixelSize, fPath, logNameMD, mode=mode, limitNplots=0,
    kernelWidth=kernelWidth, optimizeKernelWidth=False, maxDistance=maxDistance)

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

figName = "Figure_1/Supp_Fig_1_E_" + datasetName + "_" + mode #"test"
fPath = os.path.join(PDFpath, figName)
createDir(os.path.dirname(fPath), 0o755)
logNameMD = fPath + "_log.md"

pwd_sc = get_sc_pwd(dictData, datasetName)


plotDistanceHistograms(
    pwd_sc, pixelSize, fPath, logNameMD, mode=mode, limitNplots=0,
    kernelWidth=kernelWidth, optimizeKernelWidth=False, maxDistance=maxDistance)


#%% Fig Supp 1 F, difference map loRes, KDE nc14 nc11nc12

# this plot is created with Fig 1 C, see this section
vmin_diff, vmax_diff = -0.1, 0.1 # in µm

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
pwd_sc_nc11nc12 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc11nc12 = dictData[datasetName]["KDE"]


datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
pwd_sc_nc14 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc14 = dictData[datasetName]["KDE"]

pwd_KDE_diff_nc14_nc11nc12 = pwd_KDE_nc14 - pwd_KDE_nc11nc12

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi

plot_map(ax, pwd_KDE_diff_nc14_nc11nc12, cmap="bwr",
         vmin=vmin_diff, vmax=vmax_diff,
         title="KDE nc14-nc11nc12", cbar_draw=True,
         cbar_ticks=[vmin_diff, 0, vmax_diff], cbar_label="Distance change (µm)")

figName = "Figure_1/Supp_Fig_1_F"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg


#%% Fig 1 leftover, radius of gyration, nc11nc12 and nc14 (raw data)


cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[13, 21]] # for each TAD: [binStart, binEnd];
                        # numpy indexing, thus last bin won't be included

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

dictRg = {}


for datasetName in datasetNames:
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    r_gyration = np.full((pwd_sc.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
    dictRg[datasetName] = r_gyration


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []
listMeans = []


for i, datasetName in enumerate(datasetNames):
    if "_nc11nc12_" in datasetName:
        strDataset = "nc11/12"
    elif "_nc14_" in datasetName:
        strDataset = "nc14"
    else:
        strDataset = "unknown"
    
    r_gyration = dictRg[datasetName]
    
    keep = ~np.isnan(r_gyration[:, 0])
    plot = ax.violinplot(r_gyration[keep, 0], [i],
                         widths=0.5, points=100,
                         showmeans=True, showextrema=True, showmedians=False)
    plots += [plot]
    listXticks.append("{}\nn={}".format(strDataset, np.sum(keep)))
    listMeans.append(np.mean(r_gyration[keep, 0]))


txt = "R_g (raw data)\ncutoff={}µm, minFracNotNaN={}\n"
txt += "mean(nc11nc12)={:.4f}, mean(nc14)={:.4f}"
txt = txt.format(cutoff, minFracNotNaN, listMeans[0], listMeans[1])
ax.set_title(txt, fontsize=12)

ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])

ax.set_ylabel("Radius of gyration (µm)")
ax.set_ylim([0, 0.6])


colors = [myColors[0], myColors[1]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_1/Fig_1_D"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")



# statistical test on the distributions
# from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind#, cramervonmises_2samp

list_data = []
for i, datasetName in enumerate(datasetNames):
    r_gyration = dictRg[datasetName]
    keep = ~np.isnan(r_gyration[:, 0])
    list_data.append(r_gyration[keep, 0])


statistic, pvalue = ttest_ind(list_data[0], list_data[1], equal_var=False)#, permutations=1000) # needs scipy 1.7


print("statistic {}, pvalue {}".format(statistic, pvalue))


#%% Fig 2 A, meaning UMAP axes, difference map selection vs full PWD map
# (run section for Figure 1 H first)

axis_split = [0, 1] # 0 or 1 to split in x or y direction


pixelSize = 1 # in µm, as pwd_sc is in µm already, set to 1
vmin_diff, vmax_diff = -0.05, 0.05
cmap = "bwr"

c_grey = [0.6, 0.6, 0.6] # the smaller the darker
pnt_size = 1.5


# get embedding
pwd_sc_lin_cat = dictUMAP["loRes"]["pwd_sc_lin_cat"]
embedding = dictUMAP["loRes"]["embedding"]

pwd_sc_UMAP = get_mat_square(pwd_sc_lin_cat)
sel_UMAP = [True]*pwd_sc_UMAP.shape[2]
KDE_UMAP, _ = calculatesEnsemblePWDmatrix(pwd_sc_UMAP, pixelSize,
                                          sel_UMAP, mode="KDE")


# # get center of mass of embedding
q = 0.333
cutoff_lower = np.quantile(embedding, q, axis=0)
cutoff_upper = np.quantile(embedding, 1-q, axis=0)


# fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), dpi=300) # default (6.4, 4.8), 72dpi

gs_kw = dict(width_ratios=[1, 2], height_ratios=[1, 1, 1, 1])

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(5.5, 6), dpi=600,
                        constrained_layout=True, gridspec_kw=gs_kw)

for dim_UMAP in axis_split:
    # sel_below_COM = (embedding[:,dim_UMAP] < COM[dim_UMAP])
    sel_below = (embedding[:,dim_UMAP] < cutoff_lower[dim_UMAP])
    sel_above = (embedding[:,dim_UMAP] > cutoff_upper[dim_UMAP])
    
    KDE_below, _ = calculatesEnsemblePWDmatrix(pwd_sc_UMAP, pixelSize,
                                               sel_below, mode="KDE")
    
    KDE_above, _ = calculatesEnsemblePWDmatrix(pwd_sc_UMAP, pixelSize,
                                               sel_above, mode="KDE")
    
    ax = axs[dim_UMAP*2, 0] # indicate selection "below"
    ax.scatter(embedding[~sel_below,0], embedding[~sel_below,1],
               s=pnt_size, color=c_grey, alpha=1.0) # not-selected in light grey
    ax.scatter(embedding[sel_below,0], embedding[sel_below,1],
               s=pnt_size, color=[0.0,0.0,1.0], alpha=1.0)
    ax.set_title("n={}".format(np.sum(sel_below)), fontsize=12)
    ax.axis("off")
    
    ax = axs[dim_UMAP*2, 1] # ensemble map "below"
    plot_map_tria(ax, KDE_UMAP-KDE_below, cmap, vmin_diff, vmax_diff,
                  cbar_draw=True, cbar_label="Dist. diff.\n(µm)",
                  cbar_ticks=[vmin_diff, vmax_diff])
                  # cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=[vmin, vmax])
    
    
    ax = axs[dim_UMAP*2+1, 0] # indicate selection "aboe"
    ax.scatter(embedding[~sel_above,0], embedding[~sel_above,1],
               s=pnt_size, color=c_grey, alpha=1.0) # not-selected in light grey
    ax.scatter(embedding[sel_above,0], embedding[sel_above,1],
               s=pnt_size, color=[0.0,0.0,1.0], alpha=1.0)
    ax.set_title("n={}".format(np.sum(sel_above)), fontsize=12)
    ax.axis("off")
    
    ax = axs[dim_UMAP*2+1, 1] # ensemble map "above"
    plot_map_tria(ax, KDE_UMAP-KDE_above, cmap, vmin_diff, vmax_diff,
                  cbar_draw=True, cbar_label="Dist. diff.\n(µm)",
                  cbar_ticks=[vmin_diff, vmax_diff])
                  # cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=[vmin, vmax])


# fig.tight_layout()


# save
figName = "Figure_2/Fig_2_A"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")
fig.savefig(fPath+".pdf")


#%% Fig 2 B, loRes, correlation Rg TAD1 vs doc-TAD


fillMode = "KDE"
cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[ 0, 13], # TAD1
              [13, 21]] # doc-TAD


datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}

dictRg = {}


for i, datasetName in enumerate(datasetNames):
    minMiss, maxMiss = minmaxMiss[datasetName]
    
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    pwd_sc_lin, _, _ = fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
    pwd_sc_fill = get_mat_square(pwd_sc_lin)
    
    
    r_gyration = np.full((pwd_sc_fill.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc_fill.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
    dictRg[datasetName] = r_gyration


# collect parameters
param_Rg = {}
param_Rg["fillMode"] = fillMode
param_Rg["cutoff"] = cutoff
param_Rg["minFracNotNaN"] = minFracNotNaN


# get center of dist for nc11nc12
center = np.nanmean(dictRg[datasetNames[0]], axis=0)


# plot (match size with Fig 2 E)
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), dpi=150) # default (6.4, 4.8), 72dpi

ax = axs[0]
plot_scatter_Rg(ax, dictRg, datasetNames[0], param_Rg, color=myColors[0],
                cross=[center, "-", "k"])

ax = axs[1]
plot_scatter_Rg(ax, dictRg, datasetNames[1], param_Rg, color=myColors[1],
                cross=[center, "--", "k"])

fig.tight_layout()


# save
figName = "Figure_2/Fig_2_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig 2 C, UMAP colorcoded by Rg (use data filled with KDE for calculation of Rg)
# (run section for Figure 1 H first)

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minBin = 13
maxBin = 21 # numpy indexing, thus last bin won't be included

vmin, vmax = 0.20, 0.35
tick_pos=[0.20, 0.25, 0.3, 0.35]


# get UMAP settings and embedding
pwd_sc_lin_cat = dictUMAP["loRes"]["pwd_sc_lin_cat"]
classID = dictUMAP["loRes"]["classID"]
embedding = dictUMAP["loRes"]["embedding"]
classes = dictUMAP["loRes"]["classes"]
fillMode = dictUMAP["loRes"]["fillMode"]
minmaxMiss = dictUMAP["loRes"]["minmaxMiss"]
classNum = dictUMAP["loRes"]["classNum"]
n_neighbors = dictUMAP["loRes"]["p"]["n_neighbors"]
min_dist = dictUMAP["loRes"]["p"]["min_dist"]


# get square version of the PWD maps that made it into the UMAP
# NaNs filled with KDE
pwd_sc = get_mat_square(pwd_sc_lin_cat)
pwd_sc = pwd_sc[minBin:maxBin,minBin:maxBin,:]


r_gyration = np.zeros(pwd_sc.shape[2])

pwd_sc_clipped = pwd_sc.copy()
pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN

for i in range(pwd_sc_clipped.shape[2]):
    r_gyration[i] = get_Rg_from_PWD(pwd_sc_clipped[:,:,i], minFracNotNaN=minFracNotNaN)




cmap = "coolwarm" # jet, viridis, winter, cool, coolwarm, RdYlBu; to reverse, add _r

sel_nc11nc12 = (np.array(classID) == "nc11nc12")
sel_nc11nc12_Rg = np.logical_and((np.array(classID) == "nc11nc12"), ~np.isnan(r_gyration))
sel_nc14_noRg = np.logical_and((np.array(classID) == "nc14"), np.isnan(r_gyration))
sel_nc14_Rg = np.logical_and((np.array(classID) == "nc14"), ~np.isnan(r_gyration))

sel_Rg = (~np.isnan(r_gyration))


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150) # default (6.4, 4.8), 72dpi

# nuclei without Rg in light grey
plot = ax.scatter(embedding[~sel_Rg,0], embedding[~sel_Rg,1],
                  s=10, marker="x", linewidths=1, facecolors=[0.2]*3, alpha=1.0) # the smaller the darker


# nuclei with Rg according to color map
plot = ax.scatter(embedding[sel_Rg,0], embedding[sel_Rg,1],
                  s=10, c=r_gyration[sel_Rg], cmap=cmap, alpha=1.0,
                  vmin=vmin, vmax=vmax)


cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, ticks=tick_pos)
cbar.set_label("Radius of gyration (µm)")

txt = "{} (imputed with {})\n"
txt += "minmaxMiss={}, n={}\n"
txt += "umap: n_neighbors={}, min_dist={}\n"
txt += "Rg: cutoff={}µm, minFracNotNaN={} (using imputed)"
txt = txt.format(list(classes), fillMode,
                 minmaxMiss,
                 list(np.unique(classNum, return_counts=True)[1]),
                 n_neighbors, min_dist,
                 cutoff, minFracNotNaN)
ax.set_title(txt, fontsize=12)

ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")

ax.set_xticks([])
ax.set_yticks([])

fig.subplots_adjust(right=0.85) # make some space on the right for the cbar label
                                # keep the same as for Fig. 2E

# save
figName = "Figure_2/Fig_2_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig 2 D, sc insulation score nc11/nc12 vs nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

# parameters for data handling; make sure they are the same as for the UMAP
minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}
fillMode = "KDE"

invertPWD = True
sqSize = 2
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49 (zero-based)


# create figures (fig2 ~0.8*width fig1)
fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(10.0, 8.45), dpi=300) # default (6.4, 4.8), 72dpi
fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=( 8.2, 1.25), dpi=150) # default (6.4, 4.8), 72dpi


# loop over datasets
for iData, datasetName in enumerate(datasetNames):
    minMiss, maxMiss = minmaxMiss[datasetName]
    
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    
    pwd_sc_fill_lin, numMissing, keep = \
        fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
    
    pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)
    
    if invertPWD:
        pwd_sc_fill = 1/pwd_sc_fill
    
    nNuclei = pwd_sc_fill.shape[2]
    nBarcodes = pwd_sc_fill.shape[0]
    
    insulation_score = np.full((nNuclei, nBarcodes), np.NaN)
    
    
    for i in range(nNuclei):
        insulation_score[i,:] = \
            get_insulation_score(pwd_sc_fill[:,:,i], sqSize,
                                 scNormalize=scNormalize)
    
    sorting = insulation_score[:, bin_border].argsort() # add [::-1] to invert
    insulation_score_sorted = insulation_score[sorting]
    
    
    # --- Figure 1 ---
    # plot single-cell insulation score
    cmap = "coolwarm_r" # viridis_r
    if scNormalize:
        vmin, vmax = 0, 2
    else:
        vmin, vmax = 0, 14 # symmetric around 7
    
    ax = axs1[iData]
    plot = ax.matshow(insulation_score_sorted, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    if scNormalize:
        cbar.set_label("Insulation score, normalized")
    else:
        cbar.set_label("Insulation score")
    
    ax.set_xlim([-0.5+sqSize, nBarcodes-0.5-sqSize])
    # ax.set_aspect(0.04) # the larger the higher the plot
    aspect = 2.2*(nBarcodes-2*sqSize)/nNuclei
    ax.set_aspect(aspect) # the larger the higher the plot
    
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    
    ax.set_xticks(np.arange(sqSize,nBarcodes-sqSize+1,3))
    ax.set_xticklabels(np.arange(sqSize,nBarcodes-sqSize+1,3)+1)
    
    ax.set_xlabel("Barcode")
    ax.set_ylabel("Single-nucleus insulation score, cell ID (n={})".format(nNuclei))
    
    ax.set_title("{}. NaNs filled with {}".format(datasetName, fillMode),
                 fontsize=12)
    
    
    # --- Figure 2 ---
    # also get a bar graph of the insulation score profile
    sFunct = "median" # mean, median; use the same as in Fig. Supp 4 A
    
    if sFunct == "mean":
        IS_ens = np.mean(insulation_score, axis=0)
    elif sFunct == "median":
        IS_ens = np.median(insulation_score, axis=0)
    else:
        raise SystemExit("Unexpected value for variable sFunct.")
    
    
    ax = axs2[iData]
    plot = ax.bar(range(nBarcodes), IS_ens, width=0.9)
    
    ax.set_xlim([-0.5+sqSize, nBarcodes-0.5-sqSize])
    if scNormalize:
        ax.set_ylim(0.8, 1.2)
    else:
        ax.set_ylim(5.5, 8.5)
    
    ax.set_xticks([])
    
    # ax.set_xlabel("Barcode")
    ax.set_ylabel("{} IS".format(sFunct))


# save both figures
figName = "Figure_2/Fig_2_D"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig1.savefig(fPath+".svg") # matrix looks crappy with svg, but thin lines might get lost with pdf

figName = "Figure_2/Fig_2_D_top"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig2.savefig(fPath+".svg")





#%% Fig 2 E, correlation sc Rg and sc insulation score nc11/nc12 vs nc14

# for each dataset (nc11nc12, nc14)
# sc Rg (TAD1 or doc-TAD)
# sc insulation at TAD border


datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

# parameters for data handling; make sure they are the same as for the UMAP
minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}
fillMode = "KDE"

# parameters for insulation score
invertPWD = True
sqSize = 2
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49 (zero-based)

# parameters for Rg
cutoff = 1.0 # in µm
minFracNotNaN = 0.33
minmaxBins = {"TAD1":    [ 0,13], # for each TAD: [binStart, binEnd]
              "doc-TAD": [13,21]} # numpy indexing, last bin won't be included

# parameters for axis
vmin_Rg, vmax_Rg = 0.2, 0.4
vmin_IS, vmax_IS = 0, 15


# create figure (match size with Fig 2 B)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10.1, 10), dpi=150) # default (6.4, 4.8), 72dpi


# loop over datasets
for iData, datasetName in enumerate(datasetNames):
    minMiss, maxMiss = minmaxMiss[datasetName]
    
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    
    pwd_sc_fill_lin, numMissing, keep = \
        fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
    
    pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)
    
    
    # --- get inulation score ---
    pwd_sc_fill_IS = pwd_sc_fill.copy()
    pwd_sc_IS = pwd_sc[:,:,keep]
    
    if invertPWD:
        pwd_sc_fill_IS = 1/pwd_sc_fill_IS
    
    nNuclei = pwd_sc_fill_IS.shape[2]
    nBarcodes = pwd_sc_fill_IS.shape[0]
    
    insulation_score = np.full((nNuclei, nBarcodes), np.NaN)
    IS_nan = np.full((nNuclei, nBarcodes), np.NaN)
    
    for i in range(nNuclei):
        insulation_score[i,:] = get_insulation_score(pwd_sc_fill_IS[:,:,i], sqSize,
                                 scNormalize=scNormalize)
        IS_nan[i,:] = get_insulation_score(pwd_sc_IS[:,:,i], sqSize,
                                 nansum=True, scNormalize=False) # to remove cells that only have imputed values
    keep_IS = (IS_nan[:, bin_border]!=0)
    
    
    # --- get radius of gyration ---
    r_gyration = np.full((pwd_sc_fill.shape[2], len(minmaxBins)), np.NaN)
    
    # for iTAD in range(len(minmaxBins)):
    for iTAD, strTAD in enumerate(["TAD1", "doc-TAD"]):
        minBin, maxBin = minmaxBins[strTAD]
        pwd_sc_clipped = pwd_sc_fill.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
        
        keep_Rg = ~np.isnan(r_gyration[:, iTAD])
        
        
        # --- plot for each combination datasetName, TAD ---
        ax = axs[iData, iTAD]
        keep_IS_Rg = keep_IS & keep_Rg
        
        v0 = r_gyration[keep_IS_Rg, iTAD]
        v1 = insulation_score[keep_IS_Rg, bin_border]
        
        pearson_corrcoef = np.corrcoef(v0, v1)[0,1]
        print("pearson_corrcoef {}: {:.6f}".format(datasetName, pearson_corrcoef))
        
        ax.scatter(v0, v1, s=10, color=myColors[iData])
        
        ax.set_xlim(vmin_Rg, vmax_Rg)
        ax.set_ylim(vmin_IS, vmax_IS)
        ax.set_xlabel("Rg {} (µm)".format(strTAD))
        ax.set_ylabel("Insulation score")
        ax.text(0.1, 0.9, "r={:.2f}".format(pearson_corrcoef), transform=ax.transAxes)
        txt = "{}\nNaNs filled with{}\nbins={}"
        ax.set_title(txt.format(datasetName, fillMode, minmaxBins[strTAD]),
                     fontsize=12)


fig.tight_layout()


# save
figName = "Figure_2/Fig_2_E"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 2 F, UMAP colorcoded by insulation score (use data filled with KDE for calculation of IS)
# run UMAP for loRes first!

sqSize = 2 # doc: 2, bintu: 5
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49
invertPWD = True
doLog = False

cmap = "coolwarm_r" # match style in Fig 2G

if doLog:
    vmin, vmax = 0, 2
else:

    vmin, vmax = 0, 14 # match values in Fig 2G
    # if "HCT116" in datasetName:
    if scNormalize:
        vmin, vmax = 0.4, 1.6


pwd_sc_lin_cat = dictUMAP["loRes"]["pwd_sc_lin_cat"]
# classID = dictUMAP["loRes"]["classID"]
embedding = dictUMAP["loRes"]["embedding"]
classes = dictUMAP["loRes"]["classes"]
fillMode = dictUMAP["loRes"]["fillMode"]
minmaxMiss = dictUMAP["loRes"]["minmaxMiss"]
classNum = dictUMAP["loRes"]["classNum"]
n_neighbors = dictUMAP["loRes"]["p"]["n_neighbors"]
min_dist = dictUMAP["loRes"]["p"]["min_dist"]



# get square version of the PWD maps that made it into the UMAP
# NaNs filled with KDE
pwd_sc = get_mat_square(pwd_sc_lin_cat)


if invertPWD:
    pwd_sc = 1/pwd_sc

numNuc = pwd_sc.shape[2]
numBarcodes = pwd_sc.shape[0]

insulation_score = np.full((numNuc, numBarcodes), np.NaN)

for i in range(numNuc):
    insulation_score[i,:] = get_insulation_score(pwd_sc[:,:,i], sqSize, scNormalize)


if doLog:
    IS = np.log10(insulation_score[:,bin_border])
else:
    IS = insulation_score[:,bin_border]


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150) # default (6.4, 4.8), 72dpi

plot = ax.scatter(embedding[:,0], embedding[:,1],
                  s=10, c=IS, cmap=cmap, alpha=1.0, vmin=vmin, vmax=vmax)


cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
if doLog:
    IS_label = "log10(Insulation score)"
else:
    if scNormalize:
        IS_label = "Insulation score, normalized"
    else:
        IS_label = "Insulation score"

cbar.set_label(IS_label)

txt = "{} (imputed with {})\n"
txt += "minmaxMiss={}, n={}\n"
txt += "UMAP: n_neighbors={}, min_dist={}\n"
txt += "IS: sqSize={}, invertPWD={}, bin_border={}"
txt = txt.format(list(classes), fillMode,
                 minmaxMiss,
                 list(np.unique(classNum, return_counts=True)[1]),
                 n_neighbors, min_dist,
                 sqSize, invertPWD, bin_border)
ax.set_title(txt, fontsize=12)

ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")

ax.set_xticks([])
ax.set_yticks([])

fig.subplots_adjust(right=0.85) # make some space on the right for the cbar label
                                # keep the same as for Fig. 2D


# save
figName = "Figure_2/Fig_2_F"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 2 A, Bintu data, 20 barcodes centered on "strong" and "weak" border


# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])

# "strong border", minmaxMissing as small as possible
selRange = [40,60] #[15,35] #[40,60] None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 0), (0, 0)] # order is aux: (min, max), wt: (min, max)
steps = [1, 3] # order is aux, wt

keyDict = "Bintu strong border"
datasetNames = ["HCT116_chr21-34-37Mb_6h auxin", "HCT116_chr21-34-37Mb_untreated"] #Chr21:34.6Mb-37.1Mb, 30kb res
listClasses = ["aux", "wt"]
sFunctEns = "median"

dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns)

random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# plot scatter of the UMAP
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)
fig.axes[0].set_title("{}, selRange={}\n{}".format(keyDict, selRange, fig.axes[0].get_title()), fontsize=12)


# save
figName = "Figure_2/Supp_Fig_2_A_strong"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")

# "weak border", minmaxMissing as small as possible
selRange = [15,35] #[15,35] #[40,60] None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 0), (0, 0)] # order is aux: (min, max), wt: (min, max)
steps = [1, 3] # order is aux, wt

keyDict = "Bintu weak border"
datasetNames = ["HCT116_chr21-34-37Mb_6h auxin", "HCT116_chr21-34-37Mb_untreated"] #Chr21:34.6Mb-37.1Mb, 30kb res
listClasses = ["aux", "wt"]
sFunctEns = "median"

dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns)

random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# plot scatter of the UMAP
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)
fig.axes[0].set_title("{}, selRange={}\n{}".format(keyDict, selRange, fig.axes[0].get_title()), fontsize=12)

# save
figName = "Figure_2/Supp_Fig_2_A_weak"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 2 B, doc loRes, scanning UMAP hyper-parameters

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

vmin, vmax = 1.5, 3.0 # in µm^-1
vmin, vmax = 0.5, 1.1 # in µm^-1

pwd_sc_KDE = dictData[datasetName]["KDE"]

arr = np.log(1/pwd_sc_KDE[6:, 6:])
idx = np.arange(arr.shape[0])
arr[idx,idx] = vmax

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150)
plot, cbar = plot_map(ax, arr, cmap="Reds", vmin=vmin, vmax=vmax,
                      cbar_label="Distance-1 (µm-1)", cbar_ticks=[vmin, vmax])

cbar.ax.set_yticklabels(["min", "max"])

# save
figName = "Figure_2/Supp_Fig_2_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf")



#%% Fig Supp 2 C, doc loRes, scanning UMAP hyper-parameters
# (run section with UMAP for loRes data first, Fig. 2C)

n_neighbors_scan = [15, 50, 250]
min_dist_scan = [0.01, 0.1, 0.8]

keyDict_orig = "loRes"
random_state = 123456 # None, 42


# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])


# make a copy of the original UMAP
keyDict = keyDict_orig + "_scan"
dictUMAP[keyDict] = dictUMAP[keyDict_orig].copy()


# loop over all parameter combinations and plot
fig, _ = plt.subplots(nrows=len(n_neighbors_scan), ncols=len(min_dist_scan),
                      figsize=(14, 12), dpi=150) # default (6.4, 4.8), 72dpi
                      # figsize=(16, 12), dpi=150) # default (6.4, 4.8), 72dpi

for i, (n_neighbors, min_dist) in enumerate(product(n_neighbors_scan, min_dist_scan)):
    print(i, n_neighbors, min_dist)
    ax = fig.axes[i]
    
    custom_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}

    dictUMAP = run_UMAP(dictUMAP, keyDict, random_state, custom_params)
    
    plot_scatter_UMAP(dictUMAP[keyDict], cmap, ax=ax, hideTicks=True)
    txt = "n_neigh={}, min_dist={}".format(n_neighbors, min_dist)
    ax.set_title(txt, fontsize=12)
    ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.tight_layout()


# save
figName = "Figure_2/Supp_Fig_2_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig Supp 2 D, violin plot radius of gyration, raw vs filled in data

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}

fillMode = "KDE"

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[13, 21]] # for each TAD: [binStart, binEnd];
                        # numpy indexing, thus last bin won't be included

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []
listXticks_top = []


for i, (datasetName, useRaw) in enumerate(product(datasetNames, [True, False])):
    if useRaw:
        sType = "raw"
        listXticks_top.append("nc11/12" if "_nc11nc12_" in datasetName else "nc14")
    else:
        sType = "imputed"
    
    minMiss, maxMiss = minmaxMiss[datasetName]
    
    # get raw data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    if not useRaw:
        # get data with missing values filled with KDE
        pwd_sc_fill_lin, numMissing, keep = \
            fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
        pwd_sc = get_mat_square(pwd_sc_fill_lin)
    
    
    r_gyration = np.full((pwd_sc.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
  
    # plot
    keep = ~np.isnan(r_gyration[:,0])
    plot = ax.violinplot(r_gyration[keep,0], [i], widths=0.5, # width of violin
                         showmeans=True, showextrema=True, showmedians=False)
    plots += [plot]
    listXticks += ["{}\nn={}".format(sType, sum(keep))]


numPlots = len(plots)
ax.set_xlim(-0.5, numPlots-0.5)
ax.set_ylim(0, 0.6)
ax.set_xticks(np.arange(numPlots))
ax.set_xticklabels(listXticks)

# set top axis
ax_top = ax.secondary_xaxis("top")
tickPos_top = np.mean(np.arange(numPlots).reshape((len(datasetNames), 2)), axis=1)
ax_top.set_xticks(tickPos_top)
ax_top.set_xticklabels(listXticks_top)

# y axis label
ax.set_ylabel("Radius of gyration (µm)")

# fix colors
colors = [myColors[0], myColors[0], myColors[1], myColors[1]]

for i in range(numPlots):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


# save
figName = "Figure_2/Supp_Fig_2_D"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig Supp 2 E, violin plot insulation score doc loRes nc11nc12 vs nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}

fillMode = "KDE"

sqSize = 2 # doc: 2, bintu: 5
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49
invertPWD = True

cutoff_dist = 0.075 # in µm


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []
listXticks_top = []


for i, (datasetName, useRaw) in enumerate(product(datasetNames, [True, False])):
    if useRaw:
        sType = "raw"
        listXticks_top.append("nc11/12" if "_nc11nc12_" in datasetName else "nc14")
    else:
        sType = "imputed"
    
    minMiss = minmaxMiss[datasetName][0]
    maxMiss = minmaxMiss[datasetName][1]
    
    # get raw data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    if not useRaw:
        # get data with missing values filled with KDE
        pwd_sc_fill_lin, numMissing, keep = \
            fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
        pwd_sc = get_mat_square(pwd_sc_fill_lin)
    
    if invertPWD:
        pwd_sc[pwd_sc < cutoff_dist] = cutoff_dist
        pwd_sc = 1/pwd_sc
    
    numNuc = pwd_sc.shape[2]
    numBarcodes = pwd_sc.shape[0]
    
    # calculate the insulation score for the full genomic region
    insulation_score = np.full((numNuc, numBarcodes), np.NaN)
    
    for iNuc in range(numNuc):
        insulation_score[iNuc,:] = get_insulation_score(pwd_sc[:,:,iNuc], sqSize, scNormalize)
    
    # get insulation score at the border
    IS = insulation_score[:,bin_border]
    
    # plot
    keep = ~np.isnan(IS)
    plot = ax.violinplot(IS[keep], [i], widths=0.5, # width of violin
                         showmeans=True, showextrema=True, showmedians=False)
    plots += [plot]
    listXticks += ["{}\nn={}".format(sType, sum(keep))]


numPlots = len(plots)
ax.set_xlim(-0.5, numPlots-0.5)
ax.set_xticks(np.arange(numPlots))
ax.set_xticklabels(listXticks)

# set top axis
ax_top = ax.secondary_xaxis("top")
tickPos_top = np.mean(np.arange(numPlots).reshape((len(datasetNames), 2)), axis=1)
ax_top.set_xticks(tickPos_top)
ax_top.set_xticklabels(listXticks_top)

# y axis label
if scNormalize:
    ax.set_ylabel("Insulation score, normalized")
else:
    ax.set_ylabel("Insulation score")

# set title
txt = "sqSize={}, scNormalize={}, bin_border={}, invertPWD={}, cutoff_dist={}"
txt = txt.format(sqSize, scNormalize, bin_border, invertPWD, cutoff_dist)
ax.set_title(txt, fontsize=12)

# fix colors
colors = [myColors[0], myColors[0], myColors[1], myColors[1]]

for i in range(numPlots):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


# save
figName = "Figure_2/Supp_Fig_2_E"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")






#%% Fig Supp 2 E, new, insulation score profile, doc loRes nc11nc12 vs nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

minmaxMiss = {"doc_wt_nc11nc12_loRes_20_perROI_3D": [0,7],
              "doc_wt_nc14_loRes_20_perROI_3D": [0,6]}

fillMode = "KDE"

sqSize = 2 # doc: 2, bintu: 5
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49
invertPWD = True

cutoff_dist = 0.075 # in µm


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi

listXticks_top = []

colors = [myColors[0], myColors[0], myColors[1], myColors[1]]
linestyles = ["-", ":", "-", ":"]

for i, (datasetName, useRaw) in enumerate(product(datasetNames, [True, False])):
    sType = "raw" if useRaw else "imputed"
    sData = "nc11/12" if "_nc11nc12_" in datasetName else "nc14"
    
    minMiss = minmaxMiss[datasetName][0]
    maxMiss = minmaxMiss[datasetName][1]
    
    # get raw data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    if not useRaw:
        # get data with missing values filled with KDE
        pwd_sc_fill_lin, numMissing, keep = \
            fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
        pwd_sc = get_mat_square(pwd_sc_fill_lin)
    
    if invertPWD:
        pwd_sc[pwd_sc < cutoff_dist] = cutoff_dist
        pwd_sc = 1/pwd_sc
    
    numNuc = pwd_sc.shape[2]
    numBarcodes = pwd_sc.shape[0]
    
    # calculate the insulation score for the full genomic region
    insulation_score = np.full((numNuc, numBarcodes), np.NaN)
    
    for iNuc in range(numNuc):
        insulation_score[iNuc,:] = get_insulation_score(pwd_sc[:,:,iNuc], sqSize, scNormalize)
    
    # get median of insulation score profile
    IS = np.nanmedian(insulation_score, axis=0)
    
    # plot
    sLabel = "{}, {}".format(sData, sType)
    plot = ax.plot(np.arange(1, IS.shape[0]+1), IS, "o-",
                   color=colors[i], linestyle=linestyles[i], label=sLabel)

ax.legend(fontsize=12, bbox_to_anchor=(1.0, 1.0))


# set axis ticks and limits
nBarcodes = IS.shape[0]
ax.set_xticks(np.arange(sqSize, nBarcodes-sqSize, 3)+1)
ax.xaxis.set_minor_locator(AutoMinorLocator(3))
ax.set_xlim(sqSize+1, nBarcodes-sqSize)


# set axis label
ax.set_xlabel("Barcode")
if scNormalize:
    ax.set_ylabel("Insulation score, normalized")
else:
    ax.set_ylabel("Insulation score")


# set title
txt = "sqSize={}, scNormalize={}, invertPWD={}, cutoff_dist={}"
txt = txt.format(sqSize, scNormalize, invertPWD, cutoff_dist)
ax.set_title(txt, fontsize=12)


# save
figName = "Figure_2/Supp_Fig_2_E_new"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")

#%% Fig Supp 2 X, Bintu data, UMAP colorcoded acc to insulation score
# run UMAP for Bintu data first!

sqSize = 5 # doc: 2, bintu: 5
scNormalize = False
bin_border = 49 # doc: 13, bintu: 49
invertPWD = True
doLog = False

cmap = "coolwarm"

if doLog:
    vmin, vmax = 0, 2
else:

    vmin, vmax = 40, 125
    
    # if "HCT116" in datasetName:
    if scNormalize:
        vmin, vmax = 0.4, 1.6


dictKey = "Bintu"
pwd_sc_lin_cat = dictUMAP[dictKey]["pwd_sc_lin_cat"]
embedding = dictUMAP[dictKey]["embedding"]
classes = dictUMAP[dictKey]["classes"]
fillMode = dictUMAP[dictKey]["fillMode"]
minmaxMiss = dictUMAP[dictKey]["minmaxMiss"]
classNum = dictUMAP[dictKey]["classNum"]
n_neighbors = dictUMAP[dictKey]["p"]["n_neighbors"]
min_dist = dictUMAP[dictKey]["p"]["min_dist"]


# get square version of the PWD maps that made it into the UMAP
# NaNs filled with KDE
pwd_sc = get_mat_square(pwd_sc_lin_cat)


if invertPWD:
    pwd_sc = 1/pwd_sc

numNuc = pwd_sc.shape[2]
numBarcodes = pwd_sc.shape[0]

insulation_score = np.full((numNuc, numBarcodes), np.NaN)

for i in range(numNuc):
    insulation_score[i,:] = get_insulation_score(pwd_sc[:,:,i], sqSize, scNormalize)


if doLog:
    IS = np.log10(insulation_score[:,bin_border])
else:
    IS = insulation_score[:,bin_border]


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

plot = ax.scatter(embedding[:,0], embedding[:,1],
                  s=10, c=IS, cmap=cmap, alpha=1.0, vmin=vmin, vmax=vmax)


cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
if doLog:
    IS_label = "log10(Insulation score)"
else:
    if scNormalize:
        IS_label = "Insulation score, normalized"
    else:
        IS_label = "Insulation score"

cbar.set_label(IS_label)

txt = "{} (imputed with {})\n"
txt += "minmaxMiss={}, n={}\n"
txt += "UMAP: n_neighbors={}, min_dist={}\n"
txt += "IS: sqSize={}, invertPWD={}, bin_border={}"
txt = txt.format(list(classes), fillMode,
                 minmaxMiss,
                 list(np.unique(classNum, return_counts=True)[1]),
                 n_neighbors, min_dist,
                 sqSize, invertPWD, bin_border)
ax.set_title(txt, fontsize=12)

ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")

ax.set_xticks([])
ax.set_yticks([])

fig.subplots_adjust(right=0.85) # make some space on the right for the cbar label
                                # keep the same as for Fig. 2D


# save
figName = "Figure_2/Supp_Fig_2_X"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 2 Y, doc loRes, colorcode by experiment
# run section for Fig. 2C first

keyDict = "loRes"

# load embedding, select nc14 (embedding includes both nc11nc12 and nc14)
embedding = dictUMAP[keyDict]["embedding"]
classNum = dictUMAP[keyDict]["classNum"]
classNum_nc14 = np.argmax(dictUMAP[keyDict]["classes"] == "nc14")
sel_emb_nc14 = (classNum==classNum_nc14)

# load sel_UMAP_*
sel_UMAP_nc14 = dictUMAP[keyDict]["sel_UMAP_nc14"]

# get array with experiment and embryo
exp_emb = dictData["doc_wt_nc14_loRes_20_perROI_3D"]["exp_emb"]
exp_emb = np.array(exp_emb)

# colorize by experiment & embryo
classes_exp_emb, classNum_exp_emb = np.unique(exp_emb, axis=0,
                                              return_inverse=True)


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi
# nc11nc12 in gray
plot = ax.scatter(embedding[~sel_emb_nc14,0], embedding[~sel_emb_nc14,1],
                  s=10, color=[0.8, 0.8, 0.8])
plot = ax.scatter(embedding[sel_emb_nc14,0], embedding[sel_emb_nc14,1],
                  s=10, c=classNum_exp_emb[sel_UMAP_nc14], cmap="tab10")

# legend
list_legend = []
for i in range(len(classes_exp_emb)):
    exp = str(classes_exp_emb[i,0])
    emb = str(classes_exp_emb[i,1])
    list_legend.append("Exp"+exp+", Emb"+emb)
ax.legend(handles=plot.legend_elements()[0], labels=list_legend,
          loc="upper left", bbox_to_anchor=(1,1))

# axes
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")

ax.set_xticks([])
ax.set_yticks([])

# title
ax.set_title("color-coded by experiment and embryo", fontsize=12)


# save
figName = "Figure_2/Supp_Fig_2_Y"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig Supp 2 Z, doc loRes, colorcode by number of missing barcodes
# run section for Fig. 2C first

keyDict = "loRes"

# load embedding and numMissing
embedding = dictUMAP[keyDict]["embedding"]
numMissing = dictUMAP[keyDict]["numMissing_cat"]

# digitize
step = 2
bins = np.arange(0, 8+1, step) # bins = np.linspace(0, 8, 5)
digitized = np.digitize(numMissing, bins)

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi
plot = ax.scatter(embedding[:,0], embedding[:,1],
                  s=10, c=digitized, cmap="plasma", alpha=1.0)


# legend
list_labels = []
for i in range(len(bins)-1):
    list_labels.append(str(bins[i])+"-"+str(bins[i]+step-1))

ax.legend(handles=plot.legend_elements()[0], labels=list_labels,
          loc="upper left", bbox_to_anchor=(1,1))

# axes
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")

ax.set_xticks([])
ax.set_yticks([])

# title
txt = "numMissing barcodes"
txt = txt.format()
ax.set_title(txt, fontsize=12)

# save
figName = "Figure_2/Supp_Fig_2_Z"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 3 B, loRes, intra- vs inter-TAD distances, ON vs OFF

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

cutoff_dist = 1.0 # in µm
bin_border = 13
sFunct = "median" # mean or median
minNumPWD = 3


# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# get intra- and inter-TAD distances
intraTAD_dist, interTAD_dist, keep_intra, keep_inter = \
    get_intra_inter_TAD(dictData, datasetName, cutoff_dist, bin_border,
                        sFunct, minNumPWD)


# get the intra- and inter-TAD distances without low detections
intraTAD_noNaN = intraTAD_dist[keep_intra]
interTAD_noNaN = interTAD_dist[keep_inter]


# plot the distance distributions
listModes = ["intra", "inter"]
listStates = [1, 0] # 1 -> ON, 0 -> OFF

tickPos = [0, 1, 2.5, 3.5]
plots = []
listXticks = []

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

for i, (mode, state) in enumerate(product(listModes, listStates)):
    if (mode=="intra"):
        curr_dist = intraTAD_noNaN
        curr_keep = keep_intra
    else:
        curr_dist = interTAD_noNaN
        curr_keep = keep_inter
    
    sState = "ON" if state==1 else "OFF"
    
    sel_transc = (label_SNDchan[curr_keep]==state)
    plot = ax.violinplot(curr_dist[sel_transc], [tickPos[i]],
                         widths=0.5, # width of violin
                         showmeans=True, showextrema=True, showmedians=False)
    
    plots.append(plot)
    # listXticks.append("{}\n{}\nN={}".format(mode, sState, sum(sel_transc)))
    listXticks.append("{}\nn={}".format(sState, sum(sel_transc)))


ax.set_ylabel("intra- or inter-TAD distance (µm)")
ax.set_xlim(min(tickPos)-0.5, max(tickPos)+0.5)
ax.set_xticks(tickPos)
ax.set_xticklabels(listXticks)
ax.set_ylim(0, 1.1)

# set top axis
ax_top = ax.secondary_xaxis("top")
tickPos_top = np.mean(np.array(tickPos).reshape((len(listModes), len(listStates))), axis=1)
ax_top.set_xticks(tickPos_top)
ax_top.set_xticklabels(listModes)

# title
txt = "intra- vs inter-TAD distances\n"
txt += "cutoff={}µm, bin_border={}, sFunct={}, minNumPWD={}"
txt = txt.format(cutoff_dist, bin_border, sFunct, minNumPWD)
ax.set_title(txt, fontsize=12)


for i in range(len(plots)):
    color = myColors[2+i%2]
    plots[i]["bodies"][0].set_facecolor(color) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    plots[i]["bodies"][0].set_alpha(0.5)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(color)
        plots[i][partname].set_linewidth(3)


# save
figName = "Figure_3/Fig_3_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")

#%% Fig 3 C, nuclei colorcoded by radius of gyration

# see separate script plots_for_paper_DAPI_masks_210625.py


#%% Fig 3 D, top, population average maps of doc loRes 20RTs, ON vs OFF

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"
vmin, vmax = 0.3, 0.9
vmin_diff, vmax_diff = -0.2, 0.2


pwd_sc = get_sc_pwd(dictData, datasetName)
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]
pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]


cells2Plot = [True]*pwd_sc_OFF.shape[2]
pwd_KDE_OFF, _ = calculatesEnsemblePWDmatrix(pwd_sc_OFF, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1

cells2Plot = [True]*pwd_sc_ON.shape[2]
pwd_KDE_ON, _ = calculatesEnsemblePWDmatrix(pwd_sc_ON, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1


# get mixed matrix
pwd_mix = get_mixed_mat(pwd_KDE_ON, pwd_KDE_OFF)


# get difference matrix
pwd_KDE_diff_ON_OFF = pwd_KDE_ON - pwd_KDE_OFF


# plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

ax = axs[0]
txt = r"OFF\ON: nc14 KDE, n={}\n={}".format(pwd_sc_OFF.shape[2], pwd_sc_ON.shape[2])
plot_map(ax, pwd_mix, cmap=cm_map, vmin=vmin, vmax=vmax, title=txt,
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=[vmin, vmax])


ax = axs[1]
txt = "nc14 KDE ON - OFF"
plot_map(ax, pwd_KDE_diff_ON_OFF, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff,
         title=txt, cbar_draw=True, cbar_label="Distance change (µm)")


fig.tight_layout()


# save
figName = "Figure_3/Fig_3_D_top"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg


#%% Fig 3 D, bottom, UMAP, doc loRes 20RTs, nc11nc12, nc14 ON, nc14 OFF
# (run section for Figure 1 H first)

keyDict = "loRes"

# load SNDchan info for doc loRes nc14
datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]



# include ON/OFF information in classID
classID = dictUMAP[keyDict]["classID"]
sel_UMAP_nc14 = dictUMAP[keyDict]["sel_UMAP_nc14"]

sel_nc14 = (np.array(classID) == "nc14")
sel_nc14_SNDchan_ON = sel_nc14.copy()
sel_nc14_SNDchan_ON[sel_nc14] = (label_SNDchan[sel_UMAP_nc14] == 1) # nc14 and SNDchan ON
sel_nc14_SNDchan_OFF = sel_nc14.copy()
sel_nc14_SNDchan_OFF[sel_nc14] = (label_SNDchan[sel_UMAP_nc14] == 0) # nc14 and SNDchan OFF

classID_2 = np.array(classID)
classID_2[sel_nc14_SNDchan_ON] = "nc14 ON"
classID_2[sel_nc14_SNDchan_OFF] = "nc14 OFF"

classes_2, classNum_2 = np.unique(classID_2, return_inverse=True) # classes will be sorted alphanumerically!

dictUMAP[keyDict]["classID_2"] = classID_2
dictUMAP[keyDict]["classes_2"] = classes_2
dictUMAP[keyDict]["classNum_2"] = classNum_2


# get colormap
cmap = ListedColormap([
    (0.8, 0.8, 0.8, 1.0), # nc11nc12
    myColors[3],          # nc14 OFF
    myColors[2]           # nc14 ON
    ])


# plot
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)


# save
figName = "Figure_3/Fig_3_D_bottom"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 3 E, top, population average maps of doc hiRes 17RTs, ON vs OFF

datasetName = "doc_wt_nc14_hiRes_17_3D"
sc_label = "doc"; sc_action = "labeled"
vmin, vmax = 0.2, 0.7
vmin_diff, vmax_diff = -0.2, 0.2


pwd_sc = get_sc_pwd(dictData, datasetName)
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]
pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]


cells2Plot = [True]*pwd_sc_OFF.shape[2]
pwd_KDE_OFF, _ = calculatesEnsemblePWDmatrix(pwd_sc_OFF, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1

cells2Plot = [True]*pwd_sc_ON.shape[2]
pwd_KDE_ON, _ = calculatesEnsemblePWDmatrix(pwd_sc_ON, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1


# get mixed matrix
pwd_mix = get_mixed_mat(pwd_KDE_ON, pwd_KDE_OFF)


# get difference matrix
pwd_KDE_diff_ON_OFF = pwd_KDE_ON - pwd_KDE_OFF


# plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

ax = axs[0]
txt = r"OFF\ON: nc14 KDE, n={}\n={}".format(pwd_sc_OFF.shape[2], pwd_sc_ON.shape[2])
plot_map(ax, pwd_mix, cmap=cm_map, vmin=vmin, vmax=vmax, title=txt,
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=[vmin, vmax])


ax = axs[1]
txt = "nc14 KDE ON - OFF"
plot_map(ax, pwd_KDE_diff_ON_OFF, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff,
         title=txt, cbar_draw=True, cbar_label="Distance change (µm)")


fig.tight_layout()


# save
figName = "Figure_3/Fig_3_E_top"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg





#%% Fig 3 E, bottom, UMAP, doc hiRes 17RTs, nc11nc12, nc14 ON, nc14 OFF



selRange = None # None, or [bin start, bin end] for making a selection
minmaxMiss = [(0, 5), (0, 2)] # order is nc11nc12: (min, max), nc14: (min, max)
steps = [1, 1] # order is nc11nc12, nc14

keyDict = "hiRes"
datasetNames = ["doc_wt_nc11nc12_hiRes_17_3D", "doc_wt_nc14_hiRes_17_3D"]
listClasses = ["nc11nc12", "nc14"]
sFunctEns = "KDE"
listKeepPattern = [None, None]

dictUMAP = prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP,
                        keyDict, datasetNames, listClasses, sFunctEns, listKeepPattern)

# run UMAP
random_state = 123456 # None, 42
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)


# load SNDchan info for doc hiRes nc14
datasetName = "doc_wt_nc14_hiRes_17_3D"
sc_label = "doc"; sc_action = "labeled"

label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# include ON/OFF information in classID
classID = dictUMAP[keyDict]["classID"]
sel_UMAP_nc14 = dictUMAP[keyDict]["sel_UMAP_nc14"]

sel_nc14 = (np.array(classID) == "nc14")
sel_nc14_SNDchan_ON = sel_nc14.copy()
sel_nc14_SNDchan_ON[sel_nc14] = (label_SNDchan[sel_UMAP_nc14] == 1) # nc14 and SNDchan ON
sel_nc14_SNDchan_OFF = sel_nc14.copy()
sel_nc14_SNDchan_OFF[sel_nc14] = (label_SNDchan[sel_UMAP_nc14] == 0) # nc14 and SNDchan OFF

classID_2 = np.array(classID)
classID_2[sel_nc14_SNDchan_ON] = "nc14 ON"
classID_2[sel_nc14_SNDchan_OFF] = "nc14 OFF"

classes_2, classNum_2 = np.unique(classID_2, return_inverse=True) # classes will be sorted alphanumerically!

dictUMAP[keyDict]["classID_2"] = classID_2
dictUMAP[keyDict]["classes_2"] = classes_2
dictUMAP[keyDict]["classNum_2"] = classNum_2


# get colormap
cmap = ListedColormap([
    (0.8, 0.8, 0.8, 1.0), # nc11nc12
    myColors[3],          # nc14 OFF
    myColors[2]           # nc14 ON
    ])


# plot
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)


# save
figName = "Figure_3/Fig_3_E_bottom"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig 3 leftover, pair correlation, doc loRes 20RTs, ON vs ON, OFF vs OFF

# load data
datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

pwd_sc = get_sc_pwd(dictData, datasetName)
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]



# parameter for pair_corr
p_pc = {}
p_pc["cutoff_dist"] = [0.0, np.inf] # in µm; [0.05, 1.5]
p_pc["inverse_PWD"] = True
p_pc["minDetections"] = 13 # 0-> all, <0 percentil, >0 ratio RTs, >1 num RTs
p_pc["step"] = 1
p_pc["minRatioSharedPWD"] = 0.5
p_pc["mode"] = "pair_corr" #L1, L2, RMSD, cos, Canberra, Canberra_mod, pair_corr
p_pc["numShuffle"] = 10

bins = np.linspace(-1,1,31) # None, np.linspace(-1,1,51)



# run pair corr, loRes, all
pair_corr_all = get_pair_corr(pwd_sc, p_pc)


# run pair corr loRes, ON cells
pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]
pair_corr = get_pair_corr(pwd_sc_ON, p_pc)


# plot histogram
color = myColors[2] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, bins=bins,
                     pair_corr_all=pair_corr_all, yLim=4.0, figsize=(6, 4))


# compare distributions, get p-vale; also compare data vs shuffle
# statistic, pvalue = ks_2samp(pair_corr_all["S_data"], pair_corr["S_data"])
statistic, pvalue = mannwhitneyu(pair_corr_all["S_data"], pair_corr["S_data"])
print("mannwhitneyu loRes: all, ON: statistic, pvalue", statistic, pvalue)


# save figure
figName = "Figure_3/Fig_3_D_loRes_ON"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")



# run pair corr loRes, OFF cells
pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]
pair_corr = get_pair_corr(pwd_sc_OFF, p_pc)


# plot histogram
color = myColors[3] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, bins=bins,
                     pair_corr_all=pair_corr_all, yLim=4.0, figsize=(6, 4))


# compare distributions, get p-vale; also compare data vs shuffle

statistic, pvalue = mannwhitneyu(pair_corr_all["S_data"], pair_corr["S_data"])
print("mannwhitneyu loRes: all, OFF: statistic, pvalue", statistic, pvalue)

# save
figName = "Figure_3/Fig_3_E_loRes_OFF"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig 3 leftover, pair correlation, doc hiRes 17RTs, ON vs ON, OFF vs OFF

# load data
datasetName = "doc_wt_nc14_hiRes_17_3D"
sc_label = "doc"; sc_action = "labeled"

pwd_sc = get_sc_pwd(dictData, datasetName)
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]



# parameter for pair_corr
p_pc = {}
p_pc["cutoff_dist"] = [0.0, np.inf] # in µm; [0.05, 1.5]
p_pc["inverse_PWD"] = True
p_pc["minDetections"] = 11 # 0-> all, <0 percentil, >0 ratio RTs, >1 num RTs
p_pc["minRatioSharedPWD"] = 0.5
p_pc["mode"] = "pair_corr" #L1, L2, RMSD, cos, Canberra, Canberra_mod, pair_corr
p_pc["numShuffle"] = 10


# run pair corr, hiRes, all
p_pc["step"] = 3
pair_corr_all = get_pair_corr(pwd_sc, p_pc, doShuffle=False)


# run pair corr, hiRes, ON cells
p_pc["step"] = 1
pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]
pair_corr = get_pair_corr(pwd_sc_ON, p_pc)


# plot histogram
color = myColors[2] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName,
                     pair_corr_all=pair_corr_all, yLim=3.0, figsize=(6, 4))


# compare distributions, get p-vale; also compare data vs shuffle
statistic, pvalue = mannwhitneyu(pair_corr_all["S_data"], pair_corr["S_data"])
print("mannwhitneyu hiRes: all, ON: statistic, pvalue", statistic, pvalue)


# save
figName = "Figure_3/Fig_3_D_hiRes_ON"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")



# run pair corr, hiRes, OFF cells
p_pc["step"] = 3
pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]
pair_corr = get_pair_corr(pwd_sc_OFF, p_pc)


# plot histogram
color = myColors[3] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName,
                     pair_corr_all=pair_corr_all, yLim=3.0, figsize=(6, 4))


# compare distributions, get p-vale; also compare data vs shuffle
statistic, pvalue = mannwhitneyu(pair_corr_all["S_data"], pair_corr["S_data"])
print("mannwhitneyu hiRes: all, OFF: statistic, pvalue", statistic, pvalue)


# save
figName = "Figure_3/Fig_3_E_hiRes_OFF"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig Supp 3 A, difference map ON vs OFF, loRes and hiRes

# these plots are included in the sections for Fig 3 D, E

#%% Fig Supp 3 B, radius of gyration doc loRes, nc14 ON vs OFF (raw data)

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

# minmaxBins = [[13, 21]] # for each TAD: [binStart, binEnd];
#                         # numpy indexing, thus last bin won't be included
minmaxBins = [[13, 21],
              [ 0, 13]]
listTADs = ["doc TAD", "TAD1"]



# load data
datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
pwd_sc = get_sc_pwd(dictData, datasetName)

# load SNDchan info for doc loRes nc14
sc_label = "doc"; sc_action = "labeled"
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


listRg_data = []
listRg_labels = []

listRg_data.append(pwd_sc[:,:,(label_SNDchan==1)])
listRg_labels.append("ON")

listRg_data.append(pwd_sc[:,:,(label_SNDchan==0)])
listRg_labels.append("OFF")




dictRg = {}


for i, label in enumerate(listRg_labels):
    # pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc = listRg_data[i]
    
    r_gyration = np.full((pwd_sc.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
    dictRg[label] = r_gyration





fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []


selTAD = 0 # to get doc TAD
for i, label in enumerate(listRg_labels):
    r_gyration = dictRg[label][:, selTAD]
    
    keep = ~np.isnan(r_gyration)
    plot = ax.violinplot(r_gyration[keep], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel("Radius of gyration (µm)")
    plots += [plot]
    listXticks.append("{}\nn={}".format(label, np.sum(keep)))
    
    txt = "{}: mean +/- std : {} +/- {}"
    txt = txt.format(label, np.nanmean(r_gyration), np.nanstd(r_gyration))
    print(txt)


txt = "R_g (raw data), {}\ncutoff={}µm, minFracNotNaN={}"
ax.set_title(txt.format(listTADs[selTAD], cutoff, minFracNotNaN), fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, 0.6])


colors = [myColors[2], myColors[3]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_3/Supp_Fig_3_B_loRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # fig.savefig(fPath+".pdf")




# TAD1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []


selTAD = 1 # to select TAD1
for i, label in enumerate(listRg_labels):
    r_gyration = dictRg[label][:,selTAD]
    
    keep = ~np.isnan(r_gyration)
    plot = ax.violinplot(r_gyration[keep], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel("Radius of gyration (µm)")
    plots += [plot]
    listXticks.append("{}\nn={}".format(label, np.sum(keep)))
    
    txt = "{}: mean +/- std : {} +/- {}"
    txt = txt.format(label, np.nanmean(r_gyration), np.nanstd(r_gyration))
    print(txt)


txt = "R_g (raw data), {}\ncutoff={}µm, minFracNotNaN={}"
ax.set_title(txt.format(listTADs[selTAD], cutoff, minFracNotNaN), fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, 0.6])


colors = [myColors[2], myColors[3]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)

# significance difference in distributions
list_data = []
selTAD = 1 # doc TAD
for i, label in enumerate(["ON", "OFF"]):
    r_gyration = dictRg[label][:,selTAD]
    keep = ~np.isnan(r_gyration)
    list_data.append(r_gyration[keep])


statistic, pvalue = mannwhitneyu(list_data[0], list_data[1])

print("statistic {}, pvalue {}".format(statistic, pvalue))




#%% Fig Supp 3 C, radius of gyration doc hiRes, nc14 ON vs OFF

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[0, 18]] # for each TAD: [binStart, binEnd];
                        # numpy indexing, thus last bin won't be included


# load data
datasetName = "doc_wt_nc14_hiRes_17_3D"
pwd_sc = get_sc_pwd(dictData, datasetName)

# load SNDchan info for doc loRes nc14
sc_label = "doc"; sc_action = "labeled"
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


listRg_data = []
listRg_labels = []

listRg_data.append(pwd_sc[:,:,(label_SNDchan==1)])
listRg_labels.append("ON")

listRg_data.append(pwd_sc[:,:,(label_SNDchan==0)])
listRg_labels.append("OFF")




dictRg = {}


for i, label in enumerate(listRg_labels):
    # pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc = listRg_data[i]
    
    r_gyration = np.full((pwd_sc.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
    dictRg[label] = r_gyration
    
    txt = "{}: mean +/- std : {} +/- {}"
    txt = txt.format(label, np.nanmean(r_gyration), np.nanstd(r_gyration))
    print(txt)





fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []


# for i, datasetName in enumerate(datasetNames):
for i, label in enumerate(listRg_labels):
    r_gyration = dictRg[label]
    
    keep = ~np.isnan(r_gyration[:, 0])
    plot = ax.violinplot(r_gyration[keep, 0], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel("Radius of gyration (µm)")
    plots += [plot]
    listXticks.append("{}\nn={}".format(label, np.sum(keep)))


# ax = plt.gca()
txt = "R_g (hiRes raw data)\ncutoff={}µm, minFracNotNaN={}"
ax.set_title(txt.format(cutoff, minFracNotNaN), fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, 0.6])


colors = [myColors[2], myColors[3]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_3/Supp_Fig_3_C_hiRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # fig.savefig(fPath+".pdf")






 #%% Fig Supp 3 X, population average maps of doc hiRes 17RTs, nc11nc12 and nc14


vmin, vmax = 0.2, 0.7

datasetName = "doc_wt_nc11nc12_hiRes_17_3D"
pwd_sc_nc11nc12 = get_sc_pwd(dictData, datasetName)


datasetName = "doc_wt_nc14_hiRes_17_3D"
pwd_sc_nc14 = get_sc_pwd(dictData, datasetName)



cells2Plot = [True]*pwd_sc_nc11nc12.shape[2]
pwd_KDE_nc11nc12, _ = calculatesEnsemblePWDmatrix(pwd_sc_nc11nc12, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1

cells2Plot = [True]*pwd_sc_nc14.shape[2]
pwd_KDE_nc14, _ = calculatesEnsemblePWDmatrix(pwd_sc_nc14, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1


pwd_KDE_diff = pwd_KDE_nc14 - pwd_KDE_nc11nc12



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

ax = axs[0,0]
plot = ax.matshow(pwd_KDE_nc11nc12, cmap="PiYG", vmin=vmin, vmax=vmax) # coolwarm_r, PiYG
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Distance (µm)")
ax.set_title("hiRes KDE nc11nc12\nn={}".format(pwd_sc_nc11nc12.shape[2]))
ax.set_xticks([]); ax.set_yticks([])

ax = axs[0,1]
plot = ax.matshow(pwd_KDE_nc14, cmap="PiYG", vmin=vmin, vmax=vmax) # coolwarm_r, PiYG
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Distance (µm)")
ax.set_title("hiRes KDE nc14\nn={}".format(pwd_sc_nc14.shape[2]))
ax.set_xticks([]); ax.set_yticks([])

ax = axs[1,0]
plot = ax.matshow(pwd_KDE_diff, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff) # coolwarm, bwr
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Distance change (µm)")
ax.set_title("KDE nc14 - nc11nc12")
ax.set_xticks([]); ax.set_yticks([])

ax = axs[1,1]
ax.set_axis_off()


fig.tight_layout()


# save
figName = "Figure_3/Supp_Fig_3_X_hiRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg





#%% Fig Supp 3 Y, radius of gyration doc hiRes, nc11nc12 vs nc14

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[0, 18]] # for each TAD: [binStart, binEnd];
                        # numpy indexing, thus last bin won't be included



datasetNames = ["doc_wt_nc11nc12_hiRes_17_3D",
                "doc_wt_nc14_hiRes_17_3D"]

dictRg = {}


for datasetName in datasetNames:
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    r_gyration = np.full((pwd_sc.shape[2], len(minmaxBins)), np.NaN)
    
    for iTAD in range(len(minmaxBins)):
        minBin = minmaxBins[iTAD][0]
        maxBin = minmaxBins[iTAD][1]
        pwd_sc_clipped = pwd_sc.copy()
        pwd_sc_clipped = pwd_sc_clipped[minBin:maxBin,minBin:maxBin,:]
        pwd_sc_clipped[pwd_sc_clipped>cutoff] = np.NaN
        
        for iNuc in range(pwd_sc_clipped.shape[2]):
            r_gyration[iNuc, iTAD] = get_Rg_from_PWD(pwd_sc_clipped[:,:,iNuc],
                                                     minFracNotNaN=minFracNotNaN)
    
    dictRg[datasetName] = r_gyration





fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []


for i, datasetName in enumerate(datasetNames):
    if "_nc11nc12_" in datasetName:
        strDataset = "nc11/12"
    elif "_nc14_" in datasetName:
        strDataset = "nc14"
    else:
        strDataset = "unknown"
    
    r_gyration = dictRg[datasetName]
    
    keep = ~np.isnan(r_gyration[:, 0])
    plot = ax.violinplot(r_gyration[keep, 0], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel("Radius of gyration (µm)")
    plots += [plot]
    listXticks.append("{}\nn={}".format(strDataset, np.sum(keep)))


# ax = plt.gca()
txt = "R_g (hiRes raw data)\ncutoff={}µm, minFracNotNaN={}"
ax.set_title(txt.format(cutoff, minFracNotNaN))
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, 0.6])

colors = [myColors[0], myColors[1]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_3/Supp_Fig_3_Y_hiRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # fig.savefig(fPath+".pdf")





#%% Fig 4 A, sc insulation score next to transcription

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
minMiss, maxMiss = 0, 6
fillMode = "KDE"

invertPWD = True
sqSize = 2
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49 (zero-based)

sc_label = "doc"; sc_action = "labeled"


pwd_sc = get_sc_pwd(dictData, datasetName)
pwd_sc_KDE = dictData[datasetName]["KDE"]


pwd_sc_fill_lin, numMissing, keep = \
    fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)

pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)

if invertPWD:
    pwd_sc_fill = 1/pwd_sc_fill

nNuclei = pwd_sc_fill.shape[2]
nBarcodes = pwd_sc_fill.shape[0]

insulation_score = np.full((nNuclei, nBarcodes), np.NaN)


for i in range(nNuclei):
    insulation_score[i,:] = \
        get_insulation_score(pwd_sc_fill[:,:,i], sqSize, scNormalize=scNormalize)

sorting = insulation_score[:, bin_border].argsort() # add [::-1] to invert
insulation_score_sorted = insulation_score[sorting]



# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]
label_SNDchan_sorted = label_SNDchan[keep,None]
label_SNDchan_sorted = label_SNDchan_sorted[sorting]



fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 16), dpi=300) # default (6.4, 4.8), 72dpi


# plot single-cell insulation score
cmap = "coolwarm_r" # viridis_r
if scNormalize:
    vmin, vmax = 0, 2
else:
    vmin, vmax = 0, 14 # symmetric around 7

ax = axs[0]
plot = ax.matshow(insulation_score_sorted, cmap=cmap, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
if scNormalize:
    cbar.set_label("Insulation score, normalized")
else:
    cbar.set_label("Insulation score")

ax.set_xlim([-0.5+sqSize, nBarcodes-0.5-sqSize])
ax.set_aspect(0.04) # the larger the higher the plot

ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

ax.set_xticks(np.arange(sqSize,nBarcodes-sqSize+1,3))
ax.set_xticklabels(np.arange(sqSize,nBarcodes-sqSize+1,3)+1)

ax.set_xlabel("Barcode")
ax.set_ylabel("Single-nucleus insulation score, cell ID (n={})".format(nNuclei))

ax.set_title("NaNs filled with {}".format(fillMode))




# plot transcriptional state
cmap = ListedColormap([[1,1,1], [1,0,0]])

ax = axs[1]
plot = ax.matshow(label_SNDchan_sorted, cmap=cmap)
cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
cbar.set_label("Transcription")
cbar.set_ticklabels(["off", "on"])

ax.set_aspect(0.04/16) # the larger the higher the plot

ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

ax.set_xlabel("")
ax.set_ylabel("")


# fig.suptitle("NaNs filled with {}".format(fillMode))
# fig.tight_layout()


figName = "Figure_4/Fig_4_A"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # matrix looks crappy with svg, but thin lines might get lost with pdf





# also get a bar graph of the insulation score profile
sFunct = "median" # mean, median; use the same as in Fig. Supp 4 A

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.11, 2), dpi=150) # default (6.4, 4.8), 72dpi
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.13, 2), dpi=150) # default (6.4, 4.8), 72dpi

if sFunct == "mean":
    IS_ens = np.mean(insulation_score, axis=0)
elif sFunct == "median":
    IS_ens = np.median(insulation_score, axis=0)
else:
    raise SystemExit("Unexpected value for variable sFunct.")


plot = ax.bar(range(nBarcodes), IS_ens, width=0.9)

ax.set_xlim([-0.5+sqSize, nBarcodes-0.5-sqSize])
if scNormalize:
    ax.set_ylim(0.8, 1.2)
else:
    ax.set_ylim(5.5, 8.5)

ax.set_xticks(np.arange(sqSize,nBarcodes-sqSize+1,3))
ax.set_xticklabels(np.arange(sqSize,nBarcodes-sqSize+1,3)+1)

ax.set_xlabel("Barcode")
ax.set_ylabel("{} IS".format(sFunct))

figName = "Figure_4/Fig_4_A_top"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # matrix looks crappy with svg






#%% Fig 4 B, insulation score profile, doc loRes, ON vs OFF (NaNs filled with KDE)

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

minMiss, maxMiss = 0, 6
fillMode = "KDE"

invertPWD = True
sqSize = 2
scNormalize = False
sFunct = "nanmedian" # use the same as in Fig 4 A top
bin_border = 13 # doc: 13, bintu: 49 (zero-based)


# settings for bootstrap
nBootStraps = 2000
seed = 42
rng = np.random.default_rng(seed)

# get data and SNDchan
pwd_sc = get_sc_pwd(dictData, datasetName)
pwd_sc_KDE = dictData[datasetName]["KDE"]

label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# prepare pwd_sc and calculate insulation score
pwd_sc_fill_lin, numMissing, keep = \
    fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)

pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)

if invertPWD:
    pwd_sc_fill = 1/pwd_sc_fill

nNuclei = pwd_sc_fill.shape[2]
nBarcodes = pwd_sc_fill.shape[0]

insulation_score = np.full((nNuclei, nBarcodes), np.NaN)


for i in range(nNuclei):
    insulation_score[i,:] = \
        get_insulation_score(pwd_sc_fill[:,:,i], sqSize, scNormalize=scNormalize)


# collect insulation scores
dictIS = {}
dictIS_labels = [("ON", 1), ("OFF", 0)]

# bootstrap an error for the insulation score
for k, v in dictIS_labels:
    currIS = insulation_score[(label_SNDchan[keep]==v),:]
    
    if sFunct == "nanmean":
        currIS_ens = np.nanmean(currIS, axis=0)
        currIS_bs = np.full((nBootStraps, nBarcodes), np.NaN)
        for i in range(nBootStraps):
            currIS_sample = rng.choice(currIS, currIS.shape[0], replace=True, axis=0)
            currIS_bs[i,:] = np.nanmean(currIS_sample, axis=0)
        currIS_bs_mean = np.nanmean(currIS_bs, axis=0)
        currIS_bs_std = np.nanstd(currIS_bs, axis=0)
    elif sFunct == "nanmedian":
        currIS_ens = np.nanmedian(currIS, axis=0)
        currIS_bs = np.full((nBootStraps, nBarcodes), np.NaN)
        for i in range(nBootStraps):
            currIS_sample = rng.choice(currIS, currIS.shape[0], replace=True, axis=0)
            currIS_bs[i,:] = np.nanmedian(currIS_sample, axis=0)
        currIS_bs_mean = np.nanmean(currIS_bs, axis=0)
        currIS_bs_std = np.nanstd(currIS_bs, axis=0)
    else:
        raise SystemExit("Unexpected value for variable sFunct.")
    
    dictIS[k] = {}
    dictIS[k]["ens"] = currIS_ens
    dictIS[k]["bs_mean"] = currIS_bs_mean
    dictIS[k]["bs_std"] = currIS_bs_std


# plot insulation score
colors = [myColors[2], myColors[3]] # red, blue; order is ON, OFF

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

for i, (k, v) in enumerate(dictIS_labels):
    y = dictIS[k]["ens"]
    err = dictIS[k]["bs_std"]
    ax.fill_between(np.arange(nBarcodes), y-err, y+err,
                    alpha=0.3, edgecolor=None, facecolor=colors[i])

    ax.plot(np.arange(nBarcodes), dictIS[k]["ens"], "-o", color=colors[i],
            linewidth=2, label=k)


# highlight TAD border
yLim = ax.get_ylim() # get y limits of axis
ax.fill_between([bin_border-0.5, bin_border+0.5], yLim[0], yLim[1],
                    edgecolor=None, facecolor=[0.8,0.8,0.8], zorder=-1)


# set axis ticks and limits
ax.set_xticks(np.arange(sqSize, nBarcodes-sqSize, 3))
ax.set_xticklabels(np.arange(sqSize, nBarcodes-sqSize, 3)+1)
ax.xaxis.set_minor_locator(AutoMinorLocator(3))

ax.set_xlim(sqSize, nBarcodes-sqSize-1)
if scNormalize:
    ax.set_ylim(0.7, 1.3)
else:
    ax.set_ylim(yLim)


# set axis labels
ax.set_xlabel("Barcode")
if scNormalize:
    ax.set_ylabel("Insulation score, normalized")
else:
    ax.set_ylabel("Insulation score")


# set title and legend
txt = "IS profile (NaN filled with {})\ninvertPWD={}, sqSize={}, scNormalize={}, sFunct={}"
ax.set_title(txt.format(fillMode, invertPWD, sqSize, scNormalize, sFunct), fontsize=12)

ax.legend(loc="upper left")


# save
figName = "Figure_4/Fig_4_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # fig.savefig(fPath+".pdf")

#%% Fig 4 C, demixing score

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

cutoff_dist = 1.0 # in µm
bin_border = 13
sFunct = "median" # mean or median
minNumPWD = 3


# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# get intra- and inter-TAD distances
intraTAD_dist, interTAD_dist, keep_intra, keep_inter = \
    get_intra_inter_TAD(dictData, datasetName, cutoff_dist, bin_border,
                        sFunct, minNumPWD)


# calculate the log2 ratio of intra- and inter-TAD distances
keep_ratio = keep_intra * keep_inter
ration_log = np.log2(intraTAD_dist / interTAD_dist)
ration_log_noNaN = ration_log[keep_ratio]


# plot
plots = []
listXticks = []


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plot = ax.violinplot(ration_log_noNaN[label_SNDchan[keep_ratio]==1], [0],
                     widths=0.5, # width of violin
                     showmeans=True, showextrema=True, showmedians=False)
plots.append(plot)


plot = ax.violinplot(ration_log_noNaN[label_SNDchan[keep_ratio]==0], [1],
                     widths=0.5, # width of violin
                     showmeans=True, showextrema=True, showmedians=False)
plots.append(plot)


ax.set_ylabel("log2(intra- to inter-TAD distance)")
listXticks += ["ON\nn={}".format(sum(label_SNDchan[keep_ratio]==1))]
listXticks += ["OFF\nn={}".format(sum(label_SNDchan[keep_ratio]==0))]
ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0,1])
ax.set_xticklabels(listXticks)
ax.set_ylim(-4.0, 2.0)
txt = "demixing score\n"
txt += "cutoff={}µm, bin_border={}, sFunct={}, minNumPWD={}"
txt = txt.format(cutoff_dist, bin_border, sFunct, minNumPWD)
ax.set_title(txt, fontsize=12)

for i in range(len(plots)):
    color = myColors[i+2]
    plots[i]["bodies"][0].set_facecolor(color) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    plots[i]["bodies"][0].set_alpha(0.5)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(color)
        plots[i][partname].set_linewidth(3)


# ax.grid()


# save
figName = "Figure_4/Fig_4_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")





#%% Fig 4 D, inter-TAD contacts

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

distance_threshold = 0.25 # in µm

sFunct = "sum" # mean or median
bin_border = 13
minNumPWD = 0
showAsLog = False


# get concatenated sc PWD maps
pwd_sc = get_sc_pwd(dictData, datasetName)
nNuclei = pwd_sc.shape[2]


# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# keep only nuclei with at least minNumPWD in inter-TAD region
pwd_sc_interTAD = pwd_sc[0:bin_border, bin_border:, :]
pwd_sc_interTAD = pwd_sc_interTAD.transpose(2,0,1).reshape(nNuclei,-1)

keep = np.sum(~np.isnan(pwd_sc_interTAD), axis=1) >= minNumPWD


# convert distance to contact and count contacts
contact_sc_interTAD_lin = (pwd_sc_interTAD[keep,:] < distance_threshold)
interTAD_sum = np.sum(contact_sc_interTAD_lin, axis=1)


# plot
bins = np.arange(-0.5,20.5)
histtype = "bar" #bar, step

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi

plot = ax.hist(interTAD_sum[label_SNDchan[keep]==1], bins=bins, density=True,
               histtype=histtype, color=myColors[2], alpha=0.5, label="ON", zorder=2)

plot = ax.hist(interTAD_sum[label_SNDchan[keep]==0], bins=bins, density=True,
               histtype=histtype, color=myColors[3], alpha=0.5, label="OFF", zorder=1)


ax.legend(loc="center right")

ax.set_xlabel("Number of inter-TAD contacts")
ax.set_ylabel("Probability")
if showAsLog:
    ax.set_yscale("log")

xLim = 20
ax.set_xticks(range(0, xLim+1, 2))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(AutoMinorLocator(2)) # same as before: a single minor tick between major ticks.
ax.set_xlim(-0.5, xLim+0.5) # set axis limits after ticks

n_ON = np.sum(label_SNDchan[keep]==1)
n_OFF = np.sum(label_SNDchan[keep]==0)

txt = "Sum of sc interTAD contacts\ndistance_threshold={}, n_ON={}, n_OFF={}"
txt = txt.format(distance_threshold, n_ON, n_OFF)
ax.set_title(txt, fontsize=12)


# save
figName = "Figure_4/Fig_4_D"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 4 E, higher-order contacts, loRes, nc14 ON vs OFF

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

distance_threshold = 0.25 # in µm


sInDict = {"ON":1, "OFF":0}
sOut = "Probability" # Counts, Probability


# get concatenated sc PWD maps
pwd_sc = get_sc_pwd(dictData, datasetName)

# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


nBarcodes = pwd_sc.shape[0]
r_bc = range(nBarcodes)

contact_3way_sc = np.zeros((nBarcodes,nBarcodes,len(sInDict)))

for i, sIn in enumerate(sInDict):
    pwd_sc_tmp = pwd_sc[:,:,(label_SNDchan==sInDict[sIn])]
    
    for anchor, bait1, bait2 in product(r_bc, r_bc, r_bc):
        if (anchor==bait1) or (anchor==bait2) or (bait1==bait2):
            continue
        
        n_contacts, n_nonNaN = getMultiContact(pwd_sc_tmp, anchor, bait1, bait2, distance_threshold)
        
        if sOut == "Counts":
            contact_3way_sc[bait1, bait2, i] += n_contacts
        elif sOut == "Probability":
            contact_3way_sc[bait1, bait2, i] += n_contacts / n_nonNaN
        else:
            raise SystemExit("Unexpected value for variable sOut.")


# plot, separate
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=150) # default (6.4, 4.8), 72dpi
            
if (sOut == "Probability"):
    vmin, vmax = 0.0, 0.25
else:
    vmin, vmax = None, None

for i, sIn in enumerate(sInDict):
    ax = axs[i]
    txt = "{}, anchor={}, n={}".format(sIn, "sum all", np.sum(label_SNDchan==sInDict[sIn]))
    plot_map(ax, contact_3way_sc[:,:,i], title=txt, cbar_label=sOut, vmin=vmin, vmax=vmax)



# save
figName = "Figure_4/Fig_4_E"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg





#%% Fig Supp 4 B, insulation score at border, doc loRes, ON vs OFF (NaNs filled with KDE)

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"

minMiss, maxMiss = 0, 6
fillMode = "KDE"

invertPWD = True
sqSize = 2
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49 (zero-based)
cutoff_dist = 0.075


# get data and SNDchan
pwd_sc = get_sc_pwd(dictData, datasetName)
pwd_sc_KDE = dictData[datasetName]["KDE"]

label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


# prepare pwd_sc and calculate insulation score
pwd_sc_fill_lin, numMissing, keep = \
    fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)

pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)

if invertPWD:
    pwd_sc_fill[pwd_sc_fill<cutoff_dist] = cutoff_dist
    pwd_sc_fill = 1/pwd_sc_fill

nNuclei = pwd_sc_fill.shape[2]
nBarcodes = pwd_sc_fill.shape[0]

insulation_score = np.full((nNuclei, nBarcodes), np.NaN)


for i in range(nNuclei):
    insulation_score[i,:] = \
        get_insulation_score(pwd_sc_fill[:,:,i], sqSize, scNormalize=scNormalize)


# collect insulation scores
dictIS = {}
listIS_labels = []

listIS_labels.append("ON")
dictIS["ON"] = insulation_score[(label_SNDchan[keep]==1),bin_border]

listIS_labels.append("OFF")
dictIS["OFF"] = insulation_score[(label_SNDchan[keep]==0),bin_border]


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150) # default (6.4, 4.8), 72dpi

plots = []
listXticks = []

for i, label in enumerate(listIS_labels):
    IS = dictIS[label]
    
    keep = ~np.isnan(IS)
    plot = ax.violinplot(IS[keep], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plots += [plot]
    listXticks += ["{}\nn={}".format(label, np.sum(keep))]

if scNormalize:
    ax.set_ylabel("Insulation score, normalized")
else:
    ax.set_ylabel("Insulation score")

txt = "IS (NaN filled with {})\n"
txt += "invertPWD={}, sqSize={},\nbin_border={}, cutoff_dist={}"
txt = txt.format(fillMode, invertPWD, sqSize, bin_border, cutoff_dist)
ax.set_title(txt, fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, None])


# colors = [(1,0,0), (0,0,1)] # red, blue
colors = [myColors[2], myColors[3]] # red, blue; order is ON, OFF

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i]) # set_edgecolor(<color>), set_linewidth(<float>), set_alpha(<float>)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_4/Supp_Fig_4_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg") # fig.savefig(fPath+".pdf")






#%% Fig Supp 4 X, higher-order contacts nc11nc12

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"

distance_threshold = 0.25 # in µm

sOut = "Probability" # Counts, Probability


# get concatenated sc PWD maps
pwd_sc = get_sc_pwd(dictData, datasetName)


nBarcodes = pwd_sc.shape[0]
r_bc = range(nBarcodes)

contact_3way_sc = np.zeros((nBarcodes,nBarcodes))

    
for anchor, bait1, bait2 in product(r_bc, r_bc, r_bc):
    if (anchor==bait1) or (anchor==bait2) or (bait1==bait2):
        continue
    
    n_contacts, n_nonNaN = getMultiContact(pwd_sc, anchor, bait1, bait2, distance_threshold)
    
    if sOut == "Counts":
        contact_3way_sc[bait1, bait2] += n_contacts
    elif sOut == "Probability":
        contact_3way_sc[bait1, bait2] += n_contacts / n_nonNaN
    else:
        raise SystemExit("Unexpected value for variable sOut.")


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8), dpi=150) # default (6.4, 4.8), 72dpi
            
if (sOut == "Probability"):
    vmin, vmax = 0.0, 0.25
else:
    vmin, vmax = None, None


txt = "{}, anchor={}".format("nc11nc12", "sum all")
plot_map(ax, contact_3way_sc, title=txt, cbar_label=sOut, vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])


# save
figName = "Figure_4/Supp_Fig_4_X"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf") # matrix looks crappy with svg






#%% ####### PLAYGROUND ################################################

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    })



#%% get ensemble topological representation for loRes ON and OFF nuclei

from sklearn import manifold

def get_XYZ_from_PWD_KDE(PWD_KDE, savePath):
    # multi-dimensional scaling to get coordinates from PWDs
    pwd_KDE = PWD_KDE.copy()
    
    # make sure pwd_KDE is symmetric
    pwd_KDE = 0.5 * (pwd_KDE + np.transpose(pwd_KDE))
    # run metric mds
    verbosity = 0  # default: 0, quite verbose: 2
    mds = manifold.MDS(
        n_components=3,
        metric=True,
        n_init=20,
        max_iter=3000,
        verbose=verbosity,
        eps=1e-9,
        n_jobs=1,
        random_state=1,
        dissimilarity="precomputed",  # euclidean | precomputed
    )
    XYZ = mds.fit(pwd_KDE).embedding_
    # print(XYZ.shape)
    print(XYZ)
    write_XYZ_2_pdb(savePath, XYZ)



def write_XYZ_2_pdb(fileName, XYZ):
    # writes XYZ coordinates to a PDB file wth pseudoatoms
    # fileName : string of output file path, e.g. '/foo/bar/test2.pdb'
    # XYZ      : n-by-3 numpy array with atom coordinates

    n_atoms = XYZ.shape[0]
    with open(fileName, "w+") as fid:
        ## atom coordinates
        txt = "HETATM  {: 3d}  C{:02d} PSD P   1      {: 5.3f}  {: 5.3f}  {: 5.3f}  0.00  0.00      PSDO C  \n"
        for i in range(n_atoms):
            fid.write(txt.format(i + 1, i + 1, XYZ[i, 0], XYZ[i, 1], XYZ[i, 2]))

        ## connectivity
        txt1 = "CONECT  {: 3d}  {: 3d}\n"
        txt2 = "CONECT  {: 3d}  {: 3d}  {: 3d}\n"
        # first line of connectivity
        fid.write(txt1.format(1, 2))
        # consecutive lines
        for i in range(2, n_atoms):
            fid.write(txt2.format(i, i - 1, i + 1))
        # last line
        fid.write(txt1.format(i + 1, i))

        print("Done writing {:s} with {:d} atoms.".format(fileName, n_atoms))





datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
sc_label = "doc"; sc_action = "labeled"


# get concatenated sc PWD maps
pwd_sc = get_sc_pwd(dictData, datasetName)

# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]



pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]
pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]


cells2Plot = [True]*pwd_sc_ON.shape[2]
pwd_KDE_ON, _ = calculatesEnsemblePWDmatrix(pwd_sc_ON, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1

cells2Plot = [True]*pwd_sc_OFF.shape[2]
pwd_KDE_OFF, _ = calculatesEnsemblePWDmatrix(pwd_sc_OFF, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1


get_XYZ_from_PWD_KDE(pwd_KDE_ON, "test_ON.pdb")
get_XYZ_from_PWD_KDE(pwd_KDE_OFF, "test_OFF.pdb")


#%% imputation with dineof (version running on multiple datasets)
import subprocess
from scipy.spatial.distance import squareform #, pdist

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D", "doc_wt_nc14_loRes_20_perROI_3D"]
listMinDetections = [13, 14]


for i, datasetName in enumerate(datasetNames):
    # load data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    # filter cells with high number of missing barcodes
    minDetections = listMinDetections[i]
    keep_1_minDet = filter_by_detected_barcodes(pwd_sc, minDetections)
    pwd_sc = pwd_sc[:, :, keep_1_minDet]
    
    # make linear
    pwd_sc_lin = get_mat_linear(pwd_sc)
    
    # save as text file
    pathDINEOF = "./dineof"
    createDir(pathDINEOF, 0o775)
    np.savetxt("./dineof/PWD_test.txt", pwd_sc_lin, delimiter="\t")
    
    # ============================
    # run imputation (bottleneck, could be done parallel)
    nSeeds = 20
    print("Going to run imputation...")
    bashCmd = "Rscript ../0_dineof_210801.R PWD_test.txt {}".format(nSeeds)
    print("  "+bashCmd)
    process = subprocess.run(bashCmd.split(), cwd=pathDINEOF,
                             text=True, capture_output=True)
    
    
    # ============================
    # load imputed files
    results = np.empty((pwd_sc_lin.shape[0], pwd_sc_lin.shape[1], nSeeds))
    for seed in range(nSeeds):
        fn = "./dineof/PWD_test_rms1e-5_seed{}_dineof.txt".format(seed+1)
        results[:,:, seed] = np.loadtxt(fn, delimiter="\t")
    
    # set negative PWD to zero, calc mean and sdev
    results[results<0] = 0
    result = np.median(results, axis=2)
    
    results_sdev = np.std(results, axis=2)
    
    # filter cells with too many unstable imputations
    sdev_thresh = np.quantile(results_sdev[results_sdev > 0], 0.99)
    nStable = np.sum(results_sdev < sdev_thresh, axis=1)
    keep_2_nStable = nStable > pwd_sc_lin.shape[1]*0.9
    
    pwd_sc_imputed = np.array(list(map(squareform, result[keep_2_nStable,:])))
    pwd_sc_imputed = np.transpose(pwd_sc_imputed, axes=[1, 2, 0])
    
    # set diagonal to NaN
    idx = np.arange(pwd_sc_imputed.shape[0])
    pwd_sc_imputed[idx, idx, :] = np.NaN
    
    
    # ============================
    # calculate the KDE for the raw data (filtered for minDetections), imputed
    PWDs = [pwd_sc, pwd_sc_imputed]
    KDEs = []
    
    for i in range(len(PWDs)):
        cells2Plot = [True]*PWDs[i].shape[2]
        # pwd = np.transpose(PWDs[i], axes=[1, 2, 0])
        kde, _ = calculatesEnsemblePWDmatrix(PWDs[i], 1, cells2Plot, mode="KDE")
        # kde[np.identity(nLoci, dtype=bool)] = np.NaN # set diagonal to NaN
        KDEs.append(kde)
    
    # ============================
    # plot
    
    vmin, vmax = 0.3, 0.8
    vmin_diff, vmax_diff = -0.1, 0.1
    vmin_sdev, vmax_sdev = 1, 3
    
    fig, axs = plt.subplots(nrows=2, ncols=3)
    
    # KDE before imputation, filtered for minDetection
    ax = axs[0,0]
    plot_map(ax, KDEs[0], cmap=cm_map, vmin=vmin, vmax=vmax)
    ax.set_title("raw, n={}".format(PWDs[0].shape[2]))
    
    # KDE after imputation
    ax = axs[0,1]
    plot_map(ax, KDEs[1], cmap=cm_map, vmin=vmin, vmax=vmax)
    ax.set_title("imputed, n={}".format(PWDs[1].shape[2]))
    
    # KDE diff map
    ax = axs[0,2]
    plot_map(ax, KDEs[0]-KDEs[1], cmap="bwr", vmin=vmin_diff, vmax=vmax_diff)
    ax.set_title("raw-imputed")
    
    # sdev before imputation
    ax = axs[1,0]
    plot_map(ax, np.nanstd(pwd_sc, axis=2), vmin=vmin_sdev, vmax=vmax_sdev)
    ax.set_title("pwd sdev raw")
    
    # sdev after imputation
    ax = axs[1,1]
    plot_map(ax, np.nanstd(pwd_sc_imputed, axis=2), vmin=vmin_sdev, vmax=vmax_sdev)
    ax.set_title("pwd sdev imputed")
    
    # histogram sdev
    ax = axs[1,2]
    ax.hist(results_sdev.flatten(), log=True)
    ax.set_title("sdev seeds")
    
    fig.tight_layout()
    
    # ============================
    # save to dictData
    
    dictData[datasetName]["pwd_sc_imputed"] = pwd_sc_imputed
    dictData[datasetName]["imputation:minDetections"] = minDetections




#%% function to prepare UMAP run

def prepare_UMAP_dineof(dictData, dictUMAP, keyDict, datasetNames,
                        listClasses, steps):
    
    classID = []
    minmaxMiss = []
    dictUMAP[keyDict] = {}
    
    for i in range(len(datasetNames)):
        datasetName = datasetNames[i]
        step = steps[i]
        
        pwd_sc_imputed = dictData[datasetName]["pwd_sc_imputed"]
        pwd_sc_lin = get_mat_linear(pwd_sc_imputed)
        pwd_sc_lin = pwd_sc_lin[::int(step), :]
        
        # get minmaxMiss
        numBarcodes = pwd_sc_imputed.shape[0]
        maxMiss = numBarcodes - dictData[datasetName]["imputation:minDetections"]
        minmaxMiss.append((0, maxMiss))
        
        nCells = pwd_sc_lin.shape[0]
        classID += [listClasses[i]] * nCells
        
        txt = "Found {} nuclei for {}"
        txt = txt.format(nCells, datasetName)
        print(txt); time.sleep(0.5) # wait for print command to finish
        
        
        if (i==0):
            pwd_sc_lin_cat = pwd_sc_lin
        else:
            pwd_sc_lin_cat = np.vstack((pwd_sc_lin_cat, pwd_sc_lin))
    
    classes, classNum = np.unique(classID, return_inverse=True) # classes will be sorted alphanumerically!
    
    # save variables to dict
    dictUMAP[keyDict]["fillMode"] = "imputed"
    dictUMAP[keyDict]["minmaxMiss"] = minmaxMiss
    dictUMAP[keyDict]["steps"] = steps
    dictUMAP[keyDict]["datasetNames"] = datasetNames
    dictUMAP[keyDict]["classID"] = classID
    dictUMAP[keyDict]["classes"] = classes
    dictUMAP[keyDict]["classNum"] = classNum
    dictUMAP[keyDict]["pwd_sc_lin_cat"] = pwd_sc_lin_cat
    
    # return
    return dictUMAP





