#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:33:34 2021

@author: Markus Götz, CBS Montpellier, CNRS
"""



#%% IMPORTS

import os
import copy
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde



# change cwd
cwd = os.getcwd()
script_path = os.path.dirname(__file__)

if cwd != script_path:
    print("Changing dir to", script_path)
    os.chdir(script_path)



# pyHiM (make sure pyHiM is in path)
from HIMmatrixOperations import calculatesEnsemblePWDmatrix, plotDistanceHistograms



from functions_paper import (createDir, load_sc_data, load_sc_data_info, 
                             load_sc_data_Bintu, load_SNDchan, get_sc_pwd,
                             fill_missing, get_mat_linear, get_mat_square,
                             get_similarity_sc_distMap, get_insulation_score,
                             get_Rg_from_PWD, get_pair_corr, get_mixed_mat,
                             prepare_UMAP, run_UMAP, get_intra_inter_TAD,
                             plot_map, plot_representative_sc, plot_map_tria,
                             plot_pair_corr, plot_scatter_Rg,
                             plot_scatter_UMAP, get_leiden_cluster)



#%% INIT VARIABLES AND PATHS

if "dictUMAP" not in locals():
    dictUMAP = {}

# paths
PDFpath = "./PDFs"
createDir(PDFpath, 0o755)


dataSets = {
    "doc_wt_nc11nc12_loRes_20_perROI_3D": {"short": "loRes_nc11nc12", "minmax": [0.3, 0.9]},
    "doc_wt_nc14_loRes_20_perROI_3D": {"short": "loRes_nc14", "minmax": [0.3, 0.9]},
    "doc_wt_nc11nc12_hiRes_17_3D": {"short": "hiRes_nc11nc12", "minmax": [0.2, 0.7]},
    "doc_wt_nc14_hiRes_17_3D": {"short": "hiRes_nc14", "minmax": [0.2, 0.7]}
    }




#%% COLORS, FONTS

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


#%% LOAD DATA FROM dictData

import glob

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




#%% Fig 1 B, population averages, nc11nc12 and nc14 (export to npy & plot triangular map)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=600)


for i, datasetName in enumerate(dataSets.keys()):
    pwd_KDE = dictData[datasetName]["KDE"]
    
    # save KDE as npy
    fName = "Figure_1/Fig_1_B_" + dataSets[datasetName]["short"] + ".npy"
    fPath = os.path.join(PDFpath, fName)
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    np.save(fPath, pwd_KDE)

    ax = fig.axes[i]
    plot_map_tria(ax, pwd_KDE, cmap=cm_map, aspect=1.0,
                  vmin=dataSets[datasetName]["minmax"][0],
                  vmax=dataSets[datasetName]["minmax"][1], 
                  title=dataSets[datasetName]["short"], cbar_draw=True,
                  cbar_label="Distance (µm)", cbar_ticks=None,
                  cbar_orient="vertical")


figName = "Figure_1/Fig_1_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 C, population averages, loRes nc11nc12 and nc14

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
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), dpi=300)

ax = axs[0,0]
plot_map(ax, pwd_KDE_nc11nc12, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc11nc12.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


ax = axs[0,1]
plot_map(ax, pwd_KDE_nc14, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc14.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


ax = axs[1,0]
plot_map(ax, pwd_KDE_diff, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff,
         title="KDE(nc14) - KDE(nc11nc12)",
         cbar_draw=True, cbar_label="Distance change (µm)")

ax = axs[1,1]
plot_map(ax, pwd_KDE_log_r, cmap="bwr", vmin=vmin_log_r, vmax=vmax_log_r,
         title="log2(nc14/nc11nc12)",
         cbar_draw=True, cbar_label="log2(nc14/nc11nc12)")

fig.tight_layout()

figName = "Figure_1/Fig_1_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf")


#%% Fig 1 D (top), representative sc maps, doc_wt_nc11nc12_loRes_20_perROI_3D

showAll = False

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"

listShowSC = [522, 1262, 2910, 3022, 3773]

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
    fig.savefig(fPath+".pdf")


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
p_pc["mode"] = "pair_corr"
p_pc["numShuffle"] = 10


# run pair corr
pair_corr = get_pair_corr(pwd_sc, p_pc)

S_data = pair_corr["S_data"]

# plot histogram
color = myColors[0] # "C1"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, yLim=4.0, figsize=(8,5))


# give fraction of pairs with negative/positive pair corr
print("Fraction negative pair corr: {}".format(np.sum(S_data < 0) / len(S_data)))
print("Fraction positive pair corr: {}".format(np.sum(S_data > 0) / len(S_data)))


figName = "Figure_1/Fig_1_D_middle"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 E (top), representative sc maps, doc_wt_nc14_loRes_20_perROI_3D

showAll = False


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
    fig.savefig(fPath+".pdf")


#%% Fig 1 E (middle), pair correlation, doc_wt_nc14_loRes_20_perROI_3D

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"


# load data
pwd_sc = get_sc_pwd(dictData, datasetName)


# parameter for pair_corr
p_pc = {}
p_pc["cutoff_dist"] = [0.0, np.inf] # in µm; [0.05, 1.5]
p_pc["inverse_PWD"] = True
p_pc["minDetections"] = 13
p_pc["step"] = 1
p_pc["minRatioSharedPWD"] = 0.5
p_pc["mode"] = "pair_corr"
p_pc["numShuffle"] = 10


# run pair corr
pair_corr = get_pair_corr(pwd_sc, p_pc)


# plot histogram
color = myColors[1] #"C2"
fig = plot_pair_corr(pair_corr, p_pc, color, datasetName, yLim=4.0, figsize=(8,5))


# give fraction of pairs with negative/positive pair corr
print("Fraction negative pair corr: {}".format(np.sum(S_data < 0) / len(S_data)))
print("Fraction positive pair corr: {}".format(np.sum(S_data > 0) / len(S_data)))


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
random_state = 123456
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])

# plot scatter of the UMAP colorized according to treatement (auxin, untreated)
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True, pntS=12.5)

# save
figName = "Figure_1/Fig_1_G"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 G, H revision, UMAP Bintu data, run UMAP; add density
# get estimate for wt and aux treated regions from density of UMAP scatter

# run section for Fig 1G first

keyDict = "Bintu" # loRes, hiRes, Bintu

embedding = dictUMAP[keyDict]["embedding"]
classNum = dictUMAP[keyDict]["classNum"]

bw_method = 0.50 # bandwidth method: None, "scott", "silverman", a scalar constant
                 # scalar used as kde.factor

xmin, xmax = np.min(embedding[:,0]), np.max(embedding[:,0])
ymin, ymax = np.min(embedding[:,1]), np.max(embedding[:,1])
xdiff, ydiff = (xmax - xmin), (ymax - ymin)
xmin -= 0.1 * xdiff
xmax += 0.1 * xdiff
ymin -= 0.1 * ydiff
ymax += 0.1 * ydiff


# Peform the kernel density estimate
# (see https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python)
xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
positions = np.vstack([xx.ravel(), yy.ravel()])

# get densities for each nc separately
kernel_0 = gaussian_kde(embedding[classNum==0,:].T, bw_method=bw_method)
f_0 = kernel_0(positions).T.reshape(xx.shape)

kernel_1 = gaussian_kde(embedding[classNum==1,:].T, bw_method=bw_method)
f_1 = kernel_1(positions).T.reshape(xx.shape)

# subtract one from the other
f = f_1.copy()
mask = (f_0 < 0.02) & (f_1 < 0.02)
f = f_0 - f_1
f[mask] = 0

# plot scatter of the UMAP colorized according nuclear cycle
clist = [myColors[0], myColors[1]]
cmap = ListedColormap(clist)
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True, pntS=12.5)

ax = fig.axes[0]
xlim, ylim = ax.get_xlim(), ax.get_ylim()

levels = [-0.2, -0.02, +0.02, +0.2]
cfset = ax.contourf(xx, yy, f, levels=levels, cmap="bwr", alpha=0.15) # "filled", i.e. the shade
cset = ax.contour(xx, yy, f, levels=levels, colors="k", linestyles="-")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# save
figName = "Figure_1/Fig_1_G_revision"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


# count nuclei in the two masks 
f_count = f.copy()
f_count[:] = 0

mask = (f < -0.02)
f_count[mask] = +1 # mostly wt
mask = (f > +0.02)
f_count[mask] = -1 # mostly aux treated

# convert embedding to bins in the grid of the masks
bins = np.empty(embedding.shape, dtype=int)
bins[:,0] = np.around((embedding[:,0]-xmin) / (xmax-xmin) * (300-1))
bins[:,1] = np.around((embedding[:,1]-ymin) / (ymax-ymin) * (300-1))

counts_in_mask = np.zeros((2, 3), dtype=int) # 2 classes, 3 possible areas in UMAP (f==-1, 0, or 1)
for i in range(bins.shape[0]):
    curr_mask = f_count[bins[i,0], bins[i,1]]
    true_class = classNum[i]
    counts_in_mask[true_class, round(curr_mask+1)] += 1

print("classNum x area type")
print(counts_in_mask)


# plot the counts as pie charts
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9.6, 3.2), dpi=300)

ax = axs[0]
ax.pie(counts_in_mask[:,2], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("untreated\ncluster")
ax = axs[1]
ax.pie(counts_in_mask[:,0], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("cohesin depl.\ncluster")
ax = axs[2]
ax.pie(counts_in_mask[:,1], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("outside any\ncluster")

# save
figName = "Figure_1/Fig_1_H_revision"
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
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=300)

ax = axs[0]
plot_map(ax, pwd_median_aux, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_aux.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


ax = axs[1]
plot_map(ax, pwd_median_wt, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_wt.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


fig.tight_layout()

figName = "Figure_1/Fig_1_G_right"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf")


#%% Fig 1 I, UMAP loRes nc11nc12 vs nc14

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
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True, pntS=12.5)

# save
figName = "Figure_1/Fig_1_I"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 I revision, UMAP loRes nc11nc12 vs nc14; add density
# get estimate for nc11/12 and nc14 regions from density of UMAP scatter

# run section for Fig 1H first

keyDict = "loRes" # loRes, hiRes, Bintu

embedding = dictUMAP[keyDict]["embedding"]
classNum = dictUMAP[keyDict]["classNum"]

bw_method = 0.50 # bandwidth method: None, "scott", "silverman", a scalar constant
                 # scalar used as kde.factor

xmin, xmax = np.min(embedding[:,0]), np.max(embedding[:,0])
ymin, ymax = np.min(embedding[:,1]), np.max(embedding[:,1])
xdiff, ydiff = (xmax - xmin), (ymax - ymin)
xmin -= 0.1 * xdiff
xmax += 0.1 * xdiff
ymin -= 0.1 * ydiff
ymax += 0.1 * ydiff


# Peform the kernel density estimate
# (see https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python)
xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
positions = np.vstack([xx.ravel(), yy.ravel()])

# get densities for each nc separately
kernel_nc11nc12 = gaussian_kde(embedding[classNum==0,:].T, bw_method=bw_method)
f_nc11nc12 = kernel_nc11nc12(positions).T.reshape(xx.shape)

kernel_nc14 = gaussian_kde(embedding[classNum==1,:].T, bw_method=bw_method)
f_nc14 = kernel_nc14(positions).T.reshape(xx.shape)

# subtract one from the other
f = f_nc14.copy()
mask = (f_nc11nc12 < 0.02) & (f_nc14 < 0.02)
f = f_nc11nc12 - f_nc14
f[mask] = 0

# plot scatter of the UMAP colorized according nuclear cycle
clist = [myColors[0], myColors[1]]
cmap = ListedColormap(clist)
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True, pntS=12.5)

ax = fig.axes[0]
xlim, ylim = ax.get_xlim(), ax.get_ylim()

levels = [-0.2, -0.02, +0.02, +0.2]
cfset = ax.contourf(xx, yy, f, levels=levels, cmap="bwr", alpha=0.15) # "filled", i.e. the shade
cset = ax.contour(xx, yy, f, levels=levels, colors="k", linestyles="-")
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# save
figName = "Figure_1/Fig_1_I_revision"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


# count nuclei in the two masks 
f_count = f.copy()
f_count[:] = 0

mask = (f < -0.02)
f_count[mask] = +1 # mostly nc14
mask = (f > +0.02)
f_count[mask] = -1 # mostly nc11nc12


# convert embedding to bins in the grid of the masks
bins = np.empty(embedding.shape, dtype=int)
bins[:,0] = np.around((embedding[:,0]-xmin) / (xmax-xmin) * (300-1))
bins[:,1] = np.around((embedding[:,1]-ymin) / (ymax-ymin) * (300-1))

counts_in_mask = np.zeros((2, 3), dtype=int) # 2 classes, 3 possible areas in UMAP (f==-1, 0, or 1)
for i in range(bins.shape[0]):
    curr_mask = f_count[bins[i,0], bins[i,1]]
    true_class = classNum[i]
    counts_in_mask[true_class, round(curr_mask+1)] += 1

print("classNum x area type")
print(counts_in_mask)


# plot the counts as pie charts
N = np.sum(counts_in_mask, axis=0)
N = N / np.max(N)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9.6, 3.2), dpi=150)

ax = axs[0]
ax.pie(counts_in_mask[:,2], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("nc14\ncluster")
ax = axs[1]
ax.pie(counts_in_mask[:,0], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("nc11/12\ncluster")
ax = axs[2]
ax.pie(counts_in_mask[:,1], autopct='%1.1f%%', startangle=90, colors=clist)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("outside any\ncluster")

# save
figName = "Figure_1/Fig_1_J_revision"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 1 H, right, pwd maps loRes

vmin, vmax = 0.3, 0.8 # in µm

tickPos = np.linspace(vmin, vmax ,int(round((vmax-vmin)/0.1)+1))


datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"
pwd_sc_nc11nc12 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc11nc12 = dictData[datasetName]["KDE"]


datasetName = "doc_wt_nc14_loRes_20_perROI_3D"
pwd_sc_nc14 = get_sc_pwd(dictData, datasetName)
pwd_KDE_nc14 = dictData[datasetName]["KDE"]


# plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), dpi=300)

ax = axs[0]
plot_map(ax, pwd_KDE_nc11nc12, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc11nc12.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


ax = axs[1]
plot_map(ax, pwd_KDE_nc14, cm_map, vmin, vmax,
         title="n={}".format(pwd_sc_nc14.shape[2]),
         cbar_draw=True, cbar_label="Distance (µm)", cbar_ticks=tickPos)


fig.tight_layout()

figName = "Figure_1/Fig_1_H_right"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".pdf")


#%% Fig Supp 1 B, detection efficiency loRes nc11nc12 and nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=150)

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
figName = "Figure_1/Supp_Fig_1_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 1 B, barcode per cell, loRes nc11nc12 and nc14

datasetNames = ["doc_wt_nc11nc12_loRes_20_perROI_3D",
                "doc_wt_nc14_loRes_20_perROI_3D"]

density = True


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=150)

for i, datasetName in enumerate(datasetNames):
    ax = axs[i]
    
    # load data
    pwd_sc = get_sc_pwd(dictData, datasetName)
    
    numBarcodes = pwd_sc.shape[0]
    numNuclei = pwd_sc.shape[2]
    
    # detection when not all entries are NaN; possible values: 0, 2, 3, ... (1 barcode would not give a PWD)
    bc_per_cell = np.sum(np.sum(np.isnan(pwd_sc), axis=1) < numBarcodes, axis=0)
    
    h = ax.hist(bc_per_cell, bins=np.arange(1.5, numBarcodes+1.5, 1),
                density=True, rwidth=0.8)
    
    ax.set_xlabel("Barcodes per nucleus")
    ax.set_xlim(1.5, numBarcodes+0.5)
    step = 5
    ax.set_xticks([2] + list(range(step, numBarcodes+1, step)))
    
    if density:
        ax.set_ylabel("Probability")
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
figName = "Figure_1/Supp_Fig_1_B_bottom"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 1 C, D, plotDistanceHistograms

pixelSize = 1.0 # as PWD maps are converted to µm already, set this to 1
mode = "KDE" # hist, KDE
kernelWidth = 0.25 # in µm; 0.25 is default in pyHiM
maxDistance = 4.0 # in µm; 4.0 is default in pyHiM; removes larger distances

datasetName = "doc_wt_nc11nc12_loRes_20_perROI_3D"

figName = "Figure_1/Supp_Fig_1_C_" + datasetName + "_" + mode
fPath = os.path.join(PDFpath, figName)
createDir(os.path.dirname(fPath), 0o755)
logNameMD = fPath + "_log.md"

pwd_sc = get_sc_pwd(dictData, datasetName)

plotDistanceHistograms(
    pwd_sc, pixelSize, fPath, logNameMD, mode=mode, limitNplots=0,
    kernelWidth=kernelWidth, optimizeKernelWidth=False, maxDistance=maxDistance)

datasetName = "doc_wt_nc14_loRes_20_perROI_3D"

figName = "Figure_1/Supp_Fig_1_D_" + datasetName + "_" + mode
fPath = os.path.join(PDFpath, figName)
createDir(os.path.dirname(fPath), 0o755)
logNameMD = fPath + "_log.md"

pwd_sc = get_sc_pwd(dictData, datasetName)


plotDistanceHistograms(
    pwd_sc, pixelSize, fPath, logNameMD, mode=mode, limitNplots=0,
    kernelWidth=kernelWidth, optimizeKernelWidth=False, maxDistance=maxDistance)


#%% Fig Supp 1 E, radius of gyration, nc11nc12 and nc14 (raw data)

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minmaxBins = [[13, 21]] # for each TAD: [binStart, binEnd]
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


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150)

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
    plots[i]["bodies"][0].set_facecolor(colors[i])
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_1/Supp_Fig_1_E"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


# statistical test on the distributions
list_data = []
for i, datasetName in enumerate(datasetNames):
    r_gyration = dictRg[datasetName]
    keep = ~np.isnan(r_gyration[:, 0])
    list_data.append(r_gyration[keep, 0])

statistic, pvalue = ttest_ind(list_data[0], list_data[1], equal_var=False)


print("statistic {}, pvalue {}".format(statistic, pvalue))


#%% Fig Supp 1 F, Bintu data, 20 barcodes centered on "strong" and "weak" border

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

random_state = 123456
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# plot scatter of the UMAP
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)
fig.axes[0].set_title("{}, selRange={}\n{}".format(keyDict, selRange, fig.axes[0].get_title()), fontsize=12)


# save
figName = "Figure_1/Supp_Fig_1_F_strong"
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

random_state = 123456
dictUMAP = run_UMAP(dictUMAP, keyDict, random_state)

# plot scatter of the UMAP
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap, hideTicks=True)
fig.axes[0].set_title("{}, selRange={}\n{}".format(keyDict, selRange, fig.axes[0].get_title()), fontsize=12)

# save
figName = "Figure_1/Supp_Fig_1_F_weak"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 1 G, doc loRes, scanning UMAP hyper-parameters
# (run section with UMAP for loRes data first, Fig. 1h)

n_neighbors_scan = [15, 50, 250]
min_dist_scan = [0.01, 0.1, 0.8]

keyDict_orig = "loRes"
random_state = 123456


# get colormap
cmap = ListedColormap([myColors[0], myColors[1]])


# make a copy of the original UMAP
keyDict = keyDict_orig + "_scan"
dictUMAP[keyDict] = dictUMAP[keyDict_orig].copy()


# loop over all parameter combinations and plot
fig, _ = plt.subplots(nrows=len(n_neighbors_scan), ncols=len(min_dist_scan),
                      figsize=(14, 12), dpi=150)

for i, (n_neighbors, min_dist) in enumerate(product(n_neighbors_scan, min_dist_scan)):
    print(i, n_neighbors, min_dist)
    ax = fig.axes[i]
    
    custom_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}

    dictUMAP = run_UMAP(dictUMAP, keyDict, random_state, custom_params)
    
    plot_scatter_UMAP(dictUMAP[keyDict], cmap, ax=ax, hideTicks=True, pntS=10)
    txt = "n_neigh={}, min_dist={}".format(n_neighbors, min_dist)
    ax.set_title(txt, fontsize=12)
    ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.tight_layout()


# save
figName = "Figure_1/Supp_Fig_1_G"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 2 A, loRes, correlation Rg TAD1 vs Rg doc-TAD

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
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 10), dpi=150)

ax = axs[0]
plot_scatter_Rg(ax, dictRg, datasetNames[0], param_Rg, getRegLine=False,
                color=myColors[0], cross=[center, "-", "k"])

ax = axs[1]
plot_scatter_Rg(ax, dictRg, datasetNames[1], param_Rg, getRegLine=False,
                color=myColors[1], cross=[center, "--", "k"])

fig.tight_layout()


# save
figName = "Figure_2/Fig_2_A"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 2 B, UMAP colorcoded by Rg (use data filled with KDE for calculation of Rg)
# (run section for Figure 1 H first)

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

minBin = 13
maxBin = 21 # numpy indexing, thus last bin won't be included

vmin, vmax = 0.20, 0.35
tick_pos=[0.20, 0.25, 0.3, 0.35]
dotSize = 25


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


cmap = "coolwarm"

sel_nc11nc12 = (np.array(classID) == "nc11nc12")
sel_nc11nc12_Rg = np.logical_and((np.array(classID) == "nc11nc12"), ~np.isnan(r_gyration))
sel_nc14_noRg = np.logical_and((np.array(classID) == "nc14"), np.isnan(r_gyration))
sel_nc14_Rg = np.logical_and((np.array(classID) == "nc14"), ~np.isnan(r_gyration))

sel_Rg = (~np.isnan(r_gyration))


# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150)

# nuclei without Rg in light grey
plot = ax.scatter(embedding[~sel_Rg,0], embedding[~sel_Rg,1],
                  s=dotSize, marker="x", linewidths=1, facecolors=[0.2]*3, # the smaller the darker
                  alpha=1.0)

# nuclei with Rg according to color map
plot = ax.scatter(embedding[sel_Rg,0], embedding[sel_Rg,1],
                  s=dotSize, c=r_gyration[sel_Rg], cmap=cmap, alpha=1.0,
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
figName = "Figure_2/Fig_2_B"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig 2 C, D, sc insulation score nc11/nc12 vs nc14

dataSets = {"doc_wt_nc11nc12_loRes_20_perROI_3D": 
                {"short": "nc11nc12",
                 "minmaxMiss": [0,7]},
            "doc_wt_nc14_loRes_20_perROI_3D":
                {"short": "nc14",
                 "minmaxMiss": [0,6]}
            }

fillMode = "KDE"

invertPWD = True
sqSize = 2
scNormalize = False
bin_border = 13 # doc: 13, bintu: 49 (zero-based)


# create figures
fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(10.0, 8.45), dpi=300)
fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=( 8.2, 1.25), dpi=150)
fig3, axs3 = plt.subplots(nrows=1, ncols=1, figsize=( 5.0, 5.00), dpi=150)


# loop over datasets
for iData, datasetName in enumerate(dataSets.keys()):
    minMiss, maxMiss = dataSets[datasetName]["minmaxMiss"]
    
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc_KDE = dictData[datasetName]["KDE"]
    
    
    pwd_sc_fill_lin, numMissing, keep = \
        fill_missing(pwd_sc, pwd_sc_KDE, minMiss, maxMiss, fillMode)
    
    pwd_sc_fill = get_mat_square(pwd_sc_fill_lin)
    pwd_sc_IS = pwd_sc[:,:,keep]
    
    if invertPWD:
        pwd_sc_fill = 1/pwd_sc_fill
    
    nNuclei = pwd_sc_fill.shape[2]
    nBarcodes = pwd_sc_fill.shape[0]
    
    insulation_score = np.full((nNuclei, nBarcodes), np.NaN)
    insulation_score_NaN = np.full((nNuclei, nBarcodes), np.NaN)
    
    
    for i in range(nNuclei):
        insulation_score[i,:] = \
            get_insulation_score(pwd_sc_fill[:,:,i], sqSize,
                                 scNormalize=scNormalize)
        insulation_score_NaN[i,:] = \
            get_insulation_score(pwd_sc_IS[:,:,i], sqSize,
                                 nansum=True, scNormalize=False) # to remove cells that only have imputed values
    
    # remove nuclei that only had imputed values for border barcode
    keep_IS = (insulation_score_NaN[:, bin_border]>=0) # set to !=0 to exclude
    insulation_score = insulation_score[keep_IS,:]
    
    # sort according the IS at border barcode
    sorting = insulation_score[:, bin_border].argsort() # add [::-1] to invert
    insulation_score_sorted = insulation_score[sorting]
    
    
    # --- Figure 1 ---
    # plot single-cell insulation score
    cmap = "bwr_r"
    if scNormalize:
        vmin, vmax = 0, 2
    else:
        vmin, vmax = 0, 14 # symmetric around 7
    
    ax = axs1[iData]
    plot = ax.matshow(insulation_score_sorted, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[vmin, (vmin+vmax)/2, vmax])
    if scNormalize:
        cbar.set_label("Insulation score, normalized")
    else:
        cbar.set_label("Insulation score")
    
    ax.set_xlim([-0.5+sqSize, nBarcodes-0.5-sqSize])
    aspect = 2.2*(nBarcodes-2*sqSize)/np.sum(keep_IS)
    ax.set_aspect(aspect) # the larger the higher the plot
    
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    
    ax.set_xticks(np.arange(sqSize,nBarcodes-sqSize+1,3))
    ax.set_xticklabels(np.arange(sqSize,nBarcodes-sqSize+1,3)+1)
    
    ax.set_xlabel("Barcode")
    ax.set_ylabel("Single-nucleus insulation score, cell ID (n={})".format(np.sum(keep_IS)))
    
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


    # --- Figure 3 ---
    # cutoff "high insulation"
    cutoff_IS = 3.5
    txt = "{}: hardcoded cutoff_IS={}".format(datasetName, cutoff_IS)
    print(txt)
    IS_border = insulation_score[:, bin_border]
    N_below = np.sum(IS_border <= cutoff_IS)
    N_total = np.sum(~np.isnan(IS_border))
    print("Below cutoff:", N_below/N_total)
    N_total = np.sum(~np.isnan(IS_border))
    cutoff_IS = np.arange(0, 14.25, 0.25)
    r_below = [] # np.full(cutoff_IS.shape, np.NaN)
    for c in cutoff_IS:
        N_below = np.sum(IS_border <= c)
        r_below.append(N_below/N_total)
    axs3.plot(cutoff_IS, r_below, "-", color=myColors[iData])
    e = 0 #0.25
    axs3.set_xlim(0-e, 14+e)
    axs3.set_xticks(np.arange(0, 14.5, 2))
    axs3.set_xlabel("IS cutoff")
    axs3.set_ylabel("Fraction of nuclei with an IS at\nthe border barcode below cutoff")
    axs3.legend([dataSets[d]["short"] for d in dataSets])


# save all three figures
figName = "Figure_2/Fig_2_C_D"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig1.savefig(fPath+".svg")

figName = "Figure_2/Fig_2_C_D_top"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig2.savefig(fPath+".svg")

figName = "Figure_2/Supp_Fig_2_A"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig3.savefig(fPath+".svg")


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
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10.1, 10), dpi=150)


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
        txt = "pearson_corrcoef {} {}: {:.6f}, for n={}"
        print(txt.format(strTAD, datasetName, pearson_corrcoef, np.sum(keep_IS_Rg)))
        
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

dotSize = 25
cmap = "coolwarm_r" # match style in Fig 2G

if doLog:
    vmin, vmax = 0, 2
else:
    vmin, vmax = 0, 14 # match values in Fig 2G
    if scNormalize:
        vmin, vmax = 0.4, 1.6


pwd_sc_lin_cat = dictUMAP["loRes"]["pwd_sc_lin_cat"]
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150)

plot = ax.scatter(embedding[:,0], embedding[:,1],
                  s=dotSize, c=IS, cmap=cmap, alpha=1.0, vmin=vmin, vmax=vmax)


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


#%% Fig 2 G, H revision: KNN graph and Leiden clustering 

# run UMAP for the dataset first

keyDict = "loRes"

fig1, fig2, Rg_IS_all = get_leiden_cluster(dictUMAP, keyDict, 0.25, cm_map)
# save
figName = ["Figure_2/Fig_2_G_revision", "Figure_2/Fig_2_H_revision"]
for i, fig in enumerate([fig1, fig2]):
    fPath = os.path.join(PDFpath, figName[i])
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    fig.savefig(fPath+".svg")


fig1, fig2, Rg_IS_all = get_leiden_cluster(dictUMAP, keyDict, 0.33, cm_map)
# save
figName = ["Figure_2/Supp_Fig_2_D_top_revision", "Figure_2/Supp_Fig_2_D_bottom_revision"]
for i, fig in enumerate([fig1, fig2]):
    fPath = os.path.join(PDFpath, figName[i])
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    fig.savefig(fPath+".svg")


fig1, fig2, Rg_IS_all = get_leiden_cluster(dictUMAP, keyDict, 0.50, cm_map)
# save
figName = ["Figure_2/Supp_Fig_2_E_top_revision", "Figure_2/Supp_Fig_2_E_bottom_revision"]
for i, fig in enumerate([fig1, fig2]):
    fPath = os.path.join(PDFpath, figName[i])
    pathFigDir = os.path.dirname(fPath)
    createDir(pathFigDir, 0o755)
    fig.savefig(fPath+".svg")


#%% Fig Supp 2 A

# see section for Fig. 2C, D


#%% Fig Supp 2 B

# see section for Fig. 2E


#%% Fig Supp 2 C, Bintu data, UMAP colorcoded acc to insulation score
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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150)

plot = ax.scatter(embedding[:,0], embedding[:,1],
                  s=25, c=IS, cmap=cmap, alpha=1.0, vmin=vmin, vmax=vmax)


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
figName = "Figure_2/Supp_Fig_2_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 2 D, E: KNN graph and Leiden clustering 

# see section for Fig. 2 G, H


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

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150)

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

# fix colors and line widths for violin plots
for i in range(len(plots)):
    color = myColors[2+i%2]
    plots[i]["bodies"][0].set_facecolor(color)
    plots[i]["bodies"][0].set_alpha(0.5)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(color)
        plots[i][partname].set_linewidth(3)

# calculate the p-value
for i, mode in enumerate(listModes):
    if (mode=="intra"):
        curr_dist = intraTAD_noNaN
        curr_keep = keep_intra
    else:
        curr_dist = interTAD_noNaN
        curr_keep = keep_inter
    
    sel_transc = (label_SNDchan[curr_keep]==1)
    dist_ON = curr_dist[sel_transc]
    dist_OFF = curr_dist[~sel_transc]
    
    statistic, pvalue = ttest_ind(dist_ON, dist_OFF, equal_var=False)
    
    print("{}: ttest_ind, statistic {}, pvalue {}".format(mode, statistic, pvalue))


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
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), dpi=150)

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
fig.savefig(fPath+".pdf")


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
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), dpi=150)

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
fig.savefig(fPath+".pdf")


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


#%% Fig 3 E revision, bottom, UMAP, doc hiRes 17RTs, nc11nc12, nc14 ON, nc14 OFF; add density
# get estimate for nc11/12 and nc14 treated regions from density of UMAP scatter

# run section for Fig 3 E bottom first

keyDict = "hiRes"

embedding = dictUMAP[keyDict]["embedding"]
classNum = dictUMAP[keyDict]["classNum"]

bw_method = 0.50 # bandwidth method: None, "scott", "silverman", a scalar constant
                 # scalar used as kde.factor

cmap_scatter = ListedColormap([
    (0.8, 0.8, 0.8, 1.0), # nc11nc12
    myColors[3],          # nc14 OFF
    myColors[2]           # nc14 ON
    ])

xmin, xmax = np.min(embedding[:,0]), np.max(embedding[:,0])
ymin, ymax = np.min(embedding[:,1]), np.max(embedding[:,1])
xdiff, ydiff = (xmax - xmin), (ymax - ymin)
xmin -= 0.1 * xdiff
xmax += 0.1 * xdiff
ymin -= 0.1 * ydiff
ymax += 0.1 * ydiff


# Peform the kernel density estimate
# (see https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python)
xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
positions = np.vstack([xx.ravel(), yy.ravel()])

# get densities for each nc separately
kernel_0 = gaussian_kde(embedding[classNum==0,:].T, bw_method=bw_method)
f_0 = kernel_0(positions).T.reshape(xx.shape)

kernel_1 = gaussian_kde(embedding[classNum==1,:].T, bw_method=bw_method)
f_1 = kernel_1(positions).T.reshape(xx.shape)

# subtract one from the other
f = f_1.copy()
mask = (f_0 < 0.02) & (f_1 < 0.02)
f = f_0 - f_1
f[mask] = 0

# plot scatter of the UMAP colorized according nuclear cycle
fig = plot_scatter_UMAP(dictUMAP[keyDict], cmap_scatter, hideTicks=True)

ax = fig.axes[0]
xlim, ylim = ax.get_xlim(), ax.get_ylim()

levels = [-0.2, -0.01, +0.01, +0.2]
cfset = ax.contourf(xx, yy, f, levels=levels, cmap="bwr", alpha=0.15) # "filled", i.e. the shade
cset = ax.contour(xx, yy, f, levels=levels, colors="k", linestyles="-")
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# save
figName = "Figure_3/Fig_3_E_bottom_revision"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 3 A, difference map ON vs OFF, loRes

# this plot is included in the section for Fig 3 D


#%% Fig Supp 3 B, radius of gyration doc loRes, nc14 ON vs OFF (raw data)

cutoff = 1.0 # in µm
minFracNotNaN = 0.33

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


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150)

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
    plots[i]["bodies"][0].set_facecolor(colors[i])
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_3/Supp_Fig_3_B_loRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 3 C, contact probability map doc hiRes, ON and OFF

datasetName = "doc_wt_nc14_hiRes_17_3D"
sc_label = "doc"; sc_action = "labeled"

cutoff_contact = 0.25 # threshold (in µm) for binarizing distances
vmin, vmax = 0.0, 0.25


pwd_sc = get_sc_pwd(dictData, datasetName)
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]


pwd_sc_OFF = pwd_sc[:,:,(label_SNDchan==0)]
pwd_sc_ON = pwd_sc[:,:,(label_SNDchan==1)]


contacts_OFF = np.nansum((pwd_sc_OFF < cutoff_contact), axis=2)
contacts_ON = np.nansum((pwd_sc_ON < cutoff_contact), axis=2)

n_OFF = np.sum(~np.isnan(pwd_sc_OFF), axis=2)
n_ON = np.sum(~np.isnan(pwd_sc_ON), axis=2)

contact_prob_OFF = contacts_OFF/n_OFF
contact_prob_ON = contacts_ON/n_ON

contact_prob_ON_OFF = get_mixed_mat(contact_prob_ON, contact_prob_OFF)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), dpi=150)

plot_map(axs[0], contact_prob_ON_OFF, cmap="viridis", vmin=vmin, vmax=vmax,
         title="", cbar_draw=True, cbar_label="Contact probability",
         cbar_ticks=None, cbar_orient="vertical")

axs[1].set_axis_off()

# save
figName = "Figure_3/Supp_Fig_3_C"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


#%% Fig Supp 3 D, radius of gyration doc hiRes, nc14 ON vs OFF

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


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150)

plots = []
listXticks = []


for i, label in enumerate(listRg_labels):
    r_gyration = dictRg[label]
    
    keep = ~np.isnan(r_gyration[:, 0])
    plot = ax.violinplot(r_gyration[keep, 0], [i], widths=0.5,
                         showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel("Radius of gyration (µm)")
    plots += [plot]
    listXticks.append("{}\nn={}".format(label, np.sum(keep)))


txt = "R_g (hiRes raw data)\ncutoff={}µm, minFracNotNaN={}"
ax.set_title(txt.format(cutoff, minFracNotNaN), fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(listXticks)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([0, 0.6])


colors = [myColors[2], myColors[3]]

for i in range(len(plots)):
    plots[i]["bodies"][0].set_facecolor(colors[i])
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(colors[i])
        plots[i][partname].set_linewidth(3)


figName = "Figure_3/Supp_Fig_3_D_hiRes"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


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
pwd_sc_IS = pwd_sc[:,:,keep]

if invertPWD:
    pwd_sc_fill = 1/pwd_sc_fill

nNuclei = pwd_sc_fill.shape[2]
nBarcodes = pwd_sc_fill.shape[0]

insulation_score = np.full((nNuclei, nBarcodes), np.NaN)
insulation_score_NaN = np.full((nNuclei, nBarcodes), np.NaN)


for i in range(nNuclei):
    insulation_score[i,:] = \
        get_insulation_score(pwd_sc_fill[:,:,i], sqSize, scNormalize=scNormalize)

    insulation_score_NaN[i,:] = \
        get_insulation_score(pwd_sc_IS[:,:,i], sqSize,
                             nansum=True, scNormalize=False) # to remove cells that only have imputed values
    
# remove nuclei that only had imputed values for border barcode
keep_IS = (insulation_score_NaN[:, bin_border]>=0) # set to !=0 to exclude
insulation_score = insulation_score[keep_IS,:]

sorting = insulation_score[:, bin_border].argsort() # add [::-1] to invert
insulation_score_sorted = insulation_score[sorting]



# get transcriptional state
label_SNDchan = dictData[datasetName]["SNDchan"][sc_label+"_"+sc_action]
label_SNDchan_sorted = label_SNDchan[keep,None]
label_SNDchan_sorted = label_SNDchan_sorted[sorting]


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 16), dpi=300)


# plot single-cell insulation score
cmap = "bwr_r"
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
ax.set_ylabel("Single-nucleus insulation score, cell ID (n={})".format(np.sum(keep_IS)))

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


figName = "Figure_4/Fig_4_A"
fPath = os.path.join(PDFpath, figName)
pathFigDir = os.path.dirname(fPath)
createDir(pathFigDir, 0o755)
fig.savefig(fPath+".svg")


# also get a bar graph of the insulation score profile
sFunct = "median" # mean, median; use the same as in Fig. Supp 4 A

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.13, 2), dpi=150)

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
fig.savefig(fPath+".svg")


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
pwd_sc_IS = pwd_sc[:,:,keep]


if invertPWD:
    pwd_sc_fill = 1/pwd_sc_fill

nNuclei = pwd_sc_fill.shape[2]
nBarcodes = pwd_sc_fill.shape[0]

insulation_score = np.full((nNuclei, nBarcodes), np.NaN)
insulation_score_NaN = np.full((nNuclei, nBarcodes), np.NaN)


for i in range(nNuclei):
    insulation_score[i,:] = \
        get_insulation_score(pwd_sc_fill[:,:,i], sqSize, scNormalize=scNormalize)

    insulation_score_NaN[i,:] = \
        get_insulation_score(pwd_sc_IS[:,:,i], sqSize,
                             nansum=True, scNormalize=False) # to remove cells that only have imputed values
    
# remove nuclei that only had imputed values for border barcode
keep_IS = (insulation_score_NaN[:, bin_border]>=0) # set to !=0 to exclude
insulation_score = insulation_score[keep_IS,:]


# collect insulation scores
dictIS = {}
dictIS_labels = [("ON", 1), ("OFF", 0)]

# bootstrap an error for the insulation score
for k, v in dictIS_labels:
    label_SNDchan_keep = (label_SNDchan[keep])[keep_IS]
    currIS = insulation_score[(label_SNDchan_keep==v),:]
    
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

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150)

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
fig.savefig(fPath+".svg")


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


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6), dpi=150)

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
ax.set_ylim(-4.0, 3.0)
txt = "demixing score\n"
txt += "cutoff={}µm, bin_border={}, sFunct={}, minNumPWD={}"
txt = txt.format(cutoff_dist, bin_border, sFunct, minNumPWD)
ax.set_title(txt, fontsize=12)

for i in range(len(plots)):
    color = myColors[i+2]
    plots[i]["bodies"][0].set_facecolor(color)
    plots[i]["bodies"][0].set_alpha(0.5)
    for partname in ("cmeans","cmaxes","cmins","cbars"):
        plots[i][partname].set_edgecolor(color)
        plots[i][partname].set_linewidth(3)


# calculate the p-value
dist_ON = ration_log_noNaN[label_SNDchan[keep_ratio]==1]
dist_OFF = ration_log_noNaN[label_SNDchan[keep_ratio]==0]

statistic, pvalue = ttest_ind(dist_ON, dist_OFF, equal_var=False)
    
print("ttest_ind, statistic {}, pvalue {}".format(statistic, pvalue))


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
histtype = "bar"

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150)

plot = ax.hist(interTAD_sum[label_SNDchan[keep]==1], bins=bins, density=True,
               histtype=histtype, color=myColors[2], alpha=0.5, label="ON",
               width=0.9, zorder=2)

plot = ax.hist(interTAD_sum[label_SNDchan[keep]==0], bins=bins, density=True,
               histtype=histtype, color=myColors[3], alpha=0.5, label="OFF",
               width=0.9, zorder=1)


ax.legend(loc="center right")

ax.set_xlabel("Number of inter-TAD contacts")
ax.set_ylabel("Probability")
if showAsLog:
    ax.set_yscale("log")
    xLim = 20
    ax.set_xticks(range(0, xLim+1, 2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2)) # a single minor tick between major ticks.
else:
    xLim = 6
    ax.set_xticks(range(0, xLim+1, 1))


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



