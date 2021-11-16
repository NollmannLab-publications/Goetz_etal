#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 19:36:24 2021

@author: Markus Götz, CBS Montpellier, CNRS
"""


import os
import copy
import time
import numpy as np
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from matplotlib import transforms
from tqdm import trange
import umap

from HIMmatrixOperations import calculatesEnsemblePWDmatrix





def createDir(dirPath, permission=0o755):
    if not os.path.exists(dirPath):
        try:
            os.mkdir(dirPath)
            os.chmod(dirPath, permission)
        except OSError:
            print("Creation of the directory failed.")
    else:
        # print("")
        pass





def get_sc_pwd(dictData, datasetName):
    # load data
    pwd_sc_raw = dictData[datasetName]["pwd_sc_raw"]
    convertToMicron = dictData[datasetName]["conversion_to_um"]
    try:
        shuffle = dictData[datasetName]["shuffle"]
    except KeyError:
        print("Attention: Couldn't find shuffle vector. Not going to perform any shuffling.")
        shuffle = np.arange(pwd_sc_raw.shape[0])
    
    # convert to microns and shuffle
    pwd_sc = pwd_sc_raw*convertToMicron
    pwd_sc = shuffleMatrix2(pwd_sc, shuffle)
    
    return pwd_sc





def load_sc_data(dictData, datasetName):
    sc_label="doc"; sc_action="all"
    
    embryos2Load = dictData[datasetName]["embryos2Load"]
    rootFolder = os.path.dirname(embryos2Load)
    rootFN = os.path.basename(rootFolder)
    fname = "{}_label:{}_action:{}_SCmatrixCollated.npy"
    fname = fname.format(rootFN, sc_label, sc_action)
    fpath = os.path.join(rootFolder, "scHiMmatrices", fname)
    
    pwd_sc_raw = np.load(fpath)
    dictData[datasetName]["pwd_sc_raw"] = pwd_sc_raw
    
    # return dictData
    # scHiMmatrices/wt_doc_hiRes_17RTs_nc14_pyHiM_v0.5_label:doc_action:all_SCmatrixCollated.npy





def load_sc_data_info(dictData, datasetName):
    import json, re
    # sc_label="doc"; sc_action="all"
    
    embryos2Load = dictData[datasetName]["embryos2Load"]
    
    # load json file that contains all the paths to the data
    with open(embryos2Load) as json_file:
        dictEmbryo = json.load(json_file)
    
    keyDatasetName = list(dictEmbryo.keys())[0]
    print("Dataset: {}".format(keyDatasetName))
    
    listDataPaths = dictEmbryo[keyDatasetName]["Folders"]
    
    exp_emb = []
    
    for i in range(len(listDataPaths)):
        dataPath = listDataPaths[i]
        fName = "buildsPWDmatrix_3D_HiMscMatrix.npy"
        fName = os.path.join(dataPath, fName)
        pwd = np.load(fName)
        
        # re.search() checks for a match anywhere in the string
        strExp = re.search(r"(?<=Experiment_)\d+", dataPath).group()
        strEmb = re.search(r"(?<=Embryo_)\d+", dataPath)
        if strEmb == None:
            strEmb = re.search(r"\d+(?=_Embryo)", dataPath)
        strEmb = strEmb.group()
    
        # sInfo = "Found {} sc in {}\nExperiment {}, Embryo {}"
        # sInfo = sInfo.format(pwd.shape[2], dataPath, strExp, strEmb)
        # print(sInfo)
        
        exp_emb.extend([[strExp, strEmb]] * pwd.shape[2])
        
        # if i == 0:
        #     exp_emb = [[strExp, strEmb]] * pwd.shape[2]
        # else:
        #     exp_emb.extend([[strExp, strEmb]] * pwd.shape[2])
    
    return exp_emb

    


    
def load_SNDchan(dictData, datasetName, sc_label, sc_action):
    """
    Load SClabeledCollated.npy for given label and action. This is an 1D ndarray
    with 0/1, indicating if this nuclei matches the label/action criterion.

    Parameters
    ----------
    dictData : dict
        dict that holds the data. See pair_correlation.py for details.
    datasetName : str
        Name of the dataset.
    sc_label : str
        Name of the SND channel mask, e.g. "doc".
    sc_action : string
        Type of the selection, options are "all", "labeled", "unlabeled".

    Returns
    -------
    label_SNDchan : ndarray, 1D
        Gives the truth if nuclei was in selected mask. Same lenght and order as
        the full sc PWD matrix stack.

    """
    
    embryos2Load = dictData[datasetName]["embryos2Load"]
    rootFolder = os.path.dirname(embryos2Load)
    rootFN = os.path.basename(rootFolder)
    fn = "{}_label:{}_action:{}_SClabeledCollated.npy"
    fn = fn.format(rootFN, sc_label, sc_action)
    fn = os.path.join(rootFolder, "scHiMmatrices", fn)
    
    
    label_SNDchan = np.load(fn)
    # label_SNDchan = (sel_SNDchan == 1)
    print("Loaded SClabeledCollated.npy from ", fn)
    
    return label_SNDchan





def load_sc_data_Bintu(dictData, datasetName):
    # read file
    datadir = dictData[datasetName]["datadir"]
    fpath = os.path.join(datadir, datasetName+".csv")
    lines = [ln[:-1].split(',') for ln in open(fpath,"r")]
    # keep only data enties and reorganize the data
    keep = np.array(list(map(len,lines)))>1
    data = np.array([line for line,kp in zip(lines,keep) if kp][1:],dtype=float)
    chromosomes = data[:,0]
    nchr = len(np.unique(chromosomes))
    zxys = data[:,2:].reshape([nchr,-1,3])
    # print(zxys.shape)
    
    pwd_sc_raw = np.array(list(map(squareform,map(pdist,zxys))))
    pwd_sc_raw = np.transpose(pwd_sc_raw, axes=[1,2,0])
    
    dictData[datasetName]["pwd_sc_raw"] = pwd_sc_raw





def shuffleMatrix2(matrix, index):
    # if matrix is 2D, add a third dim
    numDims = len(matrix.shape)
    if numDims == 2:
        matrix = matrix[:,:,None]
    
    newSize = len(index)
    newMatrix = np.full((newSize,newSize,matrix.shape[2]), np.NaN)
    

    if not (newSize <= matrix.shape[0]):
        print("Error: shuffle size {} is larger than matrix dimensions {}".format(newSize, matrix.shape[0]))
        print("Shuffle: {} ".format(index))
        return newMatrix
    
    
    for i in range(newSize):
        for j in range(newSize):
            if (index[i] < matrix.shape[0]) and (index[j] < matrix.shape[0]):
                newMatrix[i, j, :] = matrix[index[i], index[j], :]
    
    if numDims == 2:
        newMatrix = newMatrix[:,:,0]
    
    return newMatrix





def get_num_detected_barcodes(pwd_sc):
    nBarcodes = pwd_sc.shape[0]
    nan_per_barcode = np.sum(np.isnan(pwd_sc), axis=1)
    barcodes_detected = np.sum(nan_per_barcode < nBarcodes, axis=0)
    
    return barcodes_detected





def get_subrange(pwd_sc, xy_start=None, xy_stop=None, xy_step=None):
    # zero-based indices of the matrix
    # xy_start = 0
    # xy_stop = 45
    # xy_step = 1
    
    if (xy_start is None):
        xy_start = 0
    if (xy_stop is None):
        xy_stop = pwd_sc.shape[0]
    if (xy_step is None):
        xy_step = 1
    
    subrange = np.arange(xy_start, xy_stop, xy_step)
    
    print("Num barcodes selected:", subrange.size)
    print(list(subrange))
    time.sleep(0.5) # make sure print command is executed before get_paircorr_sc_distMap
    
    sel_subrange = np.ix_(subrange, subrange, np.arange(pwd_sc.shape[2]))
    pwd_sc_sub = pwd_sc[sel_subrange]
    
    return pwd_sc_sub, "[{}:{}:{}]".format(xy_start, xy_stop, xy_step)





def filter_by_detected_barcodes(pwd_sc, minDetections):
    """
    Filter concatenated single cell PWD matrices based on the number of detected
    barcodes. Retruns a 1D bool ndarray to select matching PWD matrices.

    Parameters
    ----------
    pwd_sc : ndarray
        Concatenated sc PWD matrices.
    minDetections : float
        0            : Select all.
        in ]-1, 0[   : minDetections is percentile of distribution of number of detected barcodes.
        in ]0, 1[    : minDetections is fraction of detected barcodes.
        else         : minDetections is minimal number of detected barcodes.

    Returns
    -------
    sel : ndarray, bool.
        True for cells matching the filter criterium.

    """
    
    nBarcodes = pwd_sc.shape[0]
    
    # make sure the diagonal is set to NaN
    idx = np.arange(nBarcodes)
    pwd_sc[idx, idx, :] = np.NaN
    
        
    if (minDetections == 0):
        print("Selecting all cells.")
        time.sleep(0.5) # wait for print command to finish
        sel = [True] * pwd_sc.shape[2]
    else:
        nan_per_barcode = np.sum(np.isnan(pwd_sc), axis=1)
        barcodes_detected = np.sum(nan_per_barcode < nBarcodes, axis=0)
        if (-1 < minDetections < 0): # percentil
            cutoff = np.quantile(barcodes_detected, minDetections)
            print("Cutoff for number of detected barcodes used: {}".format(cutoff))
            time.sleep(0.5) # wait for print command to finish
            sel = barcodes_detected > cutoff
        elif (0 < minDetections < 1): # fraction of barcodes
            cutoff = np.ceil(nBarcodes * minDetections)
            print("Cutoff for number of detected barcodes used: {}".format(cutoff))
            time.sleep(0.5) # wait for print command to finish
            sel = barcodes_detected >= cutoff
        else:
            print("Cutoff for number of detected barcodes used: {}".format(minDetections))
            time.sleep(0.5) # wait for print command to finish
            sel = barcodes_detected >= minDetections
    
    return sel





def fill_missing(PWD_sc, pwd_sc_full_data=None, minMissing=None, maxMissing=None,
                 fillMode="mean", selRange=None, keepPattern=None):
    """
    Calculated population average from pwd_sc_full_data (mean, median or KDE).
    Fill in missing values in pwd_sc with this population average.
    Flatten the sc PWD maps and return the ones that match the number of missing barcodes.
    
    Parameters
    ----------
    PWD_sc : ndarray, numBarcodes x numBarcodes x numNuclei
        array with concatenated sc PWD maps
    pwd_sc_full_data : ndarray, numBarcodes x numBarcodes x numNuclei_all
        Use this array to calculated the population average. If None, use pwd_sc.
        If this is a 2D matrix (numBarcodes x numBarcodes), then use this instead
        of calculating the KDE.
    minMissing : int, optional
        DESCRIPTION. The default is None.
    maxMissing : int, optional
        DESCRIPTION. The default is None.
    fillMode : str, optional
        Control what missing values should be replaced with.
        Options are: mean, median, or KDE. The default is "mean".
    selRange : iterable of len 2, optional
        Control which barcodes of the PWD map should be selected.
        Use full map if None.
    keepPattern : list of 0 and 1. Will get expanded to PWD_sc.shape[2].
        Gives the pattern of which nuclei to keep. Default is None, which
        will keep all nuclei (that are not rejected due to minMissing or maxMissing).
        To keep every 2nd nucleus, use keepPattern=[0,1].

    Returns
    -------
    pwd_sc_lin : ndarray, shape (numNuclei_keep, numPWD)
        DESCRIPTION.
    missing_barcodes : ndarray, shape (numNuclei_keep,)
        DESCRIPTION.
    keep : ndarray, shape (numNuclei,)
        Boolean array which cells of the input are kept.

    """
    
    pwd_sc = PWD_sc.copy()
    fullIsKDE = False
    
    
    if (pwd_sc_full_data is None):
        print("Setting pwd_sc_full_data to pwd_sc")
        pwd_sc_full_data = pwd_sc.copy()
    elif (len(pwd_sc_full_data.shape)==2):
        fullIsKDE = True
    
    if (minMissing is None):
        minMissing = 0
    
    if (maxMissing is None):
        maxMissing = np.inf
    
    
    nBarcodes = pwd_sc.shape[0]
    nCells = pwd_sc.shape[2]
    if not fullIsKDE:
        nCells_full_data = pwd_sc_full_data.shape[2]
    
    if selRange is not None:
        a = int(selRange[0])
        b = int(selRange[1])
        print("Selecting barcodes [{}:{}] (numpy notation)".format(a,b))
        pwd_sc = pwd_sc[a:b, a:b, :]
        nBarcodes = pwd_sc.shape[0]
        # also take care of pwd_sc_full_data
        if fullIsKDE:
            pwd_sc_full_data = pwd_sc_full_data[a:b, a:b]
        else:
            pwd_sc_full_data = pwd_sc_full_data[a:b, a:b, :]
    
    
    # make sure the diagonal is set to NaN
    idx = np.arange(nBarcodes)
    pwd_sc[idx, idx, :] = np.NaN
    
    
    # get number of missing barcodes per cell
    det_per_barcode = np.sum(~np.isnan(pwd_sc), axis=1)
    missing_barcodes = np.sum((det_per_barcode == 0), axis=0)
    
    
    # fill missing values with mean, median, or KDE
    if (fillMode=="mean"):
        pwd_sc_fill = np.nanmean(pwd_sc_full_data, axis=2)
    elif (fillMode=="median"):
        pwd_sc_fill = np.nanmedian(pwd_sc_full_data, axis=2)
    elif ( (fillMode=="KDE") and (fullIsKDE) ):
        print("Using supplied KDE map")
        pwd_sc_fill = pwd_sc_full_data
    elif ( (fillMode=="KDE") and not (fullIsKDE) ):
        # get estiamte of PWDs that are not NaN
        fractNotNaN = np.sum(~np.isnan(pwd_sc_full_data)) / np.product(pwd_sc_full_data.shape)
        nCells_corrected = nCells_full_data * fractNotNaN
        if (nCells_corrected < 2000):
            cells2Plot = [True]*pwd_sc_full_data.shape[2]
        else:
            # target_num_cells = 200
            target_num_cells = 50000/nBarcodes
            step = round(nCells_corrected/target_num_cells)
            step = max(step, 1) # make sure step is >= 1
            txt = "Attention: Huge number of nuclei (n={}). Calculating KDE with step size: {}\n"
            print(txt.format(nCells_full_data, step))
            time.sleep(0.5) # wait for print command to finish
            cells2Plot = [False]*pwd_sc_full_data.shape[2]
            cells2Plot[0::step] = [True]*(1+(len(cells2Plot)-1)//step)
        pwd_sc_fill, _ = calculatesEnsemblePWDmatrix(pwd_sc_full_data, 1, cells2Plot, mode="KDE") # pwd_sc is in µm already, thus pixel size = 1
    
    
    for iCell in range(nCells):
        mask = np.isnan(pwd_sc[:,:,iCell])
        xy = np.where(mask)
        pwd_sc[xy[0],xy[1],iCell] = pwd_sc_fill[mask]
        # if iCell < 10:
        #     print("filled in {} values".format(xy[0].shape))
    
    pwd_sc_lin = get_mat_linear(pwd_sc) # shape=(nCells,nPWD)
    
    
    # expand keepPattern
    if (keepPattern is None):
        keepPattern = [1]
    else:
        print("Using keepPattern {}".format(keepPattern))
    lenKeep = len(keepPattern)
    repeats = np.ceil(nCells/lenKeep)
    keepPattern = keepPattern*int(repeats)
    keepPattern = (np.array(keepPattern)[0:nCells] == 1)
    
    
    keep1 = (missing_barcodes >= minMissing)
    keep2 = (missing_barcodes <= maxMissing)
    # sel = sel1 * sel2
    keep = keep1 * keep2 * keepPattern
    # print(keepPattern)
    
    pwd_sc_lin = pwd_sc_lin[keep,:]
    missing_barcodes = missing_barcodes[keep]
    
    return pwd_sc_lin, missing_barcodes, keep





def get_mat_linear(pwd_sc):
    """
    Returns a linearized version of concatenated sc PWD map.
    Only consider upper triangle of the PWD map.

    Parameters
    ----------
    pwd_sc : ndarray, 3D
        Concatenated sc PWD maps. nBarcodes x nBarcodes x nNuclei

    Returns
    -------
    pwd_sc_lin : ndarray, 2D
        Linearized form of pwd_sc. nNuclei x nPWD
    """
    
    numBarcodes = pwd_sc.shape[0]
    numPWD = int(numBarcodes*(numBarcodes-1)/2)
    numNuc = pwd_sc.shape[2]
    
    pwd_sc_lin = np.full((numNuc, numPWD), np.NaN)
    
    ii = 0
    for i in range(numBarcodes):
        for j in range(i+1, numBarcodes):
            pwd_sc_lin[:,ii] = pwd_sc[i,j,:]
            ii += 1
    
    return pwd_sc_lin





def get_mat_square(pwd_sc_lin):
    numNuc = pwd_sc_lin.shape[0]
    lenLin = pwd_sc_lin.shape[1]
    
    # n**2 - n - 2*lenLin
    
    numBarcodes = (1 + np.sqrt(1 + 4*2*lenLin))/2
    
    if (int(numBarcodes) - numBarcodes > 0.001):
        raise SystemExit("get_mat_square: Check shape of input array.")
    
    numBarcodes = int(numBarcodes)
    print("get_mat_square: numNuc={}, numBarcodes={}".format(numNuc, numBarcodes))
    time.sleep(0.5) # wait for print command to finish
    
    pwd_sc = np.full((numBarcodes, numBarcodes, numNuc), np.NaN)
    
    ii = 0
    for i in range(numBarcodes):
        for j in range(i+1, numBarcodes):
            pwd_sc[i,j,:] = pwd_sc_lin[:,ii]
            pwd_sc[j,i,:] = pwd_sc_lin[:,ii]
            ii += 1
    
    return pwd_sc





def get_similarity_sc_distMap(PWD, minRatioSharedPWD=0.2, distNorm=False,
                              mode="pair_corr", numShuffle=0):
    """
    Parameters
    ----------
    PWD : 3D ndarray (n x n x N)
        n barcodes, N cells. single cell pairwise distance maps.
    minRatioSharedPWD : float from [0,1], optional
        Fraction of PWDs that are detected in both cells to calculate the
        Pearson corr coef. The default is 0.2.
    distNorm : bool, or string, optional
        "diag": Normalize PWD map by subtracting the per-diagonal mean.
        "zscore": For each pair of barcodes, transform to zscore.
        The default is False.
    mode: string, optional
        Which distance or similarity measure to use: pair_corr, L1, L2, RMSD, cos, Canberra
        The default is "pair_corr"
    numShuffle: integer >= 0, optional
        Shuffle one of the single cell maps to get a "null hypothesis" distribution.
        Repeat numShuffle times. If numShuffle is zero, don't shuffle.

    Returns
    -------
    s : ndarray of floats
        values of the distance or similarity measure for pairs of nuclei that
        fullfil the minRatioSharedPWD criterion.
    o : overlap (=ratioSharedPWD) for the same pairs of nuclei as in s.
    p : list of strings giving the index of the pairs in s, o.
    """
    
    # parse numShuffle
    if numShuffle <= 0:
        doShuffle = False
        numIters = 1
    else:
        doShuffle = True
        numIters = numShuffle
    print("doShuffle {}, numIters {}".format(doShuffle, numIters))
    time.sleep(0.5) # wait for print command to finish
    
    # cnostruct a 2D matrix with the mean dist for each diagonal
    if (distNorm == "diag"):
        expected = np.full(PWD.shape[0:2], np.NaN)
        
        for i in range(1, PWD.shape[0]):
            e = np.nanmean(np.diagonal(PWD, offset=i))
            mask = np.diag(np.repeat(True, PWD.shape[0]-i), k=-i)
            expected[mask] = e
        PWD = PWD - expected[:,:,None] #create empty 3rd dim for expected
        # print(expected)
    elif (distNorm == "zscore"):
        avg = np.nanmean(PWD, axis=2)
        std = np.nanstd(PWD, axis=2)
        PWD = (PWD - avg[:,:,None])/std[:,:,None] # #create empty 3rd dim for avg and std
    elif (distNorm != False):
        print("distNorm={} not supported.".format(distNorm))
    
    
    # only consider half of the PWD map
    mask = np.tril(np.full(PWD.shape[0:2], 1, dtype=bool), k=-1)
    numPWD = np.sum(mask)
    numNuc = PWD.shape[2]
    
    PWD_lin = np.full((numPWD, numNuc), np.NaN)
    
    ii = 0
    for i in range(PWD.shape[0]):
        for j in range(i+1, PWD.shape[1]):
            PWD_lin[ii,:] = PWD[j,i,:]
            ii += 1
    
    # r = np.full(numNuc*(numNuc-1)//2, np.NaN)
    simty = []; overlp = []; pair = []
    
    
    for i in trange(numNuc):
        # sc1 = PWD[:,:,i][mask]
        sc1 = PWD_lin[:,i]
        sc1_isnan = np.isnan(sc1)
        
        # for j in range(i+1, PWD.shape[2]):
        for j in range(i+1, numNuc):
            # ii += 1
            # sc1 = PWD[:,:,i][mask]
            
            # sc2 = PWD[:,:,j][mask]
            sc2 = PWD_lin[:,j]
            sc2_isnan = np.isnan(sc2)
            
            
            # mask_not_nan = ((~sc1_isnan) & (~sc2_isnan))
            mask_not_nan  = ~np.logical_or(sc1_isnan, sc2_isnan)
            ratioSharedPWD = np.sum(mask_not_nan)/numPWD
            
            if ratioSharedPWD >= minRatioSharedPWD:
                sc1_not_nan = sc1[mask_not_nan]
                sc2_not_nan = sc2[mask_not_nan]
                s_collected = []
                
                np.random.seed(seed=424242) # make shuffle sequence reproducible
                for _ in range(numIters):
                    if doShuffle:
                        np.random.shuffle(sc2_not_nan) # shuffle in place
                        # sc2_not_nan = sc2[mask_not_nan][::-1]
                    
                    if (mode == "pair_corr"):
                        s = np.corrcoef(sc1_not_nan, sc2_not_nan)[0,1]
                    elif (mode == "L1"):
                        d = np.abs(sc1_not_nan - sc2_not_nan)
                        s = np.sum(d)
                    elif (mode == "L2"):
                        d = np.abs(sc1_not_nan - sc2_not_nan)
                        s = np.sum(d**2)**0.5
                    elif (mode == "RMSD"):
                        d = np.abs(sc1_not_nan - sc2_not_nan)
                        s = np.mean(d**2)**0.5
                    elif (mode == "cos"):
                        a = np.sum(sc1_not_nan * sc2_not_nan)
                        b = np.sqrt(np.sum(np.square(sc1_not_nan)))
                        c = np.sqrt(np.sum(np.square(sc2_not_nan)))
                        s = a/(b*c)
                    elif (mode == "Canberra"): #known to be very sensitive to small changes near zero
                        d = np.abs(sc1_not_nan - sc2_not_nan)
                        su = sc1_not_nan + sc2_not_nan
                        s = np.sum(d/su)
                    else:
                        print("Mode {} not supported. Returning NaN.".format(mode))
                        s = np.NaN
                    
                    s_collected.append(s)
                
                # simty.append(np.mean(s_collected))
                simty += s_collected
                overlp.append(ratioSharedPWD)
                pair.append("{}, {}".format(i, j))
                
    
    return np.array(simty), np.array(overlp), pair #, PWD





def get_insulation_score(PWDmatrix, sqSize, nansum=False, scNormalize=False):
    """
    Calculates something similar to the insulation score from a 2D pairwise distance matrix
    assumes that none of the entries in the PWD map is NaN
    see Crane et al., 2015, Nature. https://doi.org/10.1038/nature14450
    
    PWDmatrix   : ndarray, NxN, sc PWD map
    nansum      : ignore NaNs in the PWD map. does not correct for lower number of summands
    sqSize      : size of the square to consider
    scNormalize : optional, divide by sc nanmedian IS to normalize for overall compaction
    """
    
    numBarcodes = PWDmatrix.shape[0]
    
    IS = np.full(numBarcodes, np.NaN)
    
    for i in range(sqSize, numBarcodes-sqSize):
        square = PWDmatrix[i-sqSize:i, i+1:i+1+sqSize]
        
        if nansum:
            IS[i] = np.nansum(square)
        else:
            IS[i] = np.sum(square)
    
    if scNormalize:
        IS /= np.nanmean(IS)
    return IS





def get_Rg_from_PWD(PWDmatrix, minFracNotNaN=0.8):
    """
    Calculates the Rg from a 2D pairwise distance matrix
    while taking into account that some of the PWD might be NaN
    
    PWDmatrix:       ndarray, NxN
    minFracNotNaN:   require a minimal fraction of PWDs to be not NaN, return NaN otherwise
    
    for the math, see https://en.wikipedia.org/wiki/Radius_of_gyration#Molecular_applications
    """
    
    # check that PWDmatrix is of right shape
    if (PWDmatrix.ndim != 2):
        raise SystemExit("getRgFromPWD: Expected 2D input but got {}D.".format(PWDmatrix.ndim))
    if (PWDmatrix.shape[0] != PWDmatrix.shape[1]):
        raise SystemExit("getRgFromPWD: Expected square matrix as input.")
    
    # make sure the diagonal is NaN
    np.fill_diagonal(PWDmatrix, np.NaN)
    
    # get the number of PWDs that are not NaN
    numPWDs = PWDmatrix.shape[0]*(PWDmatrix.shape[0]-1)/2
    numNotNan = np.sum(~np.isnan(PWDmatrix)) / 2 # default is to compute the sum of the flattened array
    #print("numNotNaN", numNotNan)
    if (numNotNan/numPWDs < minFracNotNaN):
        return np.NaN
    
    # calculate Rg
    sq = np.square(PWDmatrix)
    sq = np.nansum(sq) # default is to compute the sum of the flattened array
    
    Rg_sq = sq / (2 * (2*numNotNan + PWDmatrix.shape[0])) # replaces 1/(2*N^2)
    
    Rg = np.sqrt(Rg_sq)
    
    return Rg





def get_mixed_mat(mat_upper, mat_lower):
    s_upper = mat_upper.shape
    s_lower = mat_lower.shape
    
    if not (s_upper == s_lower):
        raise SystemExit("Matrices are not of same size.")
    
    if not (s_upper[0] == s_upper[1]):
        raise SystemExit("Matrices are not square.")
    
    nBins = s_upper[0]
    mat_mix = np.zeros((nBins,nBins))
    mat_true = np.full((nBins,nBins), True)
    sel_upper = np.triu(mat_true, k=+1) # exclude diagonal
    sel_lower = np.tril(mat_true, k=-1)
    mat_mix[sel_upper] = mat_upper[sel_upper]
    mat_mix[sel_lower] = mat_lower[sel_lower]
    
    return mat_mix





def prepare_UMAP(dictData, selRange, minmaxMiss, steps, dictUMAP, keyDict,
                 datasetNames, listClasses, sFunctEns, listKeepPattern=None):
    
    fillMode = "KDE" # for Bintu data: set to KDE even if ensemble avg is median
    
    classID = []
    dictUMAP[keyDict] = {}
    
    if listKeepPattern is None:
        listKeepPattern = [None]*len(datasetNames)
    
    for i in range(len(datasetNames)):
        datasetName = datasetNames[i]
        minMiss, maxMiss = minmaxMiss[i]
        keepPattern = listKeepPattern[i]
        step = steps[i]
        
        pwd_sc = get_sc_pwd(dictData, datasetName)
        pwd_sc_ens_avg = dictData[datasetName][sFunctEns]
        
        pwd_sc_lin, numMissing, sel_UMAP = \
            fill_missing(pwd_sc, pwd_sc_ens_avg, minMiss, maxMiss, fillMode,
                         selRange=selRange, keepPattern=keepPattern)
        
        pwd_sc_lin = pwd_sc_lin[::int(step), :]
        
        nCells = pwd_sc_lin.shape[0]
        classID += [listClasses[i]] * nCells
        
        txt = "Found {} nuclei with {} to {} missing barcodes step size {}, for {}"
        txt = txt.format(nCells, minMiss, maxMiss, step, datasetName)
        print(txt); time.sleep(0.5) # wait for print command to finish
        
        dictUMAP[keyDict]["sel_UMAP_"+listClasses[i]] = sel_UMAP
        
        if (i==0):
            pwd_sc_lin_cat = pwd_sc_lin
            numMissing_cat = numMissing
        else:
            pwd_sc_lin_cat = np.vstack((pwd_sc_lin_cat, pwd_sc_lin))
            numMissing_cat = np.concatenate((numMissing_cat, numMissing), axis=0)
    
    classes, classNum = np.unique(classID, return_inverse=True) # classes will be sorted alphanumerically!
    
    # save variables to dict
    dictUMAP[keyDict]["fillMode"] = fillMode
    dictUMAP[keyDict]["datasetNames"] = datasetNames
    dictUMAP[keyDict]["classID"] = classID
    dictUMAP[keyDict]["classes"] = classes
    dictUMAP[keyDict]["classNum"] = classNum
    dictUMAP[keyDict]["minmaxMiss"] = minmaxMiss
    dictUMAP[keyDict]["steps"] = steps
    # dictUMAP[keyDict]["numMiss"] = numMiss
    # dictUMAP[keyDict]["sel_UMAP_aux"] = sel_UMAP_aux
    # dictUMAP[keyDict]["sel_UMAP_wt"] = sel_UMAP_wt
    dictUMAP[keyDict]["pwd_sc_lin_cat"] = pwd_sc_lin_cat
    dictUMAP[keyDict]["numMissing_cat"] = numMissing_cat
    
    # return
    return dictUMAP





def run_UMAP(dictUMAP, keyDict, random_state, custom_params=None):
    # get the vectorized sc pwd maps
    pwd_sc_lin_cat = dictUMAP[keyDict]["pwd_sc_lin_cat"]
    
    # set default UMAP parameters
    p = {}
    p["n_neighbors"] = 50     # default: 15
    p["min_dist"] = 0.1       # default: 0.1
    p["n_components"] = 2     # default: 2; number of dimension
    p["n_epochs"] = 500       # default: None (i.e. 200 for large datasets, 500 for small)
    p["metric"] = "canberra"  # default: euclidean; manhattan, euclidean, chebyshev, minkowski
                              # canberra, braycurtis, haversine
                              # mahalanobis, wminkowski, seuclidean
                              # cosine, correlation
    # p["random_state"] = 123456 # None, 42
    
    if custom_params is not None:
        for key in custom_params:
            p[key] = custom_params[key]
    
    
    reducer = umap.UMAP(
            n_neighbors=p["n_neighbors"],
            n_components=p["n_components"],
            metric=p["metric"],
            n_epochs=p["n_epochs"],
            min_dist=p["min_dist"],
            random_state=random_state,
            verbose=False
            )
    # embedding = reducer.fit_transform(pwd_sc_lin_cat)
    trans = reducer.fit(pwd_sc_lin_cat)
    embedding = trans.embedding_
    
    print(embedding.shape)
    
    # keep this embedding for later
    dictUMAP[keyDict]["trans"] = trans
    dictUMAP[keyDict]["embedding"] = embedding
    dictUMAP[keyDict]["p"] = {}
    dictUMAP[keyDict]["p"]["n_neighbors"] = p["n_neighbors"]
    dictUMAP[keyDict]["p"]["min_dist"] = p["min_dist"]
    dictUMAP[keyDict]["p"]["n_components"] = p["n_components"]
    dictUMAP[keyDict]["p"]["n_epochs"] = p["n_epochs"]
    dictUMAP[keyDict]["p"]["metric"] = p["metric"]
    dictUMAP[keyDict]["p"]["random_state"] = random_state
    
    return dictUMAP





def get_intra_inter_TAD(dictData, datasetName, cutoff_dist, bin_border,
                        sFunct, minNumPWD):
    # get concatenated sc PWD maps
    pwd_sc = get_sc_pwd(dictData, datasetName)
    pwd_sc[pwd_sc > cutoff_dist] = np.NaN
    
    nNuclei = pwd_sc.shape[2]
    
    
    # get intra- and inter-TAD distances, calculate the mean/median for each
    pwd_sc_intraTAD = pwd_sc[bin_border:, bin_border:, :]
    pwd_sc_intraTAD_lin = get_mat_linear(pwd_sc_intraTAD)
    num_notNaN_intraTAD = np.sum(~np.isnan(pwd_sc_intraTAD_lin), axis=1)
    
    if (sFunct == "mean"):
        intraTAD_dist = np.nanmean(pwd_sc_intraTAD_lin, axis=1)
    elif (sFunct == "median"):
        intraTAD_dist = np.nanmedian(pwd_sc_intraTAD_lin, axis=1)
    
    
    pwd_sc_interTAD = pwd_sc[0:bin_border, bin_border:, :]
    pwd_sc_interTAD_lin = pwd_sc_interTAD.transpose(2,0,1).reshape(nNuclei,-1)
    num_notNaN_interTAD = np.sum(~np.isnan(pwd_sc_interTAD_lin), axis=1)
    
    if (sFunct == "mean"):
        interTAD_dist = np.nanmean(pwd_sc_interTAD_lin, axis=1)
    elif (sFunct == "median"):
        interTAD_dist = np.nanmedian(pwd_sc_interTAD_lin, axis=1)
    
    
    # remove nuclei with low detections and NaNs for violinplot
    keep_intra = (num_notNaN_intraTAD>=minNumPWD)
    keep_inter = (num_notNaN_interTAD>=minNumPWD)
    
    return intraTAD_dist, interTAD_dist, keep_intra, keep_inter





def plot_representative_sc(datasetName, listShowSC, PWD_sc, pwd_KDE,
                           maxMissing=1, showAll=False, cmap=None,
                           vmin=0.3, vmax=0.8, vmin_sc=0.0, vmax_sc=3.0):
    
    if (cmap is None):
        # cmap = "PiYG"
        cmap = copy.copy(plt.get_cmap("PiYG")) # copy of cmap to supress warning
        cmap.set_bad(color=[0.8, 0.8, 0.8]) # set color of NaNs and other invalid values
    # vmin, vmax = 0.3, 0.9 # in µm
    # vmin_sc, vmax_sc = 0.0, 3.0 # in µm
    
    
    # make a copy of PWD_sc
    pwd_sc = PWD_sc.copy()
    
    nBarcodes = pwd_sc.shape[0]
    nDetBarcodes = get_num_detected_barcodes(pwd_sc)
    
    
    # make sure the diagonal is set to zero
    idx = np.arange(nBarcodes)
    pwd_sc[idx, idx, :] = 0
    
    
    if (showAll == False):
        # show only the sc PWDs from listShowSC and also the ensemble avg (KDE)
        fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(8, 6), dpi=300) # default (6.4, 4.8), 72dpi
        
        for i in range(5):
            ax = axs[i]
            currNuc = listShowSC[i]
            plot = ax.matshow(pwd_sc[:,:,currNuc], cmap=cmap, vmin=vmin_sc, vmax=vmax_sc)
            cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04,
                                ticks=[vmin_sc, vmax_sc], orientation="horizontal")
            # cbar.set_label("Distance (µm)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("{}".format(currNuc))
        
        ax = axs[5]
        # ax.axis("off")
        ax.set_visible(False)
        
        ax = axs[6]
        plot = ax.matshow(pwd_KDE, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04,
                            ticks=[vmin, vmax], orientation="horizontal")
        # cbar.set_label("Distance (µm)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ensemble")
        
        # fig.tight_layout() # makes things worse
        
    else:
        # show all PWD maps that have most barcodes detected
        sel = (nDetBarcodes>=nBarcodes-maxMissing)
        idx_sel = np.arange(pwd_sc.shape[2])[sel]
        
        print("{}: {} nuclei in total.\n".format(datasetName, pwd_sc.shape[2]))
        print("Found {} matchting nuclei.".format(sum(sel)))
        print(np.where(sel))
        
        N = np.sum(sel)+1
        nRows = (N*2/3)**0.5
        nCols = nRows * 3/2
        nCols, nRows = int(np.ceil(nCols)), int(np.ceil(nRows))
        
        fig, axs = plt.subplots(nrows=nRows, ncols=nCols, dpi=300) # default 72dpi
        for i, ax in enumerate(fig.axes):
            if (i==len(fig.axes)-1): # last subplot
                cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
                # cbar.set_label("Distance (µm)")
            if (i>=sum(sel)):
                # break
                ax.axis("off")
                continue
            currNuc = idx_sel[i]
            plot = ax.matshow(pwd_sc[:,:,currNuc], cmap=cmap, vmin=vmin_sc, vmax=vmax_sc)
            # cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1.5, 3])
            # # cbar.set_label("Distance (µm)")
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_title("{}".format(currNuc), fontsize=8, y=0.9, color="blue")
    
    
    return fig





def plot_map(ax, PWD_map, cmap="viridis", vmin=None, vmax=None, title="",
             cbar_draw=True, cbar_label="", cbar_ticks=None,
             cbar_orient="vertical"):
    
    if (cbar_orient is None):
        cbar_orient="vertical"
    
    plot = ax.matshow(PWD_map, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if (cbar_draw):
        cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04,
                            ticks=cbar_ticks, orientation=cbar_orient)
        cbar.set_label(cbar_label)
    else:
        cbar = None
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title, fontsize=12)
    
    return plot, cbar





def plot_map_tria(ax, PWD_map, cmap="viridis", vmin=None, vmax=None, aspect=1.0,
                  title="", cbar_draw=True, cbar_label="", cbar_ticks=None,
                  cbar_orient="vertical"):
    """
    Plot the triangular version of a square, symmetric matrix.

    Parameters
    ----------
    ax : matplotlib axis.
        The axis to plot in.
    PWD_map : ndarray. NxN
        Square matrix to be plotted.
    aspect : float, optional
        Controls the ratio of width vs height of the final plot.
        The default is 1, resulting in square pixels.
    cmap : matplotlib colormap, optional
        Colormap to use for plotting. The default is "viridis".
    vmin : float, optional
        Min for colormap. The default is None, sets to min of PWD_map.
    vmax : float, optional
        Max for colormap. The default is None, sets to max of PWD_map.
    title : string, optional
        Title for the plot. The default is "".
    cbar_draw : bool, optional
        Controls if a colorbar is drawn. The default is True.
    cbar_label : string, optional
        Label of the colorbar. The default is "".
    cbar_ticks : iterable of floats, optional
        Manual setting the position of ticks on the colorbar. The default is None.
    cbar_orient : string, optional
        Set orientation of the colorbar. The default is "vertical".

    Returns
    -------
    plot : TYPE
        DESCRIPTION.
    cbar : TYPE
        DESCRIPTION.
    """
    
    r_start = 0
    r_end = PWD_map.shape[0]
    r_len = r_end - r_start
    
    scale = 1 / np.sqrt(2)
    
    
    # rotate matrix using Affine2D, reference:
    # https://github.com/GangCaoLab/CoolBox/blob/master/coolbox/core/track/hicmat/plot.py
    # https://stackoverflow.com/a/50920567/8500469
    
    tr = transforms.Affine2D().translate(-r_start, -r_start) \
        .rotate_deg_around(0, 0, 45) \
        .scale(scale) \
        .translate(r_start + r_len/2, -r_len/2)
    
    plot = ax.matshow(PWD_map, cmap=cmap,
                      transform=tr+ax.transData,
                      extent=(r_start, r_end, r_start, r_end),
                      aspect=aspect, vmin=vmin, vmax=vmax)
    
    if (cbar_draw):
        cbar = plt.colorbar(plot, ax=ax, fraction=0.046/2*aspect, pad=0.04,
                            ticks=cbar_ticks, orientation=cbar_orient)
        cbar.set_label(cbar_label)
    else:
        cbar = None
    
    ax.set_ylim(r_start,r_len/2)
    ax.axis("off")
    
    if len(title)>0:
        ax.set_title(title)
    
    return plot, cbar




def get_pair_corr(PWD_sc, p_pc, doShuffle=True):
    # copy PWD_sc to work on a local copy
    pwd_sc = PWD_sc.copy()
    
    # get parameter
    cutoff_dist = p_pc["cutoff_dist"]
    inverse_PWD = p_pc["inverse_PWD"]
    minDetections = p_pc["minDetections"]
    step = p_pc["step"]
    minRatioSharedPWD = p_pc["minRatioSharedPWD"]
    mode = p_pc["mode"]
    numShuffle = p_pc["numShuffle"]
    
    # selet a subrange of barcodes
    pwd_sc, sSubR = get_subrange(pwd_sc)
    # pwd_sc, sSubR = get_subrange(pwd_sc, xy_start=30, xy_stop=72, xy_step=1)
    
    # filter cells based on number of detected barcodes per cell
    sel = filter_by_detected_barcodes(pwd_sc, minDetections)
    pwd_sc = pwd_sc[:,:,sel]
    
    # clip distances
    pwd_sc[pwd_sc>max(cutoff_dist)] = max(cutoff_dist)
    pwd_sc[pwd_sc<min(cutoff_dist)] = min(cutoff_dist)
    
    # invert PWD
    if inverse_PWD:
        pwd_sc = 1/pwd_sc
    
    # calculate pairwise similarity
    S_data, O_data, P_data = \
        get_similarity_sc_distMap(pwd_sc[:,:,::step], minRatioSharedPWD,
                                  mode=mode, distNorm=False, numShuffle=0)
    
    if doShuffle:
        S_shuffle, O_shuffle, P_shuffle = \
            get_similarity_sc_distMap(pwd_sc[:,:,::step], minRatioSharedPWD,
                                      mode=mode, distNorm=False, numShuffle=numShuffle)
    
    pair_corr = {}
    pair_corr["S_data"] = S_data
    pair_corr["O_data"] = O_data
    pair_corr["P_data"] = P_data
    if doShuffle:
        pair_corr["S_shuffle"] = S_shuffle
        pair_corr["O_shuffle"] = O_shuffle
        pair_corr["P_shuffle"] = P_shuffle
    pair_corr["sel"] = sel
    pair_corr["sSubR"] = sSubR
    
    return pair_corr






def plot_pair_corr(pair_corr, p_pc, color, datasetName, bins=None,
                   pair_corr_all=None, yLim=None, figsize=None):
    # get results
    S_data = pair_corr["S_data"]
    S_shuffle = pair_corr["S_shuffle"]
    sel = pair_corr["sel"]
    sSubR = pair_corr["sSubR"]
    
    
    # get parameter
    cutoff_dist = p_pc["cutoff_dist"]
    inverse_PWD = p_pc["inverse_PWD"]
    minDetections = p_pc["minDetections"]
    step = p_pc["step"]
    minRatioSharedPWD = p_pc["minRatioSharedPWD"]
    mode = p_pc["mode"]
    numShuffle = p_pc["numShuffle"]
    
    
    # figsize
    if figsize is None:
        figsize = (8,6)
    
    # bin location and width
    if bins is None:
        bins = np.linspace(-1,1,51) # None, np.linspace(-1,1,51)
    
    
    # settings for hist
    density = True
    histtype="stepfilled"
    
    
    # create figure
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=150) # default (6.4, 4.8), 72dpi
    ax = axs
    
    
    # plot line of the all-vs-all pair corr
    if pair_corr_all is not None:
        S_data_all = pair_corr_all["S_data"]
        
        h3 = ax.hist(S_data_all, bins=bins, density=density, alpha=0.75,
                     color="k", histtype="step", label="all vs all",
                     linestyle="-", linewidth=2)
    else:
        h3 = [np.zeros(len(bins)-1)] # ndarray of zeros to allow calculation of yLim later
    
    
    # plot hist of pair corr
    h1 = ax.hist(S_data, bins=bins, density=density, alpha=0.75, color=color,
                 histtype=histtype, label="data")
    
    
    # indicate shuffle as grey area
    shuffle_mean = np.nanmean(S_shuffle)
    shuffle_sdev = np.nanstd(S_shuffle)
    
    if yLim is None:
        yLim = 1.05*np.max([h1[0], h3[0]]) # yMax = np.max([h1[0], h2[0]])
    
    h2 = ax.fill_between([shuffle_mean-shuffle_sdev, shuffle_mean+shuffle_sdev],
                         [yLim, yLim], color=[0.4, 0.4, 0.4], label="shuffle")
    
    
    # write mean and sdev as text in plot
    c1 = h1[2][0]._original_facecolor
    c2 = h2._original_facecolor
    
    ax.axvline(np.nanmean(S_data), color="k", linestyle="-", linewidth=3)
    txt1 = "mean = {:.3f}"
    txt2 = "sdev = {:.3f}"
    ax.text(0.1, 0.95, txt1.format(np.nanmean(S_data)), color=c1,
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.90, txt2.format(np.nanstd(S_data)), color=c1,
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.85, txt1.format(np.nanmean(S_shuffle)), color=c2,
            transform=ax.transAxes, fontsize=12)
    
    if pair_corr_all is not None:
        handles, labels = ax.get_legend_handles_labels()
        order = [1,2,0]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    else:
        ax.legend()
    ax.set_ylim(0, yLim)
    ax.set_xlim(np.min(bins), np.max(bins))
    
    ax.set_xlabel("Pair correlation")
    if density:
        ax.set_ylabel("Probability density")
    else:
        ax.set_ylabel("Number of pairs")
    
    txt = "{} {}\n"
    txt += "mode={}, shuffle={}, minDetect={}\n"
    txt += "clip=[{},{}], inversePWD={}, step={}\n"
    txt += "minRatioSharedPWD={}, nCells={}, pairs={}"
    txt = txt.format(datasetName, sSubR,
                     mode, numShuffle, minDetections,
                     min(cutoff_dist), max(cutoff_dist), inverse_PWD, step,
                     minRatioSharedPWD, sum(sel), len(S_data))
    
    ax.set_title(txt, fontsize=12)
    
    # fig.tight_layout()
    
    return fig





def plot_scatter_Rg(ax, dictRg, datasetName, param_Rg, color=None, cross=None):
    
    # get parameter
    fillMode = param_Rg["fillMode"]
    cutoff = param_Rg["cutoff"]
    minFracNotNaN = param_Rg["minFracNotNaN"]
    
    # get radius of gyration
    r_gyration = dictRg[datasetName]
    ax.scatter(r_gyration[:,0], r_gyration[:,1], s=10, color=color)
    # ax.scatter(r_gyration[~sel,0], r_gyration[~sel,1], s=2, color=[0.8, 0.8, 0.8])
    # ax.plot([0,1], [0,1], "r-")
    
    # plot horizontal and vertical line through center
    if cross is not None:
        ax.axvline(cross[0][0], linestyle=cross[1], color=cross[2])
        ax.axhline(cross[0][1], linestyle=cross[1], color=cross[2])
    
    ax.set_xlabel("Rg TAD 1 (µm)")
    ax.set_ylabel("Rg doc-TAD (µm)")
    
    txt = "{}\nNaNs filled with {}\n"
    txt += "cutoff={}µm, minFracNotNaN={}"
    txt = txt.format(datasetName, fillMode, cutoff, minFracNotNaN)
    ax.set_title(txt, fontsize=12)
    
    ax.set_xlim(0.25, 0.45)
    ax.set_ylim(0.20, 0.40)
    ax.set_aspect("equal")
    
    
    # Pearson corr coef all data
    v0 = r_gyration[:,0]
    v1 = r_gyration[:,1]
    
    skip = np.isnan(v0+v1)
    
    pearson_corrcoef = np.corrcoef(v0[~skip], v1[~skip])[0,1]
    print("pearson_corrcoef {}: {:.6f}".format(datasetName, pearson_corrcoef))
    
    ax.text(0.1, 0.9, "r={:.2f}".format(pearson_corrcoef), transform=ax.transAxes)





def plot_scatter_UMAP(dictUMAP_data, cmap, ax=None, hideTicks=False):
    embedding = dictUMAP_data["embedding"]
    classNum = dictUMAP_data["classNum"]
    classes = dictUMAP_data["classes"]
    fillMode = dictUMAP_data["fillMode"]
    minmaxMiss = dictUMAP_data["minmaxMiss"]
    steps = dictUMAP_data["steps"]
    n_neighbors = dictUMAP_data["p"]["n_neighbors"]
    min_dist = dictUMAP_data["p"]["min_dist"]
    metric = dictUMAP_data["p"]["metric"]
    
    if "classNum_2" in dictUMAP_data:
        print("Using classNum_2.")
        classNum = dictUMAP_data["classNum_2"]
        classes = dictUMAP_data["classes_2"]
    
    if ax is None:
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=150) # default (6.4, 4.8), 72dpi
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=150) # default (6.4, 4.8), 72dpi
        fig.subplots_adjust(right=0.85) # make some space on the right for the cbar label
                                        # keep the same as for Fig. 2E
    else:
        fig = None
    
    plot = ax.scatter(embedding[:,0], embedding[:,1],
                      s=10, c=classNum, cmap=cmap, alpha=1.0)
    
    # ax.legend(handles=plot.legend_elements()[0], labels=list(classes))
    ax.legend(handles=plot.legend_elements()[0], labels=list(classes),
              loc="upper left", bbox_to_anchor=(1,1))
    
    
    txt = "{} (imputed with {})\n"
    txt += "minmaxMiss={}, step={}, n={}\n"
    txt += "umap: n_neigh={}, min_dist={}, metric={}"
    txt = txt.format(list(classes), fillMode,
                     minmaxMiss, steps, list(np.unique(classNum, return_counts=True)[1]),
                     n_neighbors, min_dist, metric)
    ax.set_title(txt, fontsize=12)
    
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    
    if (hideTicks):
        ax.set_xticks([])
        ax.set_yticks([])
    
    return fig




