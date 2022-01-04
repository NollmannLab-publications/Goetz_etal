# pyHiM, single-cell

This repository contains the code and data needed to generate the figures for Götz *et al*.

## Setup

1. Make sure you have a running installation of pyHiM and the path `/path/to/pyHiM/src/matrixOperations` is in your `$PYTHONPATH`
2. Install UMAP
   1. `pip install umap-learn`
3. Currently, the script is intended for interactive execution and does not accept any arguments. Make sure the `dictData_small_211107.npy` is in the same folder as `plots_for_paper.py`

## plots_for_paper.py

This is the main script that contains most of the code needed to generate the figures.

## functions_paper.py

## dictData_small_211107.npy

This file contains all the necessary sc pairwise distance maps and auxiliary data to generate the figures.

The file can be downloaded from Zenodo under DOI: 10.5281/zenodo.5815196

Datasets available:
1. Dm, doc locus, low resolution, nc11/12
2. Dm, doc locus, low resolution, nc14
3. Dm, doc locus, high resolution, nc11/12
4. Dm, doc locus, high resolution, nc14
5. HCT116_chr21-34-37Mb_untreated
6. HCT116_chr21-34-37Mb_6h auxin

Datasets 5 and 6 are taken from [Bintu et al. 2018](https://doi.org/10.1126/science.aau1783).
 

