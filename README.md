# Mutual Nearest Neighbor dMRI-based parcellation
[![Build Status](https://travis-ci.org/BBELAOUCHA/dMRIParcellation.svg?branch=master)](https://travis-ci.org/BBELAOUCHA/dMRIParcellation)
[![codecov](https://codecov.io/gh/BBELAOUCHA/dMRIParcellation/branch/master/graph/badge.svg)](https://codecov.io/gh/BBELAOUCHA/dMRIParcellation)

dMRIParcellation is a python toolbox for parcellating whole cortical surface using
diffusion Magnetic Resonanse Imaging (dMRI) information and the Mutual Nearest 
Neighbor condition.

The parcellation algorithm needs the mesh connectivity, the tractograms of the
seeds (vertices of the mesh) in NIfTi format ".nii.gz", and the coordinate of 
the seeds in the diffusion space in ".txt" file. The file name of tractogram,
of seed i with coordinate (x,y,z), is be in the format of the
probabilistic tractography of FSL i.e. file_prefix_x_y_z.nii.gz.

# Requirements
1-h5py

2-scipy

3-numpy

4-nibabel

# Cite

Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface 
Parcellation via dMRI Using Mutual Nearset Neighbor Condition”, International
Symposium on Biomedical Imaging, Apr 2016, Prague, Czech Republic. 2016.


# Author

Belaoucha Brahim 

Papadopoulo Théodore


# Installation
python setup.py install

# Parcellating whole cortex
MNNparcellation is the command used to parcellate the cortical surface. Check 
the file "./details/how_to_use.pdf" for more details or MNNparcellation --help.

# Example
An example of parcellating a part of temporal lobe can be obrained by running:

sh example_parcellation.sh

The results are saved in ./data/Results/
