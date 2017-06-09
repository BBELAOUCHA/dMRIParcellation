# Mutual Nearest Neighbor dMRI-based parcellation
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d82ad6541e214a04b3fc5f142cfa9cbf)](https://www.codacy.com/app/BBELAOUCHA/dMRIParcellation?utm_source=github.com&utm_medium=referral&utm_content=BBELAOUCHA/dMRIParcellation&utm_campaign=badger)
[![Build Status](https://travis-ci.org/BBELAOUCHA/dMRIParcellation.svg?branch=master)](https://travis-ci.org/BBELAOUCHA/dMRIParcellation)
[![codecov](https://codecov.io/gh/BBELAOUCHA/dMRIParcellation/branch/master/graph/badge.svg)](https://codecov.io/gh/BBELAOUCHA/dMRIParcellation)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d82ad6541e214a04b3fc5f142cfa9cbf)](https://www.codacy.com/app/BBELAOUCHA/dMRIParcellation?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BBELAOUCHA/dMRIParcellation&amp;utm_campaign=Badge_Grade)

dMRIParcellation is a python toolbox for parcellating whole cortical surface using
diffusion Magnetic Resonanse Imaging (dMRI) information and the Mutual Nearest 
Neighbor condition.

The parcellation algorithm needs the mesh connectivity, the tractograms of the
seeds (vertices of the mesh) in NIfTi format ".nii.gz", and the coordinate of 
the seeds in the diffusion space in ".txt" file. The file name of tractogram,
of seed i with coordinate (x,y,z), is be in the format of the
probabilistic tractography of FSL i.e. file_prefix_x_y_z.nii.gz.

# Requirements
1-h5py(>=2.6.0)

2-scipy(>=0.17.1)

3-numpy(>=1.11.3)

4-nibabel(>=2.1.0)

# Cite

Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface 
Parcellation via dMRI Using Mutual Nearset Neighbor Condition”, International
Symposium on Biomedical Imaging, Apr 2016, Prague, Czech Republic. 2016.


# Author

Belaoucha Brahim 

# Installation
python setup.py install

# Parcellating whole cortex
MNNparcellation is the command used to parcellate the cortical surface. Check 
the file "./details/how_to_use.pdf" for more details or MNNparcellation --help.

# Example
An example of parcellating a part of temporal lobe can be obtained by running:

sh example_parcellation.sh

The results are saved in ./data/Results/
