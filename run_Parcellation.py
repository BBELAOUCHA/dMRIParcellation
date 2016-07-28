#!/usr/bin/env python
# -*- coding: utf-8 -*-
#####################################################################################
#####################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, you have to cite:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface Parcellation via dMRI Using Mutual
#    Nearset Neighbor Condition”, International Symposium on Biomedical Imaging: From Nano to Macro, Prague,
#    Czech Republic. pp. 903-906, April 2016.
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
#   International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), Utrecht, Netherlands, September 2015.
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################
######################################################################################

import sys
sys.path.append("./inc/")
import h5py
import scipy
import numpy as np
import argparse
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP

# How to run:
# python run_Parcellation.py -i input -o output -t tractogram -tb tracto prefix -seed coordinate -Ex excluded -sm Similarity measures
#                    -NR number of regions -cvth coeficient of variance -nodif mask


parser = argparse.ArgumentParser()  # Parse the input to the variables
parser.add_argument('-i', action="store", dest='input') # input file "mat" file containing Faces, Normal, Vertices, VertConn (mesh connectivity)
parser.add_argument('-o', action="store", dest='save') # folder where to save the resulting parcellation in vtk format
parser.add_argument('-t', action="store", dest='tractograms') # folder that contains the tractograms of the seeds in nifti format ".nii.gz"
parser.add_argument('-tb', action="store", dest='tract_name') # prefix of the tractogram name (tractogram is assumed to have the follwong name tract_name_x_y_z.nii.gz)
parser.add_argument('-seed', action="store", dest='coordinates') # path to the ascii file that contains the coordinates of seeds in the diffusion space
parser.add_argument('-Ex', action="store", dest='excluded') # path to file that contain the list of seeds to be excluded from the parcellation
parser.add_argument('-sm', '--SM', help='delimited list input', type=str) # list of similarity measures to be used in the parcellation, see ref.2
parser.add_argument('-NR', '--list', help='delimited list input', type=str) # list of the number of regions, it is used to stop merging big regions, see ref.1
parser.add_argument('-cvth', action="store", dest='cv', type=float) # variation coefficient is used to stop merging regions with low homogeneity
parser.add_argument('-nodif', action="store", dest='nodif') # path to the brain mask, the parcellation algorithm consider only voxels inside the brain mask
parser.add_argument('-v', action="store", dest='verbose', type=int) # parameter used to enable results display
parser.add_argument('-m', action="store", dest='merge', type=int) # parameter used to enable results display
Arg = parser.parse_args() # read different parameters
coordinate = np.loadtxt(str(Arg.coordinates), unpack=True, delimiter='\t', dtype=int).T # read the diffusion space coordinate of the seeds
Cortex = h5py.File(str(Arg.input), 'r') # load the details of the mesh, coordinate, faces, normal, mesh connecticity.
vertices_plot = np.array(Cortex['Vertices']) # get the coordinate in the anatomy image
normal_plot=[]
if "VertNormals" in Cortex.keys():
    normal_plot = np.array(Cortex['VertNormals']) # get the normals in the anatomical space
faces_plot=[]
if "Faces" in Cortex.keys():
    faces_plot = np.array(Cortex["Faces"], dtype=int)  # get faces of the mesh in the anatomical space.
    faces_plot = faces_plot - 1
Connectivity=np.eye(np.max(np.shape(coordinate)),dtype=int)
if "VertConn" in Cortex.keys():
    C = Cortex['VertConn'] # get the tess connectivity matrix
    D_conenct = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))#
    Connectivity = np.array(D_conenct.todense(), np.int8)
    del D_conenct, C, Cortex # delete unused dat
Excluded_seeds=[] # default excluded seeds
if Arg.excluded:
    Excluded_seeds = np.loadtxt(Arg.excluded, dtype=int) # get the list of the excluded seeds
################ Parcellation starts here #########################################
Verbose = False # by default dont display any results
if Arg.verbose:
	Verbose	= True # display results

cvth=np.Inf # by default variation coefficient is set to infinity i.e is not included in the stoping criteria
if Arg.cv:
	cvth = Arg.cv # default threshold used to stop merging regions with low homogeneity

Regions	= [len(coordinate[:,0])-len(Excluded_seeds)] # default number of regions
if Arg.list:
	Regions = [int(item) for item in Arg.list.split(',')]

SM = ['Cosine'] # Default similarity measure, cosine similarity
if Arg.SM:
	SM = [item for item in Arg.SM.split(',')] # list conatining the wanted similarity measures
merge = 2
if Arg.merge is not None:
	merge = Arg.merge

Parcel = CSP(Arg.tractograms, Arg.tract_name, Arg.save, Arg.nodif, Verbose, merge) # initialize the parcellation by specifying the different paths
Mesh_plot = RP.Mesh(vertices_plot, faces_plot, normal_plot) # define the mesh to be used to generate the vtk file
del vertices_plot, faces_plot, normal_plot
Parcel.Parcellation_agg(coordinate, Connectivity, Excluded_seeds, Regions, SM, Mesh_plot, cvth) # run the parcellation algorithm
