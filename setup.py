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
sys.path.append("/home/bbelaouc/dMRIParcellation/inc/")
import h5py
import scipy
import numpy as np
import argparse
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP

# How to run:
# python run_Parcellation.py -i input -o output -t tractogram -tb tracto prefix -seed coordinate -Ex excluded -sm Similarity measures
#                    -NR number of regions -cvth coeficient of variance -nodif mask

Subject_id = [1,2,3,4,5,6,9,12,13,14,15]
for Sub in Subject_id:
    print Sub
    CurrentF="/user/bbelaouc/home/Data/WorkShop/Pre-processing/eddy_correction/S"+str(Sub)
    SaveF="/user/bbelaouc/home/Data/WorkShop/Results_Thesis/With_postprocessing"
    Nodiff_path="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S"+str(Sub)
    input_ = CurrentF+"/W"+str(Sub)+"_cgal.mat"
    Argsave = SaveF+"/Sub"+str(Sub)+"/"
    Argtractograms= CurrentF+"/tract"
    Argtract_name = "tract_"
    Argcoordinates = CurrentF+"/tract/fdt_coordinates.txt"
    Arglist= [1000]
    Argexcluded =CurrentF+"/Excluded_points.txt"
    Argnodif = Nodiff_path+"/bedpostx.bedpostX/nodif_brain_mask.nii.gz"
    ArgSM = ['Cosine']
    Argverbose, Argmerge = 1, 0

    coordinate = np.loadtxt(str(Argcoordinates), unpack=True, delimiter='\t', dtype=int).T # read the diffusion space coordinate of the seeds
    Cortex = h5py.File(input_, 'r') # load the details of the mesh, coordinate, faces, normal, mesh connecticity.
    vertices_plot = np.array(Cortex['Vertices']) # get the coordinate in the anatomy image
    normal_plot=[]
    if "VertNormals" in Cortex.keys():
        normal_plot = np.array(Cortex['VertNormals']) # get the normals in the anatomical space
    faces_plot=[]
    if "Faces" in Cortex.keys():
        faces_plot = np.array(Cortex["Faces"], dtype=int)  # get faces of the mesh in the anatomical space.
        if faces_plot.min() > 0:
            faces_plot = faces_plot - 1
    Connectivity=np.eye(np.max(np.shape(coordinate)),dtype=int)
    if "VertConn" in Cortex.keys():
        C = Cortex['VertConn'] # get the tess connectivity matrix
        D_conenct = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))#
        Connectivity = np.array(D_conenct.todense(), np.int8)
        del D_conenct, C, Cortex # delete unused dat
    Excluded_seeds=[] # default excluded seeds
    if Argexcluded:
        Excluded_seeds = np.loadtxt(Argexcluded, dtype=int) # get the list of the excluded seeds
    ################ Parcellation starts here #########################################
    Verbose = False # by default dont display any results
    if Argverbose:
   	Verbose	= True # display results

    cvth=np.Inf # by default variation coefficient is set to infinity i.e is not included in the stoping criteria
    Argcv = None
    if Argcv:
   	cvth = Argcv # default threshold used to stop merging regions with low homogeneity

    Regions	= [len(coordinate[:,0])-len(Excluded_seeds)] # default number of regions
    #if Arglist:
    Regions = Arglist #[int(item) for item in Arglist.split(',')]

    SM = ['Cosine'] # Default similarity measure, cosine similarity
    #if ArgSM:
    #	SM = [item for item in ArgSM.split(',')] # list conatining the wanted similarity measures
    merge = 2
    if Argmerge is not None:
   	merge = Argmerge

    Parcel = CSP(Argtractograms, Argtract_name, Argsave, Argnodif, Verbose, merge) # initialize the parcellation by specifying the different paths
    Mesh_plot = RP.Mesh(vertices_plot, faces_plot, normal_plot) # define the mesh to be used to generate the vtk file
    del vertices_plot, faces_plot, normal_plot
    Parcel.Parcellation_agg(coordinate, Connectivity, Excluded_seeds, Regions, SM, Mesh_plot, cvth) # run the parcellation algorithm
