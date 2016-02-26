#!/usr/bin/env python
# -*- coding: utf-8 -*-
#####################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
#
# Whole brain parcellation algorithm based on Mutual Nearest Neighbors.


# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
# # INRIA Sophia Antipolis Méditerranée # #
######################################################################################
import sys
sys.path.append("./inc")
import h5py
import scipy
import numpy as np
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP
##########

CurrentF = "./test"
path_tractogram = CurrentF+"/tract"  # Folder which contains the tractograms
Prefix_name = 'tract_'		# Prefix of the tractogram file name
save_path = CurrentF+"/parcellation"      # Folder in which the results will be saved
coord_path = CurrentF+"/tract/fdt_coordinates.txt"  # The file that contain the coordinates of the seeds
cortex = CurrentF+"/W1_cgal.mat"                   # The file (Mat format) that contain the mesh details (coordinates, normal, mesh connectivity, Faces)
excluded = CurrentF+"/Excluded_points.txt"         # The seeds that will be excluded from the computation. In this case; the Thalamus
coordinate = np.loadtxt(coord_path, unpack=True, delimiter='\t', dtype=float).T  # read coord
Excluded_seeds = np.loadtxt(excluded, unpack=True, delimiter='\t', dtype=int)  # read coord
Cortex = h5py.File(cortex, 'r')
Vertices = np.array(Cortex['Vertices']).T
Normal = np.array(Cortex['VertNormals']).T
Faces = np.array(Cortex["Faces"], dtype=int).T  # used t
C = Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity = np.array(D.todense())  # mesh connectivity matrix
del D, C, Cortex
vertices_plot = Vertices
faces_plot = Faces
normal_plot = Normal
SM_method = ['Cosine']  #,'Tanimoto','Motyka','Ruzicka','Roberts']  # The similarity measures that will be used.
Regions = [1000, 800, 600, 400, 200, 100]  # number of regions, used to stop merging big regions
cvth = np.Inf  # parameter used to stop merging regions.
Verbose = True
nodif_path = CurrentF+"/nodif_brain_mask.nii.gz"  # mask used to reduce the required memory. The tractogram's voxels outside the mask are zeros.
Par = CSP(path_tractogram, Prefix_name, save_path, nodif_path, Verbose)  # prepare to read tractograms
Mesh_plot = RP.Mesh(vertices_plot.T, faces_plot.T, normal_plot.T)  # prepare mesh to visualize
Par.Parcellation_agg(coordinate, Connectivity, Excluded_seeds, Regions, SM_method, Mesh_plot, cvth)  # run parcellation
