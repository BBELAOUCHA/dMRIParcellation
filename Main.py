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

# this peace of code shows how you can use parcellation algorithm on python

# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
# # INRIA Sophia Antipolis Méditerranée # #
######################################################################################
import sys
sys.path.append("/home/bbelaouc/dMRIParcellation/inc/")
import h5py
import scipy
import numpy as np
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP
##########

path_tractogram="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S1/tract"
Prefix_name='tract_'
save_path="/home/bbelaouc/Data/WorkShop/Parcellation2/Cosine/S1"
coord_path="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S1/tract/fdt_coordinates.txt"
cortex="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S1/W1_cgal.mat"
excluded="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S1/Excluded_points.txt"
coordinate=np.loadtxt(coord_path,unpack=True,delimiter='\t',dtype=float).T#read coord
Excluded_seeds=np.loadtxt(excluded,unpack=True,delimiter='\t',dtype=int)#read coord
Cortex = h5py.File(cortex,'r')
Vertices=np.array(Cortex['Vertices']).T
Normal=np.array(Cortex['VertNormals']).T
Faces=np.array(Cortex["Faces"],dtype=int).T # used t
C=Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity=np.array(D.todense()) # mesh connectivity matrix
del D, C, Cortex
vertices_plot=Vertices
faces_plot=Faces
normal_plot=Normal
SM_method=['Cosine','Tanimoto','Motyka','Ruzicka','Roberts']
Regions=[1000,800,600,400,200,100]# number of regions, used to stop merging big regions
nbr_sample = 6000
cvth=10000# parameter used to stop merging regions that have high variance of the SM
nodif_path="/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S1/bedpostx.bedpostX/nodif_brain_mask.nii.gz"
Par=CSP(path_tractogram,Prefix_name,save_path,nodif_path,nbr_sample)# prepare to read tractograms
Mesh_plot=RP.Mesh(vertices_plot.T,[],faces_plot.T,normal_plot.T)# prepare mesh to visualize
Par.Parcellation_agg(coordinate,Connectivity, Excluded_seeds,Regions,SM_method,Mesh_plot,cvth)# main code to parcellate
