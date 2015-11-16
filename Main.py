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
# Arguments:
#          -i : Input to the cortical surface '.mat' file containing the Vertices, Faces, and Normals
#          -o : Path to the save results
#          -t : Path to the tractograms '.nii.gz' files
#          -tb: The prefix of the tracto's name: name_x(i)_y(i)_z(i).nii.gz
#          -seed: Path to the seeds coordinate in the diffusion space.
#          -NR: Requested number of regions list r1,r2,r3,......,rn
#          -Sth: Path to the smoothed cortical surface
# NOTE: number of iteration is fixed to 100.
# If you use this code, you have to cite:
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
# Proceeding of International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical surface parcellation via dMRI using Mutual Nearset
# Neighbor condition”,  Submitted, 2015.
# Brahim Belaoucha and Théodore Papadopoulo, “Comparision of dMRI-based cortical surface parcellation
# with different similarity measures”,  Submitted, 2015.

# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
# # INRIA Sophia Antipolis Méditerranée # #
######################################################################################
import sys
sys.path.append("./inc/")
import h5py
import scipy
import numpy as np
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP
##########

path_tractogram='/user/bbelaouc/home/Data/WorkShop/Pre-processing/CGAL/LHS2/tract/'
Prefix_name='tract_'
save_path='/user/bbelaouc/home/Data/WorkShop/Pre-processing/CGAL/LHS2/Results/'
coord_path='/user/bbelaouc/home/Data/WorkShop/Pre-processing/CGAL/LHS2/fdt_coordinates_fsl.txt'
cortex="/user/bbelaouc/home/Data/WorkShop/Pre-processing/CGAL/LHS2/lhS2.mat"
coordinate=np.loadtxt(coord_path,unpack=True,delimiter='\t',dtype=float).T#read coord
Cortex = h5py.File(cortex,'r')
Vertices=np.array(Cortex['Vertices']).T
Normal=np.array(Cortex['VertNormals']).T
Faces=np.array(Cortex["Faces"],dtype=int).T # used t
C=Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity=np.array(D.todense())
del D, C, Cortex
Excluded_seeds=[]#range(4000)
vertices_plot=Vertices
faces_plot=Faces
normal_plot=Normal
Regions=[500]
SM_method='Cosine'
cvth=10
Par=CSP(path_tractogram,Prefix_name,save_path)
Mesh_plot=RP.Mesh(vertices_plot.T,[],faces_plot.T,normal_plot.T)
Par.Parcellation_agg(coordinate,Connectivity, Excluded_seeds,Regions,SM_method,Mesh_plot,cvth)
