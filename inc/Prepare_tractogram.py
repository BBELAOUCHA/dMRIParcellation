# -*- coding: utf-8 -*-
#####################################################################################
#
# This code is used to prepare the parcellation of the cortical surface from dMRI information
# (tractograms in nii.gz files) using the Mutual Nearest Neighbor Condition "see ref."

# There are 4 function in the Parcellation_data class:
#  1-Read_Tracto: read and return the tractogram in vector form
#  2-Read_tractograms: read and return all tractogram in vector form.
#  3-Detect_void_tracto: detect viod tractogram (sum <3*nbr_samples)
#  4-Replace_void_tracto: replace the viod tractograms with the nearest non viod tractograms
#  5-Logit_function: compute the logit function of the tractogram (probability of structural connectivity)
#####################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, you have to cite 2 of the following:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface Parcellation via dMRI Using Mutual
#Nearset Neighbor Condition”, International Symposium on Biomedical Imaging, Apr 2016, Prague, Czech Republic. 2016.

# Brahim Belaoucha and Théodore Papadopoulo, “Comparision of dMRI-based cortical surface parcellation
# with different similarity measures”,  Submitted, 2016.

# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
#   International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################

from copy import deepcopy
import numpy as np
import nibabel as nl


class Parcellation_data():

	# class of operations on the tractogram images
    def __init__(self, tract_folder_path, tract_name, mesh, nodif_mask):

	self.tract_path = tract_folder_path  # path to tractogram
	self.tract_name = tract_name  # prefix tractogram
	self.mesh = mesh	 # mesh details coordinates and tess connectivity
	self.Similarity_Matrix = np.eye(len(mesh.vertices[:, 1]))   # matrix contains similarity measure values
	self.excluded = []  # excluded seeds
	self.nodif_mask = nodif_mask	 # path to the brain's mask, used to reduce the memory since all voxels of the tractograms outside the mask are0
	MASK = nl.load(self.nodif_mask).get_data()
	M = np.array(MASK.reshape(-1))
	self.non_zeroMask = np.array(np.nonzero(M)[0])  # get the voxels of only the mask.
	del MASK, M

    def Read_Tracto(self, V): # function used to read and return the tractogram in 3D
	# read the nii.gz tractogram files one by one
	x, y, z = V # extract the x,y,z coordinate
	filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
	return nl.load(filename).get_data() # read and return the tractogram in 3D

    def Read_tractograms(self, V):
	#read all the tractograms used in the cluster
	self.tractograms = [] # array contains the tractograms in vector form
	for i in range(len(V[:, 0])): # loop over the coordinates
	   x, y, z = V[i, :] # read the ith coordinate (x,y,z)
	   filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
	   A = nl.load(filename).get_data() # read the tractogram in nifti format (.nii.gz)
	   T1 = A.reshape(-1) # from 3D to vector form
	   T2 = T1[np.array(self.non_zeroMask)] # read only the voxels inside the brain mask
	   self.tractograms.append(T2) # add the ith tractogram
	return self.tractograms # return the tractograms

    def Detect_void_tracto(self):	 # detecte the void tractograms
	# zero void is defined as the tracto that is less than 3*nbr_samples.
	self.zero_tracto = []	 # will contain void tractogram (tractograms with sum < 3*n_samples)
	self.nonzero_tracto = []	 # will contain the non void tractogram
	self.tractograms = []
	vertices = list(self.mesh.vertices)# the coordinate of the seeds in the diffusion space
	for i in range(len(vertices)): # loop over the coordinates
            seed = vertices[i] # extract the x,y,z coordinate
            T = self.Read_Tracto(seed)	 # read the tractogram
            self.nbr_sample = np.max(T.reshape(-1)) # if all voxel = 0, nbr_sample = 0
            Sm = np.sum(T)
            if (Sm <= self.nbr_sample*3):	 # detect void tractogram. It defined as the tractogram that has a sum
                self.zero_tracto.extend([i]) # less then 3* number of samples
            else:
		self.nonzero_tracto.extend([i]) # add the ith seed to the non viod tractogram
		T1 = T.reshape(-1)  # from 3D to a vector form
		T2 = T1[np.array(self.non_zeroMask)] # read only the voxels that are inside the (brain) mask
		self.tractograms.append(T2) # add the tractogram into an array of arrays

    def Replace_void_tracto(self):  # replace the void tractograms by the nearest neighbor non void

	self.replacement = {} 	# dictionaty contain the (key:value) the void tracto and its correspedning
	for k in self.zero_tracto:  # nearest non void tracto/ loop over void tracto
	   neighbors = [k]	 # initialize the neighbors of k
	   visited_neighbors = deepcopy(self.zero_tracto)  # already visited seeds
	   while (len(neighbors) > 0):	 # loop over the neighbors
		visited_neighbors.extend(neighbors)
		neighbors = np.unique(np.where(self.mesh.connectivity[neighbors, :] == 1)[1])	 # get the neighbors from the tess connectivity matrix
		A = set(neighbors)
		B = A.difference(set(visited_neighbors))	 # only non visited seeds
		valid_neighbors = list(B)
		if len(valid_neighbors) > 0:	 # if there are non void neighbors
                    distances = np.zeros(len(valid_neighbors))	 # vector contains the euclidian distance
                    distances = []
                    for neighbor in valid_neighbors:
                        distances.append(np.linalg.norm(self.mesh.vertices[k, :]-self.mesh.vertices[neighbor, :]))	 # distance
                    best_neighbor = valid_neighbors[np.array(distances).argmin()]	 # chose the non void tracto which is the closest in euclidian distance
                    self.replacement[k] = best_neighbor
                    break	 # exit the while , go the next void tractogram

    def Logit_function(self): # function used to transfer the tractogram from prbability space to logit space

        self.tractograms_logit=[]
        zero_value, one_value = 1e-6,0.9999999
        nbr_non_zero=len(self.nonzero_tracto)
        for i in range(nbr_non_zero):
            tracto = np.array(self.tractograms[i])
            tracto = tracto/np.max(tracto)
            ind_zero = np.where(tracto == 0)[0]
            ind_one = np.where(tracto == 1)[0]
            tracto[np.array(ind_zero)] = zero_value
            tracto[np.array(ind_one)] = one_value
            logit_p=np.add(np.log(tracto), -np.log(1-tracto))  # logit(p)=ln(p/(1-p))
            self.tractograms_logit.append(logit_p)
        self.tractograms = self.tractograms_logit # replace the probability space by the logit space
