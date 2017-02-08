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
#  5-Logit_function: compute the logit function of the tractogram (probability of structural
# connectivity)
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
#    Nearset Neighbor Condition”, International Symposium on Biomedical Imaging: From Nano to Macro, Prague,
#    Czech Republic. pp. 903-906, April 2016.
# Brahim Belaoucha and Théodore Papadopoulo,“MEG/EEG reconstruction in the reduced source space”, in
#   International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), Utrecht,
#    Netherlands, September 2015.
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
	if nodif_mask != '':
           MASK = nl.load(self.nodif_mask).get_data()
	   M = np.array(MASK.reshape(-1))
	   self.non_zeroMask = np.array(np.nonzero(M)[0])  # get the voxels of only the mask.
	   del MASK, M

    def Repeated_Coordinate(self, V):

        self.non_duplicated={}
        for i in np.arange(np.max(np.shape(V))):
            v=V[i,:]
            B=np.subtract(V, v)
            C=np.add(np.add(np.absolute(B[:,0]),np.absolute(B[:,1])),np.absolute(B[:,2]))
            D=np.where(C == 0)[0]
            for j in range(len(D)):
                if D[j] not in self.non_duplicated.keys():
                        self.non_duplicated[D[j]] = i
        Q2=np.unique(self.non_duplicated.values())
        self.coordinate_2_read=[]
        self.ind_2_loaded={}
        for i in range(len(Q2)):
            self.ind_2_loaded[Q2[i]] = i
            self.coordinate_2_read.append(V[Q2[i],:])
        #return self.non_duplicated,self.ind_2_loaded[self.non_duplicated]


    def Read_tracto(self, V): # function used to read and return the tractogram in 3D
	# read the nii.gz tractogram files one by one
	x, y, z = V # extract the x,y,z coordinate
	filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
	return nl.load(filename).get_data() # read and return the tractogram in 3D

    def Read_tractograms(self, V):
	#read all the tractograms used in the cluster
	# zero void is defined as the tracto that is less than 3*nbr_samples.
	self.zero_tracto = []	 # will contain void tractogram (tractograms with sum < 3*n_samples)
	self.nonzero_tracto = []	 # will contain the non void tractogram
	self.tractograms = []
	for i in range(len(V[:, 0])): # loop over the coordinates
	   x, y, z = V[i, :] # read the ith coordinate (x,y,z)
	   filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
	   A = nl.load(filename).get_data() # read the tractogram in nifti format (.nii.gz)
	   T1 = A.reshape(-1) # from 3D to vector form
	   self.nbr_sample = np.max(T1.reshape(-1)) # if all voxel = 0, nbr_sample = 0
	   Sm = np.sum(T1)
	   if (Sm <= self.nbr_sample*5):	 # detect void tractogram. It defined as the tractogram that has a sum
                self.zero_tracto.extend([i]) # less then 5* number of samples
           else:
		self.nonzero_tracto.extend([i]) # add the ith seed to the non viod tractogram
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
            T = self.Read_tracto(seed)	 # read the tractogram
            self.nbr_sample = np.max(T.reshape(-1)) # if all voxel = 0, nbr_sample = 0
            Sm = np.sum(T)
            if (Sm <= self.nbr_sample*5):	 # detect void tractogram. It defined as the tractogram that has a sum
                self.zero_tracto.extend([i]) # less then 5* number of samples
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
