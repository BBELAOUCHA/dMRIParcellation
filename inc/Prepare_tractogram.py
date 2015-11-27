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
#####################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, you have to cite:
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
# Proceeding of International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.

# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical surface parcellation via dMRI using Mutual Nearset
# Neighbor condition”,  Submitted, 2015.

# Brahim Belaoucha and Théodore Papadopoulo, “Comparision of dMRI-based cortical surface parcellation
# with different similarity measures”,  Submitted, 2015.

# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################

from copy import deepcopy
import numpy as np
import nibabel as nl
class Parcellation_data():# class of operations on the tractogram images
    def __init__(self,tract_folder_path,tract_name,mesh,nodif_mask,nbr_sample=6000):
        self.tract_path  = tract_folder_path# path to tractogram
        self.tract_name  = tract_name# prefix tractogram
        self.nbr_sample  = nbr_sample# number of samples used in the probabilistic tractography
        self.mesh        = mesh# mesh details coordinates and tess connectivity
        self.Similarity_Matrix =np.eye(len(mesh.vertices[:,1]))# matrix contains similarity measure values
        self.excluded = []# excluded seeds
        self.nodif_mask=nodif_mask# path to the mask of the brain , used to reduce the memory since all voxels of the tractograms outside the mask are0
        MASK=nl.load(self.nodif_mask).get_data()
        M=np.array(MASK.reshape(-1))
        self.non_zeroMask=np.array(np.nonzero(M)[0]) # get the voxels of only the mask.
        del MASK, M
    def Read_Tracto(self,V): # read the nii.gz tractogram files one by one
        x, y, z  = V
        filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
        return nl.load(filename).get_data()
        #T1=Tract.reshape(-1)   # To move with correlation
    def Read_tractograms(self,V): #read all the tractograms used in the cluster
        self.tractograms=[]
        for i in range(len(V[:,0])):
            x, y, z  = V[i,:]
            filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
            A= nl.load(filename).get_data()
            T1=A.reshape(-1)
            T2=T1[np.array(self.non_zeroMask)]
            self.tractograms.append(T2)
        return self.tractograms
    def Detect_void_tracto(self):# detecte the void tractograms
        # zero void is defined as the tracto that is less than 3*nbr_samples.
        self.zero_tracto = []# will contain void tractogram (tractograms with sum < 3*n_samples)
        self.nonzero_tracto = []# will contain the non void tractogram
        self.tractograms=[]
        vertices = list(self.mesh.vertices)
        for i in range(len(vertices)):
            seed = vertices[i]
            T = self.Read_Tracto(seed)# read the tractogram
            Sm=np.sum(T)# sum of the tractogram
            if (Sm < self.nbr_sample*3): # detect void tractogram
                self.zero_tracto.extend([i])
            else:
                self.nonzero_tracto.extend([i])
                T1=T.reshape(-1)#,dtype=np.float16)
                #ind=np.where(T1<=0.1*np.max(T1))[0]
                #T1[ind]=0
                T2=T1[np.array(self.non_zeroMask)]
                self.tractograms.append(T2)

    def Replace_void_tracto(self):#replace the void tractograms by the nearest neighbor non void
        self.replacement = {} # dictionaty contain the (key:value) the void tracto and its correspedning
        for k in self.zero_tracto:#nearest non void tracto/ loop over void tracto
            neighbors = [k]# initialize the neighbors of k
            visited_neighbors = deepcopy(self.zero_tracto)# already visited seeds
            while (len(neighbors)>0):# loop over the neighbors
                visited_neighbors.extend(neighbors)
                neighbors       = np.unique(np.where(self.mesh.connectivity[neighbors,:]==1)[1])# get the neighbors from the tess connectivity matrix
                A=set(neighbors) #
                B=A.difference(set(visited_neighbors))# only non visited seeds
                valid_neighbors = list(B)
                if len(valid_neighbors)>0:# if there are non void neighbors
                    distances = np.zeros(len(valid_neighbors)) # vector contains the euclidian distance
                    distances = []
                    for neighbor in valid_neighbors:
                        distances.append(np.linalg.norm(self.mesh.vertices[k,:]-self.mesh.vertices[neighbor,:]))# distance
                    best_neighbor = valid_neighbors[np.array(distances).argmin()]# chose the non void tracto which is the closest in euclidian distance
                    self.replacement[k] = best_neighbor
                    break # exit the while , go the next void tractogram
