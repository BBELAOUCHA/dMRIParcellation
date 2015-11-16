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
    def __init__(self,tract_folder_path,tract_name,mesh,nbr_sample=5000):
        self.tract_path  = tract_folder_path
        self.tract_name  = tract_name
        self.nbr_sample  = nbr_sample
        self.mesh        = mesh
        self.Similarity_Matrix = np.eye(len(mesh.vertices[:,1]))
        self.excluded = []
    def Read_Tracto(self,V): # read the nii.gz tractogram files
        x, y, z  = V
        filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
        return nl.load(filename).get_data()
        #T1=Tract.reshape(-1)   # To move with correlation
    def Read_tractograms(self,V):
        self.tractograms=[]
        for i in range(len(V[:,0])):
            x, y, z  = V[i,:]
            filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
            A= nl.load(filename).get_data()
            A=A.reshape(-1)
            self.tractograms.append(A)
        return self.tractograms
    def Detect_void_tracto(self):# detecte the void tractograms
        # zero void is defined as the tracto that is less than 3*nbr_samples.
        self.zero_tracto = []
        self.nonzero_tracto = []
        self.tractograms=[]
        vertices = list(self.mesh.vertices)
        for i in range(len(vertices)):
            seed = vertices[i]
            T = self.Read_Tracto(seed)
            Sm=np.sum(T)
            if (Sm < self.nbr_sample*3):
                self.zero_tracto.extend([i])
            else:
                self.nonzero_tracto.extend([i])
                T=T.reshape(-1)
                self.tractograms.append(T)

    def Replace_void_tracto(self):#replace the void tractograms by the nearest neighbor non viod
        self.replacement = {}
        for k in self.zero_tracto:
            neighbors = [k]
            visited_neighbors = deepcopy(self.zero_tracto)
            while (len(neighbors)>0):
                visited_neighbors.extend(neighbors)
                neighbors       = np.unique(np.where(self.mesh.connectivity[neighbors,:]==1)[1])
                A=set(neighbors)
                B=A.difference(set(visited_neighbors))
                valid_neighbors = list(B)
                if len(valid_neighbors)>0:
                    distances = np.zeros(len(valid_neighbors))
                    distances = []
                    for neighbor in valid_neighbors:
                        distances.append(np.linalg.norm(self.mesh.vertices[k,:]-self.mesh.vertices[neighbor,:]))
                    best_neighbor = valid_neighbors[np.array(distances).argmin()]
                    self.replacement[k] = best_neighbor
                    break
