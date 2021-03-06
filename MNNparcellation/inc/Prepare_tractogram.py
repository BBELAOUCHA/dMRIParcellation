# -*- coding: utf-8 -*-
'''
###############################################################################
#
# This code is used to parcellate the cortical surface from dMRI information
# (tractograms in nii.gz files) using the Mutual Nearest Neighbor Condition
#
###############################################################################
# BELAOUCHA Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, please acknowledge Brahim Belaoucha.
# The best single reference is:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface
# Parcellation via dMRI Using Mutual Nearset Neighbor Condition”, International
# Symposium on Biomedical Imaging: From Nano to Macro, Prague, Czech Republic.
# pp. 903-906, April 2016.

# Author: Brahim Belaoucha 2015
# Any questions, please contact brahim.belaoucha@gmail.com
###############################################################################
'''
from copy import deepcopy
import numpy as np
import nibabel as nl
import os.path


class Parcellation_data():

    # class of operations on the tractogram images
    def __init__(self, tract_folder_path, tract_name, mesh, nodif_mask):

        self.tract_path = tract_folder_path  # path to tractogram
        self.tract_name = tract_name  # prefix tractogram
        if mesh:
            self.mesh = mesh  # mesh details coordinates and tess connectivity
            nbr = len(mesh.vertices[:, 1])
            self.Similarity_Matrix = np.zeros(nbr*(nbr-1)//2, dtype=np.float16)
            # matrix contains similarity measure values

        self.excluded = []  # excluded seeds
        self.nodif_mask = nodif_mask
        # path to the brain's mask, used to reduce the memory since all voxels
        # the tractograms outside the mask are0
        if nodif_mask != '':
            MASK = nl.load(self.nodif_mask).get_data()
            M = np.array(MASK.reshape(-1))
            self.non_zeroMask = np.array(np.nonzero(M)[0])
            # get the voxels of only the mask.
            del MASK, M

    def Read_tracto(self, V):
        # function used to read  tractogram in 3D
        # read the nii.gz tractogram files one by one

        x, y, z = V  # extract the x,y,z coordinate
        st = self.tract_path + '/' + self.tract_name + str(int(x))
        filename = st + '_' + str(int(y)) + '_' + str(int(z)) + '.nii.gz'
        if os.path.exists(filename):
            return nl.load(filename).get_data()  # return the tractogram in 3D
        else:
            return np.array([0])

    def Detect_void_tracto(self, min_vox=5):
        # detecte the void tractograms
        # zero void is defined as the tracto that is less than
        # min_vox*nbr_samples. void tractogram (tractograms with
        # sum<min_vox*n_samples)

        self.zero_tracto = []
        self.nonzero_tracto = []  # will contain the non void tractogram
        self.tractograms = []
        vertices = list(self.mesh.vertices)
        # the coordinate of the seeds in the diffusion space
        for i in range(len(vertices)):  # loop over the coordinates
            seed = vertices[i]  # extract the x,y,z coordinate
            T = self.Read_tracto(seed)  # read the tractogram
            self.nbr_sample = np.max(T.reshape(-1))
            # if all voxel = 0, nbr_sample = 0
            Sm = np.sum(T)
            if (Sm <= self.nbr_sample*min_vox):
                # detect void tractogram. It defined as the tractogram that has
                # a sum less then 5* number of samples
                self.zero_tracto.extend([i])
            else:
                self.nonzero_tracto.extend([i])
                # add the ith seed to the non viod tractogram
                T1 = T.reshape(-1)  # from 3D to a vector form
                # read only the voxels that are inside the (brain) mask
                T1 = T1[np.array(self.non_zeroMask)]
                self.tractograms.append(T1.astype('float32'))
                # add the tractogram into an array of arrays

    def Replace_void_tracto(self):
        # replace the void tractograms by the nearest neighbor non void

        self.replacement = {}
        if len(self.nonzero_tracto) == 0:
            return
        # dictionaty contain the (key:value) of the void tracto
        # and its correspedning
        for k in self.zero_tracto:
            # nearest non void tracto/ loop over void tracto
            neighbors = [k]  # initialize the neighbors of k
            visited_neighbors = deepcopy(self.zero_tracto)
            # already visited seeds
            while (len(neighbors) > 0):  # loop over the neighbors
                visited_neighbors.extend(neighbors)
                st = self.mesh.connectivity[neighbors, :]
                neighbors = np.unique(np.where(st == 1)[1])
                # get the neighbors from the tess connectivity matrix
                A = set(neighbors)
                B = A.difference(set(visited_neighbors))
                # only non visited seeds
                valid_neighbors = list(B)
                if len(valid_neighbors) > 0:  # if there are non void neighbors
                    distances = np.zeros(len(valid_neighbors))
                    # vector contains the euclidian distance
                    distances = []
                    for neighbor in valid_neighbors:
                        st = self.mesh.vertices[k, :]
                        dt = self.mesh.vertices[neighbor, :]
                        distances.append(np.linalg.norm(st - dt))  # distance

                    st = np.array(distances).argmin()
                    best_neighbor = valid_neighbors[st]
                    # choose the closest non void tracto which is the closest
                    self.replacement[k] = best_neighbor
                    break  # exit the while , go the next void tractogram
