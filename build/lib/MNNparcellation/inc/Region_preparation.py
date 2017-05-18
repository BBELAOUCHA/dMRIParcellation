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
import numpy as np


def Neighbor_region(Regions, index, mesh):
    # Neighboring regions of region index
    # vector containg labels, region of interest, mesh details (coordinate
    # and tess connectivity)

    insideregion = np.where(np.array(Regions) == index)[0]  # neighbors index
    I = mesh.connectivity[insideregion]  # get connection to region  i
    I = np.squeeze(I)
    q = set(np.unique(np.where(I == 1)[len(insideregion) > 1]))
    q.difference_update(set(insideregion))
    # extract only idices connected to Region that dont belong to i
    ConnRegions = [Regions[np.array(list(q))[k]] for k in range(len(q))]
    # get the regions label that are connected to region == i
    connected_regions = np.unique(ConnRegions)  # the regions connected to i
    return insideregion, connected_regions
    # return the index of the region, and it neighbors


def Add_zero_tracto_label(check_tracto, Regions):
    # this function add the label of the detected
    # Prepare_tractograms, label of regions
    nbr_seeds = len(check_tracto.zero_tracto)+len(check_tracto.nonzero_tracto)
    # void tract as the nearest label of non void tracto
    All_seeds_label = np.array(np.zeros(nbr_seeds, dtype=int))
    for i in range(len(check_tracto.nonzero_tracto)):
        All_seeds_label[check_tracto.nonzero_tracto[i]] = Regions[i]

    for i in check_tracto.zero_tracto:  # add the label of void seeds
        All_seeds_label[i] = All_seeds_label[check_tracto.replacement[i]]
        # label of void tractogram is the same as its nearest non void tracto

    return All_seeds_label
    # add the label of all seeds including the void tractogram


def Excluded_label(excluded, Labelnonexcluded, label_orig):
    # function used to add the label of the # excluded seeds
    # (0 label) to the label of the non void seeds
    # label of excluded seeds, label of non excluded seeds, seeds non excluded

    if len(excluded) == 0:
        return Labelnonexcluded

    Label = np.zeros(len(excluded)+len(Labelnonexcluded))
    for i in xrange(len(Labelnonexcluded)):
        Label[label_orig[i]] = Labelnonexcluded[i]

    return Label  # return the label with 0 for excluded seeds


class Mesh():  # class definition of mesh:-coordinates and tess connectivity

    def __init__(self, vertices, faces=None, normal=None, connectivity=None):
        if faces is not None:
            faces = []
        if normal is not None:
            normal = []
        if connectivity is not None:
            connectivity = []
        self.vertices = vertices  # coordinates in the diffusion space
        self.connectivity = connectivity  # mesh tess connectivity matrix
        self.faces = faces  # faces of the mesh
        self.normal = normal  # normal of the mesh

    def Remove_void_tracto(self, zero_tracto, nonzero_tracto):
        # remove the void seeds from the parcellation

        self.connectivity = self.connectivity[nonzero_tracto, :]
        # to speed up the computation and for less memory.
        self.connectivity = self.connectivity[:, nonzero_tracto]
        self.vertices = self.vertices[nonzero_tracto, :]  # remove void seeds
        # void tractograms are removed from the computation


class Regions_processing():  # functions on the regions

    def __init__(self, N):   # initialization
        self.regions = []
        self.regions = np.arange(N)
        # for i in range(N):
        # self.regions.append(i)
        # create a vector with singlton regions (each seed is a region)
