# -*- coding: utf-8 -*-
'''
###################################################################################
#
# This code is used to prepare the regions of the cortical surface from dMRI
# (tractograms in nii.gz) using the Mutual Nearest Neighbor Condition "see ref."

# There are 2 classes:
#  1-Mesh:
#         1.1 Remove_void_tracto: removed the void tractograms from the coordinates
#             and tess connectivity matrix to reduce memory and computation time.
#  2-Regions_processing:
#     2.1 Neighbor_region: get the neighbors of region i from tess connectivity.
#     2.2 Add_zero_tracto_label: add the label of the void to the non void seeds.
#     2.3 Small_region: merge small regions with the big ones.
#     2.4 Excluded_label: add label (0) of excluded seeds to the parcellated ones.
#     2.5 Write2file: function used to write some results in ./results.txt
#      at each iteration save in row nbr of regions,time of execution, STD of SM
###################################################################################
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
# Symposium on Biomedical Imaging: From Nano to Macro, Prague, Czech Republic. pp.
# 903-906, April 2016.

# Author: Brahim Belaoucha 2015
# Any questions, please contact brahim.belaoucha@gmail.com
###################################################################################
'''
import numpy as np


class Mesh():  # class definition of mesh:-coordinates and tess connectivity

    def __init__(self, vertices, faces=[], normal=[], connectivity=[]):

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
        #for i in range(N):
        #    self.regions.append(i)
            # create a vector with singlton regions (each seed is a region)

    def Neighbor_region(self, Regions, index, mesh):
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

    def Add_zero_tracto_label(self, check_tracto, Regions):
        # this function add the label of the detected
        #Prepare_tractograms, label of regions
        nbr_seeds = len(check_tracto.zero_tracto)+len(check_tracto.nonzero_tracto)
        # void tract as the nearest label of non void tracto
        All_seeds_label = np.array(np.zeros(nbr_seeds, dtype=int))
        for i in range(len(check_tracto.nonzero_tracto)):
            All_seeds_label[check_tracto.nonzero_tracto[i]] = Regions[i]
        for i in check_tracto.zero_tracto:  # add the label of void seeds
            All_seeds_label[i] = All_seeds_label[check_tracto.replacement[i]]
            #label of void tractogram is the same as its nearest non void tracto
        return  All_seeds_label
        # add the label of all seeds including the void tractogram

    #def Small_region(self, SizeRegion, Regions, Connectivity, integrated_regions):
    #    # function used to merge small regions
    #    #vector contains size of regions, label of seeds, tess connectivity,
    #    # threshold to merge small regions
    #    Un = np.unique(Regions) # the uniqe labels
    #    RegionsX = np.array(Regions)
    #    Z = np.where(SizeRegion <= integrated_regions)[0]
    #    # get regions that have a cordinal less than integrated_regions
    #    X = np.zeros(len(Z))
    #    Q = len(Z)    # number of small regions
    #    while Q > 0:  # loop to merge small regions with bigger ones
    #        for i in xrange(len(Z)): # loop over the number of small regions
    #            insideregion, connected_regions = self.Neighbor_region(RegionsX, Z[i],
    #                                              Connectivity)
    #            sth = 0
    #   	        CX = Connectivity[np.array(insideregion), :]
    #   	        # get the tess connectivity mesh of only region Z[i]
    #            for j in xrange(len(connected_regions)): # loop over all regions
    #                ind = np.where(RegionsX == connected_regions[j])[0]
    #                C = CX[:, ind]
    #                # get the tess connectivity between region Un[j] and Z[i]
    #                A = np.sum(C) # check if there is edges between Un[j] and Z[i]
    #                if (A > sth) and (connected_regions[j] != 0):
    #                    # if there are edges and are different regions and region is
    #                    # different from mask
    #                    sth = np.sum(C) #add region with the highest number of edges
    #                    X[i] = connected_regions[j]#merge Z[i] connected_regions[j]
    #        RegionsX2 = np.array(RegionsX)
    #        for i in xrange(Q):# change the labeling  after the merging
    #            indx = np.where(RegionsX == Z[i])[0]
    #            RegionsX2[np.array(indx)] = X[i]
    #        RegionsX = RegionsX2
    #        Un = np.unique(RegionsX)# new unique labels
    #        nbr_r = len(Un) # new number of regions
    #        SizeRegion = np.zeros(nbr_r)
    #        RegionX_ = np.zeros(len(RegionsX))
    #        for i in xrange(nbr_r): # get the size of the new regions
    #            ind = np.where(RegionsX == Un[i])[0]
    #            SizeRegion[i] = len(ind)
    #            RegionX_[np.array(ind)] = i
    #        RegionsX = RegionX_
    #        Z = np.where(SizeRegion <= integrated_regions)[0]
    #        # get the regions with small size
    #        X = np.zeros(len(Z))
    #   	    if len(Z) == Q:
    #   	    # break the loop if the pre and actual number of small regions are equal
    #  		break
    #        Q = len(Z)
    #    return RegionsX  # label of seeds after merging small regions with big ones.

    def Excluded_label(self, excluded, Labelnonexcluded, label_orig):
        # function used to add the label of the # excluded seeds
        #(0 label) to the label of the non void seeds
        #label of excluded seeds, label of non excluded seeds, seeds non excluded
        if len(excluded) == 0:  # excluded 0 label to the label of the non void seeds
            return Labelnonexcluded
        Label = np.zeros(len(excluded)+len(Labelnonexcluded))
        for i in xrange(len(Labelnonexcluded)):
            Label[label_orig[i]] = Labelnonexcluded[i]
        return Label  # return the label with 0 for excluded seeds
