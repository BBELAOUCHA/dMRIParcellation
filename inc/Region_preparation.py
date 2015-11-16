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

import numpy as np
class Mesh(): # class definition of mesh:-coordinates and tess connectivity

    def __init__(self,vertices,connectivity,faces=[],normal=[]):
        self.vertices     = vertices
        self.connectivity = connectivity
        self.faces=faces
        self.normal=normal
    def Remove_void_tracto(self,zero_tracto, nonzero_tracto):
        self.connectivity=self.connectivity[nonzero_tracto , :]
        self.connectivity=self.connectivity[: , nonzero_tracto]
        self.vertices=self.vertices[nonzero_tracto,:]

class Regions_processing(): # class process on regions

    def __init__(self,N):   # initialization
        self.regions = []
        for i in range(N):
            self.regions.append(i)

    def Neighbor_region(self, Regions, index, mesh): # Neighboring regions of region index
            insideregion=np.where(np.array(Regions) == index)[0]
            I=mesh.connectivity[insideregion] # get tess connection to region Region == i
            I=np.squeeze(I)
            q=set(np.unique(np.where(I == 1)[len(insideregion)>1]))
            q.difference_update(set(insideregion)) # extract only idices connected to region == i that doesnt belong to i during this iter
            ConnRegions=[Regions[np.array(list(q))[k]] for k in range(len(q))] # get the regions label that are connected to region == i
            connected_regions= np.unique(ConnRegions) # the regions connected to i
            return insideregion,connected_regions

    def Add_zero_tracto_label(self,check_tracto,Regions):# this function add the label of the detected
        nbr_seeds=len(check_tracto.zero_tracto)+len(check_tracto.nonzero_tracto) # void tract as the nearest label of non void tracto
        All_seeds_label=np.array(np.zeros(nbr_seeds,dtype=int))
        for i in range(len(check_tracto.nonzero_tracto)):
            All_seeds_label[check_tracto.nonzero_tracto[i]]=Regions[i]
        for i in check_tracto.zero_tracto:
            All_seeds_label[i]=All_seeds_label[check_tracto.replacement[i]]
        return  All_seeds_label

    def Small_region(self,SizeRegion,Regions,Connectivity,integrated_regions):# function used to merge
        Un=np.unique(Regions)
        Z=np.where(SizeRegion < integrated_regions)[0]# small regiosn with bigger one
        X=np.zeros(len(Z))
        Q=len(Z)    # number of small regions
        while Q > 0: # loop to merge small regions with bigger ones
                for i in range(len(Z)):
                    sth=0
                    indx=np.where(Regions == Z[i])[0]
       	            CX=Connectivity[indx,:]
                    for j in range(len(Un)):
                        ind=np.where(Regions == Un[j])[0]
                        C=CX[:,ind]
                        A=np.sum(C)
                        if A > sth and (i!=j):  # NOT THE SAME REGION
                            sth=np.sum(C)
                            X[i] = Un[j]
                for i in range(Q):
                    indx=np.where(Regions == Z[i])[0]
                    for j in range(len(indx)):
                        Regions[indx[j]] = X[i]
                Un=np.unique(Regions)
                nbr_r=len(Un)
                SizeRegion=np.zeros(nbr_r)
                RegionX=np.zeros(len(Regions))
                for i in range(nbr_r):
                    SizeRegion[i]=len(np.where(Regions== Un[i])[0])
                    ind=np.where(Regions== Un[i])[0]
                    for j in range(len(ind)):
                        RegionX[ind[j]]=i
                Regions=RegionX
                Z=np.where(SizeRegion < int(integrated_regions))[0]
                X=np.zeros(len(Z))
       	        if len(Z) == Q:  # break the loop if the pre and actual number of small regions are equal
      		    break
                Q=len(Z)
        return Regions
    def Excluded_label(self,excluded,Labelnonexcluded,label_orig):
        if not excluded:
            return Labelnonexcluded
        Label=np.zeros(len(excluded)+len(Labelnonexcluded))
        for i in range(len(Labelnonexcluded)):
            Label[label_orig[i]] =Labelnonexcluded[i]
        return Label
