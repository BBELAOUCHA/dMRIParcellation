# -*- coding: utf-8 -*-
#####################################################################################
#
# This code is used to prepare the regions of the cortical surface from dMRI information
# (tractograms in nii.gz files) using the Mutual Nearest Neighbor Condition "see ref."

# There are 2 classes:
#  1-Mesh:
#         1.1 Remove_void_tracto: removed the void tractograms from the coordinates
#                 and tess connectivity matrix to reduce thye memory and computation time.
#  2-Regions_processing:
#         2.1 Neighbor_region: get the neighboring regions of region i from the tess connectivity.
#         2.2 Add_zero_tracto_label: add the label of the void to the non void seeds.
#         2.3 Small_region: merge small regions with the big ones.
#         2.4 Excluded_label: add the label (0) of the excluded seeds to the parcellated ones.
#         2.5 Write2file: function used to write some results in ./results.txt
#              at each iteration save in row way nbr of regions,time of execution ,mean of SM,STD of SM
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

import numpy as np
class Mesh(): # class definition of mesh:-coordinates and tess connectivity

    def __init__(self,vertices,connectivity,faces=[],normal=[]):
        self.vertices     = vertices
        self.connectivity = connectivity
        self.faces=faces
        self.normal=normal
    def Remove_void_tracto(self,zero_tracto, nonzero_tracto):# remove the void seeds from the parcellation
        self.connectivity=self.connectivity[nonzero_tracto , :] # to speed up the computation and for less memory.
        self.connectivity=self.connectivity[: , nonzero_tracto]
        self.vertices=self.vertices[nonzero_tracto,:]

class Regions_processing(): # class process on regions

    def __init__(self,N):   # initialization
        self.regions = []
        for i in range(N):
            self.regions.append(i)# create a vector with singlton regions (each seed is a region)

    def Neighbor_region(self, Regions, index, mesh): # Neighboring regions of region index
            # vector containg labels, region of interest, mesh details (coordinate and tess connectivity)
            insideregion=np.where(np.array(Regions) == index)[0]# neighbors of region index
            I=mesh.connectivity[insideregion] # get tess connection to region Region == i
            I=np.squeeze(I)
            q=set(np.unique(np.where(I == 1)[len(insideregion)>1]))
            q.difference_update(set(insideregion)) # extract only idices connected to region == i that doesnt belong to i during this iter
            ConnRegions=[Regions[np.array(list(q))[k]] for k in range(len(q))] # get the regions label that are connected to region == i
            connected_regions= np.unique(ConnRegions) # the regions connected to i
            return insideregion,connected_regions# return the index of the region, and it neighbors

    def Add_zero_tracto_label(self,check_tracto,Regions):# this function add the label of the detected
        #Prepare_tractograms, label of regions
        nbr_seeds=len(check_tracto.zero_tracto)+len(check_tracto.nonzero_tracto) # void tract as the nearest label of non void tracto
        All_seeds_label=np.array(np.zeros(nbr_seeds,dtype=int))
        for i in range(len(check_tracto.nonzero_tracto)):
            All_seeds_label[check_tracto.nonzero_tracto[i]]=Regions[i]
        for i in check_tracto.zero_tracto:# add the label of void seeds
            All_seeds_label[i]=All_seeds_label[check_tracto.replacement[i]]
        return  All_seeds_label

    def Small_region(self,SizeRegion,Regions,Connectivity,integrated_regions):# function used to merge small regions
        #vector contains size of regions, label of seeds, tess connectivity, threshold to merge small regions
        Un=np.unique(Regions)
        RegionsX=np.array(Regions)
        Z=np.where(SizeRegion <= integrated_regions)[0]# small regiosn with bigger one
        X=np.zeros(len(Z))
        Q=len(Z)    # number of small regions
        while Q > 0: # loop to merge small regions with bigger ones
                for i in range(len(Z)):
                    sth=0
                    indx=np.where(RegionsX == Z[i])[0]
       	            CX=Connectivity[np.array(indx),:]
                    for j in range(len(Un)):
                        ind=np.where(RegionsX == Un[j])[0]
                        C=CX[:,ind]
                        A=np.sum(C)
                        if A > sth and (Z[i]!=Un[j]) and (Un[j]!=0):  # NOT THE SAME REGION
                            sth=np.sum(C)
                            X[i] = Un[j]
                RegionsX2=np.array(RegionsX)
                for i in range(Q):
                    indx=np.where(RegionsX == Z[i])[0]
                    RegionsX2[np.array(indx)] = X[i]
                RegionsX=RegionsX2
                Un=np.unique(RegionsX)
                nbr_r=len(Un)
                SizeRegion=np.zeros(nbr_r)
                RegionX_=np.zeros(len(RegionsX))
                for i in range(nbr_r):
                    ind=np.where(RegionsX == Un[i])[0]
                    SizeRegion[i]=len(ind)
                    RegionX_[np.array(ind)]=i
                RegionsX=RegionX_
                Z=np.where(SizeRegion <= integrated_regions)[0]
                X=np.zeros(len(Z))
       	        if len(Z) == Q:  # break the loop if the pre and actual number of small regions are equal
      		    break
                Q=len(Z)
        return RegionsX # label of seeds after merging small regions with big ones.
    def Excluded_label(self,excluded,Labelnonexcluded,label_orig):# function used to add the label of the
        #label of excluded seeds, label of non excluded seeds, seeds non excluded
        if len(excluded) == 0:# excluded seeds (0 label) to the label of the non void seeds
            return Labelnonexcluded
        Label=np.zeros(len(excluded)+len(Labelnonexcluded))
        for i in range(len(Labelnonexcluded)):
            Label[label_orig[i]] =Labelnonexcluded[i]
        return Label # return the label with 0 for excluded seeds

    def Write2file(self,save_results_path,sm,nbr_r,t,mean_v,std_v,Rw): #writesome results of the regions
            # path to save, Similarity measure, nbr of regions, time of execution, mean values of SM, std of SM, stopping condition R.
            resultfile=open(save_results_path+'/results.txt','w')
            resultfile.write('Similarity Measure: '+sm+'\t R:='+str(Rw)+'\n')
            resultfile.write('nbr i \t nbr R \t t(min) \t mean SM \t STD SM: \n')
            for i in range(len(nbr_r)):
                resultfile.write(str(i+1)+'\t'+str(nbr_r[i]) + '\t'+ str(t[i])+'\t'+str(mean_v[i])+'\t'+str(std_v[i])+'\n')
            resultfile.close()
