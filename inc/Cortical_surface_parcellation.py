# -*- coding: utf-8 -*-
#####################################################################################
#
# This code is used to parcellate the cortical surface from dMRI information
# (tractograms in nii.gz files) using the Mutual Nearest Neighbor Condition "see ref."
#
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

import Region_preparation as RP
import Prepare_tractogram as PT
import Similarity_Measures
import numpy as np
from Python2Vtk import WritePython2Vtk
from copy import deepcopy
import os
from scipy.stats import variation as cv
import time
class Parcellation():# main class to parcellate the cortical surface
    def __init__(self,path_tractogram,Prefix_name,save_path,nodif_mask,nbr_sample=6000):# initialize; prepare the paths
        self.path_tractogram=path_tractogram # path where the tractogram are located
        self.Prefix_name=Prefix_name         # prefix of the tractogram: Prefix_name_x_y_z.nii.gz
        self.save_path=save_path             # folder that will contain the results
        self.nodif_mask=nodif_mask           # path to mask of the brain fron b0 image
        self.nbr_sample=nbr_sample           # number of samples used in the probabilistic tractography
        self.Time=[]                          # array contains the execution time
        self.save_results_path='    '        # folder of each execution
    def Parcellation_agg(self,coordinate,Connectivity, Excluded_seeds,NbrRegion,SM_method,Mesh_plot,cvth):
        #FileName should contain path to tracto grams prefix of tractograms and save path
        Connectivity_X=deepcopy(Connectivity)
        if len(Excluded_seeds)>0: # if you exclude some seeds from the parcellation
            all_S=range(len(coordinate[:,0]))# they will be removed from the coordinates
            LABEL_ORIG=list(set(all_S) - set(Excluded_seeds))#and the tess conneectivity matrix
            COORD_ORIG=coordinate[np.array(LABEL_ORIG),:]
            Connectivity=Connectivity[np.array(LABEL_ORIG),:]
            Connectivity=Connectivity[:,np.array(LABEL_ORIG)]
            coordinate=COORD_ORIG

        mesh = RP.Mesh(coordinate,Connectivity)# save coordinates and tess connectivity on mesh
        Parc=PT.Parcellation_data(self.path_tractogram,self.Prefix_name,mesh,self.nodif_mask,self.nbr_sample) # prepare the parcellation by seeting the different paths
        Parc.Detect_void_tracto()# detect zero tracto ( tractograms that have sum less than 3*nbr_samples)
        if len(Parc.zero_tracto)>0:# if there are void tractograms
                Parc.Replace_void_tracto() # replace void tractograms
                mesh.Remove_void_tracto(Parc.zero_tracto, Parc.nonzero_tracto)# remove the void tractogram to speed up the computation.

        Connectivity=deepcopy(mesh.connectivity) # hard (not shallow) copy
        nbr_iteration=100 # total number of iterations
        for SM in SM_method: # loop over the list containing the name of the similarity measures
            sm=getattr(Similarity_Measures,SM+'_SM') #the function that implements the similarity measures SM
            Parc.Similarity_Matrix=np.eye(np.shape(Parc.Similarity_Matrix)[0]) # re initialize the Similarity_Matrix for the next similarity measure
            for R in NbrRegion: # loop over the list containing the number of regions "R"
                ATime=time.time() # execution time for each loop
                integrated_regions=int(round(0.1*(len(Parc.zero_tracto)+len(Parc.nonzero_tracto))/R))# merge at the end
                self.save_results_path=self.save_path+"/"+SM+"/"+str(R)                                   # regions that have size less than 10 %
                if not os.path.exists(self.save_results_path+'/LabelSurface'): # the cortical regions
                    os.makedirs(self.save_results_path+'/LabelSurface')
                nbr_seeds=len(mesh.vertices[:,0]) # number of vertices
                Reg=RP.Regions_processing(nbr_seeds)# initialize the class to handle regions
                Regions=Reg.regions # initialize regions, each seed is a region.
                NBR_REGIONS=nbr_seeds # init the nbr of regions
                region_th = np.float32(nbr_seeds)/R        # number used to stop growing big regions
                nbr_remaining=nbr_iteration
                region_labels=range(nbr_seeds)
                mesh.connectivity=deepcopy(Connectivity)
                nbr_r,t,mean_v,std_v =[],[],[],[] # vectors that conatin nbr regions, execution time, mean and std of the similarity values at each iteration
                while nbr_remaining > 0: # nbr of iteration
                    #   For each region find the best candidate to merge with it. (DRegion)
                    Dregion=[] # vector contains the Mutual nearest N
                    SM_vector=[] # vector that contain the similarity values between all pairs of regions
                    for i in range(NBR_REGIONS): # loop over the regions
                            #   Find neigboring regions of region i (connected_regions) and the indices of the ROI (insideregion)
                        insideregion,connected_regions=Reg.Neighbor_region(Regions, i, mesh)
                        nbr_connected=len(connected_regions)
                            # Calculate the row S of correlations of region i with its neighbors
                        if nbr_connected>0:
                            S = np.zeros(nbr_connected) # S contain the mean of the SM between i and all its neighbors
                            for l in range(nbr_connected):# loop over neighbors of i
                                outeregion=np.where(np.array(Regions) == connected_regions[l])[0] # get idices of region j that is connected to i
                                S[l] = sm(Parc,insideregion,outeregion)
                            Dregion.append(connected_regions[S.argmax()])# get the neighbors with the max SM value
                        else: # if no neighbor i is merged with itself.
                            Dregion.append(i)
                        #   Idea: do the fusion in the order of the quality of the correlation.
                    region_labels = np.unique(Regions)
                    RegionsX=np.array(Regions)
                    for i in range(len(region_labels)): # check if the mutual nearest neighbor condition is valid
                        if  (region_labels[i] == Dregion[Dregion[i]]):     # if they belong to the same region
                                #   Fuse region i and Dregion[i]
                            a=np.where(np.array(Regions) == region_labels[i])[0]
                            b=np.where(np.array(Regions) == Dregion[i])[0]
                            c=list(a)
                            c.extend(list(b))
                            RegionsX[np.array(c)] = Dregion[i]
                            #   Stop region growing.
                            cv_array=Parc.Similarity_Matrix[np.array(c),:]
                            cv_array=cv_array[:,np.array(c)]
                            cv_array=cv_array.reshape(-1)
                            if len(c) >= region_th or cv(cv_array) >= cvth: # stop growing region i if it contains more than region_th seeds
                                mesh.connectivity[c,:]=np.zeros(np.shape(mesh.connectivity[c,:]))# by setting rows and columns of region i
                                mesh.connectivity[:,c]=np.zeros(np.shape(mesh.connectivity[:,c]))# to zero
                        else:
                            a=np.where(np.array(Regions) == region_labels[i])[0]
                            c=list(a)
                            cv_array=Parc.Similarity_Matrix[np.array(c),:]
                            cv_array=cv_array[:,np.array(c)]
                            cv_array=cv_array.reshape(-1)
                        SM_vector.extend(cv_array) # save the SM values

                    Regions=np.array(RegionsX)
                    #Stopping criterion if the proper number of regions is obtained
                    region_labels=np.unique(Regions)
                    if (len(region_labels) == NBR_REGIONS) or (len(region_labels) <= R) or (int(mesh.connectivity.sum()) == 0):   # condition to stop the code. if the same nbr of region before and after stop iterating
                        RX,NBR_REGIONS=self.Add_void(Parc,Reg,region_labels,Regions,Excluded_seeds,LABEL_ORIG)
                        break # exit the while loop
                    RX,NBR_REGIONS=self.Add_void(Parc,Reg,region_labels,Regions,Excluded_seeds,LABEL_ORIG)
                    WritePython2Vtk(self.save_results_path+'/LabelSurface/Label_'+str(int(nbr_iteration-nbr_remaining+1))+'.vtk', Mesh_plot.vertices.T, Mesh_plot.faces.T,Mesh_plot.faces.T, RX, "Cluster")# save in vtk
                    nbr_remaining-=1 # in (de) crease iteration
                    nbr_r.append(NBR_REGIONS) # append the nbr of regions
                    t.append((time.time()-ATime)/60) # add execution time to the current parcellation
                    mean_v.append(np.mean(SM_vector)) # add the mean of the SM values
                    std_v.append(np.std(SM_vector))   # add the std of the SM values
                ##### add the zero tractogram to the nearest non zero tractogram that is labeled
                Reg.Write2file(self.save_results_path,SM,nbr_r,t,mean_v,std_v,R) # save results in ./results.txt
                np.savetxt(self.save_results_path+'/LabelSurface/Labels.txt',RX)
                # merge small regions
                mesh.connectivity=Connectivity_X
                NBR=np.unique(RX)
                #SMv=[]# Similarity measures values vector contain the SM values of all pair of the regions i.e. blocks of the Similarity_Matrix
                Regions=RX
                SizeRegion=np.zeros(len(NBR))
                for i in range(len(NBR)):
                    index = np.where(RX == NBR[i])[0]
                    SizeRegion[i]=len(index)
                    Regions[np.array(index)] = i

                Regions=Reg.Small_region(SizeRegion,Regions,Connectivity_X,integrated_regions) # function used to merge small regions (<10% of nbr_seeds/R) with big ones
                WritePython2Vtk(self.save_results_path+'/Parcellated.vtk', Mesh_plot.vertices.T, Mesh_plot.faces.T,Mesh_plot.normal.T, Regions, "Cluster")# save the final result in vtk
                np.savetxt(self.save_results_path+'/Parcellated.txt',Regions,fmt='%i')# save the final result of the parcellation in txt
                self.Time.append((time.time()-ATime)/60)
    def Add_void(self,Parc,Reg,region_labels,Regions,Excluded_seeds,LABEL_ORIG): # function used to add the labels to the viod tractograms
        NBR_REGIONS=len(region_labels)
        SizeRegion=np.zeros(NBR_REGIONS)
        for ii in range(NBR_REGIONS):
            insideregion=np.where(np.array(Regions) == region_labels[ii])[0]
            SizeRegion[ii]=len(insideregion)
            Regions[np.array(insideregion)]=ii
        if len(Parc.zero_tracto)>0:
            RX=Reg.Add_zero_tracto_label(Parc,Regions) # add the label of the void tractograms
            RX=RX+1  # label {1,..,NBR_REGIONS}
        else:
            RX=Regions
        if len(Excluded_seeds)>0:
            RX=Reg.Excluded_label(Excluded_seeds,RX,LABEL_ORIG)
        return RX,NBR_REGIONS
