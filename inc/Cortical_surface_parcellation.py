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

import Region_preparation as RP
import Prepare_tractogram as PT
import Similarity_Measures
import numpy as np
from Python2Vtk import WritePython2Vtk
from copy import deepcopy
import os
from scipy.stats import variation as cv


class Parcellation():
    def __init__(self,path_tractogram,Prefix_name,save_path):
        self.path_tractogram=path_tractogram
        self.Prefix_name=Prefix_name
        self.save_path=save_path
    def Parcellation_agg(self,coordinate,Connectivity, Excluded_seeds,ATLAS,SM_method,Mesh_plot,cvth):
        #FileName should contain path to tracto grams prefix of tractograms and save path
        sm=getattr(Similarity_Measures,SM_method+'_SM') ## use the similarity measures
        if len(Excluded_seeds)>0:
            print len(Excluded_seeds), "Was removed from computation"
            all_S=range(len(coordinate[:,0]))
            LABEL_ORIG=list(set(all_S) - set(Excluded_seeds))
            COORD_ORIG=coordinate[np.array(LABEL_ORIG),:]
            Connectivity=Connectivity[np.array(LABEL_ORIG),:]
            Connectivity=Connectivity[:,np.array(LABEL_ORIG)]
            coordinate=COORD_ORIG
        print np.shape(coordinate)
        mesh = RP.Mesh(coordinate,Connectivity)
        Parc=PT.Parcellation_data(self.path_tractogram,self.Prefix_name,mesh,12000)
        print "detect void"
        Parc.Detect_void_tracto()# detect zero tracto
        if len(Parc.zero_tracto)>0:
                Parc.Replace_void_tracto() # replace void tractograms
                mesh.Remove_void_tracto(Parc.zero_tracto, Parc.nonzero_tracto)
        print "read tracto"
        Connectivity=deepcopy(mesh.connectivity)
        #Parc.Read_tractograms(mesh.vertices)
        for AT in ATLAS:
            needed_nbr_region=AT
            integrated_regions=int(0.1*(len(Parc.zero_tracto)+len(Parc.nonzero_tracto))/AT)# merge at the end
            save_results_path=self.save_path+"/"+str(AT)                                   # regions that have size less than 10 %
            if not os.path.exists(save_results_path+'/LabelSurface'): # the cortical regions
                os.makedirs(save_results_path+'/LabelSurface')
                print "Folder is created:     ./LabelSurface"
            nbr_iteration=100
            nbr_seeds=len(mesh.vertices[:,0]) # number of vertices
            Reg=RP.Regions_processing(nbr_seeds)
            Regions=Reg.regions # initialize regions, each seed is a region.
            NBR_REGIONS=nbr_seeds # init the nbr of regions
            region_th = np.float32(nbr_seeds)/needed_nbr_region        # number used to stop growing big regions
            nbr_remaining=nbr_iteration
            region_labels=range(nbr_seeds)
            Region_iteration=[]
            mesh.connectivity=deepcopy(Connectivity)
            print "Start", NBR_REGIONS
            while nbr_remaining > 0: # nbr of iteration
                #   For each region find the best candidate to merge with it. (DRegion)
                Dregion=[] # vector contains the Mutual nearest N
                SizeRegion=np.zeros(NBR_REGIONS)# initialize the size of regions to zeros
                AS=[]
                for i in range(NBR_REGIONS): # loop over the regions
                        #   Find neigboring regions of region i (connected_regions) and the indices of the ROI (insideregion)
                        insideregion,connected_regions=Reg.Neighbor_region(Regions, i, mesh)
                        nbr_connected=len(connected_regions)
                        # Calculate the row S of correlations of region i with its neighbors
                        if nbr_connected>0:
                            S = np.zeros(nbr_connected)
                            outeregion=[]
                            for l in range(nbr_connected):
                                    outeregion.append(np.where(np.array(Regions) == connected_regions[l])[0]) # get idices of region j that is connected to i
                                    S[l] = sm(Parc,insideregion,outeregion[l])

                            Dregion.append(connected_regions[S.argmax()])
                            AS.append(S[S.argmax()])
                        else:
                            Dregion.append(i)
                    #   Idea: do the fusion in the order of the quality of the correlation.
                print "Merging"
                region_labels = np.unique(Regions)
                RegionsX=np.array(Regions)
                for i in range(len(region_labels)):
                    if  (region_labels[i] == Dregion[Dregion[i]]):     # if they belong to the same region
                            #   Fuse region i and Dregion[i]
                            a=np.where(np.array(Regions) == region_labels[i])[0]
                            b=np.where(np.array(Regions) == Dregion[i])[0]
                            c=list(a)
                            c.extend(list(b))
                            #for k in range(len(c)):
                            RegionsX[np.array(c)] = Dregion[i]
                            #   Stop region growing.
                            cv_array=Parc.Similarity_Matrix[np.array(c),:]
                            cv_array=cv_array[:,np.array(c)]
                            cv_array=cv_array.reshape(-1)
                            if len(c) >= region_th or cv(cv_array) >= cvth: # stop growing region i if it contains more than region_th dipoles
                                mesh.connectivity[c,:]=np.zeros(np.shape(mesh.connectivity[c,:]))
                                mesh.connectivity[:,c]=np.zeros(np.shape(mesh.connectivity[:,c]))
                                #print "Region ", i, " is completed "
                Regions=np.array(RegionsX)
                    #   Stopping criterion if the proper number of regions is obtained
                region_labels=np.unique(Regions)
                if (len(region_labels) == NBR_REGIONS) or (len(region_labels) <= needed_nbr_region) or (int(mesh.connectivity.sum()) == 0):   # condition to stop the code. if the same nbr of region before and after stop iterating
                        NBR_REGIONS=len(region_labels)
                        SizeRegion=np.zeros(NBR_REGIONS)
                        for ii in range(NBR_REGIONS):
                            insideregion=np.where(np.array(Regions) == region_labels[ii])[0]
                            SizeRegion[ii]=len(insideregion)
                            #for k in range(len(insideregion)):
                            Regions[np.array(insideregion)]=ii
                        RX=Reg.Add_zero_tracto_label(Parc,Regions) # add the label of the void tractograms
                        RX=RX+1
                        if len(Excluded_seeds)>0:
                            RX=Reg.Excluded_label(Excluded_seeds,RX,LABEL_ORIG)

                        break
                NBR_REGIONS=len(region_labels)
                SizeRegion=np.zeros(NBR_REGIONS)
                for i in range(NBR_REGIONS):
                        insideregion=np.where(np.array(Regions) == region_labels[i])[0]
                        SizeRegion[i]=len(insideregion)
                        #for k in range(len(insideregion)):
                        RegionsX[np.array(insideregion)]=i
                print "New number of regions ", NBR_REGIONS
                NBR_REGIONS=len(np.unique(RegionsX))
                Regions=RegionsX
                RX=Reg.Add_zero_tracto_label(Parc,Regions) # add the label of the void tractograms
                RX=RX+1
                if len(Excluded_seeds)>0:
                    RX=Reg.Excluded_label(Excluded_seeds,RX,LABEL_ORIG)
                Region_iteration.append(RX)
                np.savetxt(save_results_path+'/LabelSurface/Label_'+str(int(nbr_iteration-nbr_remaining))+'.txt',RX)
                WritePython2Vtk(save_results_path+'/LabelSurface/Label_'+str(int(nbr_iteration-nbr_remaining))+'.vtk', Mesh_plot.vertices.T, Mesh_plot.faces.T,Mesh_plot.faces.T, RX, "Cluster")
                nbr_remaining-=1 # in (de) crease iteration
            ##### add the zero tractogram to the nearest non zero tractogram that is labeled
            np.savetxt(save_results_path+'/LabelSurface/Labels.txt',RX)
            NBR=np.unique(Regions)
            Si=np.zeros(len(NBR))
            Sq=[]
            for i in range(len(NBR)):
                    index = np.where(Regions == NBR[i])[0]
                    A=Parc.Similarity_Matrix[index,:]
                    A=A[:,index]
                    Sq.extend(A.reshape(-1))
                    Si[i] = np.mean(A)
            np.savetxt(save_results_path+'/SM_mean.txt',Si)
            np.savetxt(save_results_path+'/SM_all.txt',Sq)
            Regions=RX
            Un=np.unique(Regions)
            SizeRegion=np.zeros(len(Un)-1)
            print "---------> Start computing the number of dipoles per region"
            for i in range(1,len(Un)):
       	        SizeRegion[i-1]=len(np.where(Regions== Un[i])[0])
        # merge small regions
            Regions=Reg.Small_region(SizeRegion,Regions,Connectivity,integrated_regions)
            region_labels=np.unique(Regions)
            NBR_REGIONS=len(region_labels)
            WritePython2Vtk(save_results_path+'/Corrected.vtk', Mesh_plot.vertices.T, Mesh_plot.faces.T,Mesh_plot.normal.T, Regions, "Cluster")
            np.savetxt(save_results_path+'/Corrected.txt',Regions,fmt='%i')
            nbr_iteration-1
            pass # mesh.vertices mesh.connectivity
