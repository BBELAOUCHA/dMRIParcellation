# -*- coding: utf-8 -*-
'''
################################################################################
#
# This code is used to parcellate the cortical surface from dMRI information
# (tractograms in nii.gz files) using the Mutual Nearest Neighbor Condition
#
################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, you have to cite:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface
# Parcellation via dMRI Using Mutual Nearset Neighbor Condition”, International
# Symposium on Biomedical Imaging: From Nano to Macro, Prague, Czech Republic.
#                                                       pp. 903-906, April 2016.
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the
#       reduced source space”, in International Conference on Basic and Clinical
#          Multimodal Imaging (BaCi 2015), Utrecht, Netherlands, September 2015.
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
################################################################################
'''
import Region_preparation as RP
import Prepare_tractogram as PT
import Similarity_Measures
import numpy as np
from Python2Vtk import WritePython2Vtk
from copy import deepcopy
import os
from scipy.stats import variation as cv
import time


class Parcellation():  # main class to parcellate the cortical surface
    def __init__(self, path_tractogram, Prefix_name, save_path, nodif_mask,
                 VERBOSE = False, merge = 0):# initialize; prepare the paths
	self.path_tractogram = path_tractogram# path tractogram's location
	self.Prefix_name = Prefix_name# prefix     Prefix_name_x_y_z.nii.gz
	self.save_path = save_path# folder that will contain the results
	self.nodif_mask = nodif_mask# path to mask of the brain fron b0 image
	self.Time = []	 # array contains the execution time
	self.save_results_path = '_'	 # folder of each execution
	self.verbose = VERBOSE #enable terminal display of the results
        self.merge = merge # type of postprocessing (after the MNN parcellation)

    def PrintResults(self, Data):  # print the different results in the terminal
	if self.verbose:  # The result is saved in a dictionary
	   for i in Data.keys():
                print i, ' = ', Data[i] #print the dictionary

    def Add_void(self, Parc, Reg, Regions, Excluded_seeds, Label_non_excluded):
        # function used to add the labels to the viod tractograms
	# Object of the parcellation, Object to region processing class, labels,
        # excluded seeds, non excluded seeds
	region_labels = np.unique(Regions)
        NBR_REGIONS = len(region_labels)
	SizeRegion = np.zeros(NBR_REGIONS)
	for ii in range(NBR_REGIONS):
	   insideregion = np.where(np.array(Regions) == region_labels[ii])[0]
	   SizeRegion[ii] = len(insideregion)
	   Regions[np.array(insideregion)] = ii

	if len(Parc.zero_tracto) > 0:
	   RX = Reg.Add_zero_tracto_label(Parc, Regions)#add void tractograms
	   RX = RX+1	 # label {1,..,NBR_REGIONS}
	else:
		RX = Regions

	if len(Excluded_seeds) > 0:
		RX = Reg.Excluded_label(Excluded_seeds, RX, Label_non_excluded)
	return RX, NBR_REGIONS

    def Write2file_results(self, Similarity_Measure, nbr_r, t, mean_v, std_v, R):
        # writesome results of the regions
        # path to save, Similarity measure, nbr of regions, time of execution,
        # mean values of SM, std of SM, stopping condition R.
        resultfile = open(self.save_results_path + '/results.txt', 'w')
        resultfile.write('Similarity Measure:\t' + Similarity_Measure +
                         '\t R_th \t'+str(self.region_th)+'\t R:='+str(R)+'\n')
        resultfile.write('nbr i \t nbr R \t t(min) \t mean SM \t STD SM: \n')
        for i in range(len(nbr_r)):
            nbr= "%03d" % nbr_r[i]
            resultfile.write(str(i + 1) + '\t' + nbr + '\t' + str(t[i]) + '\t' +
                             str(mean_v[i])+'\t'+str(std_v[i])+'\n')
        resultfile.close()

    def Write2file_zero_tracto(self, Parc):
        # writesome results of the regions
        # path to save, Similarity measure, nbr of regions, time of execution,
        # mean values of SM, std of SM, stopping condition R.
        resultfile = open(self.save_path + '/zero_tractogram.txt', 'w')
        resultfile.write('index_zero_tracto\t' + 'index_replacement'+'\n')
        zero_t, repla_c = Parc.zero_tracto,Parc.replacement
        for i in range(len(zero_t)):
            resultfile.write(str(zero_t[i])+'\t'+ str(repla_c[zero_t[i]]) +'\n')
        resultfile.close()

    def Parcellation_agg(self, coordinate, Connectivity, Excluded_seeds,
                         NbrRegion, SM_method, Mesh_plot, cvth):
	# diffusion coordinates, mesh connectivity , array of exclueded seeds,
        # array of nbr of regions, array of similarity measures, mesh used
	# to generate vtk files, variation coefficient threshold
        all_Seeds = range(len(coordinate[:, 0]))
        # they will be removed from the coordinates
        self.nbr_seedsX = len(all_Seeds)  # number of vertices
        self.Label_non_excluded = list(set(all_Seeds))
        Mesh_back_up = RP.Mesh(coordinate,[],[], Connectivity)
        if len(Excluded_seeds) > 0:
           # if some seeds are exclude from the parcellation
	   self.Label_non_excluded = list(set(all_Seeds) - set(Excluded_seeds))
	   # and the tess conneectivity matrix
	   Coord_non_excluded = coordinate[np.array(self.Label_non_excluded), :]
	   Connectivity = Connectivity[np.array(self.Label_non_excluded), :]
	   Connectivity = Connectivity[:, np.array(self.Label_non_excluded)]
	   coordinate = Coord_non_excluded
        nbr_seeds = len(coordinate[:, 0])  # number of vertices
        Connectivity_X = deepcopy(Connectivity)# this mesh connectivity will not
        #be modefied used at the end of the code to merge the small regions
	mesh = RP.Mesh(coordinate,[],[], Connectivity)
	# create an object containing the coordinate and mesh connectivity
	# Prepare the parcellation by seeting the different paths
	printData = {}
	printData['Loading tractograms '] = str(self.nbr_seedsX)
	self.PrintResults(printData)
	Parc = PT.Parcellation_data(self.path_tractogram, self.Prefix_name,
	                            mesh, self.nodif_mask)
	# Parc.Repeated_Coordinate(coordinate)
	Parc.Detect_void_tracto()
        # detect zero tracto(tractogram that has sum < 3*max(tractogram))
	if len(Parc.zero_tracto) > 0:  # if there are void tractograms
		Parc.Replace_void_tracto()
		# replace void tractograms by the nearest neighbor non void
		mesh.Remove_void_tracto(Parc.zero_tracto, Parc.nonzero_tracto)
		# remove the void tractogram to speed up the computation.
	printData = {} # This dictionary is used to save the different results
	# to be printed in the terminal if verbose == true
	printData['# Excluded seeds:'] = len(Excluded_seeds)
	printData['Path to tractogram:'] = self.path_tractogram
	printData['Prefix name:'] = self.Prefix_name
	printData['Path to nodif mask:'] = self.nodif_mask
	printData['Save path:'] = self.save_path
	printData['# Tracto, # Void tracto'] = nbr_seeds, len(Parc.zero_tracto)
	self.Write2file_zero_tracto(Parc)
        Connectivity = deepcopy(mesh.connectivity)  # hard (not shallow) copy
        #new mesh connec after removing void tractogram used at each R and SM
        nbr_iteration = 100
        # total number of iterations fixed, generally 50 is enough
        self.PrintResults(printData) # print the results so far
        nbr_seeds = len(mesh.vertices[:, 0])  # number of vertices
        for SM in SM_method:
            # loop over the list containing the name of the similarity measures
            SimilarityMeasure = getattr(Similarity_Measures, SM+'_SM')
            # call the function of the similarity measures SM
            Parc.Similarity_Matrix = np.eye(nbr_seeds)
            # reinitialize the Similarity_Matrix for the next similarity measure
            for R in NbrRegion:
                # loop over the list containing the number of regions "R"
                ATime = time.time()  # execution time for each loop
                self.save_results_path = self.save_path+ "/" + SM + "/" + str(R)
                # folder where to save results
                if not os.path.exists(self.save_results_path):#cortical regions
                    os.makedirs(self.save_results_path)
                    # create the folder if not found
                Reg = RP.Regions_processing(nbr_seeds)
                # initialize the class to handle regions
                Regions = Reg.regions #initialize regions,each seed is a region.
                NBR_REGIONS = nbr_seeds  # init the nbr of regions
                self.region_th = round(np.float32(self.nbr_seedsX-
                                 len(Excluded_seeds) - len(Parc.zero_tracto))/R)
                                 # number used to stop growing big regions
                nbr_remaining = nbr_iteration
                # initialize the remaining number of iterations
                region_labels = range(nbr_seeds) # inital labeling
                mesh.connectivity = deepcopy(Connectivity)
                # re initialize the mesh connectivity
                # vectors that conatin nbr regions, execution time, mean and std
                #of the similarity values at each iteration
                nbr_r, t, mean_v, std_v = [], [], [], []
                Labels = [] # list of list contains the labels at each iteration
                printData = {}#reempty dictionary that is used todisplay results
                printData['# Region'] = R
                printData['Similarity Measure'] = SM
                printData['# Iterations'] = nbr_iteration
                printData['Stop merging at'] = self.region_th
                self.PrintResults(printData)#disply results if verbose is active
                while nbr_remaining > 0:  # nbr of iteration
                    printData = {} # dictionary used to display results
                    Merg_Condidates = []
                    # vector contains the Mutual nearest N condidates
                    SM_vector = []  # vector that contain the similarity values
                    # between all pairs of regions
                    for i in range(NBR_REGIONS):  # loop over the regions
                        #Find neigboring regions of region i (connected_regions)
                        #and the indices of the ROI (insideregion)
                        insideregion, connected_regions = Reg.Neighbor_region(
                                                          Regions, i, mesh)
                        nbr_connected = len(connected_regions)
                        # Calculate the row S of correlations of region i with
                        #its neighbors
                        if nbr_connected > 0:
                            S = np.zeros(nbr_connected)  # S contain the mean of
                            #the SM between i and all its neighbors
                            for l in range(nbr_connected):#loop over i neighbors
                                outeregion = np.where(np.array(Regions) ==
                                                        connected_regions[l])[0]
                                S[l] = SimilarityMeasure(Parc, insideregion,
                                                         outeregion)
                            Reg_cond = list(np.where(S == S.max())[0])
                            Reg_list = [connected_regions[u] for u in Reg_cond]
                            # get the neighbors with the max SM value
                            Merg_Condidates.append(Reg_list)
                        else:  # if no neighbor i is merged with itself.
                            Merg_Condidates.append([i])

                    region_labels = np.unique(Regions)
                    RegionsX = np.array(Regions)
                    for i in range(len(region_labels)):
                        # check if the mutual nearest neighbor condition is valid
                        Reg_cond = Merg_Condidates[i]
                        # candidates of merging to region i
                        a = np.where(np.array(Regions) == region_labels[i])[0]
                        # get seeds with label  region_labels[i]
                        if len(a) < self.region_th:
                            for u in Reg_cond:
                                Reg_list = Merg_Condidates[u]
                                if i in Reg_list:
                             # if region i is also a condidate of merging to region u
                                    a = np.where(np.array(Regions) ==
                                                region_labels[i])[0]
                                    # get seeds with label  region_labels[i]
                                    b = np.where(np.array(Regions) == u)[0]
                                    # get seeds with label  u
                                    c = list(a)
                                    # merge region  region_labels[i] and u
                                    c.extend(list(b))
                                    RegionsX[np.array(c)] = region_labels[i]
            # change the label of region u and region_labels[i] to region_labels[i]
                                    #   Stop region growing.
                                    cv_array = Parc.Similarity_Matrix[np.array(c), :]
                                    # get the similarity measure of the new merged region
                                    cv_array = cv_array[:, np.array(c)]
                                    cv_array = cv_array.reshape(-1)
                                    if (len(c) >= self.region_th or \
                                        cv(cv_array) > cvth):
                                        z_shape = np.shape(mesh.connectivity[c, :])
                                        z_shapeT = np.shape(mesh.connectivity[:, c])
            # stop growing region i if it contains more than region_th seeds
                                        mesh.connectivity[c, :] = np.zeros(z_shape)
                                        # by setting rows and columns of region i
                                        mesh.connectivity[:, c] = np.zeros(z_shapeT)
                                        # to zero
                                    break # go to the next region

                    Regions = np.array(RegionsX)
                    SM_vector=self.Statistics_SM(Parc,Regions)
                    # get the similarity values of all pairs inside each region
                    #Stopping criterion if no more merging is found
                    region_labels = np.unique(Regions)
                    if (len(region_labels) == NBR_REGIONS)  or \
                       (int(mesh.connectivity.sum()) == 0):
                        #or (len(region_labels) <= R)  # condition to stop the code.
                        #if the same nbr of region before and after stop iterating
                        Label_all, NBR_REGIONS = self.Add_void(Parc, Reg, Regions,
                                                 Excluded_seeds, self.Label_non_excluded)
                        # add the labels of the excluded and void seeds
                        Labels.append(Label_all) # save the labels at each iteration
                        break  # exit the while loop

                    Label_all, NBR_REGIONS = self.Add_void(Parc, Reg, Regions,
                                             Excluded_seeds, self.Label_non_excluded)
                    # add the labels of the excluded and void seeds
                    nbr_remaining -= 1  # decrease iteration
                    nbr_r.append(NBR_REGIONS)  # append the nbr of regions
                    t.append((time.time()-ATime)/60)
                    # add execution time to the current parcellation
                    mean_v.append(np.mean(SM_vector))  # add the mean of the SM values
                    std_v.append(np.std(SM_vector))   # add the std of the SM values
                    Labels.append(Label_all)
                    nbr="%03d" % (nbr_iteration-nbr_remaining)
                    printData['Iter ,# Reg ,time(m) ,mean ,std'] = nbr ,NBR_REGIONS,\
                    format(t[-1], '.3f'), format(mean_v[-1], '.3f'), \
                    format(std_v[-1], '.3f')
                    self.PrintResults(printData)
                    # print results at each iteration if verbose is true

                # add the zero tractogram to the nearest non zero tractogram
                # merge small regions to the neighbrs with high similarity

                Connectivityx = deepcopy(Connectivity_X[:, np.array(Parc.nonzero_tracto)])
                mesh.connectivity = deepcopy(Connectivityx[np.array(Parc.nonzero_tracto), :])
                NBR = np.unique(Regions)
                Label_all = Regions
                SizeRegion = np.zeros(len(NBR))
                for i in range(len(NBR)):
                    index = np.where(Label_all == NBR[i])[0]
                    SizeRegion[i] = len(index) # get the size of the regions
                    Regions[np.array(index)] = i

                if self.merge == 1:
                    Regions = self.Merge_till_R(Parc, SimilarityMeasure, Reg, SizeRegion,
                              Regions, mesh, R)
                    # merge small regions with the nearest region (highest SM) to have R nbrR
                elif self.merge == 0:
                    Regions = self.Small_Region_st(Parc, SimilarityMeasure, Reg, SizeRegion,
                              Regions, mesh)
                    # merge small regions with the nearest region (highest SM)
                else:
                    pass

                Reg_sm = self.Statistics_SM(Parc,Regions)
                # extract the SM as a vector of the last label
                nbr_r.append(len(np.unique(Regions)))  # append the nbr of regions
                t.append((time.time()-ATime)/60)
                # add execution time to the current parcellation
                mean_v.append(np.mean(Reg_sm))  # add the mean of the SM values
                std_v.append(np.std(Reg_sm))   # add the std of the SM values
                region_labels = np.unique(Regions)
                Regions, NBR_REGIONS = self.Add_void(Parc, Reg, Regions,
                                       Excluded_seeds, self.Label_non_excluded)
                # add the label to void seeds
                Labels.append(Regions) # save the labels at each iteration
                self.Write2file_results(SM, nbr_r, t, mean_v, std_v, R)
                # save results in ./results.txt
                np.savetxt(self.save_results_path+'/Labels_per_iteration.txt',
                           np.transpose(Labels), fmt='%i', delimiter='\t')
                WritePython2Vtk(self.save_results_path+'/Parcellation.vtk',
                Mesh_plot.vertices.T, Mesh_plot.faces.T, Mesh_plot.normal.T, Regions)
                # save the final result in vtk
                nbr="%03d" % (nbr_iteration-nbr_remaining+1)
                key_str = 'Iter ,# Reg ,time(m) ,mean ,std'
                printData[key_str] = nbr , nbr_r[-1], format(t[-1], '.3f'), \
                            format(mean_v[-1], '.3f'), format(std_v[-1], '.3f')
                self.PrintResults(printData)
                # print results at each iteration if verbose is true

    def Small_Region_st(self, Parc, SimilarityMeasures, Reg, SizeRegion,
                        Regions, mesh):  # function used to merge the small regions
        # so that in total you will have regions with the highest structural connecitvity
        Un = np.unique(Regions) # the uniqe labels
        RegionsX = np.array(Regions)
        Reg_small = np.where(SizeRegion < self.region_th)[0]
        # get regions that have a cardinal less than integrated_regions
        X = np.zeros(len(Reg_small))
        nbr_small_rg = len(Reg_small)    # number of small regions
        while nbr_small_rg > 0:  # loop to merge small regions with bigger ones
            for i in range(len(Reg_small)): # loop over the number of small regions
                sth = 0
                insideregion, connected_regions = Reg.Neighbor_region(RegionsX,
                                                  Reg_small[i], mesh)
                # get the neighbors and seeds of region Z[i]
                for j in range(len(connected_regions)): # loop over all regions
                    if (connected_regions[j] != 0):
                        outeregion = np.where(RegionsX == connected_regions[j])[0]
                        S_mean = SimilarityMeasures(Parc, insideregion, outeregion)
                        if (S_mean > sth):
                  # if  connected_regions[j] and Z[i] have high similarity measure
                            sth = S_mean #   merge Z[i] to connected_regions[j]
                            X[i] = connected_regions[j]# merge Z[i] to connected_regions[j]
            RegionsX2 = np.array(RegionsX)
            for i in range(nbr_small_rg): # change the labeling  after the merging
                indx = np.where(RegionsX == Reg_small[i])[0]
                RegionsX2[np.array(indx)] = X[i]
            RegionsX = RegionsX2
            Un = np.unique(RegionsX)# new unique labels
            nbr_r = len(Un) # new number of regions
            SizeRegion = np.zeros(nbr_r)
            RegionX_ = np.zeros(len(RegionsX))
            for i in range(nbr_r): # get the size of the new regions
                ind = np.where(RegionsX == Un[i])[0]
                SizeRegion[i] = len(ind)
                RegionX_[np.array(ind)] = i
            RegionsX = RegionX_
            Reg_small = np.where(SizeRegion <= self.region_th)[0]
            # get the regions with small size
            X = np.zeros(len(Reg_small))
            if len(Reg_small) == nbr_small_rg:
                # break the loop if the pre and actual number of small regions are equal
          	break # stop merging small regions
            nbr_small_rg = len(Reg_small)
        return RegionsX  # label of seeds after merging small regions with big ones.

    def Merge_till_R(self,Parc,SimilarityMeasures,Reg,SizeRegion, Regions, mesh,
                     R_coef):  # function used to merge the small regions
        # so that in total you will have regions with the highest structural connecitvity
        # and total number of regions == R_coef
        Un = np.unique(Regions) # the uniqe labels
        RegionsX = np.array(Regions)
        while len(Un) > R_coef:  # loop to merge small regions with bigger ones
            Reg_small = SizeRegion.argmin()
            sth, X = 0, Reg_small
            insideregion, connected_regions = Reg.Neighbor_region(RegionsX,
                                              Reg_small, mesh)
            # get the neighbors and seeds of region Z[i]
            for j in range(len(connected_regions)): # loop over all regions
                if connected_regions[j] != 0:
                    outeregion = np.where(RegionsX == connected_regions[j])[0]
                    S_mean = SimilarityMeasures(Parc, insideregion, outeregion)
                    if (S_mean > sth):
                  # if  connected_regions[j] and Z[i] have high similarity measure
                        sth = S_mean #   merge Z[i] to connected_regions[j]
                        X = connected_regions[j] # merge Z[i] to connected_regions[j]
            RegionsX2 = np.array(RegionsX)
            RegionsX2[np.array(insideregion)] = X
            Un = np.unique(RegionsX2)# new unique labels
            nbr_r = len(Un) # new number of regions
            SizeRegion = np.zeros(nbr_r)
            RegionX_ = np.zeros(len(RegionsX2))
            for i in range(nbr_r): # get the size of the new regions
                ind = np.where(RegionsX2 == Un[i])[0]
                SizeRegion[i] = len(ind)
                RegionX_[np.array(ind)] = i
            RegionsX = RegionX_
        return RegionsX  # label of seeds after merging small regions with big ones.

    def Statistics_SM(self, Parc, Regions):
        # function used to extract the the similarity measure values between all the pairs
        # in each region as a vector
        Un=np.unique(Regions)
        Reg_SM=[]
        for i in Un:
            ind=np.where(Regions == i)[0]
            cv_array = Parc.Similarity_Matrix[np.array(ind), :]
            cv_array = cv_array[:, np.array(ind)]
            # extract the submatrix of the similarity measure
            cv_array = cv_array.reshape(-1)
            Reg_SM.extend(cv_array) # save in vector form
        return Reg_SM # return the similarity measures values of all pairs inside each region

    def SimMea_all_seed(self, Parc, Mesh_bg):
        # create bigger SM matrix of zero and non zero seeds
        nbr_seeds = np.shape(Mesh_bg.connectivity)[0]
        Similarity_measure = np.eye(nbr_seeds)
        for i in range(len(Parc.nonzero_tracto)):
            index = Parc.nonzero_tracto[i]
            Similarity_measure[index, np.array(Parc.nonzero_tracto)] = Parc.Similarity_Matrix[i, :]
            Similarity_measure[np.array(Parc.nonzero_tracto), index] = Parc.Similarity_Matrix[:, i]

        for i in range(len(Parc.zero_tracto)):
            index = Parc.replacement[Parc.zero_tracto[i]]
            in_list = np.where(np.array(Parc.nonzero_tracto) ==index)[0]
            Similarity_measure[index, np.array(Parc.nonzero_tracto)] = Parc.Similarity_Matrix[in_list, :]
            Similarity_measure[np.array(Parc.nonzero_tracto), index] = Parc.Similarity_Matrix[:, in_list]
        return Similarity_measure