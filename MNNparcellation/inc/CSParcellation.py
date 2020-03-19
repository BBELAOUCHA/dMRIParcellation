# -*- coding: utf-8 -*-
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
from . import Region_preparation as RP
from . import Prepare_tractogram as PT
from . import Similarity_Measures
import numpy as np
from .Python2Vtk import WritePython2Vtk
from copy import deepcopy
import os
from scipy.stats import variation as cv
import time
from .util import mat2cond_index


def Add_void(Parc, Reg, Regions, Excluded_seeds, Label_non_excluded):
    ''' function used to add the labels to the viod tractograms
    Object of the parcellation, Object to region processing class,
    labels, excluded seeds, non excluded seeds'''

    region_labels = np.unique(Regions)
    NBR_REGIONS = len(region_labels)
    SizeRegion = np.zeros(NBR_REGIONS)
    for ii in range(NBR_REGIONS):
        insideregion = np.where(np.array(Regions) == region_labels[ii])[0]
        SizeRegion[ii] = len(insideregion)
        Regions[np.array(insideregion)] = ii

    if len(Parc.zero_tracto) > 0:
        RX = RP.Add_zero_tracto_label(Parc, Regions)
        # add void tractograms
        RX = RX+1  # label {1,..,NBR_REGIONS}
    else:
        RX = Regions

    if len(Excluded_seeds) > 0:
        RX = RP.Excluded_label(Excluded_seeds, RX, Label_non_excluded)
    return RX, NBR_REGIONS


def Merge_till_R(Parc, SimilarityMeasures, Reg, SizeRegion, Regions, mesh,
                 R_coef):
    '''# function used to merge the small regions
    # so that in total you will have regions with the highest structural
    # connecitvity and total number of regions == R_coef'''

    Un = np.unique(Regions)
    # the uniqe labels
    RegionsX = np.array(Regions)
    while len(Un) > R_coef:
        # loop to merge small regions with bigger ones
        Reg_small = SizeRegion.argmin()
        sth, X = 0, Reg_small
        insideregion, connected_regions = RP.Neighbor_region(RegionsX,
                                                             Reg_small, mesh)
        # get the neighbors and seeds of region Z[i]
        for j in range(len(connected_regions)):
            # loop over all regions
            if connected_regions[j] != 0:
                outeregion = np.where(RegionsX == connected_regions[j])[0]
                S_mean = SimilarityMeasures(Parc, insideregion, outeregion)
                if (S_mean > sth):
                    # if  connected_r[j] and Z[i] have high similarity measure
                    sth = S_mean
                    # merge Z[i] to connected_regions[j]
                    X = connected_regions[j]
                    # merge Z[i] to connected_regions[j]
        RegionsX2 = np.array(RegionsX)
        RegionsX2[np.array(insideregion)] = X
        Un = np.unique(RegionsX2)  # new unique labels
        nbr_r = len(Un)  # new number of regions
        SizeRegion = np.zeros(nbr_r)
        RegionX_ = np.zeros(len(RegionsX2))
        for i in range(nbr_r):  # get the size of the new regions
            ind = np.where(RegionsX2 == Un[i])[0]
            SizeRegion[i] = len(ind)
            RegionX_[np.array(ind)] = i
        RegionsX = RegionX_
    return RegionsX  # label of seeds after merging small regions with big one


def Read_from_SM(Parc, ind):
    # this function is used to read sm values from the SM vector (n(n-1)/2)

    cv_array = []
    for iix in ind:
        for jjx in ind:
            if iix != jjx:
                ix = mat2cond_index(Parc.nbr_seeds, iix, jjx)
                cv_array.append(Parc.Similarity_Matrix[ix])

    return np.array(cv_array)


def Statistics_SM(Parc, Regions):
    '''function used to extract the the similarity measure values between all
    the pairs in each region as a vector'''
    Un = np.unique(Regions)
    Reg_SM = []
    for i in Un:
        ind = np.array(np.where(Regions == i)[0])
        cv_array = Read_from_SM(Parc, ind)
        if len(cv_array):
            Reg_SM.extend(cv_array)
        # return the similarity measures values of all pairs inside each
    return np.array(Reg_SM)


class Parcellation():
    # main class to parcellate the cortical surface

    def __init__(self, path_tractogram, Prefix_name, save_path, nodif_mask,
                 VERBOSE=False, merge=0, write_data=True):

        # initialize; prepare the paths
        self.path_tractogram = path_tractogram
        # path tractogram's location
        self.Prefix_name = Prefix_name
        # prefix     Prefix_name_x_y_z.nii.gz
        self.save_path = save_path
        # folder that will contain the results
        self.nodif_mask = nodif_mask
        # path to mask of the brain fron b0 image
        self.Time = []
        # array contains the execution time
        self.save_results_path = '_'
        # folder of each execution
        self.verbose = VERBOSE
        # enable terminal display of the results
        self.merge = merge
        # type of postprocessing (after the MNN parcellation)
        self.Labels = []
        self.write_data = write_data
        self.cvth = np.Inf
    def PrintResults(self, Data):
        # print the different results in the terminal
        if self.verbose:  # The result is saved in a dictionary
            for i in Data.keys():
                print(i, ' = ', Data[i])  # print the dictionary

    def Write2file_results(self, Similarity_Measure, nbr_r, t, mean_v,
                           std_v, R):
        # writesome results of the regions
        # path to save, Similarity measure, nbr of regions, time of execution,
        # mean values of SM, std of SM, stopping condition R.

        if self.write_data:
            resultfile = open(self.save_results_path + '/results.txt', 'w')
            resultfile.write('Similarity Measure:\t' + Similarity_Measure +
                             '\t R_th \t' + str(self.region_th) +
                             '\t R:=' + str(R)+'\n')
            hd = 'nbr i \t nbr R \t t(min) \t mean SM \t STD SM: \n'
            resultfile.write(hd)
            for i in range(len(nbr_r)):
                nbr = "%03d" % nbr_r[i]
                resultfile.write(str(i + 1) + '\t' + nbr + '\t' +
                                 str(t[i]) + '\t' + str(mean_v[i]) +
                                 '\t'+str(std_v[i]) + '\n')
            resultfile.close()

    def Write2file_zero_tracto(self):
        ''' writesome results of the regions
        path to save, Similarity measure, nbr of regions, time of execution,
        mean values of SM, std of SM, stopping condition R.'''

        if self.write_data:
            if len(self.Parc.zero_tracto) > 0:
                resultfile = open(self.save_path + '/zero_tractogram.txt', 'w')
                resultfile.write('index_zero_tracto\t' +
                                 'index_replacement'+'\n')
                zero_t, repla_c = self.Parc.zero_tracto, self.Parc.replacement
                for i in range(len(zero_t)):
                    st = str(zero_t[i]) + '\t' + str(repla_c[zero_t[i]]) + '\n'
                    resultfile.write(st)
                resultfile.close()

    def PrepareData(self, coordinate, Connectivity, Excluded_seeds):
        all_Seeds = np.array([i for i in range(len(coordinate[:, 0]))])
        # they will be removed from the coordinates
        self.nbr_seedsX = len(all_Seeds)  # number of vertices
        self.Label_non_excluded = list(set(all_Seeds))
        # Mesh_back_up = RP.Mesh(coordinate,[],[], Connectivity)
        if len(Excluded_seeds) > 0:
            # if some seeds are exclude from the parcellation
            self.Label_non_excluded = list(set(all_Seeds) -
                                           set(Excluded_seeds))
            # and the tess conneectivity matrix
            self.Label_non_excluded = np.array(self.Label_non_excluded)
            Coord_non_excluded = coordinate[self.Label_non_excluded, :]
            Connectivity = Connectivity[self.Label_non_excluded, :]
            Connectivity = Connectivity[:, self.Label_non_excluded]
            coordinate = Coord_non_excluded

        self.nbr = len(coordinate[:, 0])
        # number of vertices
        self.Connectivity_X = deepcopy(Connectivity)
        # this mesh connectivity will not
        # be modefied used at the end of the code to merge the small regions
        self.mesh = RP.Mesh(coordinate, [], [], Connectivity)
        # create an object containing the coordinate and mesh connectivity
        # Prepare the parcellation by seeting the different paths
        printData = {}
        printData['Loading tractograms '] = str(self.nbr_seedsX)
        self.PrintResults(printData)
        self.Parc = PT.Parcellation_data(self.path_tractogram,
                                         self.Prefix_name, self.mesh,
                                         self.nodif_mask)
        # Parc.Repeated_Coordinate(coordinate)
        self.Parc.Detect_void_tracto()
        # detect zero tracto(tractogram that has sum < 3*max(tractogram))
        if len(self.Parc.zero_tracto) > 0:  # if there are void tractograms
            self.Parc.Replace_void_tracto()
            # replace void tractograms by the nearest neighbor non void
            self.mesh.Remove_void_tracto(np.array(self.Parc.zero_tracto),
                                         np.array(self.Parc.nonzero_tracto))


    def data2bprinted(self, Excluded_seeds, nbr_seeds):
        # This function is used to disp the input info
        # This dictionary is used to save the different results
        printData = {}
        printData['# Excluded seeds:'] = len(Excluded_seeds)
        printData['Path to tractogram:'] = self.path_tractogram
        printData['Prefix name:'] = self.Prefix_name
        printData['Path to nodif mask:'] = self.nodif_mask
        printData['Save path:'] = self.save_path
        n_zero_t = len(self.Parc.zero_tracto)
        printData['# Tracto, # Void tracto'] = nbr_seeds, n_zero_t
        return printData

    def result2bprinted(self, R, SM, nbr_iteration):
        # This function is used to print info of the parcellation

        printData = {}
        printData[' # Region '] = R
        printData['Similarity Measure'] = SM
        printData['# Iterations'] = nbr_iteration
        printData['Stop merging at'] = self.region_th
        return printData

    def find_mergingcondidate(self, NBR_REGIONS, Regions, SimilarityMeasure):
        # dictionary used to display results
        Merg_Condidates = []
        # vector contains the Mutual nearest N condidates
        # between all pairs of regions
        for i in range(NBR_REGIONS):  # loop over the regions
            insideregion, connected_regions = RP.Neighbor_region(Regions, i,
                                                                 self.mesh)
            nbr_connected = len(connected_regions)
            if nbr_connected > 0:
                S = np.zeros(nbr_connected)
                for l in range(nbr_connected):
                                # loop over i neighbors
                    outeregion = np.where(np.array(Regions) ==
                                          connected_regions[l])[0]
                    S[l] = SimilarityMeasure(self.Parc, insideregion,
                                             outeregion)
                Reg_cond = list(np.where(S == S.max())[0])
                Reg_list = [connected_regions[u] for u in Reg_cond]
                Merg_Condidates.append(Reg_list)
            else:
                # if no neighbor i is merged with itself.
                Merg_Condidates.append([i])

        return Merg_Condidates

    def merging_step(self, region_labels, Merg_Condidates, Regions):
        # this function is used to merge condiate regions
        RegionsX = np.array(Regions)
        for i in range(len(region_labels)):
            # check if the mutual nearest neighbor is valid
            Reg_cond = Merg_Condidates[i]
            # candidates of merging to region i
            a = np.where(np.array(Regions) == region_labels[i])[0]
            # get seeds with label  region_labels[i]
            if len(a) < self.region_th:
                for u in Reg_cond:
                    Reg_list = Merg_Condidates[u]
                    if i in Reg_list:
                        # if region i is also a cond merging to u
                        a = np.where(np.array(Regions) ==
                                     region_labels[i])[0]
                        # get seeds with label  region_labels[i]
                        b = np.where(np.array(Regions) == u)[0]
                        # get seeds with label  u
                        c = list(a)
                        # merge region  region_labels[i] and u
                        c.extend(list(b))
                        c = np.array(c)
                        RegionsX[c] = region_labels[i]
                        cv_array = Read_from_SM(self.Parc, c)
                        c_v = cv(cv_array)
                        n_c = len(c)
                        if (n_c >= self.region_th or c_v > self.cvth):
                            dum = self.mesh.connectivity[c, :]
                            z_shape = np.shape(dum)
                            dum = np.zeros(z_shape)
                            self.mesh.connectivity[c, :] = dum
                            # by setting rows and columns of i
                            self.mesh.connectivity[:, c] = dum.T
                            # to zero
                        break
                        # go to the next region
        return RegionsX

    def agg_over_region(self, nbr_seeds,  R, SimilarityMeasure, SM,
                        Excluded_seeds, Connectivity, Mesh_plot,
                        nbr_iteration = 100):
        # run the MNN parcellation over parameter R of the MNN
        key_str = 'Iter, # Reg, time(m), mean, std'
        exe_Time = time.time()  # execution time for each loop
        save_folder = self.save_path + "/" + SM + "/" + str(R)
        self.save_results_path = save_folder
        # folder where to save results
        file_ex = os.path.exists(self.save_results_path)
        if (self.write_data and not file_ex):  # cortical regions
            os.makedirs(self.save_results_path)
            # create the folder if not found
        Reg = RP.Regions_processing(nbr_seeds)
        # initialize the class to handle regions
        Regions = Reg.regions
        # initialize regions,each seed is a region.
        NBR_REGIONS = nbr_seeds
        s_value = np.float32(self.nbr_seedsX - len(Excluded_seeds) -
                             len(self.Parc.zero_tracto))/R
        self.region_th = round(s_value)
        # number used to stop growing big regions
        nbr_remaining = nbr_iteration
        # initialize the remaining number of iterations
        region_labels = np.array([x for x in range(nbr_seeds)])  # inital labeling
        self.mesh.connectivity = deepcopy(Connectivity)
        # re initialize the mesh connectivity
        # vectors that conatin nbrregions, execution time, mean and std
        # of the similarity values at each iteration
        nbr_r, t, mean_v, std_v = [], [], [], []
        Labels = []
        printData = self.result2bprinted(R, SM, nbr_iteration)
        self.PrintResults(printData)
        # disply results if verbose is active
        while nbr_remaining > 0:
            Merg_Condidates = self.find_mergingcondidate(NBR_REGIONS, Regions,
                                                         SimilarityMeasure)
            region_labels = np.unique(Regions)
            RegionsX = self.merging_step(region_labels, Merg_Condidates,
                                         Regions)
            Regions = np.array(RegionsX)
            SM_vector = Statistics_SM(self.Parc, Regions)
            region_labels = np.unique(Regions)
            if (len(region_labels) == NBR_REGIONS):
                L, N = Add_void(self.Parc, Reg, Regions,
                                Excluded_seeds,
                                self.Label_non_excluded)
                Label_all, NBR_REGIONS = L, N
                # add the labels of the excluded and void seeds
                Labels.append(Label_all)
                # save the labels at each iteration
                break  # exit the while loop

            Label_all, NBR_REGIONS = Add_void(self.Parc, Reg, Regions,
                                              Excluded_seeds,
                                              self.Label_non_excluded)
            # add the labels of the excluded and void seeds
            nbr_remaining -= 1  # decrease iteration
            nbr_r.append(NBR_REGIONS)  # append the nbr of regions
            t.append((time.time()-exe_Time)/60)
            # add execution time to the current parcellation
            mean_v.append(np.mean(SM_vector, dtype=np.float64))
            # add the mean of the SM values
            std_v.append(np.std(SM_vector, dtype=np.float64))
            # add the std of the SM values
            Labels.append(Label_all)
            nbr = "%03d" % (nbr_iteration-nbr_remaining)
            t_s, m_v = format(t[-1], '.3f'), format(mean_v[-1], '.3f')
            s_v = format(std_v[-1], '.3f')
            printData[key_str] = nbr, NBR_REGIONS, t_s, m_v, s_v
            self.PrintResults(printData)
            # print results at each iteration if verbose is true

        # add the zero tractogram to the nearest non zero tractogram
        # merge small regions to the neighbrs with high similarity
        X = self.Connectivity_X[:, np.array(self.Parc.nonzero_tracto)]
        Connectivityx = deepcopy(X)
        Y = Connectivityx[np.array(self.Parc.nonzero_tracto), :]
        self.mesh.connectivity = deepcopy(Y)
        NBR = np.unique(Regions)
        Label_all = Regions
        SizeRegion = np.zeros(len(NBR))
        for i in range(len(NBR)):
            index = np.where(Label_all == NBR[i])[0]
            SizeRegion[i] = len(index)  # get the size of the regions
            Regions[np.array(index)] = i

        if self.merge == 1:
            Regions = Merge_till_R(self.Parc, SimilarityMeasure, Reg,
                                   SizeRegion, Regions, self.mesh, R)
        # merge small regions with the highest SM to have R nbrR
        elif self.merge == 0:
            Regions = self.Small_Region_st(self.Parc,
                                           SimilarityMeasure, Reg,
                                           SizeRegion, Regions,
                                           self.mesh)
            # merge small regions with the nearest region (highest SM)
        else:
            pass

        Reg_sm = Statistics_SM(self.Parc, Regions)
        # extract the SM as a vector of the last label
        nbr_r.append(len(np.unique(Regions)))
        # append the nbr of regions
        t.append((time.time()-exe_Time)/60)
        # add execution time to the current parcellation
        mean_v.append(np.mean(Reg_sm, dtype=np.float64))
        # add the mean of the SM values
        std_v.append(np.std(Reg_sm, dtype=np.float64))
        # add the std of the SM values
        region_labels = np.unique(Regions)
        Regions, NBR_REGIONS = Add_void(self.Parc, Reg, Regions,
                                        Excluded_seeds,
                                        self.Label_non_excluded)
        # add the label to void seeds
        Labels.append(Regions)
        # save the labels at each iteration
        if self.write_data:
            self.Write2file_results(SM, nbr_r, t, mean_v, std_v, R)
        # save results in ./results.txt
            path_dum = self.save_results_path
            path_dum += '/Labels_per_iteration.txt'
            np.savetxt(path_dum, np.array(Labels).T, fmt='%i',
                       delimiter='\t')
            WritePython2Vtk(self.save_results_path+'/Parcellation.vtk',
                            Mesh_plot.vertices.T, Mesh_plot.faces.T,
                            Mesh_plot.normal.T, Regions)
        # save the final result in vtk
        nbr = "%03d" % (nbr_iteration-nbr_remaining+1)
        t_s, m_v = format(t[-1], '.3f'), format(mean_v[-1], '.3f')
        s_v = format(std_v[-1], '.3f')
        printData[key_str] = nbr, nbr_r[-1], t_s, m_v, s_v
        self.PrintResults(printData)
        self.Labels = Regions

    def Parcellation_agg(self, coordinate, Connectivity, Excluded_seeds,
                         NbrRegion, SM_method, Mesh_plot, cvth):
        # diffusion coordinates, mesh connectivity , array of exclueded seeds,
        # array of nbr of regions, array of similarity measures, mesh used
        # to generate vtk files, variation coefficient threshold
        # remove the void tractogram to speed up the computation.
        nbr_seeds = self.nbr
        printData = self.data2bprinted(Excluded_seeds, nbr_seeds)
        self.Write2file_zero_tracto()
        self.cvth = cvth
        Connectivity = deepcopy(self.mesh.connectivity)
        # hard (not shallow) copy
        # new mesh connec after removing void tractogram used at each R and SM
        nbr_iteration = 100
        # total number of iterations fixed, generally 50 is enough
        self.PrintResults(printData)  # print the results so far
        nbr_seeds = len(self.mesh.vertices[:, 0])  # number of vertices
        self.Parc.nbr_seeds = nbr_seeds
        for SM in SM_method:
            # loop over the list containing the name of the similarity measures
            SimilarityMeasure = getattr(Similarity_Measures, SM+'_SM')
            # call the function of the similarity measures SM
            self.Parc.Similarity_Matrix = np.zeros(nbr_seeds*(nbr_seeds-1)//2,
                                                   dtype=np.float16)
            # reinit the Similarity_Matrix for the next similarity measure
            for R in NbrRegion:
                # loop over the list containing the number of regions "R"
                self.agg_over_region(nbr_seeds, R, SimilarityMeasure, SM,
                                     Excluded_seeds, Connectivity, Mesh_plot,
                                     nbr_iteration)

    def Small_Region_st(self, Parc, SimilarityMeasures, Reg, SizeRegion,
                        Regions, mesh):
        '''function used to merge the small regions
         so that in total you will have regions with the highest
        structural connecitvity'''
        Un = np.unique(Regions)
        RegionsX = np.array(Regions)
        Reg_small = np.where(SizeRegion < self.region_th)[0]
        # get regions that have a cardinal less than integrated_regions
        X = np.zeros(len(Reg_small))
        nbr_small_rg = len(Reg_small)
        while nbr_small_rg > 0:
            # loop to merge small regions with bigger ones
            for i in range(len(Reg_small)):
                # loop over the number of small regions
                sth = 0
                insideregion, connected_r = RP.Neighbor_region(RegionsX,
                                                               Reg_small[i],
                                                               mesh)
                # get the neighbors and seeds of region Z[i]
                for j in range(len(connected_r)):  # loop over all regions
                    if (connected_r[j] != 0):
                        outeregion = np.where(RegionsX == connected_r[j])[0]
                        S_mean = SimilarityMeasures(Parc, insideregion,
                                                    outeregion)
                        if (S_mean > sth):
                            sth = S_mean
                            # merge Z[i] to connected_r[j]
                            X[i] = connected_r[j]
                            # merge Z[i] to connected_r[j]
            RegionsX2 = np.array(RegionsX)
            for i in range(nbr_small_rg):
                # change the labeling  after the merging
                indx = np.where(RegionsX == Reg_small[i])[0]
                RegionsX2[np.array(indx)] = X[i]

            RegionsX = RegionsX2
            Un = np.unique(RegionsX)
            # new unique labels
            nbr_r = len(Un)
            # new number of regions
            SizeRegion = np.zeros(nbr_r)
            RegionX_ = np.zeros(len(RegionsX))
            for i in range(nbr_r):
                # get the size of the new regions
                ind = np.where(RegionsX == Un[i])[0]
                SizeRegion[i] = len(ind)
                RegionX_[np.array(ind)] = i

            RegionsX = RegionX_
            Reg_small = np.where(SizeRegion <= self.region_th)[0]
            # get the regions with small size
            X = np.zeros(len(Reg_small))
            if len(Reg_small) == nbr_small_rg:
                # break the loop if the pre and actual number is equal
                break
            # stop merging small regions
            nbr_small_rg = len(Reg_small)

        return RegionsX
        # label of seeds after merging small regions with big ones.
