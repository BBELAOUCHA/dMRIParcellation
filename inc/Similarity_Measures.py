# -*- coding: utf-8 -*-
#####################################################################################
#
# This code contain the different similarity measure used in the cortical surface
#parcellation from dMRI information (tractograms in nii.gz files) using the Mutual
# Nearest Neighbor Condition "see ref.3"

# There are 6 functions (Similarity measures):
#    1.Correlation_SM (Pearson's)     4.Tanimoto_SM
#    2.Ruzicka_SM                  5.Motyka_SM
#    3.Cosine_SM                   6.Roberts
#
#####################################################################################
# BELAOUCHA Brahim
# Copyright (C) 2015 Belaoucha Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, you have to cite 2 of the following work:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface Parcellation via dMRI Using Mutual
#    Nearset Neighbor Condition”, International Symposium on Biomedical Imaging: From Nano to Macro, Prague,
#    Czech Republic. pp. 903-906, April 2016.
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
#   International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), Utrecht, Netherlands, September 2015.

# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################
import numpy as np
from scipy.stats import pearsonr


def Correlation_SM(Parc, region1, region2):  # this function computes the average correlation between regions region1, region2

    # Sum (corr(region1_i,region2_j))/total number of combinations
    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1: # loop over the region 1
        for j in region2: # loop over the region 2
            if Parc.Similarity_Matrix[i, j] != 0.0:# if already computed
                S += Parc.Similarity_Matrix[i, j]
            else:                      # if it was not computed before
                T1 = Parc.tractograms[i] # get the ith tractogram
                T2 = Parc.tractograms[j] # get the jth tractogram
                a, b = pearsonr(T1, T2)  # compute the Pearson's correlation
                if np.isnan(a): # if one of the tractogram is viod similarity is
                    a = 0       # zero
                S += a          # add value to compute the mean similarity measure
                Parc.Similarity_Matrix[i, j] = a # write the similarity value in
                Parc.Similarity_Matrix[j, i] = a # the similarity matrix (symmetric)
    return S/(len(region1)*len(region2)) # return the mean similarity value


def Ruzicka_SM(Parc, region1, region2):# Ruzicka similarity measures [0,..,1]

    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1: # loop over the 1st region
        for j in region2: # loop over the 2nd region
            if Parc.Similarity_Matrix[i, j] != 0.0: #if similarity is already
                S += Parc.Similarity_Matrix[i, j]# computed between ith and jth tractogram
            else: # if similarity was not computed before
                T1 = Parc.tractograms[i] # get the ith tractogram
                T2 = Parc.tractograms[j] # get the jth tractogram
                a = np.minimum(T1, T2)   # compute the Ruzicka similarity measure
		b = np.maximum(T1, T2)
                q = np.sum(a)/np.sum(b)
                if np.isnan(q): # if one of the tractogram is void
                    q = 0       # similarity measure value = 0
                S += q          # sum of the similarity values between region 1 and 2
                Parc.Similarity_Matrix[i, j] = q # write value in the similarity matrix
                Parc.Similarity_Matrix[j, i] = q # matrix is symmetric
    return S/(len(region1)*len(region2)) # return the mean similarity value


def Cosine_SM(Parc, region1, region2):# Cosine similarity measures [0,..,1]

    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1: # loop over region 1
        for j in region2: # loop over region 2
            if Parc.Similarity_Matrix[i, j] != 0.0: # if similarity was computed before
                S += Parc.Similarity_Matrix[i, j]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                A = np.dot(np.transpose(T1), T2) # compute the Cosine similarity between the
                a = np.linalg.norm(T1) #ith and the jth tractogram
                b = np.linalg.norm(T2)
                q = A/(a*b)
                if np.isnan(q): # if one of the tractogram is void
                    q = 0 # similarity value = 0
                S += q # sum of similarity measure
                Parc.Similarity_Matrix[i, j] = q # write the similarity measure in the similarity
                Parc.Similarity_Matrix[j, i] = q # matrix
    return S/(len(region1)*len(region2)) # return the mean similarity measure between region 1 and 2


def Tanimoto_SM(Parc, region1, region2):# Tanimoto similarity measures [0,..,1]

    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1:
        for j in region2:
            if Parc.Similarity_Matrix[i, j] != 0.0:
                S += Parc.Similarity_Matrix[i, j]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                A = np.dot(np.transpose(T1), T2) # compute the Tanimoto similarity
                a = np.linalg.norm(T1)**2
                b = np.linalg.norm(T2)**2
                q = A/(a+b-A)
                if np.isnan(q):
                    q = 0
                S += q
                Parc.Similarity_Matrix[i, j] = q
                Parc.Similarity_Matrix[j, i] = q
    return S/(len(region1)*len(region2))


def Motyka_SM(Parc, region1, region2):# Motyka similarity measures [0,..,1]

    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1:
        for j in region2:
            if Parc.Similarity_Matrix[i, j] != 0.0:
                S += Parc.Similarity_Matrix[i, j]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                a = np.minimum(T1, T2) # Compute Motyka similarity
                b = np.add(T1, T2)
                q = np.sum(a)/np.sum(b)
                if np.isnan(q):
                    q = 0
                S += q
                Parc.Similarity_Matrix[i, j] = q*2
                Parc.Similarity_Matrix[j, i] = q*2
    return S/(len(region1)*len(region2))


def Roberts_SM(Parc, region1, region2):# Roberts similarity measures [0,..,1]

    S = 0 # initialize the mean similarity value between region1 and region2
    for i in region1:
        for j in region2:
            if Parc.Similarity_Matrix[i, j] != 0.0:
                S += Parc.Similarity_Matrix[i, j]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                b = np.add(T1, T2) # compute Roberts similarity
		Q1 = np.minimum(T1, T2)
		Q2 = np.maximum(T1, T2)
                c = np.divide(Q1, Q2)
		c[np.isnan(c)] = 0.0
                a = np.multiply(b, c)
                q = np.sum(a)/np.sum(b)
                if np.isnan(q):
                    q = 0
                S += q
                Parc.Similarity_Matrix[i, j] = q
                Parc.Similarity_Matrix[j, i] = q
    return S/(len(region1)*len(region2))
