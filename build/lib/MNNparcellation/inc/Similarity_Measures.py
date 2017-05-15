# -*- coding: utf-8 -*-
'''
###################################################################################
#
#This code contain the different similarity measure used in the cortical surface
#parcellation from dMRI information (tractograms in nii.gz files) using the
# Mutual Nearest Neighbor Condition

# There are 6 functions (Similarity measures):
#    1. Correlation_SM (Pearson's)  4.Tanimoto_SM
#    2. Ruzicka_SM                  5. Motyka_SM
#    3. Cosine_SM                   6. Roberts_SM
#
###################################################################################
# BELAOUCHA Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, please acknowledge Brahim Belaoucha. The best single reference is:
# Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface Parcell
# ation via dMRI Using Mutual Nearset Neighbor Condition”, International Symposium on
# Biomedical Imaging: From Nano to Macro, Prague, Czech Republic. pp. 903-906,Apr 2016.

# Author: Brahim Belaoucha 2015
# Any questions, please contact brahim.belaoucha@gmail.com
###################################################################################
'''
import numpy as np
from scipy.stats import pearsonr
from util import mat2cond_index


def Correlation_SM(Parc, region1, region2):
    # computes the average correlation between regions region1, region2
    # Sum(corr(region1_i,region2_j))/total number of combinations
    S = 0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:  # loop over the region 1
        for j in region2:  # loop over the region 2
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:  # if already computed
                S += Parc.Similarity_Matrix[ix]
            else:                      # if it was not computed before
                T1 = Parc.tractograms[i]  # get the ith tractogram
                T2 = Parc.tractograms[j]  # get the jth tractogram
                a = pearsonr(T1, T2)[0]  # compute the Pearson's correlation
                if np.isnan(a):  # if one of the tractogram is viod similarity is
                    a = 0       # zero
                S += a          # add value to compute the mean similarity measure
                Parc.Similarity_Matrix[ix] = a  # write the similarity value in

    return S/(len(region1)*len(region2))  # return the mean similarity value


def Ruzicka_SM(Parc, region1, region2):
    # Compute Ruzicka similarity measures between two regions. values in [0,..,1]
    S = 0.0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:  # loop over the 1st region
        for j in region2:  # loop over the 2nd region
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:  # if similarity is already
                S += Parc.Similarity_Matrix[ix]  # comp between i and j tractogram
            else:  # if similarity was not computed before
                T1 = Parc.tractograms[i]  # get the ith tractogram
                T2 = Parc.tractograms[j]  # get the jth tractogram
                a = np.minimum(T1, T2)   # compute the Ruzicka similarity measure
		b = np.maximum(T1, T2)
		q = 0.0
		if np.sum(b) is not 0.0:
		    q = np.sum(a)/np.sum(b)
                S += q  # sum of the similarity values between region 1 and 2
                Parc.Similarity_Matrix[ix] = q  # write value in similarity matrix
                #Parc.Similarity_Matrix[j, i] = q # matrix is symmetric
    return S/(len(region1)*len(region2))  # return the mean similarity value


def Cosine_SM(Parc, region1, region2):
    # Cosine similarity measures [0,..,1]
    S = 0.0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:  # loop over region 1
        for j in region2:  # loop over region 2
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:  # if similarity was computed before
                S += Parc.Similarity_Matrix[ix]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                A = np.dot(np.transpose(T1), T2)  # compute the Cosine between i and j
                a = np.linalg.norm(T1)
                b = np.linalg.norm(T2)
                q = 0.0
                if a*b != 0.0:
                    q = A/(a * b)

                S += q  # sum of similarity measure
                Parc.Similarity_Matrix[ix] = q  # write into similarity matrix

    return S/(len(region1)*len(region2))  # mean similarity between region 1&2


def Tanimoto_SM(Parc, region1, region2):
    # Tanimoto similarity measures [0,..,1]
    S = 0.0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:
        for j in region2:
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:
                S += Parc.Similarity_Matrix[ix]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                A = np.dot(np.transpose(T1), T2)  # compute the Tanimoto similarity
                a = np.linalg.norm(T1)**2
                b = np.linalg.norm(T2)**2
                x = (a + b - A)
                q = 0.0
                if x != 0.0:
                    q = A/x

                S += q
                Parc.Similarity_Matrix[ix] = np.float16(q)

    return S/(len(region1)*len(region2))


def Motyka_SM(Parc, region1, region2):
    # Motyka similarity measures [0,..,1]
    S = 0.0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:
        for j in region2:
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:
                S += Parc.Similarity_Matrix[ix]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                a = np.minimum(T1, T2)  # Compute Motyka similarity
                b = np.add(T1, T2)
                q = 0.0
                if np.sum(b) is not 0.0:
                    q = np.sum(a)/np.sum(b)
                S += q
                Parc.Similarity_Matrix[ix] = q*2

    return S/(len(region1)*len(region2))


def Roberts_SM(Parc, region1, region2):
    # Roberts similarity measures [0,..,1]
    S = 0.0  # initialize the mean similarity value between region1 and region2
    n = Parc.nbr_seeds
    for i in region1:
        for j in region2:
            ix = mat2cond_index(n,  i,  j)
            if Parc.Similarity_Matrix[ix] != 0.0:
                S += Parc.Similarity_Matrix[ix]
            else:
                T1 = Parc.tractograms[i]
                T2 = Parc.tractograms[j]
                b = np.add(T1, T2)  # compute Roberts similarity
		Q1 = np.minimum(T1, T2)
		Q2 = np.maximum(T1, T2)
                c = np.divide(Q1, Q2, dtype=float)
		c[np.isnan(c)] = 0.0
                a = np.multiply(b, c)
                b = np.sum(b)
                q = 0.0
                if b is not 0.0:
                    q = np.sum(a)/b

                S += q
                Parc.Similarity_Matrix[ix] = q

    return S/(len(region1)*len(region2))
