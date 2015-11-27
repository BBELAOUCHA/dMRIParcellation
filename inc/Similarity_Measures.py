# -*- coding: utf-8 -*-
#####################################################################################
#
# This code contain the different similarity measure used in the cortical surface
#parcellation from dMRI information (tractograms in nii.gz files) using the Mutual
# Nearest Neighbor Condition "see ref.3"

# There are 6 functions (Similarity measures):
#    1.Pearson correlation  4.Tanimoto_SM
#    2.Ruzicka_SM           5.Motyka_SM
#    3.Cosine_SM            6.Roberts
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
# Neighbor condition”,  Submitted, 2016.

# Brahim Belaoucha and Théodore Papadopoulo, “Comparision of dMRI-based cortical surface parcellation
# with different similarity measures”,  Submitted, 2016.
# for more details about the similarity measures you can refer to # Encyclopedia of Distances by: Elena Deza, Michel Marie Deza,2009 #
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################
import numpy as np
from scipy.stats import pearsonr
def Correlation_SM(Parc,region1,region2): # this function computes the average correlation between regions region1,region2
                        # Sum (corr(region1_i,region2_j))/total number of combinations
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    a,b = pearsonr(T1,T2)
                    if np.isnan(a):
                        a=0
                    S+= a
                    Parc.Similarity_Matrix[i,j] = a
                    Parc.Similarity_Matrix[j,i] = a
        return S/(len(region1)*len(region2))
def Ruzicka_SM(Parc,region1,region2):
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    a=np.minimum(T1, T2)
		    b=np.maximum(T1, T2)
                    q=np.sum(a)/np.sum(b)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))
def Cosine_SM(Parc,region1,region2):
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    A=np.dot(np.transpose(T1),T2)
                    a =np.linalg.norm(T1)
                    b =np.linalg.norm(T2)
                    q=A/(a*b)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))
def Tanimoto_SM(Parc,region1,region2):
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    A=np.dot(np.transpose(T1),T2)
                    a =np.linalg.norm(T1)**2
                    b =np.linalg.norm(T2)**2
                    q=A/(a+b-A)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))

def Motyka_SM(Parc,region1,region2):
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    a=np.minimum(T1,T2)
                    b=np.add(T1,T2)
                    q=np.sum(a)/np.sum(b)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))
def Roberts_SM(Parc,region1,region2):
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    b = np.add(T1,T2)
		    Q1=np.minimum(T1,T2)
		    Q2=np.maximum(T1,T2)
                    c=np.divide(Q1,Q2)
		    c[np.isnan(c)]=0.0
                    a=np.multiply(b,c)
                    q=np.sum(a)/np.sum(b)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))
