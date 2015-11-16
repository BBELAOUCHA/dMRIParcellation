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
# Different similarity measures that was used to parcellate the cortical surface from dMRI information.


# If you use this code, you have to cite:
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
# Proceeding of International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.
#Belaoucha and Théodore Papadopoulo, “”, in
# Proceeding of ISMRM (ISMRM 2016), 2016.
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################
## import module
import numpy as np
from scipy.stats import pearsonr
# In this code there is the following similarity measures:
# Pearson_Correlation
# Ruzicka
# Roberts
# Motycka
# Tanimoto
# Logit

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
def Ruzicka_SM(Parc,region1,region2):   # Encyclopedia of Distances 2009
        S = 0
        for i in region1:
            for j in region2:
                if Parc.Similarity_Matrix[i,j]!=0.0:
                    S += Parc.Similarity_Matrix[i,j]
                else:
                    T1 = Parc.tractograms[i]
                    T2  = Parc.tractograms[j]
                    a = [np.amin([T1[ix],T2[ix]]) for ix in range(len(T1))]
                    b = [np.amax([T1[ix],T2[ix]]) for ix in range(len(T2))]
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
                    a = [np.amin([T1[ix],T2[ix]]) for ix in range(len(T1))]
                    b = [T1[ix]+T2[ix]            for ix in range(len(T1))]
                    q=2*np.sum(a)/(np.sum(b))
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
                    a=np.zeros(len(T1))
                    for ix in range(len(T1)):
                        if np.amax([T1[ix],T2[ix]]) == 0:
                            a[ix] = 0
                        else:
                            a[ix] = (T1[ix]+T2[ix])*np.amin([T1[ix],T2[ix]])/(np.amax([T1[ix],T2[ix]]))
                    b = [T1[ix]+T2[ix] for ix in range(len(T2))]
                    q=np.sum(a)/np.sum(b)
                    if np.isnan(q):
                        q=0
                    S+= q
                    Parc.Similarity_Matrix[i,j] = q
                    Parc.Similarity_Matrix[j,i] = q
        return S/(len(region1)*len(region2))
