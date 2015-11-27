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
# Whole brain parcellation algorithm based on Mutual Nearest Neighbors.
# This peace of code shows how you can call the parcellation method from terminal

# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
# # INRIA Sophia Antipolis Méditerranée # #
######################################################################################

import sys
sys.path.append("./inc/")
import time
import h5py
import scipy
import numpy as np
import argparse
import Region_preparation as RP
from Cortical_surface_parcellation import Parcellation as CSP

parser = argparse.ArgumentParser() # Parse the input to the variables
parser.add_argument('-i', action="store",dest='input')
parser.add_argument('-o', action="store",dest='save')
parser.add_argument('-t', action="store",dest='tractograms')
parser.add_argument('-tb', action="store",dest='tract_name')
parser.add_argument('-seed', action="store",dest='coordinates')
parser.add_argument('-Ex', action="store",dest='excluded')
parser.add_argument('-sm', '--SM', help='delimited list input', type=str)
parser.add_argument('-NR', '--list', help='delimited list input', type=str)
parser.add_argument('-cvth', action="store",dest='cv')
parser.add_argument('-nodif', action="store",dest='nodif')
Arg=parser.parse_args()
Regions = [int(item) for item in Arg.list.split(',')]
SM = [item for item in Arg.SM.split(',')]
coordinate=np.loadtxt(str(Arg.coordinates),unpack=True,delimiter='\t',dtype=float).T
Cortex = h5py.File(str(Arg.input),'r')
Vertices=np.array(Cortex['Vertices'])
Normal=np.array(Cortex['VertNormals'])
Faces=np.array(Cortex["Faces"],dtype=int) # used t
C=Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity=np.array(D.todense())
del D, C, Cortex
Excluded_seeds=np.loadtxt(Arg.excluded)
vertices_plot=Vertices
faces_plot=Faces
normal_plot=Normal
del Normal, Faces, Vertices
################ Parcellation starts here #########################################
Parcel=CSP(Arg.tractograms,Arg.tract_name,Arg.save, Arg.nodif,6000)
Mesh_plot=RP.Mesh(vertices_plot,[],faces_plot,normal_plot)
Parcel.Parcellation_agg(coordinate,Connectivity, Excluded_seeds,Regions,SM,Mesh_plot,float(Arg.cv))
