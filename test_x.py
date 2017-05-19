#!/usr/bin/env python
'''
################################################################################
################################################################################
# BELAOUCHA Brahim
# Version 1.0
# Inria Sophia Antipolis
# University of Nice Sophia Antipolis
# brahim.belaoucha@inria.fr
# belaoucha.brahim@etu.unice.fr
# If you use this code, please acknowledge Brahim Belaoucha.
# The best single reference is:
# Brahim Belaoucha, Maurren Clerc and Th\'eodore Papadopoulo, "Cortical Surface
# Parcellation via dMRI Using Mutual Nearset Neighbor Condition", International
# Symposium on Biomedical Imaging: From Nano to Macro, Prague, Czech Republic.
# pp. 903-906, Apr.2016.
# Author: Brahim Belaoucha 2015
# Any questions, please contact brahim.belaoucha@gmail.com
################################################################################
################################################################################
'''
import h5py
import scipy
import numpy as np
import argparse
from MNNparcellation.inc import Region_preparation as RP
from MNNparcellation.inc.CSParcellation import Parcellation as CSP


def check_input(args):

    test_none = False
    if (args.save is None or args.tractograms is None or
        args.tract_name is None or args.coordinates is None or
        args.SM is None or args.NR is None):
        test_none = True
    return test_none

def run_parcellation(mesh_file, out_folder, tract_folder, tract_prefix,
                     coord_file, exec_file, sim_m, nbr_r, coef_var, file_mask,
                     merge_v = 0, verbose = 0):
        """run_parcellation is used to parcellate the whole/part of the brain
        cortical surface using structural information based on dMRI information
        input: 1-mesh_file: mat file which contains the surface mesh details
               2-output_folder: path to folder where to save results
               3-tract_folder: folder to tractograms in nifti format
               4-tract_prefix: prefix of the tracto files prefix_x_y_z.nii,gz
               5-coord_file: ascii file containing the(x,y,z) coord of seeds
               6-exec_file: excluded points from the parcellation algo
               7-sim_m: similarity measure to compare tractograms
               8-nbr_r: parameter of algo used to stop merging big regions
               9-coef_var: parameter used to stop merging imhomogenous regions
               10-file_mask: nifti mask used to reduce the memory usage
               11-merge_v: postprocessing choice
               12-verbose: display results"""

        coordinate = np.loadtxt(str(coord_file), unpack=True, delimiter='\t',
                                dtype=int).T
        x, y = np.shape(coordinate)
        if y > x:
            coordinate = coordinate.T
        '''# read the diffusion space coordinate of the seeds'''
        Cortex = h5py.File(str(mesh_file), 'r')
        '''#load details of mesh, coordinate, faces, normal,connecticity.'''
        vertices_plot = np.array(Cortex['Vertices'])
        '''#get the coordinate in the anatomy image'''
        normal_plot, faces_plot, Excluded_seeds = None, None, None
        if "VertNormals" in Cortex.keys():
            normal_plot = np.array(Cortex['VertNormals'])
            '''# get the normals in the anatomical space'''

        if "Faces" in Cortex.keys():
            faces_plot = np.array(Cortex["Faces"], dtype=int)
            '''# get faces of the mesh in the anatomical space.'''
            if faces_plot.min() > 0:
                faces_plot = faces_plot - 1

        if "VertConn" in Cortex.keys():
            C = Cortex['VertConn']
            '''# get the tess connectivity matrix'''
            D_conenct = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
            Connectivity = np.array(D_conenct.todense(), np.int8)
            del D_conenct, C, Cortex
            '''# delete unused data for memory reason'''
        if exec_file is not None:
            Excluded_seeds = np.loadtxt(exec_file, dtype=int)
        '# get the list of the excluded seeds'
        '''############# Parcellation starts here ######################'''
        Verbose = False
        '# by default dont display any results'
        if verbose:
            Verbose = True

        cvth = np.Inf
        'default not included in the stopping criteria'
        if coef_var:
            cvth = coef_var
        'threshold used to stop merging regions with low homogeneity'

        Regions = [len(coordinate[:, 0]) - len(Excluded_seeds)]
        'default number of regions'
        if nbr_r:
            Regions = [int(item) for item in nbr_r.split(',')]

        SM = ['Cosine']
        # Default similarity measure, cosine similarity
        if sim_m:
            SM = [item for item in sim_m.split(',')]
        # list conatining the wanted similarity measures
        merge = 2
        if merge_v:
            merge = merge_v

        Parcel = CSP(tract_folder, tract_prefix, out_folder, file_mask,
                     Verbose, merge)
        # initialize the parcellation by specifying the different paths
        Mesh_plot = RP.Mesh(vertices_plot, faces_plot, normal_plot)
        # define the mesh to be used to generate the vtk file
        del vertices_plot, faces_plot, normal_plot
        Parcel.PrepareData(coordinate, Connectivity, Excluded_seeds)
        Parcel.Parcellation_agg(coordinate, Connectivity, Excluded_seeds,
                                Regions, SM, Mesh_plot, cvth)
        # run the parcellation algorithm


path = '/home/bbelaouc/dMRIParcellation/test/data/'
inputs = path + 'W_cgal.mat'
save = path + 'Results'
tractograms = path + '/tract/'
tract_name = 'tract_'
coordinates = path + '/tract/fdt_coordinates.txt'
excluded = path + '/Excluded_points.txt'
SM=0
NR='100,100'
cv=np.Inf
nodif=path + 'nodif_brain_mask.nii.gz'
run_parcellation(inputs, save, tractograms, tract_name, coordinates, excluded,
                 SM, NR, cv, nodif, 1, 1)
