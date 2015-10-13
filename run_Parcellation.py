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
# Arguments:
#          -i : Input to the cortical surface '.mat' file containing the Vertices, Faces, and Normals
#          -o : Path to the save results
#          -t : Path to the tractograms '.nii.gz' files
#          -tb: The prefix of the tracto's name: name_x(i)_y(i)_z(i).nii.gz
#          -seed: Path to the seeds coordinate in the diffusion space.
#          -NR: Requested number of regions
#          -ir: Small region threshold
#          -N:  Maximun number of iterations
#          -Sth: Path to the smoothed cortical surface

# If you use this code, you have to cite:
# Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in
# Proceeding of International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.
#
# Author: Brahim Belaoucha 2015
#         Théodore Papadopoulo 2015
######################################################################################
#   Needed packages that must be installed in your computer
from numpy import *
import time
import h5py
import scipy
import numpy as np
from Python2Vtk import WritePython2Vtk
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from mayavi import mlab

import argparse
import sys
import nibabel as nl
from copy import deepcopy
import gc
class Mesh(): # class definition of mesh:-coordinates and tess connectivity

    def __init__(self,vertices,connectivity):
        self.vertices     = vertices
        self.connectivity = connectivity
    def Remove_void_tracto(self,zero_tracto, nonzero_tracto):
        self.connectivity=self.connectivity[nonzero_tracto , :]
        self.connectivity=self.connectivity[: , nonzero_tracto]
        self.vertices=self.vertices[nonzero_tracto,:]

class Regions_processing(): # class process on regions

    def __init__(self,N):   # initialization
        self.regions = []
        for i in range(N):
            self.regions.append(i)

    def Neighbor_region(self, Regions, index, mesh): # Neighboring regions of region index
            insideregion=np.where(np.array(Regions) == index)[0]
            I=mesh.connectivity[insideregion] # get tess connection to region Region == i
            I=np.squeeze(I)
            q=set(np.unique(np.where(I == 1)[len(insideregion)>1]))
            q.difference_update(set(insideregion)) # extract only idices connected to region == i that doesnt belong to i during this iter
            ConnRegions=[Regions[np.array(list(q))[k]] for k in range(len(q))] # get the regions label that are connected to region == i
            connected_regions= np.unique(ConnRegions) # the regions connected to i
            return insideregion,connected_regions

    def Add_zero_tracto_label(self,check_tracto,Regions):# this function add the label of the detected
        nbr_seeds=len(check_tracto.zero_tracto)+len(check_tracto.nonzero_tracto) # void tract as the nearest label of non void tracto
        All_seeds_label=np.array(np.zeros(nbr_seeds,dtype=int))
        for i in range(len(check_tracto.nonzero_tracto)):
            All_seeds_label[check_tracto.nonzero_tracto[i]]=Regions[i]
        for i in check_tracto.zero_tracto:
            All_seeds_label[i]=All_seeds_label[check_tracto.replacement[i]]
        return  All_seeds_label
class Parcellation_data():# class of operations on the tractogram images

    def __init__(self,tract_folder_path,tract_name,mesh,nbr_sample=5000):
        self.tract_path  = tract_folder_path
        self.tract_name  = tract_name
        self.nbr_sample  = nbr_sample
        self.mesh        = mesh
        self.Correlation_Matrix = np.eye(len(mesh.vertices[:,1]))
        self.excluded = []
    def Read_Tracto(self,V): # read the nii.gz tractogram files
        x, y, z  = V
        filename = self.tract_path+'/'+self.tract_name+str(int(x))+'_'+str(int(y))+'_'+str(int(z))+'.nii.gz'
        return nl.load(filename).get_data()
        #T1=Tract.reshape(-1)   # To move with correlation

    def Detect_void_tracto(self):# detecte the void tractograms
        # zero void is defined as the tracto that is less than 3*nbr_samples.
        self.zero_tracto = []
        self.nonzero_tracto = []
        vertices = list(self.mesh.vertices)
        for i in range(len(vertices)):
            seed = vertices[i]
            T = self.Read_Tracto(seed)
            if (np.sum(T) < self.nbr_sample*3):
                self.zero_tracto.extend([i])
            else:
                self.nonzero_tracto.extend([i])

    def Replace_void_tracto(self):#replace the void tractograms by the nearest neighbor non viod
        self.replacement = {}
        for k in self.zero_tracto:
            neighbors = [k]
            visited_neighbors = deepcopy(self.zero_tracto)
            while (neighbors):
                visited_neighbors.extend(neighbors)
                neighbors       = np.unique(np.where(self.mesh.connectivity[neighbors,:]==1)[1])
                A=set(neighbors)
                B=A.difference(set(visited_neighbors))
                valid_neighbors = list(B)
                if len(valid_neighbors)>0:
                    distances = np.zeros(len(valid_neighbors))
                    distances = []
                    for neighbor in valid_neighbors:
                        distances.append(np.linalg.norm(self.mesh.vertices[k,:]-self.mesh.vertices[neighbor,:]))
                    best_neighbor = valid_neighbors[np.array(distances).argmin()]
                    self.replacement[k] = best_neighbor
                    break

    def CalculateSimilarity(self,region1,region2): # this function computes the average correlation between regions region1,region2
                        # Sum (corr(region1_i,region2_j))/total number of combinations
        S = 0
        for i in region1:
            T1 = self.Read_Tracto(self.mesh.vertices[i,:])
            T1=T1.reshape(-1)
            for j in region2:
                if self.Correlation_Matrix[i,j]!=0.0:
                    S += self.Correlation_Matrix[i,j]
                else:
                    T2  = self.Read_Tracto(self.mesh.vertices[j,:])
                    T2=T2.reshape(-1)
                    a,b = pearsonr(T1,T2)
                    if isnan(a):
                        a=0
                    S+= a
                    self.Correlation_Matrix[i,j] = a
                    self.Correlation_Matrix[j,i] = a
        return S/(len(region1)*len(region2))
def Exluded_label(excluded,Label_non_excluded,label_orig):
    Label=np.zeros(len(excluded)+len(Label_non_excluded))
    for i in range(len(Label_non_excluded)):
        Label[label_orig[i]] =Label_non_excluded[i]
    return Label
parser = argparse.ArgumentParser() # Parse the input to the variables
parser.add_argument('-i', action="store",dest='input')
parser.add_argument('-o', action="store",dest='save')
parser.add_argument('-t', action="store",dest='tractograms')
parser.add_argument('-tb', action="store",dest='tract_name')
parser.add_argument('-seed', action="store",dest='coordinates')
parser.add_argument('-NR', action="store",dest='Number_regions')
parser.add_argument('-ir', action="store",dest='integrated_regions')
parser.add_argument('-N', action="store",dest='iteration')
parser.add_argument('-Sth', action="store",dest='smooth')
parser.add_argument('-Ex', action="store",dest='excluded')
Arg=parser.parse_args()
# By defaulf values
if not Arg.Number_regions:
            Arg.Number_regions=200
if not Arg.iteration:
            Arg.iteration =100
if not Arg.smooth:
            SMOOTH=False
if not Arg.integrated_regions:
            Arg.integrated_regions=10
if not (Arg.input or not Arg.coordinates) or (not Arg.save or not Arg.tractograms) or (not Arg.tract_name):
   	    print "Important inputs missing. The code will not run what you required"
   	    print "But the example. stop the code and run again. Press '0' to exit and '1' "
   	    print "to continue"
  	    Arg.input="./test/W1_cgal.mat"
            Arg.smooth="./test/W1_cgal_s.mat"
  	    Arg.coordinates='./test/fdt_coordinates_fsl.txt'
            Arg.save="./test/results"
  	    Arg.tractograms="./test/tract/"
  	    Arg.tract_name="tract_"
            Arg.excluded="./test/Excluded_vertices.txt"
print  "//////////////input arguments ///////////////" # display and check
print "Path input:",Arg.input # vertices, faces and normals in the anatomy space
print "Path Output:",Arg.save # folder to save results
print "Path Tractograms: ",Arg.tractograms # folder to tracto
print "Tracto Begin Name:",Arg.tract_name  # begining of tracto name "name_x_y_z.nii.gz"
print "Path Coordinates",Arg.coordinates   # coordinates in the diffusion space
print "Number Region:",Arg.Number_regions  # Number of regions
print "Region threshold",Arg.integrated_regions # var use to merge small to large regions "now deactivated"
print "Number iteration",Arg.iteration          # number of iterations
print "Smooth Surface",Arg.smooth          # path to the smoothed surface in anatomical space.
print "/////////////////////////////////////////////"
tract_name=str(Arg.tract_name)
cortex=str(Arg.input)
needed_nbr_region=int(Arg.Number_regions)
region_integrate=int(Arg.integrated_regions)
nbr_remaining=int(Arg.iteration)# number of growing iteration
Cortex = h5py.File(cortex,'r')
Vertices=np.array(Cortex['Vertices'])
Normal=np.array(Cortex['VertNormals'])
Faces=np.array(Cortex["Faces"],dtype=int) # used t
cortex=str(Arg.smooth)
Cortex_s = h5py.File(cortex,'r')
Vertices_s=np.array(Cortex_s['Vertices'])
Faces_s=np.array(Cortex_s["Faces"],dtype=int) # used t
del Cortex_s
C=Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity=np.array(D.todense())
del D, C, Cortex
save_results_path=str(Arg.save)# folder to save results
## create sub folders
import os
if not os.path.exists(save_results_path+'/CorrelationMatrix'): # save cross correlation between regions
    os.makedirs(save_results_path+'/CorrelationMatrix')
    print "Folder is created:     ./CorrelationMatrix"
if not os.path.exists(save_results_path+'/RealSurface'): # the cortical regions
    os.makedirs(save_results_path+'/RealSurface')
    print "Folder is created:     ./RealSurface"
if not os.path.exists(save_results_path+'/Regions'):    # folder containg each of the regions separatly
    os.makedirs(save_results_path+'/Regions')
    print "Folder is created:     ./Regions"
if not os.path.exists(save_results_path+'/Region_size'): # folder containg the region size for each iteration
    os.makedirs(save_results_path+'/Region_size')
    print "Folder is created:     ./Region_size"
if not os.path.exists(save_results_path+'/SmoothSurface'):# folder the label of regions in the smoothed cortical surface
    os.makedirs(save_results_path+'/SmoothSurface')
    print "Folder is created:     ./SmoothSurface"
if not os.path.exists(save_results_path+'/RegionCorrected'): # folder containg the corrected regions "small merged to bigger regions"
    os.makedirs(save_results_path+'/RegionCorrected')
    print "Folder is created:     ./RegionCorrected"

coord_path=str(Arg.coordinates)
coordinate=np.loadtxt(coord_path,unpack=True,delimiter='\t',dtype=float).T#read coord
EXCLUDED=np.loadtxt(Arg.excluded,unpack=True,delimiter='\t',dtype=int).T#read coord
LABEL_ORIG=[]
X=len(coordinate[:,0])-len(EXCLUDED)
COORD_ORIG=np.zeros((X,3))
k=0
for i in range(len(coordinate[:,0])):
    if i not in EXCLUDED:
        LABEL_ORIG.append(i)
        COORD_ORIG[k,:]=coordinate[i,:]
        k=k+1
Connectivity=Connectivity[np.array(LABEL_ORIG),:]
Connectivity=Connectivity[:,np.array(LABEL_ORIG)]
coordinate=COORD_ORIG
## MUTUAL NEAREST NEIGHBORS
root=str(Arg.tractograms)
Vertices=Vertices*100
#### initialize Parcellation class

mesh = Mesh(coordinate,Connectivity)
del coordinate, Connectivity
Parc=Parcellation_data(root,tract_name,mesh)
print "********* Looking for zero tractograms ******** \n"
############## look for void tractograms
Parc.Detect_void_tracto() # detect void tractograms

if len(Parc.zero_tracto)>0:
    Parc.Replace_void_tracto() # replace void tractograms
    mesh.Remove_void_tracto(Parc.zero_tracto, Parc.nonzero_tracto)

nbr_seeds=len(mesh.vertices[:,0]) # number of vertices
Reg=Regions_processing(nbr_seeds)
Regions=Reg.regions # initialize regions, each seed is a region.
print "Number of Seeds", nbr_seeds
print "The program found ", len(Parc.zero_tracto)," zero tractograms"
print "********************************************************"

NBR_REGIONS=nbr_seeds # init the nbr of regions
region_th = np.float32(nbr_seeds)/needed_nbr_region        # number used to stop growing big regions
nbr_remaining_i=nbr_remaining
start_time=time.time()
while nbr_remaining > 0: # nbr of iteration
        #   For each region find the best candidate to merge with it. (DRegion)
        Dregion=[] # vector contains the Mutual nearest N
        SizeRegion=np.zeros(NBR_REGIONS)# initialize the size of regions to zeros
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
                        S[l] = Parc.CalculateSimilarity(insideregion,outeregion[l])
                Dregion.append(connected_regions[S.argmax()])
            else:
                Dregion.append(i)
        plt.imshow(Parc.Correlation_Matrix,interpolation='none')
        plt.xlabel('# Region', fontsize=14)
        plt.ylabel('# Region', fontsize=14)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_results_path+'/CorrelationMatrix/CM'+str(nbr_remaining_i-nbr_remaining)+'.pdf', format='pdf')
        plt.show(block=False)
        plt.close('all')

        print "##Now merging %03d step## %0.2f h" %((nbr_remaining_i-nbr_remaining),(-start_time+time.time())/3600)

        #   Idea: do the fusion in the order of the quality of the correlation.
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
                if len(c) > region_th: # stop growing region i if it contains more than region_th dipoles
                    mesh.connectivity[c,:]=np.zeros(np.shape(mesh.connectivity[c,:]))
                    mesh.connectivity[:,c]=np.zeros(np.shape(mesh.connectivity[:,c]))
                    print "Region ", i, " is completed "
        Regions=np.array(RegionsX)
        #   Stopping criterion if the proper number of regions is obtained
        region_labels=np.unique(Regions)
        if (len(region_labels) == NBR_REGIONS) or (len(region_labels) <= needed_nbr_region) or (int(mesh.connectivity.sum()) == 0):   # condition to stop the code. if the same nbr of region before and after stop iterating
            print "We reached the max growing at iteration : ",nbr_remaining_i-nbr_remaining # or reached the needed number of regions
            NBR_REGIONS=len(region_labels)
            SizeRegion=np.zeros(NBR_REGIONS)
            for ii in range(NBR_REGIONS):
                insideregion=np.where(np.array(Regions) == region_labels[ii])[0]
                SizeRegion[ii]=len(insideregion)
                #for k in range(len(insideregion)):
                Regions[np.array(insideregion)]=ii
            Integrated=np.where(SizeRegion <= region_integrate)
            RX=Reg.Add_zero_tracto_label(Parc,Regions) # add the label of the void tractograms
            RX=RX+1
            RX=Exluded_label(EXCLUDED,RX,LABEL_ORIG)
            print "------------- %s hours -------------" %str((time.time() - start_time)/3600)
            print "#########Execution stopped ####### "
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
        RX=Exluded_label(EXCLUDED,RX,LABEL_ORIG)
        savetxt(save_results_path+'/RealSurface/Label_'+str(int(nbr_remaining_i-nbr_remaining))+'.txt',RX)
        WritePython2Vtk(save_results_path+'/RealSurface/Label_'+str(int(nbr_remaining_i-nbr_remaining))+'.vtk', Vertices.T, Faces.T,Normal.T, RX, "Cluster")
        WritePython2Vtk(save_results_path+'/SmoothSurface/Label_'+str(int(nbr_remaining_i-nbr_remaining))+'.vtk', Vertices_s.T, Faces_s.T,Normal.T, RX, "Cluster")
        #savetxt(save_results_path+'/CorrelationMatrix/C_'+str(int(nbr_remaining_i-nbr_remaining))+'.txt',Parc.Correlation_Matrix)
# re size the correlation matrix
        nbr_remaining-=1 # in (de) crease iteration
        gc.collect()
##### add the zero tractogram to the nearest non zero tractogram that is labeled
savetxt(save_results_path+'/CCorrelation_end.txt',Parc.Correlation_Matrix)
Regions=RX
for i in range(len(np.unique(Regions))):
    t=np.zeros(len(Regions))
    insideregion=np.where(np.array(Regions) == i)[0]
    t[insideregion]=i+1
    WritePython2Vtk(save_results_path+'/Regions/R_'+str(i)+'.vtk', Vertices.T, Faces.T,Normal.T, t, "Cluster")
Un=np.unique(Regions)
SizeRegion=np.zeros(len(Un))
print "---------> Start computing the number of dipoles per region"
for i in range(len(Un)):
    SizeRegion[i]=len(np.where(Regions== Un[i])[0])

print "--- %s hours ---" % str((time.time() - start_time)/3600)
plt.figure(figsize=(10,5))
weights = np.ones_like(SizeRegion)/len(SizeRegion)
plt.hist(SizeRegion,weights=weights, histtype='bar',
                            color=['g'],
                            label=['# dipoles/ regions'])
plt.legend()

plt.savefig(save_results_path+'/SizeHisto.png')
plt.close('all')
                # draw the cortical parcellation using 3D mayavi package
t=float16(Regions)#
mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
mlab.triangular_mesh(Vertices[0]*100, Vertices[1]*100,Vertices[2]*100, Faces.T-1, scalars=t,colormap='spectral')
mlab.colorbar()
mlab.savefig(save_results_path+'/Cortex_S1.png')
#mlab.show()
del mesh, Parc
Cortex = h5py.File(str(Arg.input),'r')
C=Cortex['VertConn']
D = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))
Connectivity=np.array(D.todense())
del D, C, Cortex, cortex
Z=np.where(SizeRegion < int(Arg.integrated_regions))[0]
X=np.zeros(len(Z))
Q=len(Z)    # number of small regions
while Q > 0: # loop to merge small regions with bigger ones
        for i in range(len(Z)):
            sth=0
            indx=np.where(Regions == Z[i])[0]
	    CX=Connectivity[indx,:]
            for j in range(len(Un)):
                ind=np.where(Regions == Un[j])[0]
                C=CX[:,ind]
                A=np.sum(C)
                if A > sth and (i!=j):  # NOT THE SAME REGION
                    sth=np.sum(C)
                    X[i] = Un[j]
        for i in range(Q):
            indx=np.where(Regions == Z[i])[0]
            for j in range(len(indx)):
                Regions[indx[j]] = X[i]
        Un=np.unique(Regions)
        nbr_r=len(Un)
        SizeRegion=np.zeros(nbr_r)
        RegionX=np.zeros(len(Regions))
        for i in range(nbr_r):
            SizeRegion[i]=len(np.where(Regions== Un[i])[0])
            ind=np.where(Regions== Un[i])[0]
            for j in range(len(ind)):
                RegionX[ind[j]]=i
        Regions=RegionX
        Z=np.where(SizeRegion < int(Arg.integrated_regions))[0]
        X=np.zeros(len(Z))
	if len(Z) == Q:  # break the loop if the pre and actual number of small regions are equal
		break
        Q=len(Z)         # new number
region_labels=np.unique(Regions)
NBR_REGIONS=len(region_labels)
for k in range(NBR_REGIONS):
            ind=np.where(np.array(Regions) == int(region_labels[k]))[0]
            t=np.zeros(len(Regions))
            t[ind] = k+1
            WritePython2Vtk(save_results_path+'/RegionCorrected/R_'+str(k)+'.vtk', Vertices_s.T, Faces.T,Normal.T, t, "Cluster")
WritePython2Vtk(save_results_path+'/Corrected_Smooth.vtk', Vertices_s.T, Faces_s.T,Normal.T, Regions, "Cluster")
np.savetxt(save_results_path+'/Corrected.txt',Regions,fmt='%i')
WritePython2Vtk(save_results_path+'/Corrected.vtk', Vertices.T, Faces.T,Normal.T, Regions, "Cluster")
plt.figure(figsize=(10,5))
weights = np.ones_like(SizeRegion)/len(SizeRegion)
plt.hist(SizeRegion,weights=weights, histtype='bar',
                                color=['g'],
                                label=['# dipoles/ regions'])
plt.legend()
plt.savefig(save_results_path+'/SizeHistoCorrected.png')
plt.show(block=False)
plt.close('all')
