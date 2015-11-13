#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.

#PBS -l walltime=02:00:00,nodes=1:ppn=1,pmem=35000mb
#PBS -N Parcellation
#PBS -m abe
#PBS -e Error_Parcellation.e
#PBS -o Output_Parcellation.o


python run_Parcellation.py  -i "./test/W1_cgal.mat" -o "./test/results/" -t "./test/tract/" -tb "tract_" -seed "./test/fdt_coordinates_fsl.txt" -NR 1000,100 -Sth "./test/W1_cgal_s.mat"  -Ex "./test/Excluded_vertices.txt"

##### number of regions 1000 and 100
##### to send the job in the cluster type qsub run_parc.pbs
