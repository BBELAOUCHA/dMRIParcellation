#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.

###PBS -l walltime=02:00:00,nodes=1:ppn=1,pmem=35000mb
###PBS -N Parcellation
###PBS -m abe
###PBS -e Error_Parcellation.e
###PBS -o Output_Parcellation.o
for i in 1 
do
python run_Parcellation.py  -i "/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i/W$i""_cgal.mat" -o "/home/bbelaouc/Data/WorkShop/Parcellation2/S$i" -t "/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i/tract" -tb "tract_" -seed "/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i/tract/fdt_coordinates.txt" -NR 1000,800,600,400,200,100 -Ex "/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i/Excluded_points.txt" -nodif "/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i/bedpostx.bedpostX/nodif_brain_mask.nii.gz" -sm 'Cosine' -cvth 1000000 

done

##### number of regions 1000 and 100
##### to send the job in the cluster type qsub run_parc.pbs
