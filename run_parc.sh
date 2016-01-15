#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.
for i in 1
do
CurrentF=/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i
SaveF=/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/ISBI2016_results/S$i/Parcellation

python run_Parcellation.py  -i "$CurrentF/W${i}_cgal.mat" -o "$SaveF" -t "$CurrentF/tract" -tb "tract_" -seed "$CurrentF/tract/fdt_coordinates.txt" -NR 800,600,400,200,100 -Ex "$CurrentF/Excluded_points.txt" -nodif "$CurrentF/bedpostx.bedpostX/nodif_brain_mask.nii.gz" -sm 'Cosine' -v 0

done
