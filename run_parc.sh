#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.
for i in 1 2 3 4 5 6 9 1 12 13 14 15
do
CurrentF=/user/bbelaouc/home/Data/WorkShop/Pre-processing/eddy_correction/S$i
SaveF=/user/bbelaouc/home/Data/WorkShop/Results_Thesis
Nodiff_path=/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i
python run_Parcellation.py  -i "$CurrentF/W${i}_cgal.mat" -o "$SaveF/Sub$i/" -t "$CurrentF/tract" -tb "tract_" -seed "$CurrentF/tract/fdt_coordinates.txt" -NR 1000,800,600,400,200 -Ex "$CurrentF/Excluded_points.txt" -nodif "$Nodiff_path/bedpostx.bedpostX/nodif_brain_mask.nii.gz" -sm 'Cosine,Tanimoto,Ruzicka,Motyka,Roberts' -v 1 -m 0

done
