#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.
for i in 1
do
echo "Subject # $i"
CurrentF=/user/bbelaouc/home/Data/WorkShop/Pre-processing/eddy_correction/S$i
SaveF=/user/bbelaouc/home/Data/WorkShop/Results_Thesis/With_postprocessing/Sub$i
Nodiff_path=/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i
MNNparcellation  -i "$CurrentF/W${i}_cgal.mat" -o "$SaveF/" -t "$CurrentF/tract" -tb "tract_" -seed "$CurrentF/tract/fdt_coordinates.txt" -n 450 -Ex "$CurrentF/Excluded_points.txt" -nodif "$Nodiff_path/bedpostx.bedpostX/nodif_brain_mask.nii.gz" -sm 'Tanimoto' -v 1 -m 0
done
