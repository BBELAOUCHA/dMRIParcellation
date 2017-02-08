#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.
for i in 1
do
echo "Subject # $i"
CurrentF=/home/bbelaouc/Data/WorkShop/Results_Thesis/Weights_matrices/sWMNE
SaveF=/home/bbelaouc/Data/WorkShop/Results_Thesis/Weights_matrices/sWMNE/parcellation
Nodiff_path=/home/bbelaouc/Data/WorkShop/Pre-processing/eddy_correction/S$i
python run_Parcellation.py  -i "$CurrentF/sWMNE.mat" -o "$SaveF/" -t "$CurrentF/tract" -tb "tract_" -seed "$CurrentF/tract/fdt_coordinates.txt" -NR 100,200,300,400,500,600,700,800,900,1000 -Ex "$CurrentF/Excluded_points.txt" -nodif "$Nodiff_path/bedpostx.bedpostX/nodif_brain_mask.nii.gz" -sm 'Cosine,Tanimoto,Motyka,Ruzicka,Roberts' -v 1 -m 0
done
