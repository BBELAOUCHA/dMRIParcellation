#!/bin/sh
data_folder=${PWD}
MNNparcellation  -i "$data_folder/data/W_cgal.mat" -o "$data_folder/data/Results/" -t "$data_folder/data/tract" -tb "tract_" -seed "$data_folder/data/tract/fdt_coordinates.txt" -n 10 -Ex "$data_folder/data/Excluded_points.txt" -nodif "$data_folder/data/nodif_brain_mask.nii.gz" -sm 'Tanimoto' -v 1 -m 0

