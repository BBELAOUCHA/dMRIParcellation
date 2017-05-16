#!/bin/sh
data_folder=${PWD}
MNNparcellation  -i "$data_folder/test/data/W_cgal.mat" -o "$data_folder/test/data/Results/" -t "$data_folder/test/data/tract" -tb "tract_" -seed "$data_folder/test/data/tract/fdt_coordinates.txt" -n 10 -Ex "$data_folder/test/data/Excluded_points.txt" -nodif "$data_folder/test/data/nodif_brain_mask.nii.gz" -sm 'Tanimoto' -v 1 -m 0

