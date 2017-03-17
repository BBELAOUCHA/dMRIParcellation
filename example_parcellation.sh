#!/bin/sh
result=${PWD}
CurrentFolder=$result/data/
MNNparcellation  -i "$CurrentFolder/W_cgal.mat" -o "$CurrentFolder/Results/" -t "$CurrentFolder/tract" -tb "tract_" -seed "$CurrentFolder/tract/fdt_coordinates.txt" -n 10 -Ex "$CurrentFolder/Excluded_points.txt" -nodif "$CurrentFolder/nodif_brain_mask.nii.gz" -sm 'Tanimoto' -v 1 -m 0

