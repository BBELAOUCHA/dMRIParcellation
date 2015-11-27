#!/bin/sh
### this is an example on how to use the parcellation on the Cluster.
CurrentF=${PWD}

python run_Parcellation.py  -i "$CurrentF/test/W1_cgal.mat" -o "$CurrentF/test/results" -t "$CurrentF/test/tract" -tb "tract_" -seed "$CurrentF/test/fdt_coordinates.txt" -NR 1000,800,600,400,200,100 -Ex "$CurrentF/test/Excluded_vertices.txt" -nodif "$CurrentF/test/nodif_brain_mask.nii.gz" -sm 'Cosine' -cvth 1000000 



