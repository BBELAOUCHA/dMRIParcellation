#!/bin/sh
python run_Parcellation.py  -i "./test/W1_cgal.mat" -o "./test/results/" -t "./test/tract/" -tb "tract_" -seed "./test/fdt_coordinates_fsl.txt" -NR 1000,800,600,400,200,100 -ir 1 -N 100 -Sth "./test/W1_cgal_s.mat"  -Ex "./test/Excluded_vertices.txt"
