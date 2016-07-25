# dMRIParcellation

dMRIParcellation is a python toolbox for parcellating the cortical surface using diffusion Magnetic Resonanse Imaging (dMRI) information and the Mutual Nearest Neighbor condition.

The parcellation algorithm needs the mesh connectivity, the tractograms of the seeds (vertices of the mesh) in NIfTi format ".nii.gz", and the coordinate of the seeds in the diffusion space in ".txt" file. The file name of the tractogram of seed i with coordinate (x,y,z) is assumed to follow the results of the probabilistic tractography of FSL i.e. file_prefix_x_y_z.nii.gz.

# Requirements
1-h5py

2-scipy

3-numpy

4-nibabel

# Cite

Brahim Belaoucha, Maurren Clerc and Théodore Papadopoulo, “Cortical Surface Parcellation via dMRI Using Mutual Nearset Neighbor
Condition”, International Symposium on Biomedical Imaging, Apr 2016, Prague, Czech Republic. 2016.

Brahim Belaoucha and Théodore Papadopoulo, “MEG/EEG reconstruction in the reduced source space”, in Proceeding of International Conference on Basic and Clinical Multimodal Imaging (BaCi 2015), 2015.

# Author

Belaoucha Brahim 

Papadopoulo Théodore
