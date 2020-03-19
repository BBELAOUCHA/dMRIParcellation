# thic code is used to test prepare tracto
import numpy as np
from MNNparcellation import Prepare_tractogram as PT
from MNNparcellation import Region_preparation as RP
from MNNparcellation.inc.CSParcellation import Parcellation as CSP
from termcolor import colored
import os
import h5py
import scipy
import warnings
warnings.filterwarnings("ignore")


def test_readnibabel():
    " Test read nibabel "
    cwd = os.getcwd()
    cwd += '/data/'
    tractograms = cwd + '/tract/'
    tract_name = 'tract_'
    nodif = cwd + '/nodif_brain_mask.nii.gz'
    Test = PT.Parcellation_data(tractograms, tract_name, [], nodif)

    return len(Test.non_zeroMask) != 0


def test_preparetractogram():
    " Test prepare tracto "
    cwd = os.getcwd()
    cwd += '/data/'
    coordinates = cwd + '/tract/fdt_coordinates.txt'
    tractograms = cwd + '/tract/'
    tract_name = 'tract_'
    save = cwd
    Input = cwd + '/W_cgal.mat'
    nodif = cwd + '/nodif_brain_mask.nii.gz'
    Exclude = cwd + '/Excluded_points.txt'
    coordinate = np.loadtxt(str(coordinates), unpack=True, delimiter='\t',
                            dtype=int).T
    # read the diffusion space coordinate of the
    x, y = np.shape(coordinate)
    if y > x:
        coordinate = coordinate.T
    exclude = np.loadtxt(str(Exclude), unpack=True, delimiter='\t', dtype=int)
    Cortex = h5py.File(Input, 'r')
    # load the details of the mesh, coordinate, faces, normal, connecticity.
    normal_plot = np.array(Cortex['VertNormals'])
    # get the normals in the anatomical space
    faces_plot = np.array(Cortex["Faces"], dtype=int)
    # get faces of the mesh in the anatomical space.
    vertices_plot = np.array(Cortex['Vertices'])
    # get the coordinate in the anatomy image
    if faces_plot.min() > 0:
        faces_plot = faces_plot - 1

    C = Cortex['VertConn']  # get the tess connectivity matrix
    D_conenct = scipy.sparse.csc_matrix((C['data'], C['ir'], C['jc']))  #
    Connectivity = np.array(D_conenct.todense(), np.int8)
    Ground_t = [6, 10]
    test = [True, True]
    for i in range(2):
        Parcel = CSP(tractograms, tract_name, save, nodif, False,  i, True)
        Parcel.PrepareData(coordinate, Connectivity, exclude)
        Mesh_plot = RP.Mesh(vertices_plot, faces_plot, normal_plot)
        # define the mesh to be used to generate the vtk file
        Parcel.Parcellation_agg(coordinate, Connectivity, exclude, [10],
                                ['Cosine'], Mesh_plot, np.Inf)
        # run the parcellation algorithm
        if np.max(Parcel.Labels) != Ground_t[i]:
            test[i] = False

    return test


if __name__ == "__main__":
    t = "Test read tractogram ........................................."
    test = test_readnibabel()
    if not test:
        print(t, colored('Failed!', 'red'))
    else:
        print(t, colored('Ok', 'green'))

    test = test_preparetractogram()
    for i in range(len(test)):
        t = "Test Prepare tractogram with merge = %2d..............." % (i)
        if not test[i]:
            print(t, colored('Failed!', 'red'))
        else:
            print(t, colored('Ok', 'green'))
