# thic code is used to test parcellation
from MNNparcellation import Python2Vtk as PY
import tempfile as TP
import numpy as np
from MNNparcellation import CSParcellation as CSP
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")


def co_shape(A, B):
    if np.shape(A) == np.shape(B):
        return A, B, 1

    elif np.shape(A) == np.shape(B.T):
        return A, B.T, 1

    else:
        return A, B, 0


def test_python2vtk():
    " Test read and write vtk files "

    nbr_sources = 100
    vertices = np.random.randint(nbr_sources, size=(nbr_sources, 3))
    faces = np.random.randint(nbr_sources, size=(nbr_sources*2, 3))
    normal = np.array(np.random.randn(nbr_sources), dtype=np.float64)
    scalar = range(nbr_sources)
    f = TP.NamedTemporaryFile(delete=True, suffix='.vtk', dir='./data/')
    # write vtk file
    PY.WritePython2Vtk(f.name, vertices, faces, normal, scalar,
                       name_of_scalar="Parcels")
    # read vtk file
    Coordinates, Faces, Scalers, Normal = PY.ReadVtk2Python(f.name)
    Coordinates, vertices, t_c = co_shape(Coordinates, vertices)
    Faces, faces, t_f = co_shape(Faces, faces)
    dec = 4
    if t_f*t_c == 0:
        return False
    else:
        np.testing.assert_almost_equal(Scalers, scalar, decimal=dec)
        np.testing.assert_almost_equal(Faces, faces, decimal=dec)
        np.testing.assert_almost_equal(Coordinates, vertices, decimal=dec)
        np.testing.assert_almost_equal(Normal, normal, decimal=dec)
    return True


def test_similarity():
    " Test similarity measures "
    nbr_seeds = 100
    n_reg = 10
    seed_reg = nbr_seeds//n_reg
    Label = np.zeros(nbr_seeds, dtype=int)
    Connectivity = np.array(np.eye(nbr_seeds))
    coordinate = np.array(np.zeros((nbr_seeds, 3)))
    nbr_partical = 10
    tracto = []
    for i in range(n_reg):
        Label[int(i * seed_reg):int((i + 1)*seed_reg)] = i
        for j in range(seed_reg):
            for k in range(seed_reg):
                Connectivity[int(i * seed_reg + j), int(i * seed_reg + k)] = 1
        # Connectivity[i * seed_reg:(i + 1)*seed_reg,
        # i * seed_reg:(i + 1)*seed_reg] = 1
        for j in range(seed_reg):
            tracto.append(np.array((i+1)*np.ones(nbr_partical)))

    sim = CSP.Parcellation("", "", "", "", write_data=True)
    sim.save_path = './data/'
    Excluded_seeds = []
    sim.PrepareData(coordinate, Connectivity, Excluded_seeds)
    sim.Parc.tractograms = tracto
    sim.Parc.nonzero_tracto = np.array(range(nbr_seeds))
    sim.Parc.zero_tracto = []
    sim.Parc.Similarity_Matrix = np.zeros(nbr_seeds*(nbr_seeds - 1)//2)
    sim.nbr = nbr_seeds
    sim.Connectivity_X = Connectivity
    face = np.array(np.zeros((nbr_seeds, 3)))
    normal = np.zeros(nbr_seeds)
    sim.mesh.faces = face
    sim.mesh.normal = normal
    sim.mesh.vertices = coordinate
    sim.mesh.connectivity = Connectivity
    SM_method = ['Tanimoto', 'Ruzicka', 'Motyka', 'Correlation', 'Cosine',
                 'Roberts']
    Test = {}
    for Q in SM_method:
        sim.Parcellation_agg(coordinate, Connectivity, Excluded_seeds, [n_reg],
                             [Q], sim.mesh, np.inf)
        # SM = util.vec2symmetric_mat(sim.Parc.Similarity_Matrix, nbr_seeds)
        if np.linalg.norm(sim.Labels - Label) != 0:
            print("Test now SM ", Q)
            Test[Q] = False
    return Test

if __name__ == "__main__":
    print("Test Parcellation........................")
    if not test_similarity():
        print('similarity ', colored('Ok', 'green'))
    else:
        print('similarity ', colored('Failed !', 'red'))

    print("Test Write and Read .vtk files ..........")
    if test_python2vtk():
        print('python2vtk ', colored('Ok', 'green'))
    else:
        print('python2vtk ', colored('Failed !', 'red'))
