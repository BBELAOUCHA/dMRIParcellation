# thic code is used to test Python2Vtk.py read and write function
import sys
sys.path.append('../inc/')
from MNNparcellation import Python2Vtk as PY
import tempfile as TP
import numpy as np

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
        f = TP.NamedTemporaryFile(delete=True,suffix='.vtk', dir='./data/')
	# write vtk file
        PY.WritePython2Vtk(f.name, vertices, faces, normal, scalar, name_of_scalar="Parcels")
	# read vtk file
        Coordinates,Faces,Scalers,Normal = PY.ReadVtk2Python(f.name)
      	Coordinates, vertices, t_c =co_shape(Coordinates, vertices)
	Faces, faces, t_f =co_shape(Faces, faces)
	dec = 4
        if t_f*t_c == 0:
        	return False
        else:
		np.testing.assert_almost_equal(Scalers, scalar, decimal=dec)
		np.testing.assert_almost_equal(Faces, faces, decimal=dec)
		np.testing.assert_almost_equal(Coordinates, vertices, decimal=dec)
        	np.testing.assert_almost_equal(Normal, normal, decimal=dec)
		return True

if __name__ == "__main__":
	print "Write and Read vtk files .....", test_python2vtk()
