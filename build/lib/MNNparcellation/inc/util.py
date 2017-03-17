import numpy as np
def cond2mat_index(n, i):
    ''' Indexes (i, j) of element i in a n x n condensed matrix '''
    b = 1. - 2.*n
    x = np.floor((-b - np.sqrt(b**2. - 8.*i))/2.)
    y = i + x*(b+x+2.)/2. + 1.
    return int(x), int(y)

def mat2cond_index(n,  i,  j):
    ''' Calculate the condensed index of element (i, j) in an n x n condensed
        matrix. '''
    ix = min(i, j)
    iy = max(i, j)
    return n * ix - (ix * (ix + 1) / 2) + (iy - ix - 1)

def vec2symmetric_mat(V, n):
    ''' Write vector to matrix '''
    SM = np.eye(n)
    for x in xrange(len(V)):
	i, j = cond2mat_index(n, x)
	SM[i, j] = V[x]
        SM[j, i] = V[x]
    
    return SM
