import numpy as np
from MNNparcellation.inc.util import mat2cond_index, cond2mat_index
from MNNparcellation.inc.util import vec2symmetric_mat
from termcolor import colored


def test_mat2cond_index():
    " Test sym matrix to vector indexing "

    n = 10
    X = []
    Z = np.arange(n*(n-1)//2)
    for i in range(n):
        for j in range(i):
            nv = mat2cond_index(n, i, j)
            X.extend([nv])
    return np.array(X).sort() == Z.sort()


def test_cond2mat_index():
    " Test vector 2 sym matrix indexing "

    n = 10
    X = []
    Y = []
    Z = np.arange(n*(n-1)//2)
    for i in range(n):
        for j in range(i):
            X.append([j, i])
    for i in range(n*(n-1)//2):
        nv = cond2mat_index(n, i)
        Y.append(list(nv))
        if X[i].sort() == Y[i].sort():
            Z[i] = 1
    return np.sum(Z) == len(Z)


def test_vec2symmetric_mat():
    " Test vector 2 sym matrix writing "

    n = 3
    X = []
    a = np.random.rand(n, n)
    m = np.tril(a) + np.tril(a, -1).T
    Z = np.zeros(n*(n-1)//2)
    for i in range(n):
        m[i, i] = 1
        for j in range(i):
            nv = mat2cond_index(n, i, j)
            Z[nv] = m[i, j]
    SM = vec2symmetric_mat(Z, n)
    X = np.linalg.norm(m-SM)
    if X == 0.0:
        return True
    return False


if __name__ == "__main__":
    print("Test Util file........................")
    if test_mat2cond_index():
        print( 'mat2cond_index ', colored('Ok', 'green'))
    else:
        print(colored('mat2cond_index ', 'Failed !', 'red'))

    if test_cond2mat_index():
        print('cond2mat_index ', colored('Ok', 'green'))
    else:
        print('cond2mat_index ', colored('Failed !', 'red'))
    if test_vec2symmetric_mat():
        print('vec2symmetric_mat ', colored('Ok', 'green'))
    else:
        print('vec2symmetric_mat ', colored('Failed !', 'red'))
