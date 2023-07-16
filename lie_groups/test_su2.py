import numpy as np
import scipy.linalg as SLA

from . import su2

def test_group_rep():
    g1 = su2.GroupElt.sample()
    g2 = su2.GroupElt.sample()
    g1g2 = g1 @ g2
    for dim in range(1, 6):
        rep = su2.group_rep(dim)
        mat1 = rep(g1)
        mat2 = rep(g2)
        assert np.allclose(mat1 @ mat2, rep(g1g2))

        
def test_alg_rep_exp():
    x = su2.LieAlgElt(np.array([[2.0j, 1.3], [-1.3, -2.0j]], dtype=np.complex128))
    g = x.exp()
    for dim in range(1, 6):
        alg_rep = su2.alg_rep(dim)
        group_rep = su2.group_rep(dim)
        mat1 = SLA.expm(alg_rep(x))
        mat2 = group_rep(g)
        assert np.allclose(mat1, mat2)


def test_alg_rep_bracket():
    x = su2.LieAlgElt(np.array([[1.0j, 6.0 - 3.0j], [-6.0 - 3.0j, -1.0j]], dtype=np.complex128))
    y = su2.LieAlgElt(np.array([[2.0j, 1.3], [-1.3, -2.0j]], dtype=np.complex128))
    for dim in range(1, 6):
        alg_rep = su2.alg_rep(dim)
        mat1 = alg_rep(x)
        mat2 = alg_rep(y)
        bracket1 = mat1 @ mat2 - mat2 @ mat1
        bracket2 = alg_rep(x.bracket(y))
        assert np.allclose(bracket1, bracket2)


def test_set_diagonal():
    arr = np.zeros((2, 2), dtype=np.int64)
    su2.set_diagonal(arr, [1, 1])
    assert np.allclose(arr, np.identity(2, dtype=np.int64))
    su2.set_diagonal(arr, [1], which_diag=1)
    assert np.allclose(arr, np.array([[1, 1], [0, 1]]))
    su2.set_diagonal(arr, [2], -1)
    assert np.allclose(arr, [[1, 1], [2, 1]])
