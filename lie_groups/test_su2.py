import numpy as np

from . import su2

def test_group_rep():
    g1 = su2.Group.sample()
    g2 = su2.Group.sample()
    g1g2 = g1 * g2
    for dim in range(1, 6):
        rep = su2.group_rep(dim)
        mat1 = rep(g1)
        mat2 = rep(g2)
        assert np.allclose(mat1 @ mat2, rep(g1g2))
