import numpy as np

from . import e2


def test_of_matrix():
    mat = np.array([[0.0, -1.0, 0.5], [1.0, 0.0, -0.3], [0.0, 0.0, 1.0]])
    elt = e2.GroupElt.of_matrix(mat)
    assert np.allclose(elt.theta, np.pi / 2)
    assert np.allclose(elt.offset, np.array([0.5, -0.3]))
