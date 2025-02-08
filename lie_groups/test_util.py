import numpy as np

from .util import components


def test_components_exactly_determined():
    v = np.array([1.0, 2.0, 3.0])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(components(v, b), np.array([1.0, 2.0, 3.0]))


def test_components_underdetermined():
    v = np.array([1.0, 2.0, 3.0])
    b = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    assert np.allclose(components(v, b), np.array([1.0, 2.0]))


def test_components_complex():
    v = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
    b = np.array([[1.0, 0.0, 1.0], [0.0, 1.0j, 1.0j]])
    assert np.allclose(components(v, b), np.array([1.0 + 1.0j, 2.0 - 2.0j]))
