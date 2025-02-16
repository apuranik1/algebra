from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Type

import numpy as np
import numpy.typing as npt
import numpy.linalg as LA

from . import types

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Z = np.diag([1.0, -1.0])

PAULI = np.stack([X, Y, Z], axis=0)

# skew-hermitian matrices with the relation [S1, S2] = S3 and so on
_S = -0.5j * PAULI


def S(i):
    return _S[i - 1]


@dataclass(frozen=True)
class GroupElt(types.MatrixGroupElt):
    def __post_init__(self):
        assert self.matrix.shape == (2, 2)
        assert np.allclose(LA.det(self.matrix), 1)
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(2))

    @classmethod
    def dimension(cls) -> int:
        return 2

    @staticmethod
    def sample(rand: np.random.Generator | None = None) -> GroupElt:
        if rand is None:
            rand = np.random.default_rng()
        # sample from 3-sphere
        vec = rand.standard_normal(4)
        vec /= LA.norm(vec)
        ar, ai, br, bi = vec
        a = ar + ai * 1j
        b = br + bi * 1j
        matrix = np.array([[ar + ai * 1j, br + bi * 1j], [-br + bi * 1j, ar - ai * 1j]])
        return GroupElt(matrix)


@dataclass(frozen=True)
class LieAlgElt(types.MatrixLieAlgElt[GroupElt]):
    def __post_init__(self):
        # skew-symmetric and traceless
        assert np.allclose(self.matrix, -self.matrix.conj().T)
        assert np.allclose(self.matrix.trace(), 0.0)

    @classmethod
    def group(cls) -> Type[GroupElt]:
        return GroupElt

    @classmethod
    def basis(cls) -> list[LieAlgElt]:
        return [LieAlgElt(S(i)) for i in range(1, 4)]


def group_rep(dimension: int) -> Callable[[GroupElt], np.ndarray]:
    """Compute the unique irrep of a given dimension.

    This is based on the representation on the space of homogenous
    polynomials in two variables, but we use the dual rep since the
    standard choice doesn't specialize to the defining rep in 2D.
    """
    assert dimension >= 1

    def rep(elt: GroupElt) -> np.ndarray:
        deg = dimension - 1  # degree of polynomials
        ft = np.fft.fft(elt.matrix, dimension)
        w1, w2 = ft
        rows = []
        for i in range(dimension):
            rows.append(np.fft.ifft(w1 ** (deg - i) * w2**i))
        return np.stack(rows)

    return rep


def set_diagonal(arr, values, which_diag=0):
    """Set values of a diagonal of a square array"""
    n = len(arr)
    assert len(values) == n - abs(which_diag)
    offsets = np.arange(n - abs(which_diag))
    min_row = max(0, -which_diag)  # for negative values, start higher
    min_col = max(0, which_diag)  # for positive values, start higher
    arr[min_row + offsets, min_col + offsets] = values


def alg_rep(dimension: int) -> Callable[[LieAlgElt], npt.NDArray[np.complex_]]:
    """Compute the Lie algebra representation of a given dimension

    Like before, this one is dual to the usual rep so that in 2D it
    matches the defining rep.
    """
    assert dimension >= 1

    def rep(elt: LieAlgElt) -> npt.NDArray[np.complex_]:
        deg = dimension - 1

        X = elt.matrix
        out = np.zeros((dimension, dimension), dtype=np.complex128)
        indices = np.arange(dimension)
        main_diag = X[0, 0] * (deg - indices) + X[1, 1] * indices
        set_diagonal(out, main_diag, 0)
        lower_diag = (X[1, 0] * indices)[1:]  # skip initial 0
        set_diagonal(out, lower_diag, -1)
        upper_diag = (X[0, 1] * (deg - indices))[:-1]  # skip final 0
        set_diagonal(out, upper_diag, 1)
        return out

    return rep
