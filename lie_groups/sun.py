from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.linalg as LA

from . import types

@cache
def make_SU_n(n: int) -> type[types.MatrixGroupElt]:
    @dataclass(frozen=True)
    class GroupElt(types.MatrixGroupElt):
        def __post_init__(self):
            N = self.matrix.shape[0]
            assert np.allclose(LA.det(self.matrix), 1)
            assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(N))

        @classmethod
        def dimension(cls) -> int:
            return n

    GroupElt.__name__ = f"SU({n})"
    return GroupElt

SU3 = make_SU_n(3)

@cache
def make_su_n(n: int) -> type[types.MatrixLieAlgElt]:
    @dataclass(frozen=True)
    class LieAlgElt(types.MatrixLieAlgElt):

        def __post_init__(self):
            # skew-symmetric and traceless
            assert np.allclose(self.matrix, -self.matrix.conj().T)
            assert np.allclose(self.matrix.trace(), 0.)

        @classmethod
        def group(cls) -> type:
            return make_SU_n(n)

        @classmethod
        def basis(cls) -> list[LieAlgElt]:
            elts = []
            for k in range(n):
                for l in range(k):
                    mat = np.zeros((n, n), dtype=np.complex128)
                    mat[k, l] = mat[l, k] = 1j
                    elts.append(mat)
            return elts

    LieAlgElt.__name__ = f"su({n})"

    return LieAlgElt

su3 = make_su_n(3)