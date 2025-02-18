# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: ml
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from scipy.linalg import null_space

from lie_groups.types import MatrixLieAlgElt
from lie_groups import killing, su2, so3, sun
from lie_groups.representations import adjoint_matrices
from lie_groups.util import orth_complement


# %%
def cartan_subalgebra(adjoint_rep: list[np.ndarray]) -> list[np.ndarray]:
    # stack adjoint mats along last axis so we can take linear combinations easily
    adj_matrices = np.stack(adjoint_rep, axis=-1)
    dim = adj_matrices.shape[0]
    if dim == 0:
        return []
    # start with [1, 0, ...] as element of subalgebra
    first_subalg_vec = np.array([1.0] + [0.0] * (dim - 1))
    subalg_vecs = [first_subalg_vec]
    while True:
        subalg_matrix = np.stack(subalg_vecs, axis=1)
        subalg_adjoints = adj_matrices @ subalg_matrix  # (dim, dim, len(subalg))
        stacked_adjoint_matrix = np.moveaxis(subalg_adjoints, -1, 0).reshape(len(subalg_vecs) * dim, dim)
        ns = null_space(stacked_adjoint_matrix, rcond=1e-10)
        # find a null vector outside the span of subalg_vecs
        new_ns = orth_complement(ns, subalg_matrix)
        if new_ns.shape[1] > 0:
            subalg_vecs.append(new_ns[:, 0])
        else:
            break
    return subalg_vecs


# %%
basis = sun.su4.basis()

# %%
rep_matrices = adjoint_matrices(basis)

# %%
[np.round(a, 5) for a in cartan_subalgebra(rep_matrices)]

# %%
[np.round()]

# %%
null_space(rep_matrices[0])

# %%

# %%
basis = [su2.LieAlgElt(su2.S(i)) for i in range(1, 4)]
adj = adjoint_matrices(basis)
for mat in adj:
    print(np.round(mat, 4))

# %%
killing.killing_form(basis)

# %%
import numpy as np
from lie_groups import su2, so31
from lie_groups.util import comm

# %%
import matplotlib as mpl

# %%

import matplotlib.pyplot as plt

# %%

# %%
comm(so31.Jp(1), so31.Jp(2)) - so31.Jp(3)

# %%
comm(so31.Jp(1), so31.Jm(2))


# %%
def proj(X, e):
    return (e.conj() * X).sum() / (e.conj() * e).sum()


# %%
def vector_rep(X):
    """(1/2, 1/2) rep of so(3, 1)"""
    def proj(X, e):
        return (e.conj() * X).sum() / (e.conj() * e).sum()

    basis_in = []
    basis_out = []
    for i in range(1, 4):
        basis_in.append(so31.Jp(i))
        basis_out.append(np.kron(su2.S(i), np.eye(2)))
        basis_in.append(so31.Jm(i))
        basis_out.append(np.kron(np.eye(2), su2.S(i)))

    coefs = [proj(X, e) for e in basis_in]
    return np.sum([c * e for c, e in zip(coefs, basis_out)], axis=0)



# %%
vector_rep(so31.J1)

# %%
vector_rep(so31.J3)

# %%

# %%
elts = [so31.J1, so31.J2, so31.K1]
rep = [vector_rep(e) for e in elts]
S = solve_equivalence(elts, rep).round(4)[0]
S = canonicalize(S)
display(S)

# %%
from lie_groups.viz import complex_heatmap

# %%
fig, ax = plt.subplots(1, 4, figsize=(22, 5))
for mat, ax in zip([S, -S, S * 3, S / 3], ax):
    complex_heatmap(mat, ax=ax)

# %%
complex_heatmap(S * 5)

# %%
S @ S.conj().T

# %%
np.linalg.inv(S)

# %%
S @ so31.K2 - vector_rep(so31.K2) @ S

# %%
elts = [so31.J1, so31.K2, so31.K1]
rep = [vector_rep(e) for e in elts]
S_alt = solve_equivalence(elts, rep).round(4)[0]
S_alt /= S_alt.ravel()[np.abs(S_alt).argmax()]
S_alt

# %%

# %%
spin1 = su2.alg_rep(3)

# %%
C = [spin1(su2.LieAlgElt(su2.S(i))) for i in range(1, 4)]

# %%
comm(C[0], C[1])

# %%
C[2]

# %%
C[0]

# %%
C[1]

# %%
C[2]

# %%
