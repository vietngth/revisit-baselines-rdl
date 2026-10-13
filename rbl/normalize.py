"""Adjacency and feature normalizations, adapted from github.com/Tiiiger/SGC (Wu et al., 2019)."""
import numpy as np
import scipy.sparse as sp


def aug_normalized_adjacency(adj):
    """D^{-1/2} (A + I) D^{-1/2}, SGC's propagation operator."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).tocsr()


def absnorm_normalize(features):
    """Divide each row by its sum of absolute values.

    SGC divides by the signed row sum, which is unstable for signed embedding features.
    """
    row_sums = np.abs(features).sum(1)
    inverse = np.power(row_sums, -1, where=row_sums != 0,
                       out=np.zeros_like(row_sums, dtype=np.float64))
    inverse[np.isinf(inverse)] = 0.0
    return (features * inverse[:, None]).astype(np.float32)
