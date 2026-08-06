"""
Copied from github.com/Tiiiger/SGC (Wu et al., ICML 2019). We remove the learnable params + softmax and keep only the filtering part.
"""
import numpy as np
import scipy.sparse as sp
# TODO: filtering on GPU!

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()


def rw_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).tocsr()


def row_normalize(mx):
    rowsum = np.array(mx.sum(1)).ravel()
    r_inv = np.power(rowsum, -1, where=rowsum != 0, out=np.zeros_like(rowsum, dtype=np.float64))
    r_inv[np.isinf(r_inv)] = 0.
    return (mx * r_inv[:, None]).astype(np.float32)


def absnorm_normalize(mx):
    """Sign-safe reading of the same idea: put every node's vector on a common scale. Please see appendix on this mode for normalization."""
    s = np.abs(mx).sum(1)
    r_inv = np.power(s, -1, where=s != 0, out=np.zeros_like(s, dtype=np.float64))
    r_inv[np.isinf(r_inv)] = 0.
    return (mx * r_inv[:, None]).astype(np.float32)


