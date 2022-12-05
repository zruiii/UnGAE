import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle as pkl
import networkx as nx
import random


def sparse_to_torch(matrix):
    """ Convert a scipy sparse matrix to dense tensor

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.from_numpy(matrix.toarray()).float()


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph_sym(adj, power=-1/2, self_loop=True):
    """ symmetric adjacency matrix preprocessing

    Args:
        adj (_type_): _description_
        power (_type_, optional): _description_. Defaults to -1/2.
        self_loop (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    adj = sp.coo_matrix(adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, power).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def preprocess_graph_asym(adj):
    """ Asymmetric adjacency matrix preprocessing

    Args:
        adj (_type_): _description_

    Returns:
        _type_: _description_
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj (see section 3.3.1 of paper)
    degree_mat_inv_sqrt = sp.diags(
        np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return sparse_to_tuple(adj_normalized)


def normalize_graph(sparse_mx):
    # Find the row scalars as a Matrix_(n,1)
    sparse_mx = sparse_mx.transpose()
    rowSum = sp.csr_matrix(sparse_mx.sum(axis=1))
    rowSum.data = 1 / rowSum.data

    # Find the diagonal matrix to scale the rows
    rowSum = rowSum.transpose()
    scaling_matrix = sp.diags(rowSum.toarray()[0]).dot(sparse_mx)

    return sparse_to_tuple(scaling_matrix)


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """ Set the learning rate to the initial LR decayed by 10 every 30 epochs

    Args:
        optimizer (_type_): _description_
        epoch (_type_): _description_
        learning_rate (_type_): _description_
    """
    lr = learning_rate * (0.8 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_data(data_path):
    """ Data Loader

    Args:
        data_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(data_path + "mobility.graph", "rb") as f:
        graph = pkl.load(f)
    with open(data_path + "train_label.pkl", "rb") as f:
        train_label = pkl.load(f)
    with open(data_path + "test_label.pkl", "rb") as f:
        test_label = pkl.load(f)
    with open(data_path + "val_label.pkl", "rb") as f:
        val_label = pkl.load(f)

    return graph, train_label, test_label, val_label


def ismember(a, b, tol=5):
    """ Check whether a row in a 2D numpy array

    Args:
        a (_type_): _description_
        b (_type_): _description_
        tol (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)


def negative_sampling(graph, test_label, val_label, train_label=False):
    """ Negative Sampling for Link Prediction Valuation

    Args:
        graph (_type_): _description_
        test_label (_type_): _description_
        val_label (_type_): _description_
        train_label (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    A = nx.adjacency_matrix(graph)
    # all_edges = np.vstack(A.tocoo().row, A.tocoo().col).transpose()
    sample_edges = []
    val_num = val_label.nnz
    test_num = test_label.nnz
    print("negative sampling ...")
    while len(sample_edges) < (val_num + test_num):
        row_id = random.randint(0, A.shape[0]-1)
        col_id = random.randint(0, A.shape[0]-1)
        if A[row_id, col_id] != 0:
            continue
        if row_id == col_id:
            continue
        if sample_edges:
            if ismember([row_id, col_id], np.array(sample_edges)):
                continue
        sample_edges.append((row_id, col_id))

    sample_edges = np.array(sample_edges)
    test_neg = sample_edges[: test_num]
    val_neg = sample_edges[test_num:]
    return test_neg, val_neg


def negative_sampling_digcn(graph, test_label, val_label, train_label=False):
    """ Negative Sampling for DiGCN

    Args:
        graph (_type_): _description_
        test_label (_type_): _description_
        val_label (_type_): _description_
        train_label (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    A = nx.adjacency_matrix(graph)
    sample_edges = []
    val_num = val_label.nnz
    test_num = test_label.nnz
    print("negative sampling ...")
    train_num = train_label.nnz
    while len(sample_edges) < (val_num + test_num + train_num):
        row_id = random.randint(0, A.shape[0]-1)
        col_id = random.randint(0, A.shape[0]-1)
        if A[row_id, col_id] != 0:
            continue
        if row_id == col_id:
            continue
        if sample_edges:
            if ismember([row_id, col_id], np.array(sample_edges)):
                continue
        sample_edges.append((row_id, col_id))

    sample_edges = np.array(sample_edges)
    test_neg = sample_edges[: test_num]
    val_neg = sample_edges[test_num: test_num+val_num]
    train_neg = sample_edges[-train_num:]
    return train_neg, test_neg, val_neg


class StandScaler(object):
    def __init__(self, miu, std):
        self.miu = miu
        self.std = std

    def transform(self, data):
        return (data - self.miu) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.miu
