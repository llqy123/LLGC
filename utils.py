import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter


def coo_block_diag(arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

    shapes = torch.tensor([a.shape for a in arrs])

    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        i += [arrs[k]._indices() + torch.tensor([[r],[c]]).to(arrs[0].device)]
        v += [arrs[k]._values()]
        r += rr
        c += cc
    if arrs[0].is_cuda:
        out = torch.cuda.sparse.FloatTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), torch.sum(shapes, dim=0).tolist())
    else:
        out = torch.sparse.FloatTensor(torch.cat(i, dim=1), torch.cat(v), torch.sum(shapes, dim=0).tolist())
    return out

def gen_rand_split(samples):
    idx_train = []
    idx_val = []
    idx_test = []
    random.seed(1234)
    for i in np.unique(samples.cpu()):
        idx = np.argwhere(samples.cpu() == i)
        l = idx.reshape(1, -1).squeeze(0)
        l = l.tolist()
        random.shuffle(l)
        for j in l[:20]:
            idx_train.append(j)
        for j in l[20:50]:
            idx_val.append(j)
        for j in l[50:]:
            idx_test.append(j)
        print(i)
    random.shuffle(idx_train)
    random.shuffle(idx_val)
    random.shuffle(idx_test)
    idx_train = torch.from_numpy(np.array(idx_train))
    idx_val = torch.from_numpy(np.array(idx_val))
    idx_test = torch.from_numpy(np.array(idx_test))
    return idx_train, idx_val, idx_test


def fully_gen_rand_split(n_samples, val_ratio, test_ratio, device=torch.device('cpu')):
    rand_idx = torch.randperm(n_samples, device=device)
    n_test = int(test_ratio * n_samples)
    n_val = int(val_ratio * n_samples)

    idx_test = rand_idx[:n_test]
    idx_val = rand_idx[n_test:n_test + n_val]
    idx_train = rand_idx[n_test + n_val:]

    return idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_eye(n):
    eye = sp.eye(n).tocoo()
    eye = sparse_mx_to_torch_sparse_tensor(eye).float()
    return eye

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    adj, features = preprocess_citation(adj, features, normalization)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # print(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def load_disease_data(dataset_str, data_path, normalization="AugNormAdj", cuda=True):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str))).todense()
    adj, features = preprocess_citation(adj, features, normalization)
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    return adj, features, labels

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def load_airport_data(dataset_str, data_path, normalization="AugNormAdj", cuda=True):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    features = adj.toarray().astype(np.float32)
    features = np.triu(features, 1)
    adj, features = preprocess_citation(adj, features, normalization)
    features = torch.FloatTensor(np.array(features)).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    return adj, features, labels

def norm_feat(feature):
    feature = feature.astype(dtype=np.float32)
    if sp.issparse(feature):
        row_sum = feature.sum(axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = sp.diags(row_sum_inv, format='csc')
        norm_feature = deg_inv.dot(feature)
    else:
        row_sum_inv = np.power(np.sum(feature, axis=1), -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        norm_feature = deg_inv.dot(feature)
        norm_feature = np.array(norm_feature, dtype=np.float32)

    return norm_feature

def load_webkb_data(dataset_name, normalization="AugNormAdj", cuda=True):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    feature = sp.load_npz(os.path.join(dataset_path, 'features.npz'))
    feature = feature.tocsc()

    feature = norm_feat(feature)
    feature = feature.astype(dtype=np.float32)

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    label = np.genfromtxt(os.path.join(dataset_path, 'labels.csv'))
    labels = torch.LongTensor(label)

    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv'))
    idx_val = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv'))
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv'))

    adj, features = preprocess_citation(adj, feature, normalization)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()

    # print(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return features, adj, labels, idx_train, idx_val, idx_test

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit/reddit_adj.npz")
    data = np.load(dataset_dir+"reddit/reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
