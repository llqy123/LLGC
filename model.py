
import torch.nn as nn
from time import perf_counter

import manifolds
from layers.lorentz_layers import LorentzLinear, HypLinear
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

class PageRankAgg(MessagePassing):

    def __init__(self, K: int = 1, alpha: float = 0.2,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PageRankAgg, self).__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        t = perf_counter()
        h = x
        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        precompute_time = perf_counter() - t
        return x, precompute_time

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
# preprocessing stage
def sgc_precompute(features, adj, degree=10):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter() - t
    return features, precompute_time

# Lorentzian MODEL
class LLGC(nn.Module):
    def __init__(self, nfeat, nclass, drop_out, use_bias):
        super(LLGC, self).__init__()
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.nclass = nclass
        self.c = torch.tensor([1.0]).to("cuda")
        self.manifold = getattr(manifolds, "Lorentzian")()
        self.W = LorentzLinear(self.manifold, nfeat, nclass, self.c, self.drop_out, self.use_bias)
    def forward(self, x):
        x_loren = self.manifold.normalize_input(x, self.c)
        x_w = self.W(x_loren)
        x_tan = self.manifold.log_map_zero(x_w, self.c)
        return x_tan



