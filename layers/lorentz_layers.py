# Lorentzian neural network layers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LorentzLinear(nn.Module):
    # Lorentz Hyperbolic Graph Neural Layer
    def __init__(self, manifold, in_features, out_features, c, drop_out, use_bias):
        super(LorentzLinear, self).__init__()
        # print("LorentzLinear")
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.drop_out = drop_out
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features-1))   # -1 when use mine mat-vec multiply
        self.weight = nn.Parameter(torch.Tensor(out_features - 1, in_features))  # -1, 0 when use mine mat-vec multiply
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.drop_out, training=self.training)
        mv = self.manifold.matvec_regular(drop_weight, x, self.bias, self.c, self.use_bias)
        return mv


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res
