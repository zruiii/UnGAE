import numpy as np
from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MLPLayer


class Decoder(nn.Module):
    def __init__(self, device, weight_norm=True, link_norm=True, weight_lamb=1.0, link_lamb=1.0):

        super(Decoder, self).__init__()
        self.device = device
        self.weight_norm = weight_norm
        self.link_norm = link_norm
        self.weight_lamb = weight_lamb
        self.link_lamb = link_lamb
        self.mlp = MLPLayer(256, 32, 1)

    def forward(self, node_embed, edge_embed):
        # w_pre = self.weight_pred(edge_embed)
        w_pre = self.weight_pred_by_mlp(edge_embed)
        l_pre = self.link_pred(node_embed)
        return l_pre, w_pre

    def weight_pred_by_mlp(self, edge_embed):
        edge_in, edge_out = edge_embed
        edge_concat = torch.cat([edge_in, edge_out], dim=1)
        outputs = self.mlp(edge_concat).squeeze()
        return outputs

    def weight_pred(self, edge_embed):
        """ Predict the weight of edge(i-->j)
        p(i|j) = [m_i; h_i]
        p(j|i) = [m_j; h_j]
        d_{i->j} = m_i + log(||h_i-h_j||^2) 

        Args:
            edge_embed (_type_): _description_

        Returns:
            _type_: _description_
        """
        edge_in, edge_out = edge_embed
        dim = edge_in.shape[1]
        if self.weight_norm:
            h_in = self.l2_normalize(edge_in[:, :(dim-1)], axis=1)
            h_out = self.l2_normalize(edge_out[:, :(dim-1)], axis=1)
        else:
            h_in = edge_in[:, :(dim-1)]
            h_out = edge_out[:, :(dim-1)]

        dist = h_in - h_out
        dist = torch.norm(dist, p=2, dim=1) ** 2  # torch.Size([E])

        # Get mass parameter
        mass = edge_in[:, (dim - 1):dim].squeeze()
        outputs = mass + self.weight_lamb * torch.log(dist)
        return outputs

    def link_pred(self, node_embed):
        dim = node_embed.shape[1]
        if self.link_norm:
            inputs_z = self.l2_normalize(node_embed[:, :(dim-1)], axis=1)
        else:
            inputs_z = node_embed[:, :(dim-1)]

        dist = self.pairwise_distance(inputs_z)
        # Get mass parameter
        mass = node_embed[:, (dim - 1):dim].t()
        # Gravity-Inspired decoding
        # outputs = mass - tf.scalar_mul(FLAGS.lamb, tf.log(dist))
        outputs = mass - self.link_lamb * torch.log(dist)
        outputs = torch.sigmoid(outputs)
        return outputs

    def pairwise_distance(self, D, epsilon=0.01):
        D1 = (D * D).sum(1).unsqueeze(-1)
        D2 = torch.matmul(D, D.t())

        return D1 - 2 * D2 + D1.t() + epsilon

    def l2_normalize(self, D, axis=1):
        row_sqr_sum = (D * D).sum(axis).unsqueeze(-1)
        return torch.div(D, row_sqr_sum)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range

        return nn.Parameter(initial)
