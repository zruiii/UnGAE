import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder

class UGAE(nn.Module):
    def __init__(self, gcn_adj, adj_e, adj_n, device, args):
        super(UGAE, self).__init__()
        self.encoder = Encoder(gcn_adj, adj_e, adj_n, device, args.input_dim, args.hidden_dim, args.output_dim)

        self.decoder = Decoder(device, 
                               weight_norm = True, 
                               link_norm = True, 
                               weight_lamb = 0.1, 
                               link_lamb = 5)
    
    def forward(self, X):
        node_embed, edge_embed = self.encoder(X)
        l_pre, w_pre = self.decoder(node_embed, edge_embed)
        return l_pre, w_pre