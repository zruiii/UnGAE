import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, gcn_adj, adj_e, adj_n, device, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # 表征维度
        self.base_gcn = GraphConvSparse(input_dim, hidden_dim, gcn_adj, activation=lambda x: x)
        self.sgc = SGCL(hidden_dim, output_dim, adj_e, adj_n, device)

    def forward(self, X):
        self.hidden = self.base_gcn(X)
        assert not torch.isnan(self.hidden).any()

        node_embed, edge_embed = self.sgc(self.hidden)
        return node_embed, edge_embed

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs
    
    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range

        return nn.Parameter(initial)

class SGCL(nn.Module):
    def __init__(self, input_dim, output_dim, adj_e, adj_n, device):
        super(SGCL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_e = adj_e         # 有向图，真实存在的边，只需要结构信息
        self.adj_n = adj_n      # 无向图，获得节点表征（归一化）
        self.device = device
        self.special_spmm = SpecialSpmm()
        self.weight_miu = nn.Parameter(torch.zeros(size=(2 * input_dim, output_dim)))
        nn.init.xavier_normal_(self.weight_miu.data, gain=1.414)
        self.weight_sigma = nn.Parameter(torch.zeros(size=(2 * input_dim, output_dim)))
        nn.init.xavier_normal_(self.weight_sigma.data, gain=1.414)
    
    def forward(self, hidden):
        """
        用高斯分布来表征节点之间的条件概率，例如存在边 i-->j 则在节点i的视角下j的表征为 p(j|i) = MLP([h_j; h_i]),
        同理: 节点j的视角下i的表征为 p(i|j) = MLP([h_i; h_j]),则可以用 f[p(j|i); p(i|j)] 来刻画边 i-->j
        对于节点的表征，可以定义为不同邻居视角下表征之和，此时的邻接矩阵用无向图表示
        """
        
        di_edge = self.adj_e.coalesce().indices()         # [i, j]
        edge_in, edge_out = self.edge_embedding(hidden, di_edge)
    
        bi_edge = self.adj_n.coalesce().indices()       # 对称无向图， D^{-1}(A+I)
        node_embed = self.node_embedding(hidden, bi_edge)

        # node_embed: [N, 128]
        # edge_in, edge_out: [E, 128]
        return node_embed, (edge_in, edge_out)

    def edge_embedding(self, hidden, edges):
        '''
        建模边 i-->j 获得 edge_in = p(j|i), edge_out = p(i|j)
        hidden: 节点的隐表征
        edges: [2, E] 第一行表示行，第二行表示列
        '''
        # model edge: i-->j

        edge_h_in = torch.cat((hidden[edges[0, :], :], hidden[edges[1, :], :]), dim=1)        # [h_i; h_j] 
        edge_h_out = torch.cat((hidden[edges[1, :], :], hidden[edges[0, :], :]), dim=1)             # [h_j; h_i] 
        edge_miu_in = torch.mm(edge_h_in, self.weight_miu)    
        edge_miu_out = torch.mm(edge_h_out, self.weight_miu)    
        # edge_sigma_in = F.elu(torch.mm(edge_h_in, self.weight_sigma)) + 1         # [E, D]
        # edge_sigma_out = F.elu(torch.mm(edge_h_out, self.weight_sigma)) + 1
        edge_logstd_in = torch.mm(edge_h_in, self.weight_sigma)
        edge_logstd_out = torch.mm(edge_h_out, self.weight_sigma)
        assert not torch.isnan(edge_miu_in).any()
        assert not torch.isnan(edge_miu_out).any()
        assert not torch.isnan(edge_logstd_in).any()
        assert not torch.isnan(edge_logstd_out).any()

        # 采样得到p(h_in|h_out)和p(h_out|h_in)
        E = edges.shape[1]
        gaussian_noise_in = torch.randn(E, self.output_dim).to(self.device)
        gaussian_noise_out = torch.randn(E, self.output_dim).to(self.device)
        # edge_in = gaussian_noise_in * torch.sqrt(edge_sigma_in) + edge_miu_in    # p(i|j)  当前节点视角下邻居表征
        # edge_out = gaussian_noise_out * torch.sqrt(edge_sigma_out) + edge_miu_out     # p(j|i)  邻居视角下当前节点表征
        edge_in = gaussian_noise_in * torch.exp(edge_logstd_in) + edge_miu_in    # p(i|j)  当前节点视角下邻居表征
        edge_out = gaussian_noise_out * torch.exp(edge_logstd_out) + edge_miu_out     # p(j|i)  邻居视角下当前节点表征
        # edge_embed = torch.cat((edge_in, edge_out), dim=1)      # [E, 2*D] e_{i->j} = f{concat[p(j|i), p(i|j)]} 当前节点视角下邻居表征 + 邻居视角下当前节点表征
        # edge_embed = torch.cat((edge_out, edge_in), dim=1)
        assert not torch.isnan(edge_in).any()
        assert not torch.isnan(edge_out).any()
        return edge_in, edge_out

    def node_embedding(self, hidden, edges):
        '''
        聚合一阶邻居视角下的表征(Gaussian)从而获得当前节点的表征
        '''
        N = hidden.size()[0]
        bi_E = edges.shape[1]
        row = edges[0, :]
        col = torch.arange(bi_E).to(self.device)
        indices = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), dim=0)
        values = self.adj_n.coalesce().values().to(self.device)

        bi_edge_embed = torch.cat((hidden[edges[0, :], :], hidden[edges[1, :], :]), dim=1)
        bi_edge_miu = torch.mm(bi_edge_embed, self.weight_miu)
        bi_edge_logstd = torch.mm(bi_edge_embed, self.weight_sigma)
        assert not torch.isnan(bi_edge_miu).any()
        assert not torch.isnan(bi_edge_logstd).any()

        self.miu = self.special_spmm(indices, values, torch.Size([N, bi_E]), bi_edge_miu)
        self.logstd = 0.5 * torch.log(self.special_spmm(indices, values ** 2, torch.Size([N, bi_E]), torch.exp(bi_edge_logstd)))
        gaussian_noise = torch.randn(N, self.output_dim).to(self.device)
        node_embed = gaussian_noise * torch.exp(self.logstd) + self.miu
        assert not torch.isnan(node_embed).any()

        return node_embed

class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for only sparse region backpropataion layer.
    如果是直接用Sparse Tensor乘上张量得到的就是Tensor，内存开销大，这种写法可以直接得到相乘后的Sparse Tensor
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        """
        节省内存开销的情况下实现 torch.mm(sparse tensor, dense tensor)
        """
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)         # 将Tensor转变为Variable保存到ctx中
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output为损失函数对前向传播的返回值求导的结果

        a, b = ctx.saved_tensors
        grad_values = grad_b = None         # 权重和偏置的梯度

        # 判断forward中输入的indices, values, shape, b是否需要求导
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]          # 边的index
            grad_values = grad_a_dense.view(-1)[edge_idx]       # E
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)      # [N, D]
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

