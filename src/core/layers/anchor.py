import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import to_cuda, create_mask
from ..utils.constants import VERY_SMALL_NUMBER, INF


def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    return node_vec[idx], idx

def batch_sample_anchors(node_vec, ratio, node_mask=None, device=None):
    idx = []
    num_anchors = []
    max_num_anchors = 0
    for i in range(node_vec.size(0)):
        tmp_num_nodes = int(node_mask[i].sum().item())
        tmp_num_anchors = int(ratio * tmp_num_nodes)
        g_idx = torch.randperm(tmp_num_nodes)[:tmp_num_anchors]
        idx.append(g_idx)
        num_anchors.append(len(g_idx))

        if max_num_anchors < len(g_idx):
            max_num_anchors = len(g_idx)

    anchor_vec = batch_select_from_tensor(node_vec, idx, max_num_anchors, device)
    anchor_mask = create_mask(num_anchors, max_num_anchors, device)

    return anchor_vec, anchor_mask, idx, max_num_anchors

def batch_select_from_tensor(node_vec, idx, max_num_anchors, device=None):
    anchor_vec = []
    for i in range(node_vec.size(0)):
        tmp_anchor_vec = node_vec[i][idx[i]]
        if len(tmp_anchor_vec) < max_num_anchors:
            dummy_anchor_vec = to_cuda(torch.zeros((max_num_anchors - len(tmp_anchor_vec), node_vec.size(-1))), device)
            tmp_anchor_vec = torch.cat([tmp_anchor_vec, dummy_anchor_vec], dim=-2)
        anchor_vec.append(tmp_anchor_vec)

    anchor_vec = torch.stack(anchor_vec, 0)

    return anchor_vec

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2)   # (num_anchor, num_node)
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=VERY_SMALL_NUMBER) ** -1
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj



class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=True):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))

        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(AnchorGCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        # 这里怎么只输入两个？
        # 它没有使用这个forward
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)

        return x


class AnchorGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(AnchorGCN2, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.graph_hops = graph_hops

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, init_adj, cur_node_anchor_adj, sampled_node_idx, graph_skip_conn, first=True, first_init_agg_vec=None,
                init_agg_vec=None, update_adj_ratio=None, dropout=None, first_node_anchor_adj=None):

        if dropout is None:
            dropout = self.dropout

        # layer 1
        first_vec = self.graph_encoders[0](x, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
        if first:
            init_agg_vec = self.graph_encoders[0](x, init_adj, anchor_mp=False, batch_norm=False)
        else:
            first_vec = update_adj_ratio * first_vec + (1 - update_adj_ratio) * first_init_agg_vec
        node_vec = (1-graph_skip_conn)*first_vec+graph_skip_conn*init_agg_vec
        if self.batch_norm:
            node_vec = self.graph_encoders[0].compute_bn(node_vec)
        node_vec = F.dropout(torch.relu(node_vec), dropout, training=self.training)
        anchor_vec = node_vec[sampled_node_idx]

        # layer 2-n-1
        for encoder in self.graph_encoders[1:-1]:

            mid_cur_agg_vec = encoder(node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
            if not first:
                mid_cur_agg_vec = update_adj_ratio*mid_cur_agg_vec+(1-update_adj_ratio)*encoder(node_vec,first_node_anchor_adj,anchor_mp=True,batch_norm=False)
            node_vec = (1 - graph_skip_conn) * mid_cur_agg_vec + graph_skip_conn * encoder(node_vec, init_adj, anchor_mp=False, batch_norm=False)
            if self.batch_norm:
                node_vec = encoder.compute_bn(node_vec)
            node_vec = F.dropout(torch.relu(node_vec), dropout, training=self.training)
            anchor_vec = node_vec[sampled_node_idx]

        # layer n
        cur_agg_vec = self.graph_encoders[-1](node_vec, cur_node_anchor_adj, anchor_mp=True, batch_norm=False)
        if not first:
            cur_agg_vec = update_adj_ratio * cur_agg_vec + (1-update_adj_ratio) * self.graph_encoders[-1](node_vec, first_node_anchor_adj, anchor_mp=True, batch_norm=False)
        output = (1 - graph_skip_conn) * cur_agg_vec + graph_skip_conn * self.graph_encoders[-1](node_vec, init_adj, anchor_mp=False, batch_norm=False)
        output = F.log_softmax(output, dim=-1)
        return first_vec, init_agg_vec, node_vec, output
