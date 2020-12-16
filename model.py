from layer import *
from utils import *

import scipy as sc
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch


# GAE model

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class EdgeCut(nn.Module):
    def __init__(self, in_features, h_dims, out_dim, ps, num_pool=3, n_gcns=1):
        super(EdgeCut, self).__init__()
        self.in_features = in_features
        self.h_dim1 = h_dims[0]
        self.h_dim_fc = h_dims[1]
        self.out_dim = out_dim
        self.num_pool = num_pool
        self.ps = ps
        self.n_gcns = n_gcns

        self.add_module("encodeconv_"+str(0), GraphConvolution(in_features, self.h_dim1))
        for l in range(1, self.n_gcns):
            self.add_module("encodeconv_" + str(l), GraphConvolution(self.h_dim1, self.h_dim1))
        for k in range(1, num_pool+1):
            for l in range(self.n_gcns):
                self.add_module("encodeconv_"+str(self.n_gcns*k+l), GraphConvolution(self.h_dim1, self.h_dim1))

        for k in range(num_pool):
            self.add_module("decodeconv_"+str(k), GraphConvolution(self.h_dim1*2, self.h_dim1))
        self.add_module("decodeconv_" + str(k+1), GraphConvolution(self.h_dim1, self.out_dim))

        self.a = nn.ParameterList([])
        for k in range(num_pool):
            a = nn.Parameter(torch.zeros(size=(2 * self.h_dim_fc, 1)))
            nn.init.xavier_uniform_(a.data, gain=1.414)
            self.a.append(a)
            self.add_module("linear_"+str(k), nn.Linear(self.h_dim1, self.h_dim_fc))

        self.gcns_encode = AttrProxy(self, 'encodeconv_')
        self.gcns_decode = AttrProxy(self, 'decodeconv_')
        self.fcs = AttrProxy(self, 'linear_')

    def encode(self, x, adjC, adj):
        xs = []
        adjs = []
        adjCs = []
        atts = []
        Cs = []
        adjs.append(adj)
        adjCs.append(adjC)
        for k in range(self.num_pool):
            for l in range(self.n_gcns):
                x1 = x
                x = self.gcns_encode[self.n_gcns*k+l](x, adjCs[-1])
                x = F.relu(x)
                # x = torch.tanh(x)
                if self.n_gcns*k+l > 0:
                    x = F.normalize(x+x1, p=2, dim=1)
                else:
                    x = F.normalize(x, p=2, dim=1)
            xs.append(x)

            att = self.edge_score(x, adjs[-1], self.a[k], self.fcs[k])
            x, adjk, adjCk, C = self.pool(x, adjs[-1], adjCs[-1], att, self.ps[0])
            adjCk = self.normalize_adj(adjk, 1)
            x = F.normalize(x, p=2, dim=1)
            adjs.append(adjk)
            adjCs.append(adjCk)
            atts.append(att)
            Cs.append(C)

        x = self.gcns_encode[self.num_pool](x, adjCs[-1])
        x = F.elu(x)
        x = F.normalize(x, p=2, dim=1)

        xs.append(x)

        return xs, adjs, adjCs, Cs, atts

    def decode(self, xs, adjCs, Cs):
        x = xs[-1]
        x_classif = []
        for k in range(self.num_pool):
            x = self.unpool(x, Cs[self.num_pool-k-1])

            x = torch.cat((x, xs[self.num_pool-k-1]), 1)
            x = self.gcns_decode[k](x, adjCs[self.num_pool-k-1])
            x = F.elu(x)
            x = F.normalize(x, p=2, dim=1)
            x_clf = x

        x = self.gcns_decode[self.num_pool](x_clf, adjCs[0])
        x = torch.log_softmax(x, 1)

        return x

    def forward(self, x, adjC, adj):
        xs, adjs, adjCs, Cs, atts = self.encode(x, adjC, adj)

        x = self.decode(xs, adjCs, Cs)

        return x, Cs, atts

    def unpool(self, x, C):
        x = torch.matmul(C.transpose(0, 1), x)

        return x

    def normalize_adj(self, adj, k):
        adj = adj + k * torch.eye(adj.size(0))
        D = torch.sqrt(adj.sum(0))
        adj = adj / D
        adj = adj.transpose(0, 1)
        adj = adj / D
        adj = adj.transpose(0, 1)

        return adj

    def edge_score(self, x, adj, a, fc):
        x = fc(x)

        N = x.size()[0]

        a_input = torch.cat([x.repeat(1, N).view(N * N, -1), x.repeat(N, 1)], dim=1).view(N, -1, 2 * self.h_dim_fc)
        e = torch.matmul(a_input, a).squeeze(2)

        zero_vec = torch.zeros_like(e)
        e = torch.sigmoid(e)
        e = torch.where(adj > 0, e, zero_vec)

        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0., e, zero_vec)
        attention = 0.5 * (attention + attention.transpose(0, 1))
        ind = np.diag_indices(attention.shape[0])
        attention[ind[0], ind[1]] = torch.ones(attention.shape[0])

        return attention

    def pool(self, x, adj, adjC, attention, p):
        attention_np = attention.cpu().data.numpy()
        cut_val = np.percentile(attention_np[np.where(attention_np > 0.)], p)
        attention = attention * (attention >= cut_val)
        attention_np = attention.cpu().data.numpy()
        comp = sc.sparse.csgraph.connected_components(attention_np)[1]
        memb = np.zeros((len(comp), comp.max() + 1))
        memb[range(len(comp)), comp] = 1.
        memb = memb.T
        memb_t = torch.Tensor(memb)

        adj = torch.matmul(memb_t, adj)
        adj = torch.matmul(adj, memb_t.transpose(0, 1))
        adj = 0.5 * (adj + adj.transpose(0, 1))

        adjC = torch.matmul(memb_t, adjC)
        adjC = torch.matmul(adjC, memb_t.transpose(0, 1))
        adjC = 0.5 * (adjC + adjC.transpose(0, 1))

        attention = attention.T / (attention > 0.).sum(1)
        attention = attention.T
        x = torch.matmul(attention, x)
        x = torch.matmul(memb_t, x)

        return x, adj, adjC, memb_t

    def unpool(self, x, C):
        C = (C > 0.) * 1.
        x = torch.matmul(C.transpose(0, 1), x)

        return x
