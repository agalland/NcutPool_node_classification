import torch

import torch.optim as optim
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score
from model import EdgeCut


class graphEC(object):
    def __init__(self,
                 n_feat=None,
                 h_dims=None,
                 out_dim=None,
                 ps=None,
                 num_pool=None,
                 n_gcns=1,
                 alpha_reg=1.,
                 path_save_weights=None,
                 lr=0.01,
                 numEpochs=100,
                 epochEval=1,
                 device="cpu"):

        self._n_feat = n_feat
        self._h_dims = h_dims
        self._out_dim = out_dim
        self._ps = ps
        self._num_pool = num_pool
        self._n_gcns = n_gcns
        self.path_save_weights = path_save_weights
        self.numEpochs = numEpochs
        self.epochEval = epochEval
        self.lr = lr
        self.device = device
        self.alpha_reg = alpha_reg

        self._net = EdgeCut(self._n_feat,
                            self._h_dims,
                            self._out_dim,
                            self._ps,
                            self._num_pool,
                            self._n_gcns).to(device)

        self.criterion = nn.NLLLoss()

    def fit(self,
            x,
            adjC,
            adj,
            y_train,
            y_test,
            inds_train,
            inds_test,
            verbose=False,
            epoch_verbose=10,
            decrease_lr=False):
        label_train = y_train.astype(np.int64)
        label_train = torch.LongTensor(label_train).to(self.device)
        label_test = y_test.astype(np.int64)
        label_test = torch.LongTensor(label_test).to(self.device)
        optimizer = optim.Adam(params=self._net.parameters(),
                               lr=self.lr,
                               weight_decay=1e-4)
        inds_train = torch.LongTensor(inds_train)
        inds_test = torch.LongTensor(inds_test)

        x = torch.Tensor(np.array(x.todense()))
        adj = torch.Tensor(np.array(adj.todense()))
        adjC = torch.Tensor(np.array(adjC.todense()))

        self.loss_test_min = np.infty
        self.epochBreak = self.numEpochs
        self.loss_test_min = np.infty
        loss_test_over = 0.
        loss_test_over_nb = 50
        loss_test_prev = np.infty
        reduce_lr = True
        for epoch in range(self.numEpochs):
            #Initialize grad
            optimizer.zero_grad()

            pred, Cs, atts = self._net(x, adjC, adj)

            #Predictions for train set
            pred_train = pred[inds_train]
            pred_test = pred[inds_test]

            loss_reg_att = 0.
            for k in range(len(atts)):
                #Cut minimization
                mat = torch.matmul(torch.matmul(Cs[k], atts[k]), Cs[k].t())
                ind = np.diag_indices(mat.shape[0])
                vol = mat[ind[0], ind[1]]
                mat[ind[0], ind[1]] = 0.
                loss_reg_att += self.alpha_reg * (mat.sum(1) / vol).sum()

            loss_train_classif = self.criterion(pred_train, label_train)
            loss_train = loss_train_classif + loss_reg_att / len(atts)
            loss_test = self.criterion(pred_test, label_test)

            if decrease_lr:
                if torch.abs(loss_test - loss_test_prev) < 1e-4 and reduce_lr:
                    print("reduce lr")
                    self.lr /= 10.
                    optimizer = optim.Adam(params=self._net.parameters(),
                                           lr=self.lr,
                                           weight_decay=1e-4)
                    reduce_lr = False
                loss_test_prev = loss_test

            if epoch % epoch_verbose == 0 and verbose:
                print("Epoch: {}, "
                      "loss train: {}, "
                      "loss test: {}, "
                      "loss reg att: {} ".format(epoch,
                                                 loss_train_classif.data,
                                                 loss_test.data,
                                                 loss_reg_att.data))

            if epoch % self.epochEval == 0:
                if loss_test.data < self.loss_test_min:
                    self.loss_test_min = loss_test.data
                    self.weights = torch.save(self._net.state_dict(), self.path_save_weights)
                    loss_test_over = 0
                else:
                    loss_test_over += 1
                if loss_test_over > loss_test_over_nb:
                    self._net.load_state_dict(torch.load(self.path_save_weights))
                    print("break at epoch: {}".format(epoch - loss_test_over_nb - 1))
                    self.epochBreak = epoch
                    break

            #Loss backward
            loss_train.backward(retain_graph=True)
            optimizer.step()

    def evaluate(self, x, adj, y):
        adj = torch.Tensor(adj.todense()).to(self.device)
        x = torch.Tensor(x.todense()).to(self.device)
        output = self._net(x, adj)
        valpred, pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy()

        return np.round(accuracy_score(pred, y), 3)

    def HLoss(self, prob):
        b = -prob * torch.log(prob + 1e-10)
        b = torch.sum(b, 1)
        b = torch.mean(b)

        return b

    def class_reg(self, C):
        C_sup = (C > 0.).sum(1)
        C_sup = C_sup * C.sum(1)
        reg = torch.softmax(C_sup.unsqueeze(1), 0).transpose(0, 1)
        if self.HLoss(reg) == 0.:
            reg = torch.softmax(reg, 1)

        return 1/(self.HLoss(reg))

    def modularity_reg(self, adj, C):
        D_t = adj.sum(0, keepdim=True)
        m = adj.sum()
        ddt = torch.matmul(D_t.transpose(0, 1), D_t) / (2 * m)
        B = adj - ddt
        B = torch.matmul(C, B)
        B = torch.matmul(B, C.transpose(0, 1))

        Q = torch.trace(B) / (4 * m)

        return Q
