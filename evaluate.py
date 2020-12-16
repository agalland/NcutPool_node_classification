import torch

import numpy as np

from sklearn.metrics import accuracy_score
from utils import load_data
from train import graphEC


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

h_dims = [1024, 256]
ps = [30]
num_pool = 2
lr = 1e-3
alpha_reg = 1e-5
n_gcns = 2

dataset_names = ["cora"]
project_path = "../NcutPool_node_classif_submit/"
path_load_weights = project_path + "weightsEval/weights"

for dataset_name in dataset_names:
    print("dataset name: {}".format(dataset_name))
    adj, adjC, features, y_train, y_test, y_val, train_mask, test_mask, val_mask = load_data(dataset_name)
    n_feat = features.shape[1]
    out_dim = y_train.shape[1]

    inds_train = np.where(train_mask)[0]
    inds_test = np.where(test_mask)[0]
    inds_val = np.where(val_mask)[0]

    y_train = np.where(y_train > 0.)[1]
    y_test = np.where(y_test > 0.)[1]
    y_val = np.where(y_val > 0.)[1]

    trainEC = graphEC(n_feat=n_feat,
                      h_dims=h_dims,
                      out_dim=out_dim,
                      ps=ps,
                      num_pool=num_pool,
                      n_gcns=n_gcns,
                      alpha_reg=alpha_reg,
                      lr=lr,
                      device=device)

    trainEC._net.load_state_dict(torch.load(path_load_weights))

    adj_tensor = torch.Tensor(adj.todense())
    x_tensor = torch.Tensor(features.todense())
    adjC_tensor = torch.Tensor(adjC.todense())
    output, memb, att = trainEC._net(x_tensor, adjC_tensor, adj_tensor)
    _, pred = torch.max(output, 1)
    pred = pred.cpu().data.numpy()

    acc_train = np.round(accuracy_score(pred[inds_train], y_train), 4)
    acc_test = np.round(accuracy_score(pred[inds_test], y_test), 4)
    acc_val = np.round(accuracy_score(pred[inds_val], y_val), 4)

    print("accuracies: train {}, test {}, val {}".format(acc_train, acc_test, acc_val))