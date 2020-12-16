import torch

import numpy as np

from sklearn.metrics import accuracy_score
from utils import load_data
from train import graphEC


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

h_dims_list = []
for h_dim1 in [256, 512, 1024]:
    for h_dim2 in [64, 128]:
        h_dims_list.append([h_dim1, h_dim2])

ps_list = [[30, 50, 70]]
num_pool_list = [1, 2]
lr_list = [1e-3]
alpha_reg_list = [0.0001]
n_gcns_list = [1, 2]

dataset_names = ["cora", "citeseer", "pubmed"]
project_path = "../NcutPool_node_classif_submit/"
results_path = project_path + "results/"

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
    for lr in lr_list:
        for h_dims in h_dims_list:
            for ps in ps_list:
                for num_pool in num_pool_list:
                    for n_gcns in n_gcns_list:
                        for alpha_reg in alpha_reg_list:
                            for l in range(10):
                                # 10 iterations for mean and std
                                fname = dataset_name + "_val" + "_" + str(lr) + "_" + str(h_dims[0]) + "_" + str(h_dims[1]) + "_" + str(ps[0]) + "_" + str(num_pool) + "_" + str(alpha_reg) + "_" + str(l) + "_" + str(n_gcns) + ".npy"
                                print("lr: {}, h_dims: {}, {}, ps: {}, pool: {}, alpha reg: {}".format(lr, h_dims[0], h_dims[1], ps[0], num_pool, alpha_reg))

                                numEpochs = 2000
                                path_save_weights = project_path + "weights/" + fname[:-4]

                                trainEC = graphEC(n_feat=n_feat,
                                                  h_dims=h_dims,
                                                  out_dim=out_dim,
                                                  ps=ps,
                                                  num_pool=num_pool,
                                                  n_gcns=n_gcns,
                                                  alpha_reg=alpha_reg,
                                                  lr=lr,
                                                  numEpochs=numEpochs,
                                                  path_save_weights=path_save_weights,
                                                  device=device)

                                trainEC.fit(features,
                                            adjC,
                                            adj,
                                            y_train,
                                            y_test,
                                            inds_train,
                                            inds_test,
                                            verbose=True,
                                            epoch_verbose=1,
                                            decrease_lr=False)

                                trainEC._net.load_state_dict(torch.load(trainEC.path_save_weights))

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

                                np.save(results_path + dataset_name + "_val" + "_" + str(lr) + "_" + str(h_dims[0]) + "_" + str(h_dims[1]) + "_" + str(ps[0]) + "_" + str(num_pool) + "_" + str(alpha_reg) + "_" + str(l) + "_" + str(n_gcns) + ".npy",
                                        acc_val)
                                np.save(results_path + dataset_name + "_test" + "_" + str(lr) + "_" + str(h_dims[0]) + "_" + str(h_dims[1]) + "_" + str(ps[0]) + "_" + str(num_pool) + "_" + str(alpha_reg) + "_" + str(l) + "_" + str(n_gcns) + ".npy",
                                        acc_test)