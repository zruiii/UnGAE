from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import sys
from sklearn.metrics import mean_absolute_error
from utils import *
from model import UGAE
from evaluate import link_eval, weight_eval
from logger import Logger

device = torch.device('cpu')

def graph_train(args):
    root_path = '../data/'

    graph, train_label, test_label, val_label = load_data(
        root_path + args.data_path)

    A = nx.adjacency_matrix(graph)
    adj_train = train_label.sign()
    if os.path.exists(root_path + args.data_path + "negative_samples.pkl"):
        with open(root_path + args.data_path + "negative_samples.pkl", "rb") as f:
            test_edges_false, val_edges_false = pkl.load(f)
    else:
        test_edges_false, val_edges_false = negative_sampling(
            graph, test_label, val_label)
        with open(root_path + args.data_path + "negative_samples.pkl", "wb") as f:
            pkl.dump([test_edges_false, val_edges_false], f)

    test_edges_pos = np.vstack(
        (test_label.tocoo().row, test_label.tocoo().col)).transpose()
    val_edges_pos = np.vstack(
        (val_label.tocoo().row, val_label.tocoo().col)).transpose()

    norm = adj_train.shape[0] * adj_train.shape[0] / \
        float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    pos_weight = (
        adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()

    # Normalized Adjacency Matrix
    gcn_adj = preprocess_graph_asym(adj_train)
    gcn_adj = torch.sparse.FloatTensor(torch.LongTensor(gcn_adj[0].T),
                                       torch.FloatTensor(gcn_adj[1]),
                                       torch.Size(gcn_adj[2])).to(device)

    # Adjacency Matrix for edge weight calculation
    adj_e = sparse_to_tuple(adj_train)
    adj_e = torch.sparse.FloatTensor(torch.LongTensor(adj_e[0].T),
                                     torch.FloatTensor(adj_e[1]),
                                     torch.Size(adj_e[2])).to(device)

    # Adjacency Matrix for node embedding
    adj_n = adj_train + \
        adj_train.T.multiply(adj_train.T > adj_train) - \
        adj_train.multiply(adj_train.T > adj_train)
    adj_n = preprocess_graph_asym(adj_n)
    adj_n = torch.sparse.FloatTensor(torch.LongTensor(adj_n[0].T),
                                     torch.FloatTensor(adj_n[1]),
                                     torch.Size(adj_n[2])).to(device)

    # Link Prediction Label
    link_label = adj_train + sp.eye(adj_train.shape[0])
    link_label = sparse_to_tuple(link_label.tocoo())
    link_label = torch.sparse.FloatTensor(torch.LongTensor(link_label[0].T),
                                          torch.FloatTensor(link_label[1]),
                                          torch.Size(link_label[2])).to(device)

    weight_mask = link_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # Edge weight Labels
    weight_label = sparse_to_tuple(train_label)
    weight_label = torch.sparse.FloatTensor(torch.LongTensor(weight_label[0].T),
                                            torch.FloatTensor(weight_label[1]),
                                            torch.Size(weight_label[2])).to(device)

    # Features
    train_mean = np.mean(train_label.tocoo().data)
    train_std = np.std(train_label.tocoo().data)
    scaler = StandScaler(train_mean, train_std)

    tuple_train_label = sparse_to_tuple(train_label)
    transform_data = scaler.transform(tuple_train_label[1])
    transform_train_label = sp.csr_matrix(
        (transform_data, (tuple_train_label[0][:, 0], tuple_train_label[0][:, 1])), shape=tuple_train_label[2])

    # whether to use Bert Embedded skill features
    if args.feature == "embed":
        with open("/data/skill_feature.pkl", "rb") as f:
            features = pkl.load(f).to(device)
    else:
        features = transform_train_label + sp.identity(adj_train.shape[0])      # Norm(X) + I
        features = sparse_to_tuple(features)
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                            torch.FloatTensor(features[1]),
                                            torch.Size(features[2])).to(device)

    model = UGAE(gcn_adj, adj_e, adj_n, device, args).to(device)
    optimizer = Adam(model.parameters(), lr=args.learn_rate, weight_decay=0)

    for epoch in range(1, args.epoch+1):
        link_pred, weight_pred = model(features)        # [N, N], (E, )
        optimizer.zero_grad()
        # weight_loss = torch.sqrt(F.mse_loss(scaler.inverse_transform(weight_pred), weight_label.coalesce().values()))
        weight_loss = F.mse_loss(scaler.inverse_transform(weight_pred), weight_label.coalesce().values())
        train_mae = mean_absolute_error(weight_label.coalesce().values().detach().numpy(), 
                                        scaler.inverse_transform(weight_pred).detach().numpy())
        link_loss = norm * F.binary_cross_entropy(link_pred.view(-1), link_label.to_dense().view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / link_pred.size(0) * (1 + 2 * model.encoder.sgc.logstd -
                                                   model.encoder.sgc.miu**2 - torch.exp(model.encoder.sgc.logstd)**2).sum(1).mean()
        loss = args.beta * weight_loss + link_loss - kl_divergence
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            print("Epoch: {}".format(epoch), "Weight_Loss={:.4f}".format(weight_loss.item(
            )), "Link_loss={:.4f}".format(link_loss.item()), "KL_loss={:.4f}".format(-kl_divergence))
            val_rmse, val_mae = weight_eval(model, A, val_edges_pos, scaler, device, args)
            test_rmse, test_mae = weight_eval(model, A, test_edges_pos, scaler, device, args)
            print("Weight Prediction Task: ", "Train_MAE={:.4f}".format(train_mae),
                  "Val_RMSE={:.4f}".format(val_rmse), "Val_MAE={:.4f}".format(val_mae),
                  "Test_RMSE={:.4f}".format(test_rmse), "Test_MAE={:.4f}".format(test_mae))

            val_auc, val_ap = link_eval(link_pred, val_edges_pos, val_edges_false)
            test_auc, test_ap = link_eval(link_pred, test_edges_pos, test_edges_false)
            print("Link Prediction Task: ", "Val_AUC={:.4f}".format(val_auc), "Val_AP={:.4f}".format(val_ap),
                  "Test_AUC={:.4f}".format(test_auc), "Test_AP={:.4f}".format(test_ap))


def setup_seed(seed):
    """ Initialize Random Seed

    Args:
        seed (_type_): _description_
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Job Mobility.')
    parser.add_argument('-lr', '--learn_rate', default=0.001, type=float)
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('-clip', '--clip', default='yes', type=str)
    parser.add_argument('-dp', '--data_path', default='it_month/', type=str)
    parser.add_argument('-ep', '--epoch', default=300, type=int)
    parser.add_argument('-indim', '--input_dim', default=768, type=int)
    parser.add_argument('-hidim', '--hidden_dim', default=64, type=int)
    parser.add_argument('-outdim', '--output_dim', default=128, type=int)
    parser.add_argument('-beta', '--beta', default=0.001, type=float)
    parser.add_argument('-mode', '--mode', default="month", type=str)
    parser.add_argument('-f', '--feature', default="embed", type=str)
    parser.add_argument('-lw', '--lamb_w', default=0.1, type=float)
    parser.add_argument('-ll', '--lamb_l', default=5.0, type=float)
    args = parser.parse_args()
    sys.stdout = Logger('/log/' + 'beta_' + str(args.beta) + 'lr_' + str(args.learn_rate))
    print(args)
    setup_seed(2022)
    graph_train(args)
