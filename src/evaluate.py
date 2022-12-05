import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def link_eval(link_A, test_edges_pos, test_edges_false):
    """ Link Prediction

    Args:
        link_A (_type_): _description_
        test_edges_pos (_type_): _description_
        test_edges_false (_type_): _description_

    Returns:
        _type_: _description_
    """
    link_pred = []
    link_true = np.hstack([np.ones(len(test_edges_pos)),
                          np.zeros(len(test_edges_false))])
    for idx, idy in test_edges_pos:
        link_pred.append(link_A[idx][idy].detach().item())

    for idx, idy in test_edges_false:
        link_pred.append(link_A[idx][idy].detach().item())
    link_pred = np.array(link_pred)

    roc_score = roc_auc_score(link_true, link_pred)
    ap_score = average_precision_score(link_true, link_pred)
    f1 = f1_score(link_true, link_pred, )

    return roc_score, ap_score


def weight_eval(model, A, test_edges, scaler, device, args):
    """ Edge Weight Prediction

    Args:
        model (_type_): _description_
        A (_type_): _description_
        test_edges (_type_): _description_
        scaler (_type_): _description_
        device (_type_): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    edge = torch.from_numpy(test_edges).long().to(device)
    edge_in, edge_out = model.encoder.sgc.edge_embedding(
        model.encoder.hidden, edge.T)
    weight_pred = model.decoder.weight_pred((edge_in, edge_out))
    # weight_pred = model.decoder.weight_pred_by_mlp((edge_in, edge_out))
    weight_pred = scaler.inverse_transform(weight_pred.detach().numpy())

    weight_true = []
    for edge in test_edges:
        weight_true.append(A[edge[0], edge[1]])

    if args.mode == "month":
        weight_pred = np.round(np.array(weight_pred) / 3)
        weight_true = np.round(np.array(weight_true) / 3)

    rmse = np.sqrt(mean_squared_error(weight_true, weight_pred))
    mae = mean_absolute_error(weight_true, weight_pred)
    return rmse, mae
    