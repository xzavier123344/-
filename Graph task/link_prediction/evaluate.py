from sklearn.metrics import roc_auc_score, average_precision_score
import torch

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        val_pred = model.decode(z, data.val_pos_edge_index)
        val_neg = model.decode(z, data.val_neg_edge_index)
        preds = torch.cat([val_pred, val_neg])
        labels = torch.cat([torch.ones(val_pred.size(0)), torch.zeros(val_neg.size(0))])
        auc = roc_auc_score(labels.cpu(), preds.cpu())
        ap = average_precision_score(labels.cpu(), preds.cpu())
    return auc, ap
