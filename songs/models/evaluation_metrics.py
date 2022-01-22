import numpy as np

def Recall(pred_val, true_val, k=100):
    batch_plys = pred_val.shape[0]

    idx = np.argpartition(-pred_val, k, axis=1)
    pred_val = np.zeros_like(pred_val, dtype=bool)
    pred_val[np.arange(batch_plys)[:, np.newaxis], idx[:, :k]] = True
    true_val = (true_val > 0).toarray()
    
    tmp = (np.logical_and(true_val, pred_val).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, true_val.sum(axis=1))
    return recall
    
def NDCG(pred_val, true_val, k=100):

    n_plys = pred_val.shape[0]
    
    idx_topk_part = np.argpartition(pred_val, k, axis=1)
    topk_part = pred_val[np.arange(n_plys)[:, np.newaxis],idx_topk_part[:, :k]]
    
    idx = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(n_plys)[:, np.newaxis], idx]
    
    tmp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (true_val[np.arange(n_plys)[:, np.newaxis], idx_topk].toarray() * tmp).sum(axis=1)
    IDCG = np.array([(tmp[:min(n, k)]).sum() for n in true_val.getnnz(axis=1)])
    return DCG / IDCG