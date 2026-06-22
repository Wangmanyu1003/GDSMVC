from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.functional import normalize


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):

    model.eval()
    soft_vector = []
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, _, _, _, _,L = model(xs)

        q = L.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)

    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, labels_vector


def valid(model, device, dataset, view, data_size, epoch):

    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )

    total_pred, labels_vector = inference(test_loader, model, device, view, data_size)

    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)

    print('Epoch = {:d} =====>>>>> ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(epoch, acc, nmi, ari, pur))

    return acc, nmi, ari, pur
# metric.py
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def inn(loader, model, device, view, data_size=None):
    """
    全数据推理，收集可视化所需变量（按 loader 顺序拼接）：
    X_all : list[Tensor], len=view, each (N, ...)
    ps_all: list[Tensor], len=view, each (N, C)  (假设ps为每视图一个概率向量)
    S_all : Tensor (N, Ds) 或 (N, ...)
    L_all : Tensor (N, C) 或 (N, ...)
    y_all : Tensor (N,)
    pred  : numpy (N,)
    """
    model.eval()

    X_list  = [[] for _ in range(view)]
    ps_list = [[] for _ in range(view)]
    S_list  = []
    L_list  = []
    y_list  = []

    for step, (xs, y, _) in enumerate(loader):
        # 保存 raw（保持 CPU，保持顺序）
        for v in range(view):
            X_list[v].append(xs[v].detach().cpu())
            xs[v] = xs[v].to(device)

        y_list.append(y.detach().cpu())

        # forward
        _, _, _, ps, S, hs, L = model(xs)

        # ps 通常是 list[Tensor]，每个 view 一个
        for v in range(view):
            ps_list[v].append(ps[v].detach().cpu())

        S_list.append(S.detach().cpu())
        L_list.append(L.detach().cpu())

    # 拼接成全数据
    X_all  = [torch.cat(X_list[v], dim=0)  for v in range(view)]
    ps_all = [torch.cat(ps_list[v], dim=0) for v in range(view)]
    S_all  = torch.cat(S_list, dim=0)
    L_all  = torch.cat(L_list, dim=0)
    y_all  = torch.cat(y_list, dim=0).view(-1)

    # 可选一致性检查
    if data_size is not None:
        assert y_all.numel() == int(data_size), f"Got {y_all.numel()} samples, expect {data_size}"

    pred = torch.argmax(L_all, dim=1).cpu().numpy()

    return pred, y_all.numpy(), X_all, ps_all, S_all, L_all
def vv(model, device, dataset, view, data_size, batch_size=256):
    """
    仅用于训练结束后调用一次：返回可直接 torch.save 的 dict。
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # 必须 False，保证顺序稳定
        drop_last=False
    )

    pred, y, X_all, ps_all, S_all, L_all = inn(
        loader, model, device, view, data_size
    )

    return {
        "pred": pred,   # numpy (N,)
        "y": y,         # numpy (N,)
        "X": X_all,     # list[Tensor], len=view
        "ps": ps_all,   # list[Tensor], len=view
        "S": S_all,     # Tensor
        "L": L_all,     # Tensor
    }
