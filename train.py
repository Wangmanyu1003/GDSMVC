import torch
import torch.nn.functional as F
import numpy as np
import argparse
import random
import os
import scipy.io as sio

from network import Network
from metric import valid
from loss import Loss
from dataloader import load_data

# Supported datasets:
#   BDGP, CCV, Fashion, Caltech_5V, Hdigit, Cifar100, cifar10,
#   Youtubeface, Prokaryotic, Synthetic3d, WebKB, Cora, NGs,
#   MNIST-USPS, Noisy-MNIST, NUSWIDE, SUNRGBD, YouTubeVideo,
#   ALOI, bbcsport, DHA, handwritten, caltech101_7

Dataname = 'WebKB'

parser = argparse.ArgumentParser(description='GDS-MVC Training')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--temperature_f', default=0.8, type=float)
parser.add_argument('--temperature_l', default=0.5, type=float)
parser.add_argument('--learning_rate', default=0.0003, type=float)
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--mse_epochs', default=100, type=int)
parser.add_argument('--con_epochs', default=100, type=int)
parser.add_argument('--tune_epochs', default=3, type=int)
parser.add_argument('--low_feature_dim', default=512, type=int)
parser.add_argument('--high_feature_dim', default=128, type=int)
parser.add_argument('--layer_num', default=3, type=int)
parser.add_argument('--lambda_1', default=0.01, type=float)
parser.add_argument('--lambda_2', default=0.001, type=float)
parser.add_argument('--lambda_3', default=0.001, type=float)
parser.add_argument('--show_interval', default=5, type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not found!")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain(epoch):
    model.train()
    tot_loss = 0.0
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, pfs, sfs, _, S, _, _ = model(xs)

        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()


def tune(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, pfs, sfs, Q, S, hs, L = model(xs)

        loss_list = []
        for v in range(view):
            l = mse(xs[v], xrs[v])
            l1 = mse(torch.matmul(pfs[v], sfs[v].T), zeros)
            l2 = criterion.uniform_uncertainty_loss(Q[v])
            l33 = criterion.forward_label(L, hs[v])
            l4 = criterion.Structure_guided_Contrastive_Loss(sfs[v], S, args.batch_size)
            for w in range(view):
                l3 = criterion.distribution_alignment_loss(hs[v], hs[w])
            loss_list.extend([
                l, args.lambda_1 * l1, args.lambda_1 * l2,
                args.lambda_1 * l3, args.lambda_2 * l33, args.lambda_3 * l4
            ])

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss


# Load dataset
dataset, dims, view, data_size, class_num = load_data(args.dataset)

os.makedirs('result_8_8', exist_ok=True)

ACCS = np.zeros((8, 8, 8), dtype=np.float32)
NMIS = np.zeros((8, 8, 8), dtype=np.float32)
PURS = np.zeros((8, 8, 8), dtype=np.float32)
times = 1

# Hyperparameter grid search
for args.mse_epochs in [100]:
    for args.con_epochs in [100]:
        for seed in [10]:
            index1 = 0
            for args.lambda1 in [0.01]:
                index2 = 0
                for args.lambda2 in [0.001]:
                    index3 = 0
                    for args.lambda3 in [0.001]:
                        print(f'lambda1: {args.lambda1}, lambda2: {args.lambda2}, lambda3: {args.lambda3}')
                        setup_seed(seed)

                        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                        )

                        accs, nmis, purs = [], [], []

                        model = Network(view, dims, args.low_feature_dim, args.high_feature_dim, class_num, device)
                        model = model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                        criterion = Loss(args.batch_size, args.temperature_f, args.temperature_l, class_num, device).to(device)
                        zeros = torch.zeros(args.batch_size, args.batch_size).to(device)
                        print(times)

                        # Pre-training phase
                        epoch = 1
                        while epoch <= args.mse_epochs:
                            pretrain(epoch)
                            epoch += 1

                        # Training phase
                        while epoch <= args.mse_epochs + args.con_epochs:
                            loss = tune(epoch)
                            if (epoch - args.mse_epochs) % args.show_interval == 0:
                                acc, nmi, _, pur = valid(model, device, dataset, view, data_size, epoch)
                                accs.append(acc)
                                nmis.append(nmi)
                                purs.append(pur)
                            epoch += 1

                        # Select best performance
                        all_metrics = list(np.array(accs) + np.array(nmis) + np.array(purs))
                        max_index = all_metrics.index(max(all_metrics))
                        best_epoch = args.mse_epochs + (max_index + 1) * args.show_interval
                        print('=' * 70)
                        print('Best Epoch = {:d} | ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(
                            best_epoch, accs[max_index], nmis[max_index], purs[max_index]))

                        ACCS[index1, index2, index3] = accs[max_index]
                        NMIS[index1, index2, index3] = nmis[max_index]
                        PURS[index1, index2, index3] = purs[max_index]

                        times += 1
                        index3 += 1
                    index2 += 1
                index1 += 1

# Save results
sio.savemat('result_8_8/' + Dataname + '_acc.mat', {'ACCS': ACCS})
sio.savemat('result_8_8/' + Dataname + '_nmi.mat', {'NMIS': NMIS})
sio.savemat('result_8_8/' + Dataname + '_pur.mat', {'PURS': PURS})
print('Results saved to result_8_8/')
