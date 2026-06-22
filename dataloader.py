import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler

class bbcsport(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'bbcsport.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'bbcsport.mat')['X2'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        labels = scipy.io.loadmat(path+'bbcsport.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class DHA(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'DHA.mat')['X1']
        data2 = scipy.io.loadmat(path+'DHA.mat')['X2']
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        labels = scipy.io.loadmat(path+'DHA.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Prokaryotic(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'prokaryotic.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'prokaryotic.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'prokaryotic.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'prokaryotic.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Synthetic3d(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'synthetic3d.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'synthetic3d.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'synthetic3d.mat')['X3'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        labels = scipy.io.loadmat(path+'synthetic3d.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class SUNRGBD(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'SUNRGBD.mat')['X1'].astype(np.float32).transpose()
        data2 = scipy.io.loadmat(path+'SUNRGBD.mat')['X2'].astype(np.float32).transpose()
        labels = scipy.io.loadmat(path+'SUNRGBD.mat')['Y']
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class YouTubeVideo(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Video-3V.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Video-3V.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Video-3V.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'Video-3V.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Noisy_MNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['Y'].astype(np.int32).reshape(30000,)
        self.V1 = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NoisyMNIST-30000.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 30000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class WebKB(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'WebKB.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'WebKB.mat')['X2'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        labels = scipy.io.loadmat(path + 'WebKB.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Fashion(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Fashion.mat')

        # 读取
        Y  = data['Y'].astype(np.int32).reshape(-1,)
        X1 = np.asarray(data['X1']).astype(np.float32)
        X2 = np.asarray(data['X2']).astype(np.float32)
        X3 = np.asarray(data['X3']).astype(np.float32)

        # 去掉多余的长度为1的维度（如 (N,28,28,1) -> (N,28,28)）
        X1 = np.squeeze(X1)
        X2 = np.squeeze(X2)
        X3 = np.squeeze(X3)

        # 展平为二维 (N, D) —— 这是 MinMaxScaler 的输入要求
        N = Y.shape[0]
        X1 = X1.reshape(N, -1)
        X2 = X2.reshape(N, -1)
        X3 = X3.reshape(N, -1)

        # 分别按视图做 0-1 归一化
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        scaler3 = MinMaxScaler()
        X1 = scaler1.fit_transform(X1)
        X2 = scaler2.fit_transform(X2)
        X3 = scaler3.fit_transform(X3)

        # 保存
        self.Y  = Y
        self.V1 = X1.astype(np.float32)
        self.V2 = X2.astype(np.float32)
        self.V3 = X3.astype(np.float32)

    def __len__(self):
        return self.Y.shape[0]   # 不再写死 10000

    def __getitem__(self, idx):
        # 若特征维就是 784，会保持不变；否则按你的接口强制 reshape 成 784
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Cifar100(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Cifar100.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Cifar100.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Cifar100.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'Cifar100.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Caltech_5V(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Caltech_5V.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Caltech_5V.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path +'Caltech_5V.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path +'Caltech_5V.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path +'Caltech_5V.mat')['X5'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        data5 = min_max_scaler.fit_transform(data5.astype(np.float32))
        labels = scipy.io.loadmat(path+'Caltech_5V.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NUSWIDE(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'NUSWIDE.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'NUSWIDE.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X5'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        data5 = min_max_scaler.fit_transform(data5.astype(np.float32))
        labels = scipy.io.loadmat(path+'NUSWIDE.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class NGs(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'NGs.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'NGs.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'NGs.mat')['X3'].astype(np.float32)
        labels = scipy.io.loadmat(path+'NGs.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class cifar10(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'cifar10.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'cifar10.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'cifar10.mat')['X3'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        labels = scipy.io.loadmat(path+'cifar10.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class handwritten(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'handwritten.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'handwritten.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'handwritten.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'handwritten.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'handwritten.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'handwritten.mat')['X6'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        data5 = min_max_scaler.fit_transform(data5.astype(np.float32))
        data6 = min_max_scaler.fit_transform(data6.astype(np.float32))
        labels = scipy.io.loadmat(path+'handwritten.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx]),torch.from_numpy(
           self.x6[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')
    def __len__(self):
        return 6773
    def __getitem__(self, idx):
        x1 = self.data1[idx].reshape(5000)
        x2 = self.data2[idx].reshape(5000)
        x3 = self.data3[idx].reshape(4000)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.labels[idx], torch.from_numpy(np.array(idx)).long()
class caltech101_7(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'caltech101-7.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'caltech101-7.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'caltech101-7.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'caltech101-7.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'caltech101-7.mat')['X5'].astype(np.float32)
        data6 = scipy.io.loadmat(path + 'caltech101-7.mat')['X6'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        data5 = min_max_scaler.fit_transform(data5.astype(np.float32))
        data6 = min_max_scaler.fit_transform(data6.astype(np.float32))
        labels = scipy.io.loadmat(path+'caltech101-7.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx]),torch.from_numpy(
           self.x6[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class YoutubeFace(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Youtubeface.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Youtubeface.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Youtubeface.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Youtubeface.mat')['X4'].astype(np.float32)
        data5 = scipy.io.loadmat(path + 'Youtubeface.mat')['X5'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        data5 = min_max_scaler.fit_transform(data5.astype(np.float32))

        labels = scipy.io.loadmat(path+'Youtubeface.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.x5 = data5
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx]), torch.from_numpy(
           self.x4[idx]), torch.from_numpy(
           self.x5[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Cora(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'Cora.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'Cora.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'Cora.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'Cora.mat')['X4'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        labels = scipy.io.loadmat(path+'Cora.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(
           self.x3[idx]),torch.from_numpy(
           self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class ALOI(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'ALOI.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'ALOI.mat')['X2'].astype(np.float32)
        data3 = scipy.io.loadmat(path + 'ALOI.mat')['X3'].astype(np.float32)
        data4 = scipy.io.loadmat(path + 'ALOI.mat')['X4'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        data3 = min_max_scaler.fit_transform(data3.astype(np.float32))
        data4 = min_max_scaler.fit_transform(data4.astype(np.float32))
        labels = scipy.io.loadmat(path+'ALOI.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.x4 = data4
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]),torch.from_numpy(
           self.x3[idx]),torch.from_numpy(
           self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Hdigit():
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'Hdigit.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'Hdigit.mat')['X2'].astype(np.float32)
        data1 = min_max_scaler.fit_transform(data1.astype(np.float32))
        data2 = min_max_scaler.fit_transform(data2.astype(np.float32))
        labels = scipy.io.loadmat(path + 'Hdigit.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

# class Caltech(Dataset):
#     def __init__(self, path, view):
#         data = scipy.io.loadmat(path)
#         scaler = MinMaxScaler()
#         self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
#         self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
#         self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
#         self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
#         self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
#         self.labels = scipy.io.loadmat(path)['Y'].transpose()
#         self.view = view
#
#     def __len__(self):
#         return 1400
#
#     def __getitem__(self, idx):
#         if self.view == 2:
#             return [torch.from_numpy(
#                 self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
#         if self.view == 3:
#             return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
#                 self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
#         if self.view == 4:
#             return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
#                 self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
#         if self.view == 5:
#             return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
#                 self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
#                 self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "handwritten":
        dataset = handwritten('data/')
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "caltech101_7":
        dataset = handwritten('data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 1474
        class_num = 7
    elif dataset == "DHA":
        dataset = DHA('data/')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == "bbcsport":
        dataset = bbcsport('data/')
        dims = [3183,3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000

    elif dataset == "WebKB":
        dataset = WebKB('data/')
        dims = [2949, 334]
        view = 2
        class_num = 2
        data_size = 1051

    elif dataset == "Cora":
        dataset = Cora('data/')
        dims = [2708,1433,2706,2706]
        view = 4
        class_num = 7
        data_size = 2708

    elif dataset == "Noisy-MNIST":
        dataset = Noisy_MNIST('data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 30000
    elif dataset == "CCV":
        dataset = CCV('data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "NGs":
        dataset = NGs('data/')
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == "Cifar100":
        dataset = Cifar100('data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "cifar10":
        dataset = cifar10('data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Youtubeface":
        dataset = YoutubeFace('data/')
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == "Prokaryotic":
        dataset = Prokaryotic('data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "ALOI":
        dataset = ALOI('data/')
        dims = [77, 13, 64, 125]
        view = 4
        data_size = 10800
        class_num = 100
    elif dataset == "Synthetic3d":
        dataset = Synthetic3d('data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "YouTubeVideo":
        dataset = YouTubeVideo('data/')
        dims = [512, 647, 838]
        view = 3
        data_size = 101499
        class_num = 31
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('data/')
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 5000
        class_num = 5
    elif dataset == "Hdigit":
        dataset = Hdigit('data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "SUNRGBD":
        dataset = SUNRGBD('data/')
        dims = [4096, 4096]
        view = 2
        data_size = 10335
        class_num = 45
    elif dataset == "Fashion":
        dataset = Fashion('data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    # elif dataset == "Caltech-2V":
    #     dataset = Caltech('data/Caltech_5.mat', view=2)
    #     dims = [40, 254]
    #     view = 2
    #     data_size = 1400
    #     class_num = 7
    # elif dataset == "Caltech-3V":
    #     dataset = Caltech('data/Caltech_5.mat', view=3)
    #     dims = [40, 254, 928]
    #     view = 3
    #     data_size = 1400
    #     class_num = 7
    # elif dataset == "Caltech-4V":
    #     dataset = Caltech('data/Caltech_5.mat', view=4)
    #     dims = [40, 254, 928, 512]
    #     view = 4
    #     data_size = 1400
    #     class_num = 7
    elif dataset == "Caltech_5V":
        dataset = Caltech_5V('data/')
        dims = [40, 254, 1984, 512, 928]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
