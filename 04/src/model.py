import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
import inspect

torch._dynamo.disable()

class MyList:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train.reshape(-1, 784).astype('float32') / 255
        self.t_train = t_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), torch.tensor(self.t_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype('float32') / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)

def relu(x):
    x = torch.where(x > 0, x, torch.zeros_like(x))
    return x


def softmax(x):
    x -= torch.cat([x.max(axis=1, keepdim=True).values] * x.size()[1], dim=1)
    x_exp = torch.exp(x)
    return x_exp/torch.cat([x_exp.sum(dim=1, keepdim=True)] * x.size()[1], dim=1)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(np.random.uniform(
                        low=-np.sqrt(6/in_dim),
                        high=np.sqrt(6/in_dim),
                        size=(in_dim, out_dim)
                    ).astype('float32')))
        self.b = nn.Parameter(torch.tensor(np.zeros([out_dim]).astype('float32')))
        self.function = function

    def forward(self, x):
        return self.function(torch.matmul(x, self.W) + self.b)

class BatchNormalization(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones([in_dim]))
        self.beta = nn.Parameter(torch.zeros([in_dim]))

    def forward(self, x):
        mean = torch.mean(x, dim=0)
        var = torch.var(x, dim=0)
        x = (x - mean) / torch.sqrt(var + 1e-7)
        return self.gamma * x + self.beta
    
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size()) > self.p
            mask = mask.to("cuda")
            x = x * mask / (1 - self.p)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = Dense(in_dim, hid_dim, function=relu)
        self.dropout = Dropout(p=0.3)
        self.linear2 = Dense(hid_dim, hid_dim, function=relu)
        self.dropout2 = Dropout(p=0.3)
        self.linear3 = Dense(hid_dim, out_dim, function=softmax)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    nn_except = ["Module", "Parameter", "Sequential"]
    for m in inspect.getmembers(nn):
        if m[0] not in nn_except and m[0][0:2] != "__":
            delattr(nn, m[0])

    in_dim = 784
    hid_dim = 200
    out_dim = 10
    lr = 0.001
    n_epochs = 20
    seed = 1234
    batch_size = 32
    val_size = 10000

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #学習データ
    x_train = np.load('./04/data/x_train.npy')
    t_train = np.load('./04/data/y_train.npy')

    #テストデータ
    x_test = np.load('./04/data/x_test.npy')

    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)

    train_size = len(trainval_data) - val_size

    train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

    dataloader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_valid = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )


    mlp = MLP(in_dim, hid_dim, out_dim).to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        mlp.train()  # 訓練時には勾配を計算するtrainモードにする
        for x, t in dataloader_train:
            t_hot = torch.eye(out_dim)[t]
            optimizer.zero_grad()
            x = x.to(device)
            t_hot = t_hot.to(device)

            pred = mlp(x)
            loss = -torch.mean(torch.sum(t_hot * torch.log(pred+1e-7), dim=1))
            loss.backward()
            optimizer.step()

            losses_train.append(loss.tolist())

            pred = torch.argmax(pred, dim=1)
            acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
            train_num += acc.size()[0]
            train_true_num += acc.sum().item()

        mlp.eval()  # 評価時には勾配を計算しないevalモードにする
        for x, t in dataloader_valid:
            t_hot = torch.eye(out_dim)[t]
            x = x.to(device)
            t_hot = t_hot.to(device)

            pred = mlp(x)
            loss = -torch.mean(torch.sum(t_hot * torch.log(pred+1e-7), dim=1))

            losses_valid.append(loss.tolist())

            pred = torch.argmax(pred, dim=1)
            acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
            valid_num += acc.size()[0]
            valid_true_num += acc.sum().item()

        print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
            epoch,
            np.mean(losses_train),
            train_true_num/train_num,
            np.mean(losses_valid),
            valid_true_num/valid_num
        ))

    mlp.eval()

    t_pred = []
    for x in dataloader_test:

        x = x.to(device)

        # 順伝播
        y = mlp.forward(x)

        # モデルの出力を予測値のスカラーに変換
        pred = y.argmax(1).tolist()

        t_pred.extend(pred)

    submission = pd.Series(t_pred, name='label')
    submission.to_csv('./04/submission/submission_pred.csv', header=True, index_label='id')