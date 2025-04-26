import os
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def np_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-10, 1e10))


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (-y_true * np_log(y_pred)).sum(axis=1).mean()


def create_batch(data, batch_size):
    """
    :param data: np.ndarray: 入力データ
    :param batch_size: int: バッチサイズ
    """
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches :])

    return batched_data


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self):
        x_train = np.load(os.path.join(self.data_path, "x_train.npy"))
        y_train = np.load(os.path.join(self.data_path, "y_train.npy"))
        x_test = np.load(os.path.join(self.data_path, "x_test.npy"))

        x_train = x_train.reshape(-1, 784).astype("float32") / 255
        y_train = np.eye(10)[y_train.astype("int32")]

        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.1
        )
        x_test = x_test.reshape(-1, 784).astype("float32") / 255

        return x_train, y_train, x_valid, y_valid, x_test


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError()

    @abstractmethod
    def backward(self, out: np.ndarray) -> np.ndarray:
        NotImplementedError()


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b

        self.x: np.ndarray = np.empty(0)

        self.dW: np.ndarray = np.zeros_like(W)
        self.db: np.ndarray = np.zeros_like(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, out: np.ndarray) -> np.ndarray:
        dx: np.ndarray = np.dot(out, self.W.T)
        self.dW[...] = np.dot(self.x.T, out)
        self.db[...] = np.sum(out, axis=0)
        return dx

    def update(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Sigmoid(Layer):
    def __init__(self):
        self.out: np.ndarray = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))
        return self.out

    def backward(self, out: np.ndarray) -> np.ndarray:
        return out * (1 - self.out) * self.out
    
class ReLU(Layer):
    def __init__(self):
        self.out: np.ndarray = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.maximum(x, 0)
        return self.out

    def backward(self, out: np.ndarray) -> np.ndarray:
        return out * (self.out > 0)


class SoftMaxWithLoss:
    def __init__(self):
        self.x: np.ndarray = np.empty(0)
        self.t: np.ndarray = np.empty(0)
        self.u: np.ndarray = np.empty(0)
        self.y: np.ndarray = np.empty(0)
        self.loss: float = 0.0

    def forward(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
        self.x = x
        self.t = t

        self.u = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - self.u)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.loss = cross_entropy_loss(self.t, self.y)
        return self.y, self.loss

    def backward(self) -> np.ndarray:
        return self.y - self.t


class Model(metaclass=ABCMeta):
    def __init__(self):
        self.params: list[np.ndarray] = []
        self.grads: list[np.ndarray] = []

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError()

    @abstractmethod
    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        NotImplementedError()

    @abstractmethod
    def backward(self) -> np.ndarray:
        NotImplementedError()

    @abstractmethod
    def update(self, lr: float) -> None:
        NotImplementedError()


class MLP(Model):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        W1 = np.random.uniform(
            low=-0.08, high=0.08, size=(input_dim, hidden_dim)
        ).astype("float32")
        b1 = np.zeros(shape=(hidden_dim,)).astype("float32")
        W2 = np.random.uniform(
            low=-0.08, high=0.08, size=(hidden_dim, output_dim)
        ).astype("float32")
        b2 = np.zeros(shape=(output_dim,)).astype("float32")
        W3 = np.random.uniform(
            low=-0.08, high=0.08, size=(hidden_dim, hidden_dim)
        ).astype("float32")
        b3 = np.zeros(shape=(hidden_dim,)).astype("float32")
        W4 = np.random.uniform(
            low=-0.08, high=0.08, size=(hidden_dim, output_dim)
        ).astype("float32")
        b4 = np.zeros(shape=(output_dim,)).astype("float32")

        self.layers: list[Layer] = [
            Affine(W1, b1),
            ReLU(),
            Affine(W2, b2),
        ]
        self.loss_layer = SoftMaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            if isinstance(layer, Affine):
                self.params += [layer.W, layer.b]
                self.grads += [layer.dW, layer.db]

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        pred = self.predict(x)
        pred, loss = self.loss_layer.forward(pred, y)
        return pred, loss

    def backward(self) -> np.ndarray:
        grad = self.loss_layer.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.update(lr)


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        NotImplementedError()


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self.m is None or self.v is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for i, param in enumerate(params):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


def train_model(
    model: Model,
    optimizer: Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    n_epochs: int,
    batch_size: int,
    patience: int = 100,
):
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = None
    patience_counter = 0

    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        x_train, y_train = shuffle(x_train, y_train)  # type: ignore
        x_train_batches, y_train_batches = (
            create_batch(x_train, batch_size),
            create_batch(y_train, batch_size),
        )

        x_valid, y_valid = shuffle(x_valid, y_valid)  # type: ignore
        x_valid_batches, y_valid_batches = (
            create_batch(x_valid, batch_size),
            create_batch(y_valid, batch_size),
        )

        # モデルの訓練
        for x, t in zip(x_train_batches, y_train_batches):
            # 順伝播
            pred, loss = model.forward(x, t)

            # 損失の計算
            losses_train.append(loss)

            # パラメータの更新
            model.backward()
            optimizer.update(model.params, model.grads)

            # 精度を計算
            acc = accuracy_score(t.argmax(axis=1), pred.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += acc

        # モデルの評価
        for x, t in zip(x_valid_batches, y_valid_batches):
            # 順伝播
            pred, loss = model.forward(x, t)

            # 損失の計算
            losses_valid.append(loss)

            acc = accuracy_score(t.argmax(axis=1), pred.argmax(axis=1), normalize=False)
            valid_num += x.shape[0]
            valid_true_num += acc

        avg_val_loss = np.mean(losses_valid)
        avg_train_loss = np.mean(losses_train)
        train_acc = train_true_num / train_num
        valid_acc = valid_true_num / valid_num

        print(
            "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]".format(
                epoch,
                avg_train_loss,
                train_acc,
                avg_val_loss,
                valid_acc,
            )
        )

        # early stopping 判定
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_params = copy.deepcopy(model.params)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                if best_params is not None:
                    model.params = best_params
                break

def predict(model: Model, x_test: np.ndarray) -> None:
    pred = model.predict(x_test)
    labels = np.argmax(pred, axis=1)
    submission = pd.Series(labels, name="label")
    submission.to_csv(
        "./03/submission/submission_pred.csv", header=True, index_label="id"
    )

if __name__ == "__main__":
    data_loader = DataLoader("./03/data")
    x_train, y_train, x_valid, y_valid, x_test = data_loader.load()

    input_dim = 784
    output_dim = 10
    hidden_dim = 128
    batch_size = 100
    n_epochs = 1000

    model = MLP(input_dim, output_dim, hidden_dim)
    optimizer = Adam(lr=0.001)

    train_model(
        model, optimizer, x_train, y_train, x_valid, y_valid, n_epochs, batch_size
    )
    predict(model, x_test)
