import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class LogisticRegressionModel:
    def __init__(self, input_dim: int, output_dim: int):
        self.W = np.random.uniform(
            low=-0.08, high=0.08, size=(input_dim, output_dim)
        ).astype("float32")
        self.b = np.zeros(shape=(output_dim,)).astype("float32")

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

        self.optimizer = Optimizer(
            method="adam", shape_W=self.W.shape, shape_b=self.b.shape
        )

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.W, self.b

    def load_data(self, data_path: str) -> None:
        x_train = np.load(os.path.join(data_path, "x_train.npy"))
        y_train = np.load(os.path.join(data_path, "y_train.npy"))
        x_test = np.load(os.path.join(data_path, "x_test.npy"))

        x_train = x_train.reshape(-1, 784).astype("float32") / 255
        y_train = np.eye(10)[y_train.astype("int32")]

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            x_train, y_train, test_size=0.1
        )
        self.x_test = x_test.reshape(-1, 784).astype("float32") / 255

    def train(self, lr: float) -> float:
        batch_size = self.x_train.shape[0]

        y_hat = self.softmax(np.dot(self.x_train, self.W) + self.b)

        cost = (-self.y_train * self.np_log(y_hat)).sum(axis=1).mean()
        delta = y_hat - self.y_train

        dW = np.dot(self.x_train.T, delta) / batch_size
        db = np.sum(delta, axis=0) / batch_size

        self.W, self.b = self.optimizer.step(self.W, self.b, dW, db, lr)

        return cost

    def valid(self) -> tuple[float, np.ndarray]:
        y_hat = self.softmax(np.dot(self.x_valid, self.W) + self.b)
        cost = (-self.y_valid * self.np_log(y_hat)).sum(axis=1).mean()
        return cost, y_hat

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        y_hat = self.softmax(np.dot(self.x_test, self.W) + self.b)
        y_pred = np.argmax(y_hat, axis=1)
        return y_pred, y_hat

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def np_log(x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e10))


class Optimizer:
    def __init__(
        self,
        method="sgd",
        shape_W=None,
        shape_b=None,
        beta=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.method = method
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1

        self.v_W = np.zeros(shape_W)
        self.v_b = np.zeros(shape_b)

        self.G_W = np.zeros(shape_W)
        self.G_b = np.zeros(shape_b)

        self.m_W = np.zeros(shape_W)
        self.v2_W = np.zeros(shape_W)
        self.m_b = np.zeros(shape_b)
        self.v2_b = np.zeros(shape_b)

    def step(self, W, b, dW, db, lr):
        if self.method == "sgd":
            return self.sgd(W, b, dW, db, lr)
        elif self.method == "momentum":
            return self.momentum(W, b, dW, db, lr)
        elif self.method == "adagrad":
            return self.adagrad(W, b, dW, db, lr)
        elif self.method == "adam":
            return self.adam(W, b, dW, db, lr)
        else:
            raise ValueError(f"Unknown optimizer method: {self.method}")

    def sgd(self, W, b, dW, db, lr):
        W -= lr * dW
        b -= lr * db
        return W, b

    def momentum(self, W, b, dW, db, lr):
        self.v_W = self.beta * self.v_W - lr * dW
        self.v_b = self.beta * self.v_b - lr * db
        W += self.v_W
        b += self.v_b
        return W, b

    def adagrad(self, W, b, dW, db, lr):
        self.G_W += dW**2
        self.G_b += db**2
        W -= lr * dW / (np.sqrt(self.G_W) + self.epsilon)
        b -= lr * db / (np.sqrt(self.G_b) + self.epsilon)
        return W, b

    def adam(self, W, b, dW, db, lr):
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.v2_W = self.beta2 * self.v2_W + (1 - self.beta2) * (dW**2)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v2_b = self.beta2 * self.v2_b + (1 - self.beta2) * (db**2)

        m_W_hat = self.m_W / (1 - self.beta1**self.t)
        v_W_hat = self.v2_W / (1 - self.beta2**self.t)
        m_b_hat = self.m_b / (1 - self.beta1**self.t)
        v_b_hat = self.v2_b / (1 - self.beta2**self.t)

        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        self.t += 1
        return W, b


class Scheduler:
    def __init__(
        self,
        initial_lr: float,
        decay_rate: float = 0.95,
        decay_step: int = 10,
        plateau_patience: int = 10,
        plateau_factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.patience = plateau_patience
        self.factor = plateau_factor
        self.min_lr = min_lr
        self.val_loss_history = []
        self.current_lr = initial_lr

    def step(self, epoch: int, val_loss: float) -> float:
        scheduled_lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_step))
        self.current_lr = scheduled_lr

        self.val_loss_history.append(val_loss)
        if len(self.val_loss_history) >= self.patience + 1:
            recent = self.val_loss_history[-(self.patience + 1) :]
            if min(recent[:-1]) <= recent[-1]:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)

        return self.current_lr


if __name__ == "__main__":
    initial_lr = 0.01
    threshold = 0.00001
    prev_cost = 0.0

    model = LogisticRegressionModel(input_dim=784, output_dim=10)
    model.load_data(data_path="./02/data")

    for epoch in range(10000):
        cost = model.train(lr=initial_lr)
        cost, y_pred = model.valid()

        if epoch % 10 == 0:
            acc = accuracy_score(
                np.argmax(model.y_valid, axis=1), np.argmax(y_pred, axis=1)
            )
            print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {acc:.4f}")
            if abs(prev_cost - cost) < threshold:
                print(f"Early stopping at epoch {epoch}, Cost: {cost:.4f}")
                break
            prev_cost = cost

    y_pred, y_hat = model.predict()
    submission = pd.Series(y_pred, name="label")
    submission.to_csv(
        "./02/submission/submission_pred.csv", header=True, index_label="id"
    )
