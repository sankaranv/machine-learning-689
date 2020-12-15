import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import gzip
import pickle
import os

np.random.seed(1)
torch.manual_seed(1)


class NN(nn.Module):
    """A network architecture for simultaneous classification
    and angle regression of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """

    def __init__(self, alpha=.5, epochs=5):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.layers = nn.ModuleList([nn.Linear(784, 256, bias=True),
                                     nn.Linear(256, 64, bias=True),
                                     nn.Linear(64, 32, bias=True),
                                     nn.Linear(64, 32, bias=True),
                                     nn.Linear(32, 10, bias=True),
                                     nn.Linear(32, 1, bias=True)
                                     ])
        # for layer in self.layers:
        #     variance = math.sqrt(
        #         2.0 / (layer.in_features + layer.out_features))
        #     layer.weight = nn.Parameter(torch.Tensor(
        #         layer.out_features, layer.in_features).normal_(0.0, variance))
        #     layer.bias = nn.Parameter(torch.zeros(layer.out_features))

    def objective(self, X, y_class, y_angle):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.

        Returns:
            Composite objective function value.
        """

        X = torch.from_numpy(X).float()
        y_class = torch.from_numpy(y_class)
        y_angle = torch.from_numpy(y_angle).float().view(-1, 1)

        # MLP

        hidden1 = F.relu(self.layers[0](X))
        hidden2 = F.relu(self.layers[1](hidden1))
        hidden_prob = F.relu(self.layers[2](hidden2))
        hidden_angle = F.relu(self.layers[3](hidden2))
        out_prob = self.layers[4](hidden_prob)
        out_angle = self.layers[5](hidden_angle)

        # Loss

        loss_ce = F.cross_entropy(out_prob, y_class, reduction='sum')
        loss_angle = torch.sum(
            0.5 * (1 - torch.cos(0.01745 * (y_angle - out_angle))))
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_angle
        self.loss = loss
        return loss.detach().numpy()

    def predict(self, X):
        """Predict class labels and object angles for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                The predicted angles of the imput objects.
        """

        X = torch.from_numpy(X).float()

        # Pass through net
        hidden1 = F.relu(self.layers[0](X))
        hidden2 = F.relu(self.layers[1](hidden1))
        hidden_prob = F.relu(self.layers[2](hidden2))
        hidden_angle = F.relu(self.layers[3](hidden2))
        out_prob = self.layers[4](hidden_prob)
        out_angle = self.layers[5](hidden_angle)

        _, y_class = torch.max(F.softmax(out_prob, dim=1), dim=1)
        y_angle = out_angle.view(-1)

        y_class = y_class.detach().numpy()
        y_angle = y_angle.detach().numpy()

        return y_class, y_angle

    def fit(self, X, y_class, y_angle, step=1e-4):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.
        """
        from sklearn.utils import shuffle
        batch_size = 64
        num_samples = X.shape[0]
        optimizer = optim.Adam(self.parameters(), lr=step, weight_decay=1e-4)
        for epoch in range(self.epochs):
            batch_start_idx = 0
            batch_end_idx = batch_size
            loss = None
            X, y_class, y_angle = shuffle(X, y_class, y_angle)
            # Train a minibatch
            while(batch_end_idx != num_samples):
                # Slice minibatch
                batch_X = X[batch_start_idx:batch_end_idx]
                batch_class = y_class[batch_start_idx:batch_end_idx]
                batch_angle = y_angle[batch_start_idx:batch_end_idx]
                # Train on minibatch
                optimizer.zero_grad()
                loss = self.objective(batch_X, batch_class, batch_angle)
                self.loss.backward()
                optimizer.step()
                # Update batch
                batch_start_idx += batch_size
                if(batch_end_idx + batch_size > num_samples):
                    batch_end_idx = num_samples
                else:
                    batch_end_idx += batch_size
            print('Epoch ' + str(1 + epoch) + ': Loss = ' + str(loss))

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter).

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """
        param_list = list(self.parameters())
        for i, param in enumerate(param_list):
            param_list[i] = (param.detach().numpy()).T
        return param_list

    def set_params(self, params):
        """Set the model parameters.

        Arguments:
            params is a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter).

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """

        state_dict_new = self.state_dict()
        for i in range(6):
            state_dict_new['layers.' +
                           str(i) + '.weight'] = torch.from_numpy(params[2 * i].T)
            state_dict_new['layers.' +
                           str(i) + '.bias'] = torch.from_numpy(params[2 * i + 1].T)
        self.load_state_dict(state_dict_new, strict=True)


def zero_one_loss(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    counts = np.sum(np.array([y != y_pred]).astype(float))
    loss = counts / N
    return loss


def mean_abs_error(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    loss = np.sum(np.abs(y - y_pred)) / N
    return loss


def main():

    DATA_DIR = '../data'
    data = np.load(os.path.join(DATA_DIR, "mnist_rot_train.npz"))
    X_tr, y_tr, a_tr = data["X"], data["labels"], data["angles"]

    data = np.load(os.path.join(DATA_DIR, "mnist_rot_validation.npz"))
    X_val, y_val, a_val = data["X"], data["labels"], data["angles"]

    # Note: test class labels and angles are not provided
    # in the data set
    data = np.load(os.path.join(DATA_DIR, "mnist_rot_test.npz"))
    X_te, y_te, a_te = data["X"], data["labels"], data["angles"]

    nn = NN(0.5, 20)
    nn.fit(X_tr, y_tr, a_tr)

    print("Training Set Predictions")
    y_pred, a_pred = nn.predict(X_tr)
    print("Classification error: " + str(zero_one_loss(y_tr, y_pred)))
    print("Mean absolute angle error: " + str(mean_abs_error(a_tr, a_pred)))

    print("Validation Set Predictions")
    y_pred, a_pred = nn.predict(X_val)
    print("Classification error: " + str(zero_one_loss(y_val, y_pred)))
    print("Mean absolute angle error: " + str(mean_abs_error(a_val, a_pred)))


if __name__ == '__main__':
    main()
