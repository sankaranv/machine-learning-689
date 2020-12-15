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


class BestNN(nn.Module):
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
        self.layers = nn.ModuleList([nn.Conv2d(1, 32, kernel_size=(5, 5)),
                                     nn.Conv2d(32, 64, kernel_size=(3, 3)),
                                     nn.Linear(1600, 800, bias=True),
                                     nn.Linear(800, 400, bias=True),
                                     nn.Linear(400, 200, bias=True),
                                     nn.Linear(400, 200, bias=True),
                                     nn.Linear(200, 32, bias=True),
                                     nn.Linear(200, 32, bias=True),
                                     nn.Linear(32, 10, bias=True),
                                     nn.Linear(32, 1, bias=True)
                                     ])

    def forward(self, X):

        # LeNet mod
        conv1 = F.relu(self.layers[0](X))
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = F.relu(self.layers[1](pool1))
        pool2 = F.max_pool2d(conv2, 2)
        conv_to_dense = pool2.view(pool2.size(0), -1)
        dense1 = F.relu(self.layers[2](conv_to_dense))
        dense2 = F.relu(self.layers[3](dense1))
        hidden_prob = F.relu(self.layers[4](dense2))
        hidden_prob = F.relu(self.layers[6](hidden_prob))
        hidden_angle = F.relu(self.layers[5](dense2))
        hidden_angle = F.relu(self.layers[7](hidden_angle))
        out_prob = self.layers[8](hidden_prob)
        out_angle = self.layers[9](hidden_angle)

        return out_prob, out_angle

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
        X = X.view(-1, 1, 28, 28)
        y_class = torch.from_numpy(y_class)
        y_angle = torch.from_numpy(y_angle).float().view(-1, 1)

        out_prob, out_angle = self.forward(X)
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
        X = X.view(-1, 1, 28, 28)
        # Pass through net
        out_prob, out_angle = self.forward(X)

        _, y_class = torch.max(F.softmax(out_prob, dim=1), dim=1)
        y_angle = out_angle.view(-1)

        y_class = y_class.detach().numpy()
        y_angle = y_angle.detach().numpy()

        return y_class, y_angle

    def fit(self, X, y_class, y_angle, step=1e-3):
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
        batch_size = 256
        num_samples = X.shape[0]
        optimizer = optim.Adam(
            self.parameters(), lr=step, weight_decay=1e-4)
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

    nn = BestNN(0.1, 40)
    nn.fit(X_tr, y_tr, a_tr)

    print("Training Set Predictions")
    y_pred, a_pred = nn.predict(X_tr)
    print("Classification error: " + str(zero_one_loss(y_tr, y_pred)))
    print("Mean absolute angle error: " + str(mean_abs_error(a_tr, a_pred)))

    print("Validation Set Predictions")
    y_pred, a_pred = nn.predict(X_val)
    print("Classification error: " + str(zero_one_loss(y_val, y_pred)))
    print("Mean absolute angle error: " + str(mean_abs_error(a_val, a_pred)))

    # Save the predictions
    torch.save(nn, 'model.pt')
    print('Model Saved!')
    y_pred, a_pred = nn.predict(X_te)
    print('Saving test predictions...')
    np.save('class_predictions.npy', y_pred)
    np.save('angle_predictions.npy', a_pred)


if __name__ == '__main__':
    main()
