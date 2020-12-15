import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.nn as nn
import os
import copy
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)


class mixture_model(nn.Module):
    """A Laplace mixture model trained using marginal likelihood maximization

    Arguments:
        K: number of mixture components

    """

    def __init__(self, K=5):
        nn.Module.__init__(self)
        self.mu = None
        self.b = None
        self.pi = None
        self.K = K
        self.neg_likelihood = None

        self.mu_tensor = None
        self.b_tensor = None
        self.pi_tensor = None

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values:

            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))

        """

        b_orig = np.exp(self.b)
        pi_orig = np.exp(self.pi) / np.sum(np.exp(self.pi))
        params = [self.mu, b_orig, pi_orig]
        return params

    def set_params(self, mu, b, pi):
        """Set the model parameters.

        Arguments:
            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))
        """

        self.mu = mu
        self.b = np.log(b)
        self.pi = np.log(pi)

        self.mu_tensor = torch.nn.Parameter(torch.from_numpy(self.mu))
        self.mu_tensor.requires_grad = True
        self.b_tensor = torch.nn.Parameter(torch.from_numpy(self.b))
        self.b_tensor.requires_grad = True
        self.pi_tensor = torch.nn.Parameter(torch.from_numpy(self.pi))
        self.pi_tensor.requires_grad = True

    def marginal_likelihood(self, X):
        """log marginal likelihood function.
           Computed using the current values of the parameters
           as set via fit or set_params.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            Marginal likelihood of observed data
        """

        N = X.shape[0]
        D = X.shape[1]

        likelihood = - N * torch.logsumexp(self.pi_tensor, dim=0)

        for n in range(N):
            x_n = X[n]
            obs_idx = ~np.isnan(x_n)
            x_n = x_n[obs_idx]
            mu = self.mu_tensor[obs_idx]
            b = self.b_tensor[obs_idx]
            D_o = x_n.shape[0]
            x_n = torch.from_numpy(x_n)
            abs_term = -(torch.abs(x_n.view(-1, 1) - mu) /
                         torch.exp(b)) - b
            exp_term = torch.sum(abs_term, axis=0) + self.pi_tensor
            logsumexp_term = torch.logsumexp(exp_term, axis=0)
            likelihood += logsumexp_term - D_o * np.log(2)

        self.neg_likelihood = -likelihood
        return likelihood.detach().numpy()

    def predict_proba(self, X):
        """Predict the probability over clusters P(Z=z|X=x) for each example in X.
           Use the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N,D)):
                Input matrix where each row is a feature vector.

        Returns:
            PZ (numpy ndarray, shape = (N,K)):
                Probability distribution over classes for each
                data case in the data set.
        """

        N = X.shape[0]

        D = X.shape[1]
        K = self.K

        PZ = torch.zeros((N, K))

        for n in range(N):
            x_n = X[n]
            obs_idx = ~np.isnan(x_n)
            x_n = x_n[obs_idx]
            mu = self.mu_tensor[obs_idx]
            b = self.b_tensor[obs_idx]
            D_o = x_n.shape[0]
            x_n = torch.from_numpy(x_n)
            abs_term = -(torch.abs(x_n.view(-1, 1) - mu) /
                         torch.exp(b)) - b
            exp_term = torch.sum(abs_term, axis=0) + self.pi_tensor
            normalization_term = torch.logsumexp(exp_term, axis=0)
            PZ[n] = exp_term - normalization_term

        PZ = torch.exp(PZ)
        return PZ.detach().numpy()

    def impute(self, X):
        """Mean imputation of missing values in the input data matrix X.
           Ipmute based on the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            XI (numpy ndarray, shape = (N, D)):
                The input data matrix where the missing values on
                each row (indicated by np.nans) have been imputed
                using their conditional means given the observed
                values on each row.
        """
        N = X.shape[0]
        D = X.shape[1]
        XI = np.zeros((N, D))

        for n in range(N):
            x_n = X[n]
            prob_z_given_obs = self.predict_proba(
                x_n.reshape(1, -1)).reshape(-1)
            obs_idx = ~np.isnan(x_n)
            miss_idx = np.isnan(x_n)
            impute = np.sum(prob_z_given_obs * self.mu, axis=1)
            impute[obs_idx] = 0
            x_n[miss_idx] = 0
            XI[n] = x_n + impute

        return XI

    def fit(self, X, mu_init=None, b_init=None, pi_init=None, step=0.1, epochs=100):
        """Train the model according to the given training data
           by directly maximizing the marginal likelihood of
           the observed data. If initial parameters are specified, use those
           to initialize the model. Otherwise, use a random initialization.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
            mu_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density mean paramaeters for each mixture component
                to use for initialization
            b_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density scale parameters for each mixture component
                to use for initialization
            pi_init (None or numpy ndarray, shape = (K,)):
                Mixture proportions to use for initialization
            step (float):
                Initial step size to use during training
            epochs (int): number of epochs for training
        """

        N = X.shape[0]
        D = X.shape[1]
        K = self.K

        if mu_init is None:
            self.mu = np.random.normal(size=(D, K))
        else:
            self.mu = np.copy(mu_init)
        if b_init is None:
            self.b = np.random.normal(size=(D, K))
        else:
            self.b = np.copy(b_init)
        if pi_init is None:
            self.pi = np.random.normal(size=(K,))
        else:
            self.pi = np.copy(pi_init)

        self.mu_tensor = torch.nn.Parameter(torch.from_numpy(self.mu))
        self.mu_tensor.requires_grad = True
        self.b_tensor = torch.nn.Parameter(torch.from_numpy(self.b))
        self.b_tensor.requires_grad = True
        self.pi_tensor = torch.nn.Parameter(torch.from_numpy(self.pi))
        self.pi_tensor.requires_grad = True

        optimizer = optim.Adam(self.parameters(), lr=step)
        epochs = 100
        batch_size = 50
        loss_prev = 0
        for epoch in range(epochs):
            batch_start_idx = 0
            batch_end_idx = batch_size
            loss = None
            while(batch_end_idx != N):
                # Slice minibatch
                batch_X = X[batch_start_idx:batch_end_idx]
                # Train on minibatch
                optimizer.zero_grad()
                loss = -self.marginal_likelihood(X)
                self.neg_likelihood.backward()
                optimizer.step()
                batch_start_idx += batch_size
                if(batch_end_idx + batch_size > N):
                    batch_end_idx = N
                else:
                    batch_end_idx += batch_size
            print("Epoch " + str(epoch + 1) + " Loss = " + str(loss))
            if(np.abs(loss - loss_prev) < 0.01):
                break
            loss_prev = loss

        self.mu = self.mu_tensor.detach().numpy()
        self.b = self.b_tensor.detach().numpy()
        self.pi = self.pi_tensor.detach().numpy()


def main():

    data = np.load("../data/data.npz")
    xtr1 = data["xtr1"]
    xtr2 = data["xtr2"]
    xte1 = data["xte1"]
    xte2 = data["xte2"]

    # Experiments
    train_likelihoods = []
    test_likelihoods = []
    for K in range(1, 21):
        print("------")
        print("K = " + str(K))
        print("------")
        best_model = None
        best_likelihood = -np.inf
        for i in range(5):
            print("Training Data: run " + str(i + 1) + " of 5")
            mm = mixture_model(K=K)
            mm.fit(xtr1)
            likelihood = mm.marginal_likelihood(xtr1)
            print("Marginal likelihood on training data: " + str(likelihood))
            if(likelihood > best_likelihood):
                best_likelihood = likelihood
                best_model = mm
            print("-------------------------------------------")
        print("Best Likelihood on training data: " + str(best_likelihood))
        test_likelihood = best_model.marginal_likelihood(xte1)
        print("Likelihood on test data: " + str(test_likelihood))
        train_likelihoods.append(best_likelihood)
        test_likelihoods.append(test_likelihood)
    print("Train Likelihoods")
    print(train_likelihoods)
    print("Test Likelihoods")
    print(test_likelihoods)
    axis = np.arange(1, 21)
    plt.plot(axis, train_likelihoods)
    plt.title("Log Marginal Likelihood vs K on training data")
    plt.xlabel('Number of mixture components (K)')
    plt.ylabel('Log Marginal Likelihood')
    plt.savefig('hw4_q2_train.png')
    plt.clf()
    plt.plot(axis, test_likelihoods, color='orange')
    plt.title("Log Marginal Likelihood vs K on test data")
    plt.xlabel('Number of mixture components (K)')
    plt.ylabel('Log Marginal Likelihood')
    plt.savefig('hw4_q2_test.png')


if __name__ == '__main__':
    main()
