import numpy as np
import gzip
import pickle


class SVM:
    """SVC with subgradient descent training.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """

    def __init__(self, C=1, iterations=500):
        self.w = None
        self.b = None
        self.c = C
        self.maxiter = iterations

    def fit(self, X, y):
        """Fit the model using the training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting
        any parameter values previously set by calling set_params.

        """
        # Set up parameters
        self.w = np.zeros(X.shape[1])
        self.b = np.array([0])

        fmin = np.inf
        argmin = np.zeros(X.shape[1] + 1)
        wb = np.zeros(X.shape[1] + 1)

        for k in range(self.maxiter):
            alpha = 0.002 / np.sqrt(k + 1)
            subgrad = self.subgradient(wb, X, y)
            wb -= alpha * subgrad
            f_wb = self.objective(wb, X, y)
            if(f_wb < fmin):
                fmin = f_wb
                argmin = np.copy(wb)
            print("Step " + str(k + 1) + " : fmin = " + str(fmin))

        self.w = argmin[:-1]
        self.b = argmin[-1]

    def objective(self, wb, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """
        w = wb[:-1]
        b = wb[-1]

        obj = np.maximum(0, 1 - y * (np.dot(w, X.T) + b))
        obj = self.c * np.sum(obj)
        obj += np.linalg.norm(w, 1)

        return obj

    def subgradient(self, wb, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model
        """

        X_wb = np.hstack((X, np.ones((X.shape[0], 1))))

        threshold = (y * np.dot(wb, X_wb.T))
        subgrad_1 = -y.reshape(-1, 1) * X_wb
        subgrad_1[threshold >= 1] = 0
        subgrad_1 = self.c * np.sum(subgrad_1, axis=0)

        subgrad_2 = np.sign(wb)
        subgrad_2[-1] = 0

        subgrad = subgrad_1 + subgrad_2
        return subgrad

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        discriminant = np.dot(self.w, X.T) + self.b
        y_pred = np.array([discriminant >= 0]).astype(int)
        y_pred[y_pred == 0] = -1
        return y_pred

    def get_params(self):
        """Get the model parameters.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the model parameters.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w = w
        self.b = b


def zero_one_loss(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    counts = np.sum(np.array([y != y_pred]).astype(float))
    loss = counts / N
    return loss


def main():

    np.random.seed(0)

    with gzip.open('../data/svm_data.pkl.gz', 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)

    print(train_X.shape, test_X.shape)

    clf = SVM(C=1, iterations=10000)
    clf.fit(train_X, train_y)
    pred_y_train = clf.predict(train_X)
    print("Avg training prediction error: " +
          str(zero_one_loss(train_y, pred_y_train)))
    pred_y_test = clf.predict(test_X)
    print("Avg training prediction error: " +
          str(zero_one_loss(test_y, pred_y_test)))


if __name__ == '__main__':
    main()
