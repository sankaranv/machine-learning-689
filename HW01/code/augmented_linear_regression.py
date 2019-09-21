import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

class AugmentedLinearRegression:
    """Augmented linear regression.

    Arguments:
        delta (float): the trade-off parameter of the loss
    """
    def __init__(self, delta):
        self.delta = delta
        self.b = None
        self.w = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Real-valued output vector for training.

        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting
        any parameter values previously set by calling set_params.

        """

        # Set up parameters
        self.w = np.zeros(X.shape[1])
        self.b = np.array([0])
        wb_init = np.concatenate((self.w,self.b))

        # Add extra feature for bias
        X = np.hstack((X,np.ones((X.shape[0],1))))

        argmin, f, d = fmin_l_bfgs_b(self.objective, x0 = wb_init, fprime = self.objective_grad, args = (X, y), disp=10)

        print (argmin)

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        y_pred = np.dot(X, self.w) + self.b
        return y_pred


    def objective(self, wb, X, y):
        """Compute the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective (float):
                the objective function evaluated on wb=[w,b] and the data X,y..
        """

        objective1 = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            objective1[n] = self.delta**2 * (np.sqrt(1 + ((y[n] - np.dot(wb,X[n].T))**2 / self.delta**2)) - 1)
        objective1 = np.sum(objective1)

        # y_hat = np.dot(wb,X.T)
        # objective = self.delta**2 * (np.sqrt(1 + ((y - y_hat)**2 / self.delta**2)) - 1)
        # objective = np.sum(objective)

        return objective1


    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to wb=[w,b].
        """
        y_hat = np.dot(wb,X.T)
        objective_coeff = ((y - y_hat)/np.sqrt(1 + ((y - y_hat)**2 / self.delta**2)))
        objective_grad = np.dot(objective_coeff, X)
        return objective_grad

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray, shape = (n_features,))
            and b is the learned bias (float).
        """
        params = (self.w,self.b)
        return params

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b


def main():

    np.random.seed(0)
    train_X = np.load('data/q3_train_X.npy')
    train_y = np.load('data/q3_train_y.npy')

    lr = AugmentedLinearRegression(delta=1)
    lr.fit(train_X, train_y)

if __name__ == '__main__':
    main()
