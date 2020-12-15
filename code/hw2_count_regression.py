import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class CountRegression:
    """Count regression.

    Arguments:
       lam (float): regaularization parameter lambda
    """

    def __init__(self, lam):
        self.lam = lam
        self.w = None
        self.b = None

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
        self.w = np.zeros(X.shape[1])
        self.b = 0
        theta_init = np.concatenate((self.w, np.array([self.b])))

        argmin, f, d = fmin_l_bfgs_b(
            self.objective, x0=theta_init, fprime=self.objective_grad, args=(X, y), disp=10)

        self.w = argmin[:-1]
        self.b = argmin[-1]

    def predict(self, X):
        """Predict using the model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        pass

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
        X_wb = np.hstack((X, np.ones((X.shape[0], 1))))
        nll = np.sum(y * np.dot(wb, X_wb.T) + (y + 1) *
                     np.log(1 + np.exp(-np.dot(wb, X_wb.T))))
        objective = nll + (self.lam * np.linalg.norm(wb, 2)**2)
        return objective

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

        X_wb = np.hstack((X, np.ones((X.shape[0], 1))))
        exp_term = np.exp(-np.dot(wb, X_wb.T))
        coeff = y - (y + 1) * (exp_term / (1 + exp_term))
        objective_grad = coeff.reshape(-1, 1) * X_wb
        objective_grad = np.sum(objective_grad, axis=0) + 2 * self.lam * wb
        return objective_grad

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray, shape = (n_features,))
            and b is the learned bias (float).
        """
        return self.w, self.b

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

    def avg_predictive_log_likelihood(self, X, y):

        wb = np.concatenate((self.w, np.array([self.b])))
        N = X.shape[0]
        X_wb = np.hstack((X, np.ones((X.shape[0], 1))))
        nll = np.sum(y * np.dot(wb, X_wb.T) + (y + 1) *
                     np.log(1 + np.exp(-np.dot(wb, X_wb.T))))
        pred = (-1. / N) * nll
        return pred


def main():

    data = np.load("data/count_data.npz")
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    # Define and fit model
    cr = CountRegression(1e-4)
    cr.fit(X_train, Y_train)
    pred_y = cr.predict(X_test)
    print("Count Regression")
    print("Avg predictive log likelihood: " +
          str(cr.avg_predictive_log_likelihood(X_test, Y_test)))
    print(cr.w, cr.b)


if __name__ == '__main__':
    main()
