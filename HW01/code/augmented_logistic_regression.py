import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

class AugmentedLogisticRegression:
    """Logistic regression with optimized centering.

    Arguments:
        lambda(float): regularization parameter lambda (default: 0)
    """

    def __init__(self, lmbda=0):
        self.reg_param = lmbda  # regularization parameter (lambda)
        self.w = None
        self.c = None
        self.b = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        
        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting 
        any parameter values previously set by calling set_params.
        """
        self.w = np.zeros(X.shape[1])
        self.c = np.zeros(X.shape[1])
        self.b = 0
        theta_init = np.concatenate((self.w,self.c))
        theta_init = np.concatenate((theta_init,np.array([self.b])))

        argmin, f, d = fmin_l_bfgs_b(self.objective, x0 = wb_init, fprime = self.objective_grad, args = (X, y), disp=10)
        print(argmin)

    def predict(self, X):
        """Predict class labels for samples in X based on current parameters.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        prob_positive = 1./(1 + np.exp(-np.dot(self.w,(X-self.c).T) - self.b))
        y = (prob_positive >= 0.5).astype(int)
        return y

    def objective(self, wcb, X, y):
        """Compute the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective (float):
                the objective function evaluated at [w, b, c] and the data X, y.
        """
        
        pass

    def objective_grad(self, wcb, X, y):
        """Compute the gradient of the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective_grad (ndarray, shape = (2*n_features + 1,)):
                gradient of the objective function with respect to [w,c,b].
        """
        
        pass

    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,c,b) where w is the learned coefficients (ndarray, shape = (n_features,)),
            c  is the learned centering parameters (ndarray, shape = (n_features,)),
            and b is the learned bias (float).
        """
        params = (self.w,self.c,self.b)
        return params

    def set_params(self, w, c, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficients
            c (ndarray, shape = (n_features,)): centering parameters
            b (float): bias 
        """
        self.w = w
        self.c = c
        self.b = b


def main():
    np.random.seed(0)

    train_X = np.load('data/q2_train_X.npy')
    train_y = np.load('data/q2_train_y.npy')
    test_X = np.load('data/q2_test_X.npy')
    test_y = np.load('data/q2_test_y.npy')

    
    lr = AugmentedLogisticRegression(lmbda = 1e-6)
    lr.fit(train_X, train_y)

if __name__ == '__main__':
    main()
