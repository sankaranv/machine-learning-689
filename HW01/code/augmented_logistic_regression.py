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
        n_features = X.shape[1]
        self.w = np.zeros(X.shape[1])
        self.c = np.zeros(X.shape[1])
        self.b = 0
        theta_init = np.concatenate((self.w,self.c))
        theta_init = np.concatenate((theta_init,np.array([self.b])))

        argmin, f, d = fmin_l_bfgs_b(self.objective, x0 = theta_init, fprime = self.objective_grad, args = (X, y), disp=10)

        self.w = argmin[:n_features]
        self.c = argmin[n_features:-1]
        self.b = argmin[-1]

    def predict(self, X):
        """Predict class labels for samples in X based on current parameters.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        prob_pos = 1./(1 + np.exp(-np.dot(self.w,(X-self.c).T) - self.b))
        prob_neg = 1 - prob_pos
        y = (prob_pos >= prob_neg).astype(int)
        y[y == 0] = -1
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
        n_features = X.shape[1]
        w = wcb[:n_features]
        c = wcb[n_features:-1]
        b = wcb[-1]

        objective = np.sum(np.log(1 + np.exp(-y*(np.dot(w, (X - c).T) + b))), axis = 0)
        objective += self.reg_param*(np.linalg.norm(w,2)**2 + np.linalg.norm(c,2)**2 + b**2)
        return objective

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
        n_samples = X.shape[0]
        n_features = X.shape[1]
        w = wcb[:n_features]
        c = wcb[n_features:-1]
        b = wcb[-1]

        # grad_w = np.zeros(w.shape)
        # grad_c = np.zeros(c.shape)
        # grad_b = 0
        # for n in range(n_samples):
        #     coeff = -y[n] * np.exp(-y[n] * (np.dot(w,(X[n]-c).T) + b))/(1 + np.exp(-y[n] * (np.dot(w,(X[n]-c).T) + b)))
        #     grad_w += coeff * (X[n] - c).T
        #     grad_c += coeff * w.T
        #     grad_b += coeff

        coeff = -y * np.exp(-y * (np.dot(w,(X-c).T) + b))/(1 + np.exp(-y * (np.dot(w,(X-c).T) + b)))
        grad_w = np.sum(coeff.reshape(-1,1) * (X-c), axis = 0)
        grad_c = np.sum(coeff) * -w
        grad_b = np.sum(coeff)

        grad_w += 2*self.reg_param*w.T
        grad_c += 2*self.reg_param*c.T
        grad_b += 2*self.reg_param*b

        objective_grad = np.concatenate((grad_w,grad_c))
        objective_grad = np.concatenate((objective_grad,np.array([grad_b])))
        return objective_grad

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

def zero_one_loss(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    counts = np.sum(np.array([y != y_pred]).astype(float))
    loss = counts/N
    return loss

def main():
    np.random.seed(0)

    train_X = np.load('data/q2_train_X.npy')
    train_y = np.load('data/q2_train_y.npy')
    test_X = np.load('data/q2_test_X.npy')
    test_y = np.load('data/q2_test_y.npy')

    lr = AugmentedLogisticRegression(lmbda = 1e-6)
    lr.fit(train_X, train_y)

    pred_y = lr.predict(test_X)
    print("Augmented Logistic Regression")
    print("Avg prediction error: " + str(zero_one_loss(test_y,pred_y)))

    lr_reference = linear_model.LogisticRegression(C=1e6, solver='lbfgs')
    lr_reference.fit(train_X, train_y)
    pred_y_reference = lr_reference.predict(test_X)
    print(" \nsklearn Logistic Regression")
    print("Avg prediction error: " + str(zero_one_loss(test_y,pred_y_reference)))

if __name__ == '__main__':
    main()
