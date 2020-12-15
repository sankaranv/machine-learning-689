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

        argmin, f, d = fmin_l_bfgs_b(self.objective, x0 = wb_init, fprime = self.objective_grad, args = (X, y), disp=10)

        self.w = argmin[:-1]
        self.b = argmin[-1]

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

        X_tilde = np.hstack((X,np.ones((X.shape[0],1))))
        y_hat = np.dot(wb,X_tilde.T)

        objective = np.sum(self.delta**2 * (np.sqrt(1 + ((y - y_hat)**2 / self.delta**2)) - 1))

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
        X_tilde = np.hstack((X,np.ones((X.shape[0],1))))
        error = y - np.dot(wb,X_tilde.T)
        objective_grad = np.sum((error / np.sqrt(1 + (error/self.delta)**2)) * (-X_tilde.T), axis=1)
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

def mean_squared_error(y, y_pred):
    assert(y.size == y_pred.size)
    N = y_pred.size
    error = np.sum((y - y_pred)**2)/N
    return error

def main():

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    def plot_robust_loss():
        deltas = [0.1,1,10]
        error = np.linspace(0,10,num=50)

        # Plot squared Error
        sq_loss = error**2
        plt.plot(error, sq_loss, label='squared loss', color='green')

        # Plot robust loss
        colors = np.array([0.3,0.6,0.9])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap.set_array([])
        for color, delta in zip(colors, deltas):
            robust_loss = delta**2 * (np.sqrt(1 + error**2/delta**2) -1)
            plt.plot(error, robust_loss, label='robust loss, delta = ' + str(delta), color=cmap.to_rgba(color))

        plt.title("Loss magnitude vs. error")
        plt.legend()
        plt.xlabel("Prediction error")
        plt.ylabel("Loss value")
        plt.xticks(np.linspace(0,10,11))
        plt.yticks(np.linspace(0,100,11))
        plt.savefig("plots/loss_vs_error.png")
        plt.clf()

    def plot_regression_lines(lr1, lr2, X, y):
        plt.scatter(X, y, c='orange', alpha=0.3)
        y_pred_1 = lr1.predict(X)
        plt.plot(X,y_pred_1, label="with robust loss", color='blue')
        y_pred_2 = lr2.predict(X)
        plt.plot(X,y_pred_2, label="sklearn", color='green')
        plt.title("Regression Lines")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("y", rotation=0)
        plt.savefig("plots/regression_lines.png")
        plt.clf()

    np.random.seed(0)
    train_X = np.load('data/q3_train_X.npy')
    train_y = np.load('data/q3_train_y.npy')

    # Evaluate augmented linear regression function

    lr = AugmentedLinearRegression(delta=1)
    lr.fit(train_X, train_y)
    pred_y = lr.predict(train_X)
    print("Augmented Linear Regression")
    print("Avg prediction error: " + str(mean_squared_error(train_y,pred_y)))

    # Compare with sklearn's linear regression function

    lr_reference = linear_model.LinearRegression()
    lr_reference.fit(train_X, train_y)
    pred_y_reference = lr_reference.predict(train_X)
    print(" \nsklearn Linear Regression")
    print("Avg prediction error: " + str(mean_squared_error(train_y,pred_y_reference)))

    # Plots
    #plot_robust_loss()
    #plot_regression_lines(lr, lr_reference, train_X, train_y)

if __name__ == '__main__':
    main()
