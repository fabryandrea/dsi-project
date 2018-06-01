from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from math import sqrt

class SimpleExponentialSmoothing(BaseEstimator, RegressorMixin):
    """
    Simple Exponential Smoothing that smoothes over level by a factor of alpha.
    Parameters
    ----------
    alpha: float, required, default=0.1
        smoothing constant alpha, which will be applied to the last observation.
    Methods
    ----------
    fit():        fits
    predict():    returns predictions
    score():      returns RMSE on predictions and target values
    get_params(): inherited from BaseEstimator
    set_params(): inherited from BaseEstimator

    Notes
    -----
    This class is an example of coding to an interface, it implements the
    standard sklearn fit, predict, score interface.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y, x):
        """Implementation of a fitting function for a regressor.
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            The initializing input samples.
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        self.x_ = x

        # Return the regressor
        return self

    def predict(self, x, X, pred_n):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)

        forecasts = []
        initial_value = float(sum(x))/len(x)
        forecasts.append(initial_value)
        for n in range(1, len(X)):
            forecasts.append(self.alpha * X[n] + (1 - self.alpha) * forecasts[n-1])
        return forecasts

    def score(self, x, X, y, pred_n):
        """Return RMSE score on new data.
        Parameters
        ----------
        test: A series of true values.
        """
        forecasts = self.predict(x, X, pred_n)
        rms = sqrt(mean_squared_error(y, forecasts[-pred_n:]))
        return rms
