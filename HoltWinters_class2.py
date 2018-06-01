from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import numpy as np


class HoltWinters(BaseEstimator, RegressorMixin):
    """
    Holt-Winters Triple Exponential Smoothing forecasting method. This one has
    additive trend, additive seasonality, and no trend damping.
    Parameters
    ----------
    alpha: float, required, default=0.1
        smoothing constant alpha, which will be applied to the last observation.
    beta=0.053
    gamma=0.1
    slen=10
    n_preds=100
    Methods
    ----------
    fit():        fits
    predict():    returns predictions
    score():      returns RMSE on predictions and test dataset
    get_params(): inherited from BaseEstimator
    set_params(): inherited from BaseEstimator

    Notes
    -----
    This class is an example of coding to an interface, it implements the
    standard sklearn fit, predict, score interface.
    """

    def __init__(self, alpha=0.1, beta=0.053, gamma=0.1, slen=10, n_preds=100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.slen = slen
        self.n_preds = n_preds

    def fit(self, X, y=None):
        """Implementation of fitting function.
        Parameters
        ----------
        X : array-like, shape = [n_samples, 1_feature]
            The training input samples.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)

        self.X_ = X

        # Return the regressor
        return self

    def initial_trend(self, X, slen):
        my_sum = 0.0
        for i in range(self.slen):
            my_sum += float(X[i+self.slen] - X[i]) / self.slen
        return my_sum / self.slen

    def initial_seasonal_components(self, X, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(X)/self.slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(X[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # compute initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += X[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals

    def predict(self, X):
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

        result = []
        for i in range(len(X)+self.n_preds):
            if i == 0: # initial values
                smooth = X[0]
                trend = self.initial_trend(X, self.slen)
                seasonals = self.initial_seasonal_components(X, self.slen)
                result.append(X[0])
                continue
            if i >= len(X): # we are forecasting
                m = i - len(X) + 1
                result.append((smooth + m*trend) + seasonals[i%self.slen])
            else:
                val = X[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                result.append(smooth+trend+seasonals[i%self.slen])
        return result

    def scorer(self, X, y_true):
        """Return RMSE score on new data.
        Parameters
        ----------
        y: A series of true values.
        """
        y_pred = self.predict(X)
        rms= sqrt(np.average((y_true - y_pred[-self.n_preds:]) ** 2, axis=0))
        #rms = sqrt(mean_squared_error(y, ))
        return rms
