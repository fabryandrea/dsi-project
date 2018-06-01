from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from math import sqrt

class HoltWinters(BaseEstimator, RegressorMixin):
    """
    Holt-Winters Triple Exponential Smoothing forecasting method.
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

    def __init__(self, alpha=0.1, beta=0.053, gamma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y):
        """Implementation of a fitting function for a regressor.
        Parameters
        ----------
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

        # Return the regressor
        return self

    def initial_trend(self, X, slen):
        my_sum = 0.0
        for i in range(slen):
            my_sum += float(X[i+slen] - X[i]) / slen
        return my_sum / slen

    def initial_seasonal_components(self, X, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(X)/slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(X[slen*j:slen*j+slen])/float(slen))
        # compute initial values
        for i in range(slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += X[slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals

    def predict(self, X, slen, n_preds):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        x:
        slen:
            The number of seasons
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
        for i in range(len(X)+n_preds):
            if i == 0: # initial values
                smooth = X[0]
                trend = self.initial_trend(X, slen)
                seasonals = self.initial_seasonal_components(X, slen)
                result.append(X[0])
                continue
            if i >= len(X): # we are forecasting
                m = i - len(X) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = X[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        return result

    def score(self, X, test, slen, n_preds):
        """Return RMSE score on new data.
        Parameters
        ----------
        y: A series of true values.
        """
        forecasts = self.predict(X, slen, n_preds)
        rms = sqrt(mean_squared_error(test, forecasts[-n_preds:]))
        return rms
