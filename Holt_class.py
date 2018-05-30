
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


class HoltSmoothing(object):
    """
    Holt Smoothing that smoothes over level by a factor of alpha.
    Parameters
    ----------
    alpha: float, required, default=0.1
        smoothing constant alpha,
    Methods
    ----------
    initialize():
    fit():
    predict():
    score():

    Notes
    -----
    This class is an example of coding to an interface, it implements the
    standard sklearn fit, predict, score interface.
    """

    def __init__(self):
        pass

    def initialize(self, pretrain, label):
        """
        Takes in a dataset and initializes first data point in simple
        exponential smoothing.
        Parameters
        ---------
        pretrain : dataset
        label: object, required
            Label of the column containing 
        """
        X=pretrain.index
        y=pretrain[label]
        lin_model = LinearRegression()
        lin_model.fit(X,y)

        level = lin_model.intercept_
        trend = lin_model.coef_
        t = 1
        a_hat_zero = float(t* trend + level)
        b_hat_zero = float(trend)
        first_forecast = a_hat_zero + b_hat_zero
        return first_forecast

    def fit_predict(self, pretrain, train, alpha, pred_n):
        """
        Fit linear model.
        Parameters
        ----------
        train :
        alpha: float, required, default=0.1
            smoothing constant alpha,
        pred_n:
        Number of predictions
        Returns
        -------
        result :
        """
        result = [series[0]] # first value is same as series
        lin_model = LinearRegression()
        lin_model.fit(X, y)

        level = lin_model.intercept_
        trend = lin_model.coef_
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # we are forecasting
              value = result[-1]
            else:
              value = series[n]
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)
        return result
### old code
        result = []
        result.append(self.initialize(pretrain))
        for n in range(1, pred_n):
            result.append(alpha * train[n] + (1 - alpha) * result[n-1])
        return result

    def predict(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary
        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : returns an instance of self.
        """
        pass


    def score(self, pretrain, train, test, alpha, pred_n):
        """Return RMSE score on new data.
        Parameters
        ----------
        test: A series of true values.
        """
        result = self.fit_predict(pretrain, train, alpha, pred_n)
        rms = sqrt(mean_squared_error(test, result))
        return rms
