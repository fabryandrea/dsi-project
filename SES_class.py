
from sklearn.metrics import mean_squared_error
from math import sqrt

class SimpleExponentialSmoothing(object):
    """
    Simple Exponential Smoothing that smoothes over level by a factor of alpha.
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

    def initialize(self, pretrain):
        """
        Takes in a dataset and initializes first data point in simple
        exponential smoothing.
        """
        return float(sum(pretrain))/len(pretrain)

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
        forecasts = []
        forecasts.append(self.initialize(pretrain))
        for i, item in enumerate(train):
        for n in range(1, len(train)):
            forecasts.append(alpha * train[n] + (1 - alpha) * result[n-1])
        return forecasts[-pred_n:]

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
        forecasts = self.fit_predict(pretrain, train, alpha, pred_n)
        rms = sqrt(mean_squared_error(test, forecasts[-pred_n:]))
        return rms
