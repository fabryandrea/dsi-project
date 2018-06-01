import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data(filename):
    return pd.read_excel(filename, index_col='EntDate')

def train_test_split(data, split_n):
    ''' Takes in a time series dataset and a fraction (split_n) between 0 and 1
    to split
    '''
    X = data.values
    train_size = int(len(X) * split_n)
    train, test = X[0:train_size], X[train_size:len(X)]
    sample_size = len(X)
    print('Observations: %d' % (len(X)))
    print('Training Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))
    train_df = data[0:train_size]
    test_df = data[train_size:sample_size]
    return train_df, test_df

def ts_train_test_split(data, split_t):
    ''' Takes in a time series dataset and a fraction (split_n) between 0 and 1
    to split
    '''
    X = data.values
    train_size = int(len(X) - split_t)
    train, test = X[0:train_size], X[train_size:len(X)]
    sample_size = len(X)
    print('Observations: %d' % (len(X)))
    print('Training Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))
    train_df = data[0:train_size]
    test_df = data[train_size:sample_size]
    return train_df, test_df

def plot_train_test(train, test, item_label):
    plt.figure(figsize=(8,4))
    plt.plot(train[item_label], label='Train')
    plt.plot(test[item_label], label='Test')
    plt.legend(loc='best')
    plt.show()

def plot_time_series(train, test, item_label, yhat, yhat_label, fore_label):
    plt.figure(figsize=(8,4))
    plt.plot(train[item_label], label='Train')
    plt.plot(test[item_label], label='Test')
    plt.plot(yhat[yhat_label], label=fore_label)
    plt.legend(loc='best')
    plt.show()

def RMSE(test, item_label, yhat, yhat_label):
    return sqrt(mean_squared_error(test[item_label], yhat[yhat_label]))

def MAE(test, item_label, yhat, yhat_label):
    return mean_absolute_error(test[item_label], yhat[yhat_label])

def make_copy_df(train, item_label):
    return pd.DataFrame(train[item_label].copy())

def rename_columns(train, item_label):
    train['ds'] = train.index
    train['y'] = train[item_label]
    return train.drop([item_label], axis = 1, inplace = True)

def decompose_timeseries(train, model_kind):
    result_a = seasonal_decompose(train, model=model_kind)
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(16,10))
    result_a.trend.plot(ax=ax1, label='Trend')
    result_a.seasonal.plot(ax=ax2, label='Seasonal')
    result_a.resid.plot(ax=ax3, label='Residuals')
    plt.show()

def naive(train, split_t):
    pass

def cumulative(train):
    return float(sum(train))/len(train)

def moving_average(train, m):
    return cumulative(train[-m:])
