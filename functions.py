import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

def load_data(filename):
    data_df = pd.read_excel(filename, index_col='EntDate')
    return data_df.head()

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
