import random
import numpy as np
from math import sqrt

def prep_dataframe_for_warping(dataframe):
    dataframe = dataframe.T
    return np.asarray(dataframe)

def assign_products(dataframe, dictionary, key_n):
    for key, value in dictionary.items():
        my_products = [dataframe.index[x] for x in list(dictionary[key_n])]
    return my_products

def make_product_dataframe(dataframe, product_list, label):
    products_df = dataframe.T
    new_df = products_df[products_df.index.isin(product_list)]
    new_df['label'] = label
    return new_df

def plot():
    pass
