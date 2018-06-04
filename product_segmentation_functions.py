from datetime import datetime
import pandas as pd
import numpy as np

def identify_non_active(dataframe, product_list, year, month, day):
    # returns list of products that have not moved after a specified date
    last_tp = (dataframe[datetime(year, month, day):])

    non_active = []
    for product in product_list:
        if last_tp[product].sum() == 0:
            non_active.append(product)
    return non_active

def identify_new_product(dataframe, product_list, year, month, day):
    # returns list of products that did not exist before a specified date
    previous_tp = (dataframe[:datetime(year, month, day)])
    last_tp = (dataframe[datetime(year, month, day):])

    new_products = []
    for product in product_list:
        if previous_tp[product].sum() == 0 and last_tp[product].sum() !=0:
            new_products.append(product)
    return new_products

def identify_intermittent_product(dataframe, product_list, non_active, year, month, day, n):
    # returns list of products that had zero demand in n time period after a specified date
    last_tp = (dataframe[datetime(year, month, day):])

    products = [value for value in product_list if value not in non_active]
    intermittent = products.copy()

    for product in products:
        if last_tp[product].rolling(n).sum().dropna().nonzero():
            intermittent.remove(product)
    return intermittent

def identify_minute_demand(dataframe, product_list, n):
    # returns list of products whose demand never exceeds n units
    minute_demand = []
    for product in product_list:
        if dataframe[product].max() <= n:
            minute_demand.append(product)
    return minute_demand

def identify_repackage_product(dataframe, product_list, n):
    # returns list of products whose demand never exceeds n units
    repackage_product = []
    for product in product_list:
        if dataframe[product].max() > n:
            repackage_product.append(product)
    return repackage_product

def make_remainder_dataframe(dataframe, product_SKUs, non_active, new_products, minute_demand, repackage_product):
    result = set(non_active).union(new_products).union(minute_demand).union(repackage_product)
    col_names = [value for value in product_SKUs if value not in result]
    return dataframe[col_names]

def prep_dataframe_for_warping(dataframe):
    dataframe = dataframe.T
    return np.asarray(dataframe)

def assign_products(dataframe, dictionary, key_n):
    for key, value in dictionary.items():
        my_products = [dataframe.index[x] for x in list(dictionary[key_n])]
    return my_products
