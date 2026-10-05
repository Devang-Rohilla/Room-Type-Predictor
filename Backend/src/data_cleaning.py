import numpy as np
import pandas as pd  

# data = pd.read_csv(r"..\Datasets\RAW\AB_NYC_2019.csv")
#1. Drop Colunmns that dont help us.
def droping_useless_col(data):
    data_clean = data.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])
    return data_clean
#2. No reviews yet -> 0 reviews per month, not missing
def filling_missing(data_clean):
    data_clean['reviews_per_month'] = data_clean['reviews_per_month'].fillna(0)
    return data_clean
#3. Cap Extreme outliers instead of deleting rows.
def cap_outliers(data_clean):
    if all(col in data_clean.columns for col in ['price', 'minimum_nights']):
        price_cap  = data_clean['price'].quantile(0.99)
        nights_cap = data_clean['minimum_nights'].quantile(0.99)

        data_clean['price'] = data_clean['price'].clip(upper=price_cap)
        data_clean['minimum_nights'] = data_clean['minimum_nights'].clip(upper=nights_cap)
    return data_clean
