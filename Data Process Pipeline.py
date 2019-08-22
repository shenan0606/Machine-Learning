# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:03:00 2019
This file does the following things:
1. pre-clean the dataset
2. automatically decide the type of the column based on column names, column data object (str, int, datetime), 
column data values (continuous or categorical) and missing values
3. for categorical data, automatically check each value's frequency and make a dictionary
4. Automatically makes feature engineering using continuous and categorical variables. Categorical use one-hot-encoding
based on their frequency. (By default, frequency < 5% are neglected in the one-hot-encoding)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Clean the dataframe
def df_pre_clean(df):   
    def cell_clean(x):
        if x:          # if not none
            if isinstance(x, int):
                return(int(x))
            elif isinstance(x, float):
                return(float(x))
            elif isinstance(x, str):
                x = x.strip().upper()
                if x == '':
                    return(None)
                else:
                    return(x)
            elif isinstance(x, datetime.date):
                return(x)
            else:
                print('Can not Indentify', x)
    col_name_list = list(df.columns)                 
    for i in col_name_list:
        df[i] = df[i].apply(cell_clean)
    return df

def df_col_type_summary(df):
    col_name_list = list(df.columns)
    '''Get Different Account ID Info'''
    account_nbr_name1 = ['no', 'id', 'nbr']
    account_nbr_name2 = ['account', 'member', 'mbr', 'cust']
    # if the col names contains both the substring in account_nbr_name1 and account_nbr_name2, then it is the account_id_nbr
    account_nbr_col = [col_name for col_name in col_name_list 
                           if (any(sub_str1 in col_name for sub_str1 in account_nbr_name1)) &
                              (any(sub_str2 in col_name for sub_str2 in account_nbr_name2))]
    
    ''' Get All Date/Time Related Variable and Only Keeps Date '''
    datetime_col = [col_name for col_name in col_name_list if df[col_name].dtypes == 'datetime64[ns]']
    for col in datetime_col:
        df[col] = df[col].dt.date
    
    ''' Get Account Info: Phone, Address, Name'''
    account_info_name = ['phone', 'addr', 'name']
    exception_name = ['date', 'yrs', 'years'] # to prevent year at address variables
    account_info_col = [col_name for col_name in col_name_list 
                           if (any(sub_str1 in col_name for sub_str1 in account_info_name)) & 
                              (not any(sub_str2 in col_name for sub_str2 in exception_name))]
    for col in account_info_col:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
    
    '''Get Score Data'''
    def score_to_int(x):
        try:
            return(int(x))
        except:
            return(np.nan)
            
    score_col = [col_name for col_name in col_name_list if 'score' in col_name]
    
    for col in score_col:
        df[col] = df[col].apply(score_to_int)
        
    '''Get the Rest Column'''
    var_list = [col_name for col_name in col_name_list
                if col_name not in account_nbr_col + datetime_col + account_info_col + score_col]
       
    def is_float(val):
            try:
                float(val)
            except ValueError:
                return False
            else:
                return True
    is_numeric = lambda x: map(is_float, x) 
            
    '''For the Rest Column, determine catogory variable, continuous var and other var that are hard to determine'''
    def var_to_float(x):
        try:
            return(float(x))
        except:
            return(np.nan)
    all_missing_col = []
    most_missing_col = []
    category_col = []
    continuous_col = []
    other_col = []
    nrow = df.shape[0]
    category_threshold = 20
    for col_name in var_list:
        col_value = df[col_name][df[col_name].notnull()]
        if len(col_value) == 0:
            all_missing_col.append(col_name) 
        elif (len(col_value) / nrow < 0.05):
            most_missing_col.append(col_name)
        else:
            col_unique = col_value.unique()
            if len(col_unique) <= category_threshold:
                category_col.append(col_name)
            elif all(is_numeric(col_unique)):
                df[col_name] = df[col_name].apply(var_to_float)
                continuous_col.append(col_name)
                
            else: other_col.append(col_name)    
    
    '''Create a dictionary for different column type'''
    col_summary = {'Account ID':account_nbr_col, 'Account Info':account_info_col, 'Date Info':datetime_col, 'Score Var':score_col,
                   'Category Var':category_col, 'Continuous Var':continuous_col, 'Other Var':other_col, 
                   'Most Missing':most_missing_col, 'All Missing': all_missing_col} 
    return df, col_summary

def summary_output(df, col_summary):
    # Calculate Missing
    output = pd.DataFrame(len(df.index) - df.count())
    output['Total'] = df.shape[0]
    output.columns = ['Miss', 'Total']         
    output['Miss %'] = output['Miss'] / output['Total']
    output['NonMiss %'] = 1 - output['Miss %']
    
    # Add Column Type from col_summary dictionary
    col_type_list = []
    for col_name in output.index:
        for col_type in col_summary.keys():
            if col_name in col_summary[col_type]: col_type_find = col_type 
        col_type_list.append(col_type_find)
    output['Column Type'] = col_type_list
    # order by column type
    output['Column Type'] = pd.Categorical(output['Column Type'], 
                                           ordered=True, 
                                           categories = ['Account ID', 'Account Info','Date Info','Score Var',
                                                         'Category Var','Continuous Var', 'Other Var', 'Most Missing', 'All Missing'])
    output = output.sort_values('Column Type')
    return output

# For Categorical Variable, List all possible category value if its proportion is greater than 5% of non-missing population
def get_categorical_values(df, category_col, prop = 0.05):
    def select_categorical_values(x, prop = 0.05):
        try:
            x = list(x)
        except:
            print('Input is not a list type')
        non_missing_count = len(x)
        categorical_values = []
        for i in set(x):
            if x.count(i) / non_missing_count > prop:
                categorical_values.append(i)
        return categorical_values
    
    col_name = 'ap_loan_purpose'
    categorical_var_values = {}
    for col_name in category_col:
        x = list(df[col_name][df[col_name].notnull()])
        if len(x) > 0:
            categorical_var_values.update({col_name:select_categorical_values(x, prop = prop)})
    return categorical_var_values


'''Feature Engineering'''
def auto_feature_engineering(df, col_summary, categorical_var_values):
    # Convert Category to one-hot-encoding
    def cat_to_dummy(df, feature_name, possible_value):
        new_feature = []
        for i in possible_value:
            new_feature.append(feature_name + '_' + str(i))
            df[feature_name + '_' + str(i)] = np.where(df[feature_name] == i, 1, 0)
        if sum(pd.isna(df[feature_name]) > 0):
            df[feature_name + '_' + 'miss'] = np.where(pd.isna(df[feature_name]), 1, 0)
            new_feature.append(feature_name + '_' + 'miss')
        return df, new_feature
    
    feature_list = []
    feature_list = col_summary['Score Var'] + col_summary['Continuous Var']
    # Automatically generate categorical variables
    for i in col_summary['Category Var']:
        df, new_feature = cat_to_dummy(df, i, categorical_var_values[i])
        feature_list = feature_list + new_feature
    return df, feature_list

'''Example'''

df = pd.read_pickle('C:/Users/csltwu/Documents/Package/daily_data.pkl')
df = df_pre_clean(df)
df, col_summary = df_col_type_summary(df)
output = summary_output(df, col_summary)
categorical_var_values = get_categorical_values(df, category_col, prop = 0.05)
df, feature_list = auto_feature_engineering(df, col_summary, categorical_var_values)
X = df[feature_list]
X_summary = X.describe().T

