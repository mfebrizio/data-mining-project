# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:27:50 2022

@author: mark
"""

# import dependencies
import pandas as pd
from datetime import datetime

# function that converts single column from a dataframe into date format
def column_to_date(df, column = ''):
    ''' Convert dataframe columns in str format to datetime.date format.
    
    Dependencies
    ------------
    pandas
    datetime
    
    Parameters
    ----------
    df : Dataframe
    
    columns : str, optional
        String of column name to convert to datetime. The default is ''.
    
    Returns
    -------
    Series
        Series of converted columns.
    '''
    # copy dataframe
    df_copy = df.copy(deep=True)
    
    # convert to datetime format
    df_copy.loc[:,column] = pd.to_datetime(df_copy[column])
        
    # convert to date format (removes timestamp info)
    df_copy.loc[:,column] = df_copy[column].apply(lambda x: x.date())
    
    # return series for input column
    return df_copy.loc[:,column]

# function that handles multiple columns
def columns_to_date(df, columns = []):
    ''' Convert dataframe columns in str format to datetime.date format.
    
    Dependencies
    ------------
    pandas
    datetime
    
    Parameters
    ----------
    df : Dataframe
    
    columns : list, optional
        List of columns to convert to datetime. The default is [].
    
    Returns
    -------
    List
        List containing each converted column as a series.
    '''
    # copy dataframe
    df_copy = df.copy(deep=True)
    
    # loop over input columns
    for c in columns:
        # convert to datetime format
        df_copy.loc[:,c] = pd.to_datetime(df_copy[c])
        
        # convert to date format (removes timestamp info)
        df_copy.loc[:,c] = df_copy[c].apply(lambda x: x.date())
    
    # create list of each column as a series
    series_list = [df_copy.loc[:,series] for series in df_copy.loc[:,columns]]
    
    # return list of series for input columns
    return series_list
