# import dependencies
import itertools
import pandas as pd
import numpy as np

# Defining a function to search for string patterns within dataframe columns
def search_columns(df, patterns = [], columns = [], re_flags = None):
    """Search columns for string patterns within dataframe columns.

    Parameters
    ----------
    
    Positional arguments:
    df -- input data in format of pandas dataframe
    
    Keyword arguments:
    patterns -- list of string patterns to input, compatible with regex (default = empty list)
    columns -- list of column names to search for input patterns (default = empty list)
    re_flags -- regex flags to use, such as re.I|re.X (default = None)
    
    Dependencies
    ------------
    pandas
    numpy
    itertools
    """
    
    # create list object for appending boolean arrays
    bool_list = []
    
    # ensure that input patterns and columns are formatted as lists
    if type(patterns)==list and type(columns)==list:
        pass
    else:
        raise Exception('Inputs for "patterns" and "columns" keywords must be lists.')
        
    if len(patterns)==len(columns):
        # create list of inputs in format [(pattern1, column1),(pattern2, column2), ...]
        inputs = list(zip(patterns,columns))
        
        # loop over list of inputs
        for i in inputs:
            searchre = df[i[1]].str.contains(i[0], regex=True, case=False, flags=re_flags)
            searchbool = np.array([True if n==True else False for n in searchre])
            bool_list.append(searchbool)
        
    elif len(patterns)==1 and (len(patterns)!=len(columns)):
        # create list of inputs in format [(pattern, column1),(pattern, column2), ...]
        inputs = list(itertools.product(patterns, columns))
        
        # loop over list of inputs
        for i in inputs:
            searchre = df[i[1]].str.contains(i[0], regex=True, case=False, flags=re_flags)
            searchbool = np.array([True if n==True else False for n in searchre])
            bool_list.append(searchbool)
           
    else: # eg, patterns formatted as a list of len(n>1) but does not match len(columns)
        raise Exception('Length of inputs are incorrect. Lengths of "patterns" and "columns" must match '+
        'or a single pattern can map to multiple columns.')

    # combine each "searchbool" array elementwise
    # we want a positive match for any column to evaluate as True
    # equivalent to (bool_list[0] | bool_list[1] | bool_list[2] | ... | bool_list[n-1])
    filter_bool = np.array(bool_list).any(axis=0)

    # filter results
    dfResults = df.loc[filter_bool,:].copy(deep=True)
    print('Count: '+str(len(dfResults)))
    return dfResults
