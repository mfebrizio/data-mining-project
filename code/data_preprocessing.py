#%% --------------------------------------------------
# initialize

# import packages
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from clean_agencies import *
from columns_to_date import *

# set directory path
p = Path.cwd()
data_dir = p.parent.joinpath('data', 'raw')
if data_dir.exists():
    pass
else:
    print("Directory doesn't exist.")

# ignore warnings
warnings.filterwarnings("ignore")

#%% --------------------------------------------------
# load data
filePath = data_dir / r"documents_endpoint_rules_combo_1994_1999.json"
with open(filePath, "r") as f:
    data = json.load(f)

# convert to dataframe and check structure
df = pd.DataFrame(data['results'])
df.info()
print("#", 50 * "-")

#%% --------------------------------------------------
# Examine target labels
# categories of documents to examine
print("All document types: ", df['type'].value_counts(), sep='\n')
print("#", 50 * "-")

# filter df of core document types
core_types = ["Notice", "Rule", "Proposed Rule", "Presidential Document"]
bool_core = np.array([True if t in core_types else False for t in df['type'].tolist()])
dfCore = df.loc[bool_core, :]
print("Labeled documents (4 main types):", dfCore["type"].value_counts(), sep='\n')
print("#", 50 * "-")

# Uncategorized Documents
bool_uncat = np.array([True if t == "Uncategorized Document" else False for t in df['type'].tolist()])
dfUncat = df.loc[bool_uncat, :]
print("Unlabeled Documents:", dfUncat["type"].value_counts(), sep='\n')
print("#", 50 * "-")

#%% --------------------------------------------------
# Data cleaning

# clean up publication date column
dfCore.loc[:, 'publication_dt'] = column_to_date(dfCore, 'publication_date')
dfCore.loc[:, 'publication_year'] = dfCore['publication_dt'].apply(lambda x: x.year)

# clean up agencies column
dfCore = FR_clean_agencies(dfCore, column='agencies')

# fix negative page length; impute with length == 1
bool_fix = np.array(dfCore['page_length'] <= 0)
dfCore.loc[bool_fix, 'page_length'] = 1

#%% --------------------------------------------------
# Create new variables

# count of UQ agencies per document
dfCore.loc[:, 'agencies_count_uq'] = dfCore['agencies_slug_uq'].apply(lambda x: len(x))

# reformat agency columns
dfCore.loc[:, 'agencies_slug_uq'] = dfCore['agencies_slug_uq'].apply(lambda x: "; ".join(x))
dfCore.loc[:, 'agencies_id_uq'] = dfCore['agencies_id_uq'].apply(lambda x: "; ".join(list(map(str,x))))

# create abstract length variable
# simple tokenization of 'abstract' column using whitespace characters
dfCore.loc[:, 'abstract_tokens'] = dfCore['abstract'].str.split(pat=r'\s', regex=True)
abstract_length = [len(a) if a is not None else 0 for a in dfCore.loc[:, 'abstract_tokens']]
dfCore.loc[:, 'abstract_length'] = abstract_length

# extract page_views count
dfCore.loc[:, 'page_views_count'] = dfCore['page_views'].apply(lambda x: int(x['count']))

# convert significant to categorical: 0 (false), 1 (true), 2 (unknown/nan)
bool_na = dfCore['significant'].isna()
dfCore.loc[bool_na, 'sig'] = 2
bool_f = dfCore['significant'] == False
dfCore.loc[bool_f, 'sig'] = 0
bool_t = dfCore['significant'] == True
dfCore.loc[bool_t, 'sig'] = 1
print(dfCore['sig'].value_counts(dropna=False), '\n')

# effective date exists
bool_exists = dfCore['effective_on'].notna()
dfCore.loc[:, 'effective_date_exists'] = int(0)
dfCore.loc[bool_exists, 'effective_date_exists'] = int(1)
print(dfCore['effective_date_exists'].value_counts(dropna=False), '\n')

# comments_close_on exists
bool_exists = dfCore['comments_close_on'].notna()
dfCore.loc[:, 'comments_close_exists'] = int(0)
dfCore.loc[bool_exists, 'comments_close_exists'] = int(1)
print(dfCore['comments_close_exists'].value_counts(dropna=False), '\n')

# extract RIN count
dfCore.loc[:, 'RIN_count'] = dfCore['regulation_id_numbers'].apply(lambda x: len(x))
print(dfCore['RIN_count'].value_counts(dropna=False), '\n')

# extract CFR references count
dfCore.loc[:, 'CFR_ref_count'] = dfCore['cfr_references'].apply(lambda x: len(x))
print(dfCore['CFR_ref_count'].value_counts(dropna=False), '\n')

# regs dot gov info exists
dfCore.loc[:, 'docket_exists'] = [0 if x == {} else 1 for x in dfCore['regulations_dot_gov_info']]
print(dfCore['docket_exists'].value_counts(dropna=False), '\n')

# document issued by Executive Office of the President
bool_na = np.array(dfCore['agencies_count_uq'].notna())
bool_eop = np.array(dfCore.loc[bool_na, 'agencies_slug_uq'].apply(
    lambda x: 'executive-office-of-the-president' in x.split("; ")))
dfCore.loc[:, 'eop'] = 0
dfCore.loc[bool_eop, 'eop'] = 1
print(dfCore['eop'].value_counts(dropna=False), '\n')

#%% --------------------------------------------------
# Text column cleaning
# these columns: # action; abstract; title; [dates?]

# boolean for filtering presidential documents
# bool_prez = np.array(dfCore['type'] == "Presidential Document") -- better not to use labels to generate features
bool_eop = np.array(dfCore['eop'] == 1)

# impute missing text: action
bool_na = np.array(dfCore['action'].isna())
dfCore.loc[bool_na & bool_eop, 'action'] = 'presidential document'
dfCore.loc[bool_na & ~bool_eop, 'action'] = dfCore.loc[bool_na & ~bool_eop, 'title'].tolist()

# impute missing text: abstract/summary
bool_na = np.array(dfCore['abstract'].isna())
dfCore.loc[bool_na & bool_eop, 'abstract'] = 'presidential document'
dfCore.loc[bool_na & ~bool_eop, 'abstract'] = dfCore.loc[bool_na & ~bool_eop, 'title'].tolist()

#%% --------------------------------------------------
# Filter dataframe columns

# columns to keep for modeling
label_col = ['type']
id_cols = ['document_number', 'citation', 'agencies_id_uq', 'agencies_slug_uq', 'publication_year']
num_cols = ['page_length', 'agencies_count_uq', 'abstract_length', 'page_views_count', 'RIN_count', 'CFR_ref_count']
cat_cols = ['sig', 'effective_date_exists', 'comments_close_exists', 'docket_exists', 'eop']
text_cols = ['action', 'abstract', 'title']
keep_cols = label_col + id_cols + num_cols + cat_cols + text_cols

# create new dataframe
dfModeling = dfCore.loc[:, keep_cols].copy()

#%% --------------------------------------------------
# Export data for modeling

# set directory path
p = Path.cwd()
data_dir = p.parent.joinpath('data', 'processed')
if data_dir.exists():
    pass
else:
    try:
        data_dir.mkdir(parents=True)
    except:
        print("Cannot create data directory.")

# save as csv
filePath = data_dir / r"labeled_data_for_modeling.csv"
with open(filePath, "w", encoding="utf-8") as f:
    dfModeling.to_csv(f, index_label="index", line_terminator="\n")

# check if saved
if filePath.exists():
    print("Saved as CSV!")
else:
    print("Error saving file.")
