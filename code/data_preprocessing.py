# --------------------------------------------------
# initialize
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clean_agencies import *
from columns_to_date import *
from search_columns import *

# --------------------------------------------------
p = Path.cwd()
data_dir = p.parent.joinpath('data', 'raw')
if data_dir.exists():
    pass
else:
    print("Directory doesn't exist.")

# --------------------------------------------------
# load data
filePath = data_dir / r"documents_endpoint_rules_combo_1994_1999.json"
with open(filePath, "r") as f:
    data = json.load(f)

# convert to dataframe and check structure
df = pd.DataFrame(data['results'])
df.info()

# --------------------------------------------------
# Examine target labels
# categories of documents to examine
print(df['type'].value_counts(),'\n')

# filter df of core document types
core_types = ["Notice", "Rule", "Proposed Rule", "Presidential Document"]
bool_core = np.array([True if t in core_types else False for t in df['type'].tolist()])
dfCore = df.loc[bool_core, :]
print(dfCore["type"].value_counts(),'\n')

# ["Uncategorized Document"]
bool_uncat = np.array([True if t == "Uncategorized Document" else False for t in df['type'].tolist()])
dfUncat = df.loc[bool_uncat, :]
print(dfUncat["type"].value_counts(),'\n')

# --------------------------------------------------
# Data cleaning

# clean up publication date column
dfCore.loc[:, 'publication_dt'] = column_to_date(dfCore, 'publication_date')
dfCore.loc[:, 'publication_year'] = dfCore['publication_dt'].apply(lambda x: x.year)

# clean up agencies column
dfCore = FR_clean_agencies(dfCore, column='agencies')

# --------------------------------------------------
# Create new variables

# count of UQ agencies per document
dfCore.loc[:, 'agencies_count_uq'] = dfCore['agencies_slug_uq'].apply(lambda x: len(x))

# create abstract length variable
# simple tokenization of 'abstract' column using whitespace characters
dfCore.loc[:, 'abstract_tokens'] = dfCore['abstract'].str.split(pat = r'\\s', regex=True)
abstract_length = [len(a) if a != None else None for a in dfCore.loc[:,'abstract_tokens']]
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

# --------------------------------------------------
# Text column cleaning




docket_exists

dfCore.info()



# clean agencies column //
# len(agencies_slug_uq) //
# len(abstract) //
# extract page_views count //
# convert significant to 0, 1, nan //
# effective date exists //
# comments_close_on exists //
# RINs exist //
# cfr_references exist //
# regs dot gov exist //

# clean:
# action
# title
# dates
# abstract

