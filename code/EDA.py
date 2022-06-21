#%% --------------------------------------------------
# init
from pathlib import Path

import numpy as np
import pandas as pd

# set directory path
p = Path.cwd()
data_dir = p.parent.joinpath('data', 'processed')
if data_dir.exists():
    pass
else:
    print("Directory doesn't exist.")

#%% --------------------------------------------------
# load data
filePath = data_dir / r"labeled_data_for_modeling.csv"
with open(filePath, "r", encoding="utf-8") as f:
    df = pd.read_csv(f)

# check structure
df.info()

#%% --------------------------------------------------
# Summary Statistics

desc_cols = ['page_length', 'agencies_count_uq', 'abstract_length', 'page_views_count',
             'RIN_count', 'CFR_ref_count', 'sig',
             'effective_date_exists', 'comments_close_exists', 'docket_exists', 'eop'
             ]

# describe num and cat variables
summ_stats = df.loc[:, desc_cols].describe()
print("Summary statistics (1):", summ_stats.iloc[:, 0:3], sep="\n")
print("#", 50 * "-")
print("Summary statistics (2):", summ_stats.iloc[:, 3:6], sep="\n")
print("#", 50 * "-")
print("Summary statistics (3):", summ_stats.iloc[:, 6:9], sep="\n")
print("#", 50 * "-")
print("Summary statistics (4):", summ_stats.iloc[:, 9:], sep="\n")
print("#", 50 * "-")

#%% --------------------------------------------------
# Data viz


