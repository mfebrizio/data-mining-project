#%% --------------------------------------------------
# init
from pathlib import Path

#from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from clean_text import *
#from generate_wordcloud import *

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

# save output results
save_dir = p.parent.joinpath('data', 'analysis')
savePath = save_dir / r"summary_statistics.txt"
with open(savePath, 'w') as textfile:
    print("Summary statistics (1):", summ_stats.iloc[:, 0:3],
          "\nSummary statistics (2):", summ_stats.iloc[:, 3:6],
          "\nSummary statistics (3):", summ_stats.iloc[:, 6:9],
          "\nSummary statistics (4):", summ_stats.iloc[:, 9:],
          sep="\n",
          file=textfile)

# check if saved
if savePath.exists():
    print("Saved successfully!")
else:
    print("Error saving file.")

#%% --------------------------------------------------
# Data viz

# set interactive plot off
plt.ioff()

# path for saving figures
fig_dir = p.parent.joinpath('presentation', 'figures')

#%% -------------------------------------------------------------------------
# Document Type by Year (line)

# line graph of type across publication years
by_year_type = df.groupby(by=['publication_year', 'type'], as_index=False).agg({'document_number': 'nunique',
                                                                                'page_length': np.mean,
                                                                                'abstract_length': np.mean})
print(by_year_type)

# create objects
years = list(range(1994, 2000))  # by_year_type.index.values
values_list = []
type_list = by_year_type['type'].value_counts().index.tolist()
for t in type_list:
    bool_type = np.array(by_year_type['type'] == t)
    values_list.append(by_year_type.loc[bool_type, 'document_number'].values)
size = (10,7)  # figure size

# line plot
fig, ax = plt.subplots(figsize=size)  # Create a figure containing a single axes
ax.plot(years, values_list[0]/1000, color='blue', linestyle='solid')  # Plot some data on the axes
ax.plot(years, values_list[1]/1000, color='blue', linestyle='dashed')
ax.plot(years, values_list[2]/1000, color='blue', linestyle='dotted')
ax.plot(years, values_list[3]/1000, color='blue', linestyle='dashdot')
ax.set_xlabel('Publication Year')  # Add an x-label to the axes
ax.set_ylabel('Thousands of Documents')  # Add a y-label to the axes

fig.tight_layout(pad=2)
textstr1 = "Source: Federal Register API and authors' calculations."
fig.text(0.1, 0.075, textstr1)

plt.grid(visible=True, which='major', axis='y')
plt.title('Documents by Publication Year', fontsize=12, fontweight='normal')
plt.subplots_adjust(bottom=0.175)
plt.legend(labels=type_list)

figPath = fig_dir / r'document_type_by_year_count'
fig.savefig(figPath, facecolor='w', edgecolor='w',
            orientation='landscape',
            transparent=False, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

#%% -------------------------------------------------------------------------
# Medians & Means by Document Type (bar)

# bar graph of median by type
median_by_type = df.groupby(by=['type'], as_index=False).agg({'page_length': np.median,
                                                              'abstract_length': np.median,
                                                              'agencies_count_uq': np.median,
                                                              'page_views_count': np.median,
                                                              'RIN_count': np.median,
                                                              'CFR_ref_count': np.median})

print(median_by_type)

# create objects
var_1 = median_by_type['page_length'].values
var_2 = median_by_type['abstract_length'].values
var_3 = median_by_type['agencies_count_uq'].values
var_4 = median_by_type['page_views_count'].values
var_5 = median_by_type['RIN_count'].values
var_6 = median_by_type['CFR_ref_count'].values

# set group labels and colors
groups = median_by_type['type'].values
x = np.arange(len(groups))  # the label locations
width = 0.1  # the width of the bars
color_def = '#0190DB'  # other options: '#A2BAF5', '#033C5A', 'navy'

# grouped bar plot
# hatch options {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
size = (10, 7)  # figure size
fig, ax = plt.subplots(figsize=size)  # Create a figure containing a single axes
ax.bar(x - width*2, var_1, width, label='pages', alpha=.6, color='navy')
ax.bar(x - width*1, var_2, width, label='abstract', hatch='//', alpha=.9, color=color_def)
ax.bar(x + width*0, var_3, width, label='agencies', hatch='..', alpha=.6, color=color_def)
ax.bar(x + width*1, var_4, width, label='views', hatch='\\\\', alpha=.4, color=color_def)
ax.bar(x + width*2, var_5, width, label='RIN', hatch='..', alpha=.2, color=color_def)
ax.bar(x + width*3, var_6, width, label='CFR', hatch='-', alpha=.1, color=color_def)

# labels and legend
ax.set_ylabel('Median value of feature')  # Add a y-label to the axes
ax.set_xticks(x, groups)
ax.legend(title='Feature')

fig.tight_layout(pad=2)
textstr1 = "Source: Federal Register API and authors' calculations."
fig.text(0.1, 0.035, textstr1)
plt.title('Median by Document Type', fontsize=12, fontweight='normal')
plt.subplots_adjust(bottom=0.175)

figPath = fig_dir / r'median_values_by_document_type'
fig.savefig(figPath, facecolor='w', edgecolor='w',
            orientation='landscape',
            transparent=False, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

# bar graph of means by type
mean_by_type = df.groupby(by=['type'], as_index=False).agg({'page_length': np.mean,
                                                            'abstract_length': np.mean,
                                                            'agencies_count_uq': np.mean,
                                                            'page_views_count': np.mean,
                                                            'RIN_count': np.mean,
                                                            'CFR_ref_count': np.mean})

print(mean_by_type)

# create objects
var_1 = mean_by_type['page_length'].values
var_2 = mean_by_type['abstract_length'].values
var_3 = mean_by_type['agencies_count_uq'].values
var_4 = mean_by_type['page_views_count'].values
var_5 = mean_by_type['RIN_count'].values
var_6 = mean_by_type['CFR_ref_count'].values

# set group labels and colors
groups = mean_by_type['type'].values
x = np.arange(len(groups))  # the label locations
width = 0.1  # the width of the bars
color_def = '#0190DB'  # other options: '#A2BAF5', '#033C5A', 'navy'

# grouped bar plot
# hatch options {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
size = (10,7)  # figure size
fig, ax = plt.subplots(figsize=size)  # Create a figure containing a single axes
ax.bar(x - width*2, var_1, width, label='pages', alpha=.6, color='navy')
ax.bar(x - width*1, var_2, width, label='abstract', hatch='//', alpha=.9, color=color_def)
ax.bar(x + width*0, var_3, width, label='agencies', hatch='..', alpha=.6, color=color_def)
ax.bar(x + width*1, var_4, width, label='views', hatch='\\\\', alpha=.4, color=color_def)
ax.bar(x + width*2, var_5, width, label='RIN', hatch='..', alpha=.2, color=color_def)
ax.bar(x + width*3, var_6, width, label='CFR', hatch='-', alpha=.1, color=color_def)

# labels and legend
ax.set_ylabel('Mean value of feature')  # Add a y-label to the axes
ax.set_xticks(x, groups)
ax.legend(title='Feature')

fig.tight_layout(pad=2)
textstr1 = "Source: Federal Register API and authors' calculations."
fig.text(0.1, 0.035, textstr1)
plt.title('Mean by Document Type', fontsize=12, fontweight='normal')
plt.subplots_adjust(bottom=0.175)

figPath = fig_dir / r'mean_values_by_document_type'
fig.savefig(figPath, facecolor='w', edgecolor='w',
            orientation='landscape',
            transparent=False, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

#%% -------------------------------------------------------------------------
# Values by Document Type (scatter)





#%% misc

'''
type_corpus = {}
type_list = df['type'].value_counts().index.tolist()
type_list.remove('Presidential Document')
for t in type_list:
    bool_type = df['type'] == t
    corpus = " ".join(df.loc[bool_type, 'action'].tolist())
    type_corpus.update({t: corpus})

clean_corpus = {}
for k,v in zip(type_corpus.keys(), type_corpus.values()):
    clean_corpus.update({k: clean_text(v)})

generate_basic_wordcloud()
'''
