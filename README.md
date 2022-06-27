# data-mining-project

Building a Classifier to Identify the Type of Federal Register Documents

## Author Information
Mark Febrizio  
mfebrizio@gwu.edu  
DATS 6103 - Summer 2022  

## Project Description

The [Federal Register](https://www.federalregister.gov/) is the daily journal of the U.S. government, with a new issue published each business day. Each issue is divided into four sections containing four corresponding document types: Notices, Proposed Rules, Rules, and Presidential Documents. However, some data are missing document type labels (e.g., much of the 1990s data). When researchers conduct analysis of agency actions, this produces a severe underestimation of the frequency of document types and the content related to specific topic areas. As a solution, I used the labeled documents to build classifier for document type. After training and testing this classifier on labeled data using supervised learning models, the classifier could be applied to uncategorized data for predicting the correct labels for uncategorized documents.

## Contents

**code/**  
- directory containing the Python code used to implement the project

**data/**  
- empty directory for storing data files (does not track data files and subdirectories via .gitignore)

**presentation/**  
- presentation slides and materials

**proposal/**  
- contains Markdown file containing a description of the problem and a proposal for implementing the project

**report/**  
- contains final report for the project

## Code Sequence

The Python code should be run in the following sequence:  
- `retrieve_FR_data.py`
- `data_preprocessing.py`
	- calls modules: `clean_agencies.py`, `columns_to_date.py`
- `EDA.py`
	- calls modules: `cm_to_heatmap.py`
- `modeling_1.py`
- `modeling_2.py`
- `modeling_3.py`
- `modeling_4.py`

Executing `main.py` runs the code in this sequence.
