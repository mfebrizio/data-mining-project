# data-mining-project

## Author Information
Mark Febrizio  
mfebrizio@gwu.edu  
DATS 6103 - Summer 2022  

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
- retrieve_FR_data.py
- data_preprocessing.py
	- calls modules: clean_agencies.py, columns_to_date.py
- EDA.py
	- calls modules: cm_to_heatmap.py
- modeling_1.py
- modeling_2.py
- modeling_3.py
- modeling_4.py

Executing main.py runs the code in this sequence.
