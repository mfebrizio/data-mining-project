# Proposal

Building a Classifier to Identify the Type of Federal Register Documents

## Background

The [Federal Register](https://www.federalregister.gov/) is the daily journal of the U.S. government, with a new issue published each business day. Each issue is split into 4 sections containing 4 corresponding document types: Notices, Proposed Rules, Rules, and Presidential Documents.

Generally, documents are automatically categorized into the correct type. This is helpful for [searching](https://www.federalregister.gov/documents/search#advanced) the Federal Register for documents of a particular nature or relevance. Further, advanced users can call the [Federal Register REST API](https://www.federalregister.gov/reader-aids/developer-resources/rest-api) to retrieve large quantities of metadata on documents for additional analysis. Given that federal agencies use the Federal Register to officially announce the development and finalization of regulations (also called rules or rulemakings), as well as solicit public input on those regulations, these data are valuable when conducting research on the regulatory process, agency performance, and specific policy areas (e.g., environment, immigration, energy, housing).

## Problem: Categorizing Documents by Type

One major challenge to research on agency rulemaking is that documents, particularly from the 1990s, remain uncategorized. In the available data from the 1990s (1994-1999), more than 80% of 201,591 published documents are categorized -- specifically, 166,031 are categorized as 1 of the 4 types and 32,468 are uncategorized. For some years, such as 1994, this leads to a severe under-counting of the number of rules published.

To address this problem, I will build a classification model that categorizes Federal Register documents by their type. Although the problem stems from a set of uncategorized documents, a majority of documents are labeled by type, making it possible to train and test a supervised learning model on a target with labels. Once achieving a sufficient accuracy score, the model can be applied to the uncategorized data to help mitigate the problem of missing document types.

## Model Selection

Multinomial Naive Bayes or Complement Naive Bayes?
https://scikit-learn.org/stable/modules/naive_bayes.html

Linear SVM (class_weight='balanced', random_state=a)
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

Random forests (n_estimators=10?)
https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees


## Model Evaluation


## Anticipated Project Schedule

1. Data collection
	a. Retrieve document metadata from API
	b. Format and save data as json
2. Data preprocessing
	a. Data cleaning
		- identifying relevant features
		- formatting features
		- 
	b. Imputing missing values
	c. Data transformation and scaling
		- text
		- numeric
		- categorical
	d. Creating train/test samples
3. Exploratory data analysis
4. Modeling
	a. Training
	b. Predicting
	c. Calculating metrics
	d. Evaluation
5. Results


## References


https://stackoverflow.com/questions/65327433/machine-learning-classification-using-categorical-and-text-data-as-input

https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

