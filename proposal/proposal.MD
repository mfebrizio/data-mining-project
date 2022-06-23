# Proposal

title: Building a Classifier to Identify the Type of Federal Register Documents  
author: Mark Febrizio  
date: 2022-Jun-22  

## Author Information
Mark Febrizio  
mfebrizio@gwu.edu  
DATS 6103 - Summer 2022  

## Background

The [Federal Register](https://www.federalregister.gov/) is the daily journal of the U.S. government, with a new issue published each business day. Each issue is split into 4 sections containing 4 corresponding document types: Notices, Proposed Rules, Rules, and Presidential Documents.

Generally, documents are automatically categorized into the correct type. This is helpful for [searching](https://www.federalregister.gov/documents/search#advanced) the Federal Register for documents of a particular nature or relevance. Further, advanced users can call the [Federal Register REST API](https://www.federalregister.gov/reader-aids/developer-resources/rest-api) to retrieve large quantities of metadata on documents for additional analysis. Given that federal agencies use the Federal Register to officially announce the development and finalization of regulations (also called rules or rulemakings), as well as solicit public input on those regulations, these data are valuable when conducting research on the regulatory process, agency performance, and specific policy areas (e.g., environment, immigration, energy, housing).

## Problem: Categorizing Documents by Type

One major challenge to research on agency rulemaking is that documents, particularly from the 1990s, remain uncategorized. In the available data from the 1990s (1994-1999), more than 80% of 201,591 published documents are categorized -- specifically, 166,031 are categorized as 1 of the 4 types and 32,468 are uncategorized. For some years, such as 1994, this leads to a severe under-counting of the number of rules published.

To address this problem, I will build a classification model that categorizes Federal Register documents by their type. Although the problem stems from a set of uncategorized documents, a majority of documents are labeled by type, making it possible to train and test a supervised learning model on a target with labels. Once achieving a sufficient accuracy score, the model can be applied to the uncategorized data to help mitigate the problem of missing document types.

## Database

I will use the [Federal Register REST API](https://www.federalregister.gov/reader-aids/developer-resources/rest-api) as my data source. Containing hundreds of thousands of documents (if not millions), the database is sufficiently large to train machine learning algorithms.

I plan on restricting the data collected to the 1990s (specifically, 1994-1999) to focus on documents most similar to those missing labels and to limit the size of dataset to a reasonable amount. The Federal Register API only goes back to 1994, even though the journal was published for several decades prior (see, [Govinfo](https://www.govinfo.gov/app/collection/fr) for PDF copies dating back to 1936).

## Model Selection

I will use a several types of supervised learning models to build the classifier.

First, I will use a set of categorical and numerical features to train the classifier.

I plan on beginning with a [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) model, especially because such models are used for document classification in other contexts. I use the [Complement NB](https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes) model, rather than the Multinomial NB model, because it is more appropriate for imbalanced data sets.

To build on this initial model, I will use different ensemble methods to build a model.

I will use the boosting algorithm, [AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost), which fits a sequence of weak learners that weights the data so that future models having a higher likelihood of seeing misclassified observations. This approach will leverage many iterations of the Complement NB model. The final classifications are made via weighted majority vote.

Next, I will use a [voting classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) algorithm to combine several different types of models and then take a majority vote of the predicted labels. Here, the Complement NB model can be combined with K-Nearest Neighbors, logistic regression, tree models, or support vector machines. 

Last, I will use the text-based features of the documents, rather than numeric or categorical variables, to classify the documents. Here, I will try a simple Complement NB model again.

I will use Python and its associated libraries, such as sklearn, pandas, and numpy, to implement the machine learning algorithms. More detail on the environment and code I used is contained in the project [Github repository](https://github.com/mfebrizio/data-mining-project.git).

I will primarily rely on the sci-kit learn user guide and lecture notes to obtain the necessary background to apply these supervised learning models.

## Model Evaluation

I will use the accuracy score and the F1-score as metrics fo evaluating the models. Because the dataset is imbalanced in terms of document types (i.e., the majority of documents are type=="Notice"), the F1-score is a better metric to use, so I will weight its results more heavily.

I also will produce and examine a confusion matrix for each model. This will help me get a sense which document types the model is better or worse at classifying.

## Anticipated Project Schedule

This project can be completed in approximately 10 days.

1. Data collection
- 0.5 days
	a. Retrieve document metadata from API
	b. Format and save data as json

2. Data preprocessing
- 2.5 days
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
- 1 day

4. Modeling
- 4 days
	a. Training
	b. Predicting
	c. Calculating metrics
	d. Evaluation

5. Analyze results
- 1.5 days

## References

Li, Susan, [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568), Towards Data Science, published Sep. 25, 2018.

Pedregosa et al., [Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), JMLR 12, pp. 2825-2830, 2011.

scikit-learn 1.1.1, [User Guide](https://scikit-learn.org/stable/user_guide.html), revised May 2022.

scikit-learn 1.1.1, [API Reference](https://scikit-learn.org/stable/modules/classes.html), revised May 2022.

Stackoverflow, [Machine Learning Classification using categorical and text data as input](https://stackoverflow.com/questions/65327433/machine-learning-classification-using-categorical-and-text-data-as-input), modified Dec. 16, 2020.
