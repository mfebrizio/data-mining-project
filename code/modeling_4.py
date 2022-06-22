#%% init
from pathlib import Path

from nltk.corpus import stopwords
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder

from cm_to_heatmap import *

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
# Preprocessing

# tf-idf vectorization of corpus
corpus = df.loc[:, 'action'].tolist()
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('english'))
processed_corpus = vectorizer.fit_transform(corpus)

# identify features and target
X = processed_corpus
y = np.array(df['type']).ravel()

# encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Encoded classes: ", list(zip(range(0,4),le.classes_)), sep="\n")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=622)

# init model
clf = ComplementNB(norm=True)

#%% --------------------------------------------------
# Modeling

# fit model and make predictions
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# compare y test values against y predictions
print("Classification Report: ", classification_report(y_test, y_pred, zero_division=0),
      "Accuracy:", accuracy_score(y_test, y_pred),
      "F1 (weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=0),
      sep="\n")
print("#", 50 * "-")

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
index_list = list(map(lambda x: 'true_' + str(x), range(0, 4)))
column_list = list(map(lambda x: 'pred_' + str(x), range(0, 4)))
df_cm = pd.DataFrame(conf_matrix, index=index_list, columns=column_list)
print("Encoded classes: ", list(zip(range(0,4),le.classes_)), sep="\n")
print("Confusion Matrix: ", df_cm, sep="\n")
print("#", 50 * "-")

# save output results
save_dir = p.parent.joinpath('data', 'analysis')
savePath = save_dir / r"model_4_metrics.txt"
with open(savePath, 'w') as textfile:
    print("Model: ", clf, clf.get_params(),
          "\nClassification Report: ", classification_report(y_test, y_pred, zero_division=0),
          "\nAccuracy:", accuracy_score(y_test, y_pred),
          "\nF1 (weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=0),
          "\nEncoded classes: ", list(zip(range(0, 4), le.classes_)),
          "\nConfusion Matrix: ", df_cm,
          sep="\n",
          file=textfile)

# check if saved
if savePath.exists():
    print("Saved successfully!")
else:
    print("Error saving file.")

# save confusion matrix as heatmap
fig_dir = p.parent.joinpath('presentation', 'figures')
figPath = fig_dir / r'cm_heatmap_model_4.png'
df_cm.index = list(range(0,4))
df_cm.columns = list(range(0,4))
cm_to_heatmap(df_cm, title=r'Confusion Matrix: Model 4', figname=figPath)
