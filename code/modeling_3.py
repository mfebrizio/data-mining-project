#%% --------------------------------------------------------
# init
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

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
# reference: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

# set random state
a = 622

# column dtypes
label = 'type'
numeric_features = ['page_length', 'agencies_count_uq', 'abstract_length', 'page_views_count',
                    'RIN_count', 'CFR_ref_count']
categorical_features = ['sig', 'effective_date_exists', 'comments_close_exists', 'docket_exists', 'eop']

# distinguish X and y objects
X = df[numeric_features + categorical_features]
y = np.array(df[label]).ravel()

# encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Encoded classes: ", list(zip(range(0,4),le.classes_)), sep="\n")
print("#", 50 * "-")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=a)
print(f"X_train observations: {len(X_train)}",
      f"X_test observations: {len(X_test)}",
      f"y_train observations: {len(y_train)}",
      f"y_test observations: {len(y_test)}", sep="\n")
print("#", 50 * "-")

# ----------------------------------------------------------
# Model pipeline

# transformers
numeric_transformer = MinMaxScaler()  # use instead of StandardScaler -- complementNB does not accept negative values
categorical_transformer = OneHotEncoder(drop="if_binary")

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# base classifiers
clf1 = ComplementNB(norm=True)
clf2 = LogisticRegression(class_weight='balanced', random_state=a, max_iter=500)
clf3 = KNeighborsClassifier()

# ensemble method
eclf = VotingClassifier(estimators=[('cnb', clf1),
                                    ('lr', clf2),
                                    ('knn', clf3)], voting='hard')

# grid search
# reference: https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
# see subsection 1.11.6.4. Using the VotingClassifier with GridSearchCV
params = {'knn__n_neighbors': [5, 99, 341]}
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

# pipeline for classifier
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", grid)]
)

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
print("Confusion Matrix: ", df_cm, sep="\n")
print("#", 50 * "-")

# save output results
save_dir = p.parent.joinpath('data', 'analysis')
savePath = save_dir / r"model_3_metrics.txt"
with open(savePath, 'w') as textfile:
    print("Model steps: ", clf.named_steps,
          "\nClassification Report: ", classification_report(y_test, y_pred, zero_division=0),
          "\nAccuracy:", accuracy_score(y_test, y_pred),
          "\nF1 (weighted):", f1_score(y_test, y_pred, average='weighted', zero_division=0),
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
figPath = fig_dir / r'cm_heatmap_model_3.png'
df_cm.index = list(range(0,4))
df_cm.columns = list(range(0,4))
cm_to_heatmap(df_cm, title=r'Confusion Matrix: Model 3', figname=figPath)
