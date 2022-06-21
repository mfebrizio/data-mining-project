# --------------------------------------------------
# Import packages

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# --------------------------------------------------
# set directory path

p = Path.cwd()
data_dir = p.parent.joinpath('data', 'processed')
if data_dir.exists():
    pass
else:
    print("Directory doesn't exist.")

# --------------------------------------------------
# load data

filePath = data_dir / r"labeled_data_for_modeling.csv"
with open(filePath, "r", encoding="utf-8") as f:
    df = pd.read_csv(f)
print("Loaded dataset with shape: ", df.shape)

# --------------------------------------------------
# Preprocessing

# set random state
a = 622

# column dtypes
label = 'type'
numeric_features = ['page_length', 'agencies_count_uq', 'abstract_length', 'page_views_count',
                    'RIN_count', 'CFR_ref_count']
categorical_features = ['sig', 'effective_date_exists', 'comments_close_exists', 'docket_exists', 'eop']

# distinguish X and y objects
X = df[numeric_features+categorical_features]
y = np.array(df[label]).ravel()

# encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Encoded classes: ", list(zip(range(0,4),le.classes_)), sep="\n")

# transformers by dtype
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="if_binary")

# preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# feature selector
selector_threshold = "median"
selector = SelectFromModel(estimator=LogisticRegression(solver='lbfgs',
                                                        class_weight='balanced',
                                                        random_state=a,
                                                        max_iter=1000),
                           threshold=selector_threshold
                           )

# create pipeline
feature_selection = Pipeline(
    steps=[("preprocessor", preprocessor), ("selector", selector)]
)

# --------------------------------------------------
# Evaluate feature importance

# fit selector pipeline
feature_selection.fit(X, y_enc)

# print results
print(f"Estimator: {selector.estimator_}",
          f"No. features in: {feature_selection.n_features_in_}",
          f"Feature names in: {feature_selection.feature_names_in_}",
          f"Threshold value ({selector_threshold}): {selector.threshold_}",
          f"Mask of features selected: {selector.get_support(indices=True)}",
          f"Mask of selected feature names: {feature_selection.get_feature_names_out()}",
          sep="\n")

# save output
save_dir = p.parent.joinpath('data', 'analysis')
savePath = save_dir / r"feature_selection.txt"
with open(savePath, 'w') as textfile:
    print(f"Estimator: {selector.estimator_}",
          f"No. features in: {feature_selection.n_features_in_}",
          f"Feature names in: {feature_selection.feature_names_in_}",
          f"Threshold value ({selector_threshold}): {selector.threshold_}",
          f"Mask of features selected: {selector.get_support(indices=True)}",
          f"Mask of selected feature names: {feature_selection.get_feature_names_out()}",
          sep="\n",
          file=textfile)
