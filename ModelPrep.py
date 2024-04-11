import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv('diabetes_prediction_dataset.csv')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load

# Separate features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Define categorical and numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)],
    remainder='drop')

# Fit the preprocessor to obtain transformed column names
X_preprocessed = preprocessor.fit_transform(X)
column_names = (numerical_cols +
                list(preprocessor.named_transformers_['cat'].named_steps['onehot']
                     .get_feature_names_out(categorical_cols)))
from sklearn.utils import resample

# Re-create DataFrame with new column names for easier manipulation
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=column_names)

# Add the target variable back to the dataframe
X_preprocessed_df['diabetes'] = y

# Separate majority and minority classes
df_majority = X_preprocessed_df[X_preprocessed_df.diabetes==0]
df_minority = X_preprocessed_df[X_preprocessed_df.diabetes==1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=8500,  # to match minority class
                                   random_state=123) # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Display new class counts
# print(df_downsampled.diabetes.value_counts())

# Removing the target variable from features dataset for selection
X_res = df_downsampled.drop('diabetes', axis=1)
y_res = df_downsampled['diabetes']
from sklearn.feature_selection import SelectKBest, f_classif

# Apply SelectKBest class to extract top k best features
bestfeatures = SelectKBest(score_func=f_classif, k=5)
bestfeatures.fit(X_res, y_res)

# Get the boolean mask of the selected features
mask = bestfeatures.get_support()  # list of booleans
selected_features = []  # The list of your K best features

for bool, feature in zip(mask, X_res.columns):
    if bool:
        selected_features.append(feature)

print("Selected features:", selected_features)

from sklearn.model_selection import train_test_split

# Assuming 'selected_features' is the list of the names of the selected features
# Refine X_res to only include selected features
X_final = X_res[selected_features]
from sklearn.neural_network import MLPClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_res, test_size=0.2, random_state=42, stratify=y_res)
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42, verbose=False)
nn_clf.fit(X_train, y_train)
dump(nn_clf, 'Diabetes_MLP.joblib')