#!/opt/conda/envs/dsenv/bin/python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
# from joblib import dump

import mlflow


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  train_path = sys.argv[1] 
  model_param1 = sys.argv[2]
except:
  logging.critical("Need to pass both train dataset path and model param 1")
  sys.exit(1)


logging.info(f"TRAIN_PATH {train_path}")
logging.info(f"MODEL_PARAM1 {model_param1}")


#
# Dataset fields
#
numeric_features = ["if" + str(i) for i in range(1, 14)]
categorical_features = ["cf" + str(i) for i in range(1, 27)] 
good_cf_num = [1,2,5,6,8,9,14,17,19,20,22,23,25]
# good_cf = ["cf" + str(i) for i in range(1,27) if i in good_cf_num]
good_cf = categorical_features
bad_cf = ["cf" + str(i) for i in range(1,27) if i not in good_cf_num]
fields = ["id", "label"] + numeric_features + categorical_features

#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, good_cf)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logregression', LogisticRegression(max_iter=int(model_param1))),
])


#
# Read dataset
#
read_table_opts = dict(sep="\t", names=fields, index_col=0)
df = pd.read_table(train_path, **read_table_opts)
# df.drop(bad_cf, axis=1, inplace=True)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'label'], df.loc[:, df.columns == 'label'], test_size=0.33, random_state=42
)

#
# Train the model
#
with mlflow.start_run():
    model.fit(X_train, y_train.to_numpy().flatten())

    model_score = model.score(X_test, y_test)
    pred = model.predict_proba(X_test)

    logging.info(f"model score: {model_score:.3f}")
    logging.info(f"log loss: {log_loss(y_test, pred[:, 1])}")

    # Log with MLflow

    mlflow.sklearn.log_model(model, artifact_path="LR_model")
    mlflow.log_params(model.get_params())
    mlflow.log_metric("log_loss", log_loss(y_test, pred[:, 1]))


# save the model
# dump(model, "1a.joblib")
