import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
import pandas as pd
from joblib import dump

parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--train-in', dest="train_in")
parser.add_argument('--sklearn-model-out', dest="model_out")

args = parser.parse_args()

data = pd.read_json(args.train_in, lines=True)

y = data['label']
X = data['reviewText']

model = make_pipeline(
    HashingVectorizer(n_features=50),
    LogisticRegression()
)
model.fit(X, y)

dump(model, args.model_out)
