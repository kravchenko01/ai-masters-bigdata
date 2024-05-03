import argparse

from sklearn.linear_mode import LogisticRegression
import pandas as pd
from joblib import dump

parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--train-in', dest="train_in")
parser.add_argument('--sklearn-model-out', dest="model_out")

args = parser.parse_args()

data = pd.read_json(args.train_in, lines=True)

y = data['label']
X = data.drop(['label'])

model = LogisticRegression()
model.fit(X, y)

dump(model, args.model_out)