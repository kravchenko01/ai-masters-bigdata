#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields, bad_cf

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2a.joblib")

#read and infere
if 'label' in fields:
    fields.remove('label')

read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=100, na_values='\\N'
)

for df in pd.read_csv(sys.stdin, **read_opts):
    # df.drop(bad_cf, axis=1, inplace=True)
    ids =  df.id.copy()
    df.drop(['id'], axis=1, inplace=True)
    pred = model.predict_proba(df)
    out = zip(ids, pred[:, 1])
    print("\n".join([f"{i[0]}\t{i[1]}" for i in out]))
