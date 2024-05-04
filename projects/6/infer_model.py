import argparse

from joblib import load

from pyspark.ml.functions import vector_to_array
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType

parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--test-in', dest="test_in")
parser.add_argument('--pred-out', dest="pred_out")
parser.add_argument('--sklearn-model-in', dest="model_in")

args = parser.parse_args()


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


data = spark.read.json(args.test_in)

model = load(args.model_in)
model_broadcast = spark.sparkContext.broadcast(model)

@f.pandas_udf(FloatType())
def predict(*cols):
    predictions = model_broadcast.value.predict(cols[0])
    return pd.Series(predictions)

data = data.withColumn("prediction", predict('reviewText'))

data.select("id", "prediction").write.mode("overwrite").csv(args.pred_out)
