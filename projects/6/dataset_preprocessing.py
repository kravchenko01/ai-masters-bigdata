import argparse

from pyspark.sql import SparkSession


parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--path-in', dest="path_in")
parser.add_argument('--path-out', dest="path_out")

args = parser.parse_args()


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


train_data = spark.read.json(args.path_in).fillna("null", subset=("reviewText"))

if "label" in train_data.columns:
    train_data = train_data.select("label", "reviewText", "id")
else:
    train_data = train_data.select("reviewText", "id")

train_data.write.mode('overwrite').json(args.path_out)
