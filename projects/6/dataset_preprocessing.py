import argparse

from pyspark.ml import Pipeline
from pyspark.ml.feature import *

from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import SparkSession


parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--path-in', dest="path_in")
parser.add_argument('--path-out', dest="path_out")

args = parser.parse_args()


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


tokenizer = Tokenizer(inputCol="reviewText", outputCol="reviewText_words")

stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer1.getOutputCol(), outputCol="reviewText_words_filtered", stopWords=stop_words)

count_vectorizer = CountVectorizer(inputCol=swr1.getOutputCol(), outputCol="word_vector")

assembler = VectorAssembler(inputCols=[
    count_vectorizer1.getOutputCol(),
],outputCol="features")

pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    assembler,
])

train_data = spark.read.json(args.path_in).fillna("null", subset=("reviewText"))

train_data = pipeline.transform(train_data)

if "label" in train_data.columns:
    train_data = train_data.select("label", "features", "id")
else:
    train_data = train_data.select("features", "id")

train_data.write().overwrite().save(args.path_out)
