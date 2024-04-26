import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

data_path = sys.argv[1]
save_path = sys.argv[2]


train_data = spark.read.json(data_path)

train_data = train_data.fillna("null", subset=("reviewText", "summary", "vote")).cache()

pipeline_model = pipeline.fit(train_data)

pipeline_model.write().overwrite().save(save_path)



