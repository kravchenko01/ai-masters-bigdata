import sys

model_path = sys.argv[1]
data_path = sys.argv[2]
pred_path = sys.argv[3]


from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

model = PipelineModel.load(model_path)

data = spark.read.json(data_path).fillna("null", subset=("reviewText", "summary", "vote")).cache()
predictions = model.transform(data)
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="overall", metricName='rmse')

rmse = evaluator.evaluate(predictions)

#print("RMSE =", rmse)

predictions.write.parquet(pred_path, mode="overwrite")
