import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f

conf = SparkConf()
sc = SparkContext(appName="Pagerank", conf=conf)
spark = SparkSession(sc).builder.appName("SparkSQL").getOrCreate()

start_node = int(sys.argv[1])
end_node = int(sys.argv[2])
data_path = sys.argv[3]
output_file = sys.argv[4]

schema = StructType(fields=[
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType()),
])

df = spark.read\
        .schema(schema)\
        .format("csv")\
        .option("sep", "\t")\
        .load(data_path)


#print(df.show(5))
#print(df.printSchema())


max_path_length = 100

df_tmp = df.where(df.follower_id == start_node)
df_paths = df_tmp.select(f.concat_ws(",",  "follower_id", "user_id").alias("path"), df_tmp.user_id.alias("next"))
for i in range(max_path_length):
    if df_paths.where(df_paths.next == end_node).count() > 0:
        break

    df_join = df_paths.join(df.select(df.follower_id.alias("next"), df.user_id), on="next", how="inner")
    df_paths = df_join.select(f.concat_ws(",", "path", "user_id").alias("path"), df_join.user_id.alias("next"))

df_paths.select("path").where(df_paths.next == end_node).write.mode("overwrite").text(output_file)


spark.stop()
