from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import *

from pyspark.sql.types import *
import pyspark.sql.functions as f

lr = LinearRegression(featuresCol="features", labelCol="overall", maxIter=10, regParam=0.1)

tokenizer1 = Tokenizer(inputCol="reviewText", outputCol="reviewText_words")
tokenizer2 = Tokenizer(inputCol="summary", outputCol="summary_words")
tokenizer3 = Tokenizer(inputCol="vote", outputCol="vote_words")

stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr1 = StopWordsRemover(inputCol=tokenizer1.getOutputCol(), outputCol="reviewText_words_filtered", stopWords=stop_words)
swr2 = StopWordsRemover(inputCol=tokenizer2.getOutputCol(), outputCol="summary_words_filtered", stopWords=stop_words)

count_vectorizer1 = CountVectorizer(inputCol=swr1.getOutputCol(), outputCol="word_vector1")
count_vectorizer2 = CountVectorizer(inputCol=swr2.getOutputCol(), outputCol="word_vector2")

# hasher1 = HashingTF(numFeatures=100, inputCol=swr1.getOutputCol(), outputCol="word_vector1")
# hasher2 = HashingTF(numFeatures=100, inputCol=swr2.getOutputCol(), outputCol="word_vector2")
hasher3 = HashingTF(numFeatures=1, inputCol=tokenizer3.getOutputCol(), outputCol="vote_vector")

assembler = VectorAssembler(inputCols=[
    count_vectorizer1.getOutputCol(),
    count_vectorizer2.getOutputCol(),
    hasher3.getOutputCol(),
    "verified"
],outputCol="features", handleInvalid="keep")

pipeline = Pipeline(stages=[
    tokenizer1,
    tokenizer2,
    tokenizer3,
    swr1,
    swr2,
    count_vectorizer1,
    count_vectorizer2,
    #     hasher1,
    #     hasher2,
    hasher3,
    assembler,
    lr
])
