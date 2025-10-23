import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr, size
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisApp1").getOrCreate()

## --- Configurable parameters ---
MIN_FREQ = 5
MAX_FREQ = 900
MIN_SENT_LEN = 3
NUM_FEATURES = 3000


# 2. Load Data từ bộ dữ liệu cải tiến

# Load train and valid sets independently
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'twitter-financial-news-sentiment', 'sent_train.csv'))
valid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'twitter-financial-news-sentiment', 'sent_valid.csv'))
df_train = spark.read.csv(train_path, header=True, inferSchema=True)
df_valid = spark.read.csv(valid_path, header=True, inferSchema=True)
print(f"Train samples (raw): {df_train.count()}")
print(f"Valid samples (raw): {df_valid.count()}")

# Preprocessing function

from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

def preprocess(df_in, set_name="train"):
	print(f"\n--- Preprocessing {set_name} ---")
	print(f"Initial samples: {df_in.count()}")
	# Noise filtering
	clean_text = regexp_replace(lower(col("text")), r"https?://\S+|www\.\S+", "")
	clean_text = regexp_replace(clean_text, r"<.*?>", "")
	clean_text = regexp_replace(clean_text, r"[^a-zA-Z\s]", "")
	df = df_in.withColumn("clean_text", clean_text)
	print(f"After clean_text: {df.count()}")
	# Drop NA and map label
	df = df.dropna(subset=["label", "clean_text"])
	print(f"After dropna: {df.count()}")
	df = df.withColumn("label", expr("CASE WHEN label = 'negative' THEN 0 WHEN label = 'neutral' THEN 1 ELSE 2 END"))
	print("Label distribution after mapping:")
	df.groupBy("label").count().show()
	# Tokenize and remove stopwords
	tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
	df = tokenizer.transform(df)
	df = stopwordsRemover.transform(df)
	print(f"After stopwords removal: {df.count()}")
	# Vocabulary reduction (per set)
	word_df = df.select(F.explode(col("filtered_words")).alias("word"))
	word_freq = word_df.groupBy("word").count()
	freq_words = word_freq.filter((col("count") >= MIN_FREQ) & (col("count") <= MAX_FREQ)).select("word")
	freq_words_list = [row["word"] for row in freq_words.collect()]
	print(f"Vocabulary size after reduction ({set_name}): {len(freq_words_list)}")
	print(f"Top 10 words: {freq_words_list[:10]}")
	def filter_tokens(tokens):
		return [t for t in tokens if t in freq_words_list]
	filter_tokens_udf = udf(filter_tokens, ArrayType(StringType()))
	df = df.withColumn("filtered_words_final", filter_tokens_udf(col("filtered_words")))
	# Remove short sentences
	df = df.filter(size(col("filtered_words_final")) >= MIN_SENT_LEN)
	print(f"After removing short sentences: {df.count()}")
	print("Label distribution after filtering:")
	df.groupBy("label").count().show()
	return df, freq_words_list

train_df, train_vocab = preprocess(df_train, "train")
test_df, test_vocab = preprocess(df_valid, "test")

# Compare vocabularies
print(f"\n--- Vocabulary overlap between train and test ---")
overlap = set(train_vocab) & set(test_vocab)
print(f"Number of overlapping words: {len(overlap)}")
print(f"Sample overlapping words: {list(overlap)[:10]}")

print(f"Train samples after preprocessing: {train_df.count()}")
print(f"Test samples after preprocessing: {test_df.count()}")

# Print label distribution for debugging
print("Train label distribution:")
train_df.groupBy("label").count().show()
print("Test label distribution:")
test_df.groupBy("label").count().show()

# TF-IDF with reduced dimensionality
hashingTF = HashingTF(inputCol="filtered_words_final", outputCol="raw_features", numFeatures=NUM_FEATURES)
idf = IDF(inputCol="raw_features", outputCol="features")

# Build pipeline
lr = LogisticRegression(maxIter=2, regParam=0.001, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[hashingTF, idf, lr])

# Train and evaluate
train_start = time.time()
model = pipeline.fit(train_df)
train_time = time.time() - train_start

eval_start = time.time()
predictions = model.transform(test_df)
eval_time = time.time() - eval_start


# Tính macro F1-score
from pyspark.sql import DataFrame
import numpy as np

def compute_macro_f1(pred_df: DataFrame, label_col="label", pred_col="prediction"):
	labels = [0, 1, 2]  # negative, neutral, positive
	f1s = []
	for label in labels:
		tp = pred_df.filter((col(label_col) == label) & (col(pred_col) == label)).count()
		fp = pred_df.filter((col(label_col) != label) & (col(pred_col) == label)).count()
		fn = pred_df.filter((col(label_col) == label) & (col(pred_col) != label)).count()
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
		f1s.append(f1)
	macro_f1 = float(np.mean(f1s))
	return macro_f1

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1 = compute_macro_f1(predictions)

# Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_app1_2_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write(f"Model training time: {train_time:.4f} seconds\n")
	f.write(f"Model evaluation time: {eval_time:.4f} seconds\n")
	f.write(f"Test Accuracy: {accuracy:.4f}\n")
	f.write(f"Test F1 Score: {f1:.4f}\n")

# Also print to console
print(f"Model training time: {train_time:.4f} seconds")
print(f"Model evaluation time: {eval_time:.4f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")