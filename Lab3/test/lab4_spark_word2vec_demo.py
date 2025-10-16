import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, size
from pyspark.ml.feature import Word2Vec, StopWordsRemover, Tokenizer
import pyspark.sql.functions as F

def create_spark_session():
    """Initialize Spark session."""
    spark = SparkSession.builder \
        .appName("Lab4_PySpark_Word2Vec") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark initialized")
    return spark

def load_and_preprocess_data(spark, data_path, sample_fraction=0.05):
    """Load and preprocess C4 dataset."""
    print("Loading data...")
    
    # Read and sample data
    df = spark.read.json(data_path).sample(fraction=sample_fraction, seed=42)
    
    # Clean and tokenize
    cleaned_df = df.select(
        regexp_replace(
            regexp_replace(lower(col("text")), r"[^\w\s]", ""), 
            r"\s+", " "
        ).alias("cleaned_text")
    ).filter(F.length(col("cleaned_text")) > 50)
    
    # Tokenize and remove stop words
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="raw_words")
    stop_remover = StopWordsRemover(inputCol="raw_words", outputCol="words")
    
    processed_df = stop_remover.transform(tokenizer.transform(cleaned_df))
    final_df = processed_df.filter(size(col("words")) >= 3).select("words")
    
    count = final_df.count()
    print(f"Processed {count:,} documents")
    return final_df

def train_word2vec_model(df, vector_size=100, min_count=3, max_iter=5):
    """Train Word2Vec model."""
    print("Training Word2Vec model...")
    
    word2vec = Word2Vec(
        inputCol="words", 
        outputCol="features",
        vectorSize=vector_size,
        minCount=min_count,
        maxIter=max_iter,
        seed=42
    )
    
    model = word2vec.fit(df)
    print("Training completed")
    return model

def demonstrate_model_usage(model, test_word="computer", top_n=5):
    """Demonstrate model capabilities."""
    vectors = model.getVectors()
    vocab_size = vectors.count()
    
    print(f"Vocabulary: {vocab_size:,} words")
    
    try:
        similar_words = model.findSynonymsArray(test_word, top_n)
        print(f"Similar to '{test_word}':")
        for i, (word, sim) in enumerate(similar_words, 1):
            print(f"  {i}. {word} ({sim:.3f})")
    except:
        print(f"'{test_word}' not in vocabulary")

def save_model_results(model):
    """Save vocabulary results."""
    try:
        os.makedirs("results", exist_ok=True)
        vectors = model.getVectors()
        vocab_list = [row["word"] for row in vectors.select("word").collect()]
        
        with open("results/spark_vocab_simple.txt", 'w', encoding='utf-8') as f:
            for word in sorted(vocab_list):
                f.write(f"{word}\n")
        
        print(f"Vocabulary saved: {len(vocab_list):,} words")
    except Exception as e:
        print(f"Save error: {e}")

def main():
    """Main function for PySpark Word2Vec training."""
    
    print("Lab 4: PySpark Word2Vec Training")
    
    data_path = "../spark_labs/data/c4-train.00000-of-01024-30K.json.gz"
    
    spark = None
    try:
        # Initialize and train
        spark = create_spark_session()
        
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
        
        # Process data and train model
        processed_df = load_and_preprocess_data(spark, data_path)
        processed_df.cache()
        
        model = train_word2vec_model(processed_df, vector_size=100)
        
        # Test similarity
        demonstrate_model_usage(model, test_word="computer", top_n=5)
        
        # Try alternative words
        test_words = ["data", "system", "technology", "the"]
        for word in test_words:
            try:
                similar = model.findSynonymsArray(word, 3)
                if similar:
                    print(f"Similar to '{word}': {', '.join([w for w, s in similar[:3]])}")
                    break
            except:
                continue
        
        # Save results
        save_model_results(model)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()