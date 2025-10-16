import sys
import os
import re
from gensim.models import Word2Vec
import numpy as np
from typing import List

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src', 'representations')
sys.path.insert(0, src_dir)


class EWTDataStreamer:
    """Data loader for EWT corpus."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_sentences(self) -> List[List[str]]:
        """Load sentences from EWT corpus file."""
        if not os.path.exists(self.file_path):
            print(f"Error: File not found: {self.file_path}")
            return []
            
        sentences = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = re.findall(r'\b[a-zA-Z]+\b', line.lower())
                    if len(tokens) >= 3:
                        sentences.append(tokens)
        
        return sentences


def train_word2vec_model(data_path: str, model_save_path: str) -> Word2Vec:
    """Train a Word2Vec model on the EWT corpus."""
    
    # Load training data
    streamer = EWTDataStreamer(data_path)
    sentences = streamer.load_sentences()
    
    if not sentences:
        print("No training data available!")
        return None
    
    print(f"Training on {len(sentences)} sentences...")
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10
    )
    
    print(f"Training completed. Vocabulary: {len(model.wv.key_to_index):,} words")
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return model


def demonstrate_model_usage(model: Word2Vec):
    """Demonstrate the usage of the trained Word2Vec model."""
    
    print(f"\nModel demo - Vocab: {len(model.wv.key_to_index):,}, Dims: {model.wv.vector_size}")
    
    # Test words
    test_words = ["the", "man", "woman", "king", "good", "bad"]
    available_words = [word for word in test_words if word in model.wv.key_to_index]
    
    if len(available_words) >= 2:
        # Word similarities
        print("Similarities:")
        for i in range(min(2, len(available_words)-1)):
            word1, word2 = available_words[i], available_words[i+1]
            try:
                similarity = model.wv.similarity(word1, word2)
                print(f"  {word1}-{word2}: {similarity:.3f}")
            except KeyError:
                pass
    
    # Most similar words
    if available_words:
        word = available_words[0]
        try:
            similar = model.wv.most_similar(word, topn=3)
            print(f"Similar to '{word}': {', '.join([w for w, s in similar])}")
        except KeyError:
            pass
    
    # Analogy
    if "man" in model.wv.key_to_index and "woman" in model.wv.key_to_index and "king" in model.wv.key_to_index:
        try:
            result = model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
            if result:
                answer, score = result[0]
                print(f"Analogy man:woman :: king:{answer} ({score:.3f})")
        except:
            pass


def main():
    """Main function to run the embedding training demo."""
    
    print("Lab 4: Custom Word2Vec Training Demo")
    
    # File paths
    data_path = os.path.join("data", "UD_English-EWT", "en_ewt-ud-train.txt")
    model_save_path = os.path.join("results", "word2vec_ewt.model")
    
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        return
    
    try:
        # Train the model
        model = train_word2vec_model(data_path, model_save_path)
        
        if model:
            # Demonstrate usage
            demonstrate_model_usage(model)
            print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()