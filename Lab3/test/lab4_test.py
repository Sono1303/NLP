import sys
import os
import numpy as np

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src', 'representations')
sys.path.insert(0, src_dir)

from word_embedder import WordEmbedder

def main():
    """Main test function."""
    
    print("Lab 4: WordEmbedder Test")
    
    # Initialize WordEmbedder
    try:
        embedder = WordEmbedder('glove-wiki-gigaword-50')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 1. Get vector for 'king'
    print("\n1. Get vector for 'king':")
    king_vector = embedder.get_vector('king')
    if king_vector is not None:
        print(f"Shape: {king_vector.shape}")
        print(f"First 5 values: {king_vector[:5]}")
    else:
        print("'king' not found")
    
    # 2. Word similarities
    print("\n2. Word similarities:")
    king_queen_sim = embedder.get_similarity('king', 'queen')
    king_man_sim = embedder.get_similarity('king', 'man')
    
    if king_queen_sim and king_man_sim:
        print(f"king-queen: {king_queen_sim:.4f}")
        print(f"king-man: {king_man_sim:.4f}")
    
    # 3. Most similar words to 'computer'
    print("\n3. Most similar to 'computer':")
    similar_words = embedder.get_most_similar('computer', top_n=10)
    for i, (word, sim) in enumerate(similar_words[:5], 1):
        print(f"{i}. {word} ({sim:.3f})")
    
    # 4. Document embedding
    print("\n4. Document embedding:")
    sentence = "The queen rules the country."
    doc_vector = embedder.embed_document(sentence)
    print(f"Sentence: '{sentence}'")
    print(f"Vector shape: {doc_vector.shape}")
    print(f"First 5 values: {doc_vector[:5]}")
    print(f"Is zero vector: {np.allclose(doc_vector, 0)}")
    
    print("\nTest completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")