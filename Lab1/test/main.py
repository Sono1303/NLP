import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import RegexTokenizer
from core.dataset_loaders import load_raw_text_data

sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

simple_tokenizer = SimpleTokenizer()
regex_tokenizer = RegexTokenizer()

print("SimpleTokenizer Results:")
for s in sentences:
    print(f"Input: {s}")
    print(f"Tokens: {simple_tokenizer.tokenize(s)}\n")

print("RegexTokenizer Results:")
for s in sentences:
    print(f"Input: {s}")
    print(f"Tokens: {regex_tokenizer.tokenize(s)}\n")

dataset_path = r'E:\NLP\Lab1\UD_English-EWT\en_ewt-ud-train.txt'
raw_text = load_raw_text_data(dataset_path)

sample_text = raw_text[:500]

print(f"Original Sample: {sample_text[:100]}...\n")

simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

regex_tokens = regex_tokenizer.tokenize(sample_text)

print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

