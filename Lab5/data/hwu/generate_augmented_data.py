import pandas as pd
import numpy as np
import random
from collections import Counter

# Đọc dữ liệu gốc
df_train = pd.read_csv('train.csv')
print(f"Original data size: {len(df_train)}")
print(f"Number of categories: {df_train['category'].nunique()}")

# Phân tích category distribution
category_dist = df_train['category'].value_counts()
category_probs = category_dist / len(df_train)

print("\nCategory distribution:")
print(category_dist.head(10))
print(f"\nTotal samples to generate: 10,000")

# Tính số samples cần generate cho mỗi category
target_size = 10000
samples_per_category = {}
for cat in category_probs.index:
    samples_per_category[cat] = int(target_size * category_probs[cat])

# Điều chỉnh để đảm bảo tổng đúng 10,000
total_assigned = sum(samples_per_category.values())
diff = target_size - total_assigned

# Thêm phần dư vào category có nhiều samples nhất
if diff > 0:
    largest_cat = category_dist.index[0]
    samples_per_category[largest_cat] += diff

print(f"\nSamples per category (top 10):")
for cat in list(samples_per_category.keys())[:10]:
    print(f"  {cat}: {samples_per_category[cat]}")

# ==================== AUGMENTATION FUNCTIONS ====================

# Các từ đồng nghĩa đơn giản cho augmentation
SYNONYMS = {
    'set': ['create', 'make', 'add', 'schedule', 'put'],
    'alarm': ['alert', 'reminder', 'notification', 'wake-up call'],
    'remove': ['delete', 'cancel', 'clear', 'erase', 'take away'],
    'show': ['display', 'list', 'give me', 'tell me'],
    'please': ['could you', 'can you', 'would you', 'kindly'],
    'tomorrow': ['next day', 'the following day'],
    'today': ['this day', 'right now'],
    'meeting': ['appointment', 'conference', 'session', 'gathering'],
    'increase': ['raise', 'boost', 'turn up', 'make louder'],
    'decrease': ['lower', 'reduce', 'turn down', 'make quieter'],
    'volume': ['sound', 'audio', 'loudness'],
    'mute': ['silence', 'quiet', 'turn off sound'],
}

def synonym_replacement(text, n=1):
    """Thay thế n từ bằng từ đồng nghĩa"""
    words = text.split()
    new_words = words.copy()
    
    random_word_indices = list(range(len(words)))
    random.shuffle(random_word_indices)
    
    num_replaced = 0
    for idx in random_word_indices:
        word = words[idx].lower()
        if word in SYNONYMS and num_replaced < n:
            synonym = random.choice(SYNONYMS[word])
            new_words[idx] = synonym
            num_replaced += 1
            
    return ' '.join(new_words)

def random_insertion(text, n=1):
    """Chèn ngẫu nhiên n từ phổ biến vào câu"""
    words = text.split()
    common_words = ['please', 'just', 'now', 'today', 'really', 'very', 'quite']
    
    for _ in range(n):
        word_to_insert = random.choice(common_words)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, word_to_insert)
    
    return ' '.join(words)

def random_deletion(text, p=0.1):
    """Xóa ngẫu nhiên các từ với xác suất p"""
    words = text.split()
    
    if len(words) == 1:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)

def paraphrase_simple(text):
    """Paraphrase đơn giản bằng cách thay đổi cấu trúc câu"""
    # Thêm các biến thể câu hỏi
    question_starters = ['can you', 'could you', 'please', 'would you mind', 'i need to', 'i want to']
    
    text_lower = text.lower().strip()
    
    # Nếu câu chưa có question starter, thêm vào
    has_starter = any(text_lower.startswith(starter) for starter in question_starters)
    
    if not has_starter and random.random() > 0.5:
        starter = random.choice(question_starters)
        text = f"{starter} {text}"
    
    return text

def augment_text(text, category):
    """Augment text với nhiều kỹ thuật khác nhau"""
    aug_type = random.choice(['synonym', 'insert', 'delete', 'paraphrase', 'combination'])
    
    if aug_type == 'synonym':
        return synonym_replacement(text, n=random.randint(1, 2))
    elif aug_type == 'insert':
        return random_insertion(text, n=1)
    elif aug_type == 'delete':
        return random_deletion(text, p=0.1)
    elif aug_type == 'paraphrase':
        return paraphrase_simple(text)
    else:  # combination
        text = synonym_replacement(text, n=1)
        if random.random() > 0.5:
            text = random_insertion(text, n=1)
        return text

# ==================== GENERATE NEW DATA ====================

augmented_data = []

for category, num_samples in samples_per_category.items():
    # Lấy tất cả samples gốc của category này
    original_samples = df_train[df_train['category'] == category]['text'].tolist()
    
    print(f"\nGenerating {num_samples} samples for category: {category}")
    
    for i in range(num_samples):
        # Random chọn một sample gốc
        original_text = random.choice(original_samples)
        
        # Augment text
        augmented_text = augment_text(original_text, category)
        
        # Thêm vào list
        augmented_data.append({
            'text': augmented_text,
            'category': category
        })

# Tạo DataFrame mới
df_augmented = pd.DataFrame(augmented_data)

print(f"\n{'='*50}")
print(f"Generated {len(df_augmented)} new samples")
print(f"{'='*50}")

# Verify distribution
print("\nNew data category distribution:")
new_dist = df_augmented['category'].value_counts()
print(new_dist.head(10))

print("\nComparison with original distribution:")
comparison = pd.DataFrame({
    'Original %': category_probs * 100,
    'Generated %': (new_dist / len(df_augmented)) * 100
})
print(comparison.head(10))

# Lưu file
output_file = 'train_augmented_10k.csv'
df_augmented.to_csv(output_file, index=False)
print(f"\nSaved augmented data to: {output_file}")

# Combine với dữ liệu gốc (tùy chọn)
df_combined = pd.concat([df_train, df_augmented], ignore_index=True)
combined_file = 'train_combined.csv'
df_combined.to_csv(combined_file, index=False)
print(f"Saved combined data (original + augmented) to: {combined_file}")
print(f"Total samples: {len(df_combined)}")

print("\n" + "="*50)
print("DONE!")
print("="*50)
