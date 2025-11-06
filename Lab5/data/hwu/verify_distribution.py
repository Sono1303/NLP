"""
Script để verify category distribution giữa dữ liệu gốc và augmented
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df_original = pd.read_csv('train.csv')
df_augmented = pd.read_csv('train_augmented_10k.csv')
df_combined = pd.read_csv('train_combined.csv')

print(f"Original: {len(df_original):,}")
print(f"Augmented: {len(df_augmented):,}")
print(f"Combined: {len(df_combined):,}")
print(f"Categories: {df_original['category'].nunique()}")

# Tính phân phối
original_dist = df_original['category'].value_counts()
augmented_dist = df_augmented['category'].value_counts()
combined_dist = df_combined['category'].value_counts()

# Tạo DataFrame so sánh
comparison_df = pd.DataFrame({
    'Original Count': original_dist,
    'Original %': (original_dist / len(df_original) * 100).round(2),
    'Augmented Count': augmented_dist,
    'Augmented %': (augmented_dist / len(df_augmented) * 100).round(2),
    'Difference %': ((augmented_dist / len(df_augmented) * 100) - 
                     (original_dist / len(df_original) * 100)).round(2)
})

print("\nCategory distribution (Top 15):")
print(comparison_df.head(15).to_string())

mean_diff = abs(comparison_df['Difference %']).mean()
max_diff = abs(comparison_df['Difference %']).max()
print(f"\nMean abs diff: {mean_diff:.4f}% | Max abs diff: {max_diff:.4f}%")
print(f"Distribution preserved: {'YES' if mean_diff < 0.5 else 'NO'}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 20 categories comparison
top_20_cats = original_dist.head(20).index
comparison_top20 = comparison_df.loc[top_20_cats]

ax1 = axes[0, 0]
x = range(len(comparison_top20))
width = 0.35
ax1.bar([i - width/2 for i in x], comparison_top20['Original %'], 
        width, label='Original', color='skyblue', alpha=0.8)
ax1.bar([i + width/2 for i in x], comparison_top20['Augmented %'], 
        width, label='Augmented', color='salmon', alpha=0.8)
ax1.set_xlabel('Category')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Top 20 Categories: Original vs Augmented Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_top20.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Scatter plot - Original % vs Augmented %
ax2 = axes[0, 1]
ax2.scatter(comparison_df['Original %'], comparison_df['Augmented %'], alpha=0.6)
ax2.plot([0, comparison_df['Original %'].max()], 
         [0, comparison_df['Original %'].max()], 
         'r--', label='Perfect match')
ax2.set_xlabel('Original Distribution (%)')
ax2.set_ylabel('Augmented Distribution (%)')
ax2.set_title('Distribution Preservation Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution difference
ax3 = axes[1, 0]
sorted_diff = comparison_df.sort_values('Difference %', ascending=False)
colors = ['red' if x > 0 else 'green' for x in sorted_diff['Difference %']]
ax3.barh(range(len(sorted_diff)), sorted_diff['Difference %'], color=colors, alpha=0.6)
ax3.set_xlabel('Difference (%)')
ax3.set_ylabel('Category')
ax3.set_title('Distribution Difference (Augmented - Original)')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.grid(axis='x', alpha=0.3)
ax3.set_yticks([])  # Too many categories to show

# Plot 4: Combined data distribution (top 20)
ax4 = axes[1, 1]
combined_top20 = combined_dist.head(20)
ax4.bar(range(len(combined_top20)), combined_top20.values, color='lightgreen', alpha=0.8)
ax4.set_xlabel('Category')
ax4.set_ylabel('Count')
ax4.set_title(f'Combined Dataset Distribution (Total: {len(df_combined):,} samples)')
ax4.set_xticks(range(len(combined_top20)))
ax4.set_xticklabels(combined_top20.index, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(combined_top20.values):
    ax4.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: distribution_comparison.png")

plt.show()

print("\nSample augmented data:")
print(df_augmented[['text', 'category']].head(10).to_string(index=False))
print("\nVerification complete.")
