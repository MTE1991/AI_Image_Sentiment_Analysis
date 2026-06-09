import os
import re
import html
import pandas as pd
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# =====================================================================
# SYSTEM CONFIGURATIONS
# =====================================================================
POSTS_FILE = "insta_posts_on_ai_art.csv"
COMMENTS_FILE = "insta_comments_on_ai_art.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 16
})

print(f"--- INITIALIZING GROUP B8 MULTI-MODAL PIPELINE ---")
print(f"Hardware compute target: {DEVICE.upper()}")

# =====================================================================
# PHASE 1: DATA INGESTION & TEXT NORMALIZATION
# =====================================================================
print("\n[1/4] Executing Phase 1: Data Merging & Preprocessing...")

# Load and merge
posts_df = pd.read_csv(POSTS_FILE).rename(columns={'url': 'postUrl'})
comments_df = pd.read_csv(COMMENTS_FILE)
df = pd.merge(comments_df, posts_df, on='postUrl', how='inner', suffixes=('_comment', '_post'))

# Structural text cleaner
def rigorous_clean_text(text):
    if not isinstance(text, str): return ""
    cleaned = html.unescape(text) # Decode HTML
    cleaned = re.sub(r'http\s+://\S+|https://\S+|www\.\S+', '', cleaned) # Strip URLs
    cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Collapse whitespace
    return cleaned

df = df.dropna(subset=['text'])
df['cleaned_text'] = df['text'].apply(rigorous_clean_text)
df = df[df['cleaned_text'].str.strip() != ""] # Drop newly empty rows
print(f"Data normalization complete. Total operational rows: {len(df)}")

# =====================================================================
# PHASE 2: ROBERTA DEEP LEARNING INFERENCE
# =====================================================================
print(f"\n[2/4] Executing Phase 2: Deploying {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

ROBERTA_LABELS = ['Negative', 'Neutral', 'Positive']

def compute_roberta_metrics(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        
    probabilities = softmax(outputs.logits[0].cpu().numpy())
    predicted_idx = np.argmax(probabilities)
    
    # Calculate a continuous, weighted compound score from -1.0 to +1.0
    weighted_compound = (-1.0 * probabilities[0]) + (0.0 * probabilities[1]) + (1.0 * probabilities[2])
    
    return pd.Series([ROBERTA_LABELS[predicted_idx], probabilities[predicted_idx], weighted_compound])

print("Running sequence classification. This may take a moment depending on your CPU/GPU...")
df[['sentiment', 'confidence', 'compound_score']] = df['cleaned_text'].apply(compute_roberta_metrics)
df['comment_char_length'] = df['cleaned_text'].astype(str).apply(len)

print("Sentiment Distributions Extracted:")
print(df['sentiment'].value_counts())

# Save deep learning matrix to disk
df.to_csv("FINAL_instagram_roberta_dataset.csv", index=False)

# =====================================================================
# PHASE 3: METRIC AGGREGATION (BEHAVIORAL MAPPING)
# =====================================================================
print("\n[3/4] Executing Phase 3: Aggregating Post-Level Metrics...")

post_metrics = df.groupby('postUrl').agg({
    'compound_score': 'mean', 
    'likesCount': 'first',
    'commentsCount': 'first',
    'sentiment': lambda x: x.value_counts().index[0] 
}).reset_index()

# =====================================================================
# PHASE 4: MANUSCRIPT VISUALIZATION GENERATION (MODIFIED FOR SEPARATE EXPORTS)
# =====================================================================
print("\n[4/4] Executing Phase 4: Compiling Scientific Figures...")

color_palette = {'Positive': '#4CAF50', 'Neutral': '#9E9E9E', 'Negative': '#F44336'}
existing_sentiments = df['sentiment'].unique()
current_palette = {k: color_palette.get(k, '#000000') for k in existing_sentiments}

# --- GENERATE FIGURE 1 ---
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=post_metrics, x='compound_score', y='likesCount', hue='sentiment',
                palette=current_palette, s=140, edgecolor='black', alpha=0.8, ax=axes1[0])
axes1[0].set_title("A: Post Appreciation (Likes) vs. Mean RoBERTa Valence", fontweight='bold')
axes1[0].set_xlabel("Mean Comment Sentiment Compound (-1.0 to +1.0)")
axes1[0].set_ylabel("Total Post Likes Count")

sns.scatterplot(data=post_metrics, x='compound_score', y='commentsCount', hue='sentiment',
                palette=current_palette, s=140, edgecolor='black', alpha=0.8, ax=axes1[1])
axes1[1].set_title("B: Discursive Volatility (Comments) vs. Mean RoBERTa Valence", fontweight='bold')
axes1[1].set_xlabel("Mean Comment Sentiment Compound (-1.0 to +1.0)")
axes1[1].set_ylabel("Total Post Comments Volume")

plt.suptitle("Instagram User Engagement Metrics mapped against Deep Learning Sentiment", y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig("FINAL_FIG1_Behavior_vs_Sentiment.png", dpi=300, bbox_inches='tight')
plt.close(fig1) # Clear memory for Fig 1

# --- GENERATE FIGURE 2 ---
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))

# Plot B1: Discursive Depth
sns.boxplot(data=df, x='sentiment', y='comment_char_length', palette=current_palette, ax=axes2[0])
axes2[0].set_title("A: Discursive Depth (Comment Length by Sentiment)", fontweight='bold')
axes2[0].set_xlabel("Sentiment Class")
axes2[0].set_ylabel("Comment Length (Characters)")
axes2[0].set_ylim(0, df['comment_char_length'].quantile(0.95))

# Plot B2: 100% Stacked Composition
top_posts = df['postUrl'].value_counts().nlargest(10).index
df_top = df[df['postUrl'].isin(top_posts)]
composition = pd.crosstab(df_top['postUrl'], df_top['sentiment'], normalize='index') * 100
composition.index = [f"Post {i+1}" for i in range(len(composition))]

composition.plot(kind='bar', stacked=True, color=[current_palette.get(c) for c in composition.columns], ax=axes2[1], edgecolor='black')
axes2[1].set_title("B: Sentiment Composition Across Top 10 Active Posts", fontweight='bold')
axes2[1].set_xlabel("Anonymized Post ID")
axes2[1].set_ylabel("Percentage of Comments (%)")
axes2[1].tick_params(axis='x', rotation=45)
axes2[1].legend(title="Sentiment Class", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle("Advanced Linguistic and Compositional Sentiment Analytics", y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig("FINAL_FIG2_Advanced_Analytics.png", dpi=300, bbox_inches='tight')
plt.close(fig2) # Clear memory for Fig 2

# --- GENERATE FIGURE 3: Vocabulary Analysis ---
print("[4/4] Generating WordCloud Visualization...")

# Define custom stopwords to ensure cleaner thematic focus
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['post', 'ai', 'image', 'generated', 'using', 'will'])

# Filter data for Negative and Positive sentiments
pos_text = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
neg_text = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])

fig3, axes3 = plt.subplots(2, 1, figsize=(16, 7))

# Create and plot WordClouds
wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
axes3[0].imshow(wc_pos, interpolation='bilinear')
axes3[0].set_title("Positive Sentiment Lexicon", fontweight='bold')
axes3[0].axis('off')

wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)
axes3[1].imshow(wc_neg, interpolation='bilinear')
axes3[1].set_title("Negative Sentiment Lexicon", fontweight='bold')
axes3[1].axis('off')

plt.tight_layout()
plt.savefig("FINAL_FIG3_Vocabulary_Analysis.png", dpi=300, bbox_inches='tight')
plt.close(fig3)

print("\n=== PIPELINE SUCCESS ===")
