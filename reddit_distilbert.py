import os
import html
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Set visualization styles for an academic manuscript
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Initialize NLTK Assets
nltk.download('vader_lexicon', quiet=True)

# Define Core Configurations - Swapped to a lightweight, fast DistilBERT student model
DATASET_PATH = "preprocessed_manuscript_data.csv"
TRANSFORMER_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Executing pipeline utilizing compute device: {DEVICE.upper()}")

# =====================================================================
# 1. DATA INGESTION & ADVANCED PREPROCESSING
# =====================================================================
print("\n--- Phase 1: Ingestion & Text Normalization ---")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Source dataset target path '{DATASET_PATH}' not found.")

df = pd.read_csv(DATASET_PATH)

# Drop explicit missing value vectors
df = df.dropna(subset=['body'])

# Normalize string configurations and decode HTML entities
df['cleaned_text'] = df['body'].apply(lambda x: html.unescape(str(x)))

# Filter out low-density semantic data noise (comments <= 3 words)
df = df[df['cleaned_text'].apply(lambda x: len(x.split()) > 3)].copy()
print(f"Data ingestion successful. Total corpus density: {len(df)} operational text rows.")

# =====================================================================
# 2. BASELINE SENTIMENT EXTRACTION ENGINE (VADER)
# =====================================================================
print("\n--- Phase 2: Compiling Lexicon Baseline Engine (VADER) ---")
vader_analyzer = SentimentIntensityAnalyzer()

df['vader_compound'] = df['cleaned_text'].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])

def classify_vader(score):
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    return 'Neutral'

df['vader_sentiment'] = df['vader_compound'].apply(classify_vader)

# =====================================================================
# 3. LIGHTWEIGHT TRANSFORMER SENTIMENT ENGINE (DistilBERT)
# =====================================================================
print(f"\n--- Phase 3: Deploying DistilBERT Architecture from HF ({TRANSFORMER_MODEL_NAME}) ---")
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_NAME).to(DEVICE)
transformer_model.eval() # Freeze layers for pure inference mode

# Map labels according to the specific configuration schema of lxyuan's DistilBERT student card
# Index positions: 0 -> positive, 1 -> neutral, 2 -> negative
DISTILBERT_LABELS = ['Positive', 'Neutral', 'Negative']

def compute_distilbert_sentiment(text):
    # Enforce text truncation within standard 512-token sequence lengths
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    
    # Process logits back to CPU space and resolve probabilistic arrays via Softmax
    logits = outputs.logits[0].cpu().numpy()
    probabilities = softmax(logits)
    
    predicted_idx = np.argmax(probabilities)
    
    # Return structured metadata mapping label name and confidence probability
    return pd.Series([DISTILBERT_LABELS[predicted_idx], probabilities[predicted_idx]])

# Apply high-velocity inference loop over the corpus
print("Computing deep-learning sequence classifications via DistilBERT CPU engine...")
df[['distilbert_sentiment', 'distilbert_confidence']] = df['cleaned_text'].apply(compute_distilbert_sentiment)

# =====================================================================
# 4. ASPECT-BASED SENTIMENT ANALYSIS (ABSA) MAPPING ENGINE
# =====================================================================
print("\n--- Phase 4: Constructing Domain-Specific ABSA Matrix ---")

# Operationalize cross-cutting academic aspects identified in your literature review
ASPECT_TAXONOMY = {
    'Ethics_Legality': ['theft', 'copyright', 'stolen', 'plagiarism', 'ethics', 'unethical', 'scam', 'fraud', 'consent'],
    'Technology_Utility': ['model', 'v6', 'tool', 'prompt', 'software', 'generator', 'algorithm', 'midjourney', 'chatgpt'],
    'Aesthetics_Quality': ['beautiful', 'slop', 'looks', 'artistic', 'style', 'aesthetic', 'ugly', 'rendering', 'pretty'],
    'Socio_Economics': ['jobs', 'artists', 'livelihood', 'work', 'money', 'career', 'industry', 'market', 'replace']
}

def map_aspect_sentiments(text, sentiment_score):
    normalized_text = text.lower()
    scores = {}
    for aspect, terms in ASPECT_TAXONOMY.items():
        # Map localized sentence score if aspect keyword features exist inside the string
        if any(term in normalized_text for term in terms):
            scores[aspect] = sentiment_score
        else:
            scores[aspect] = np.nan # Assign null vector if aspect is omitted
    return pd.Series(scores)

# Use DistilBERT compound weight proxies (normalized scaling from -1 to 1) for ABSA mapping
df['distilbert_numeric_score'] = df['distilbert_sentiment'].map({'Negative': -1.0, 'Neutral': 0.0, 'Positive': 1.0})
aspect_matrix = df.apply(lambda row: map_aspect_sentiments(row['cleaned_text'], row['distilbert_numeric_score']), axis=1)
df = pd.concat([df, aspect_matrix], axis=1)

# Persist finalized analytical model outputs to filesystem
df.to_csv("manuscript_analyzed_results.csv", index=False)
print("Pipeline datasets serialized successfully to: 'manuscript_analyzed_results.csv'")

# =====================================================================
# 5. SCIENTIFIC VISUALIZATION GENERATION (EXPANDED MANUSCRIPT EDITION)
# =====================================================================
print("\n--- Phase 5: Generating Manuscript Figures ---")
community_order = ['r/DefendingAIArt', 'r/aiArt', 'r/antiai']
existing_communities = [c for c in community_order if c in df['communityName'].unique()]

# ---------------------------------------------------------------------
# FIG 1: Cross-Sectional Comparative Model Classifications
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

sns.countplot(data=df, x='communityName', hue='vader_sentiment', order=existing_communities,
              palette=['#4CAF50', '#9E9E9E', '#F44336'], hue_order=['Positive', 'Neutral', 'Negative'], ax=axes[0])
axes[0].set_title("A: Baseline Lexicon (VADER) Distribution", fontweight='bold')
axes[0].set_xlabel("Community")
axes[0].set_ylabel("Observation Frequency Count")
axes[0].tick_params(axis='x', rotation=15)

sns.countplot(data=df, x='communityName', hue='distilbert_sentiment', order=existing_communities,
              palette=['#4CAF50', '#9E9E9E', '#F44336'], hue_order=['Positive', 'Neutral', 'Negative'], ax=axes[1])
axes[1].set_title("B: Contextual Transformer (DistilBERT) Distribution", fontweight='bold')
axes[1].set_xlabel("Community")
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle("Model Sentiment Classification Polarization Matrix across Target Subreddits", y=1.02)
plt.tight_layout()
plt.savefig("manuscript_fig1_sentiment_polarization.png", dpi=300, bbox_inches='tight')
print("Generated Figure 1: 'manuscript_fig1_sentiment_polarization.png'")


# ---------------------------------------------------------------------
# FIG 2: Granular Heatmap of Aspect-Based Sentiments
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 6))
heatmap_data = df.groupby('communityName')[list(ASPECT_TAXONOMY.keys())].mean()

sns.heatmap(heatmap_data.reindex(existing_communities), annot=True, cmap="RdYlGn", vmin=-0.6, vmax=0.6, 
            linewidths=.5, cbar_kws={'label': 'Mean Sentiment Valence Metric (-1.0 to +1.0)'})
plt.title("Granular Structural Heatmap of Aspect-Based Sentiment Scores", fontweight='bold', pad=15)
plt.ylabel("Subreddit Target Segment")
plt.xlabel("Linguistic Aspect Domain")
plt.tight_layout()
plt.savefig("manuscript_fig2_absa_heatmap.png", dpi=300, bbox_inches='tight')
print("Generated Figure 2: 'manuscript_fig2_absa_heatmap.png'")


# ---------------------------------------------------------------------
# FIG 3: WordCloud Analysis isolating r/antiai Ethical/Economic Anxieties
# ---------------------------------------------------------------------
antiai_neg_corpus = " ".join(
    text for text in df[(df['communityName'] == 'r/antiai') & (df['distilbert_sentiment'] == 'Negative')]['cleaned_text']
).strip()

if len(antiai_neg_corpus) > 0:
    academic_stopwords = set(STOPWORDS).union(["ai", "art", "image", "like", "people", "would", "one", "get", "even", "think", "reddit", "comments", "sub", "post"])

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        max_words=80, 
        stopwords=academic_stopwords, 
        colormap='copper'
    ).generate(antiai_neg_corpus)
        
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Linguistic Extraction of Core Public Anxieties inside Counter-AI Spaces", fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig("manuscript_fig3_anxiety_wordcloud.png", dpi=300, bbox_inches='tight')
    print("Generated Figure 3: 'manuscript_fig3_anxiety_wordcloud.png'")
else:
    print("[SYSTEM NOTICE] Skipping Figure 3: Negative corpus density yields zero values inside r/antiai slice.")


# ---------------------------------------------------------------------
# NEW FIG 4: Transformer Model Prediction Confidence Threshold Distribution
# ---------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df, 
    x='communityName', 
    y='distilbert_confidence', 
    order=existing_communities,
    palette="Pastel1"
)
plt.title("Distribution of DistilBERT Prediction Confidence Across Communities", fontweight='bold', pad=12)
plt.xlabel("Subreddit Community Target")
plt.ylabel("Classifier Prediction Confidence (0.0 - 1.0)")
plt.ylim(0.4, 1.05)
plt.tight_layout()
plt.savefig("manuscript_fig4_model_confidence.png", dpi=300, bbox_inches='tight')
print("Generated Figure 4: 'manuscript_fig4_model_confidence.png'")


# ---------------------------------------------------------------------
# NEW FIG 5: Absolute Topic Volume/Salience Density (Stacked Bar Chart)
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Count non-null entries for each aspect grouped by community
volume_data = df.groupby('communityName')[list(ASPECT_TAXONOMY.keys())].count()
volume_data = volume_data.reindex(existing_communities)

# Calculate percentages to show relative domain prominence per community
volume_pct = volume_data.div(volume_data.sum(axis=1), axis=0) * 100

volume_pct.plot(
    kind='bar', 
    stacked=True, 
    ax=plt.gca(), 
    colormap='viridis', 
    edgecolor='black', 
    linewidth=0.7
)

plt.title("Relative Domain Salience: Distribution of Aspect Discussion Volume", fontweight='bold', pad=12)
plt.xlabel("Subreddit Community")
plt.ylabel("Share of Total Extracted Aspect Mentions (%)")
plt.xticks(rotation=15)
plt.legend(title="Linguistic Aspect Vector", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("manuscript_fig5_aspect_salience_volume.png", dpi=300, bbox_inches='tight')
print("Generated Figure 5: 'manuscript_fig5_aspect_salience_volume.png'")

print("\n--- Pipeline Complete. Expanded Technical Manuscript Artifacts Compiled and Saved. ---")