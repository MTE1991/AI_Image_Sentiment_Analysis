## Overview

These three scripts collectively form a **multi-platform sentiment analysis pipeline** for social media research, specifically analyzing public discourse around AI-generated art. The pipeline processes data from **Instagram** and **Reddit**, applying both lexicon-based and transformer-based deep learning models to extract sentiment, generate visualizations, and produce academic manuscript-ready figures.

---

## Script 1: `insta_sentiment.py` - Instagram Sentiment Analysis Pipeline

### Purpose
Analyzes Instagram comments on AI art posts using **RoBERTa** (a robustly optimized BERT approach) for sentiment classification and correlates findings with engagement metrics (likes, comments).

### Key Components

#### Data Processing
- **Input Files**: `insta_posts_on_ai_art.csv` (posts with URLs) and `insta_comments_on_ai_art.csv` (comments)
- **Merge Strategy**: Inner join on `postUrl` to link comments to their parent posts
- **Text Normalization Function** (`rigorous_clean_text`):
  - Decodes HTML entities (`&amp;` → `&`, `&#39;` → `'`)
  - Removes URLs using regex patterns
  - Collapses multiple whitespace characters

#### Sentiment Model - RoBERTa
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - a Twitter-trained RoBERTa variant optimized for short social media text
- **Labels**: Negative, Neutral, Positive (3-class classification)
- **Output Metrics**:
  - `sentiment`: Predicted class label
  - `confidence`: Softmax probability of the predicted class
  - `compound_score`: Weighted continuous score from -1.0 (negative) to +1.0 (positive), calculated as: `(-1.0 * p_neg) + (0.0 * p_neu) + (1.0 * p_pos)`

#### Aggregation & Analysis
- **Post-Level Metrics**: Groups by `postUrl` to calculate:
  - Mean compound sentiment score per post
  - Dominant sentiment (mode)
  - Original likes and comments counts
- **Comment Length Analysis**: Character-length of cleaned comments as a proxy for "discursive depth"

#### Visualization Outputs (300 DPI, manuscript-ready)

| Figure | Content | Purpose |
|--------|---------|---------|
| `FINAL_FIG1_Behavior_vs_Sentiment.png` | Two scatterplots: Likes vs. sentiment, Comments vs. sentiment | Maps user engagement against comment sentiment |
| `FINAL_FIG2_Advanced_Analytics.png` | Boxplot of comment length by sentiment + stacked bar chart of sentiment composition for top 10 active posts | Analyzes discursive depth and post-level sentiment diversity |
| `FINAL_FIG3_Vocabulary_Analysis.png` | Side-by-side word clouds for positive and negative sentiments | Visualizes contrasting lexicons |

#### Hardware Optimization
- Automatically detects CUDA GPU availability; falls back to CPU
- Uses `model.eval()` and `torch.no_grad()` for inference-only mode

---

## Script 2: `reddit_preprocess.py` - Reddit Data Cleaning Pipeline

### Purpose
Preprocesses raw Reddit scrape data from Apify (a web scraping platform) into a clean, normalized format suitable for downstream sentiment analysis.

### Key Operations

#### Input
- `dataset_reddit-scraper-lite_2026-06-05_16-09-06-559.csv` - Raw Apify export containing Reddit posts/comments

#### Cleaning Pipeline (`rigorous_clean_text` function)

| Step | Operation | Example Transformation |
|------|-----------|------------------------|
| 1 | HTML unescape | `&amp;` → `&`, `&#39;` → `'` |
| 2 | URL removal | `https://redd.it/abc123` → (removed) |
| 3 | Reddit user mention removal | `/u/username` → (removed) |
| 4 | Subreddit reference removal | `/r/subreddit` → (removed) |
| 5 | Markdown artifact removal | `[link]`, `[comments]`, `submitted by` → (removed) |
| 6 | Whitespace collapse | Multiple spaces/line breaks → single space |

#### Quality Filtering
- Drops rows with empty/missing `body` field
- Removes comments with **≤ 3 words** (eliminates low-information content like "lol", "nice", "agree")

#### Output
- `preprocessed_manuscript_data.csv` - Cleaned, filtered corpus ready for analysis

---

## Script 3: `reddit_distilbert.py` - Multi-Model Reddit Analysis

### Purpose
Performs comparative sentiment analysis on Reddit data using **two distinct models** (VADER lexicon + DistilBERT transformer) plus **aspect-based sentiment analysis (ABSA)**. Generates expanded manuscript figures (5 total).

### Three-Tier Analysis Architecture

#### Tier 1: VADER (Lexicon Baseline)
- **Method**: Rule-based sentiment analyzer from NLTK
- **Output**: `vader_compound` score (-1.0 to +1.0)
- **Classification Thresholds**:
  - `≥ 0.05` → Positive
  - `≤ -0.05` → Negative
  - Between → Neutral
- **Purpose**: Provides a lightweight, interpretable baseline for comparison

#### Tier 2: DistilBERT (Transformer Model)
- **Model**: `lxyuan/distilbert-base-multilingual-cased-sentiments-student` - A distilled, faster variant of BERT
- **Advantages over RoBERTa**: Lighter weight, faster inference, multilingual support
- **Outputs**:
  - `distilbert_sentiment`: Positive/Neutral/Negative
  - `distilbert_confidence`: Prediction probability (0.0-1.0)
  - `distilbert_numeric_score`: Converted to continuous (-1, 0, +1)

#### Tier 3: Aspect-Based Sentiment Analysis (ABSA)
- **Aspect Taxonomy**:

| Aspect | Keywords | Focus |
|--------|----------|-------|
| Ethics_Legality | theft, copyright, stolen, plagiarism, ethics, scam, consent | Legal/moral concerns |
| Technology_Utility | model, v6, tool, prompt, generator, algorithm, midjourney | Technical capabilities |
| Aesthetics_Quality | beautiful, slop, artistic, style, ugly, rendering | Visual evaluation |
| Socio_Economics | jobs, artists, livelihood, work, money, career, industry | Economic impact |

- **Method**: For each comment, checks if any aspect keyword appears; if yes, assigns the DistilBERT numeric sentiment score to that aspect; otherwise assigns `NaN`
- **Output**: Four new columns (`Ethics_Legality`, `Technology_Utility`, `Aesthetics_Quality`, `Socio_Economics`)

### Visualization Outputs (5 Figures)

| Figure | Type | Content |
|--------|------|---------|
| `manuscript_fig1_sentiment_polarization.png` | Side-by-side count plots | VADER vs. DistilBERT sentiment distributions across 3 subreddits (r/DefendingAIArt, r/aiArt, r/antiai) |
| `manuscript_fig2_absa_heatmap.png` | Heatmap | Mean aspect sentiment scores by subreddit (Red-Yellow-Green colormap, -0.6 to +0.6 range) |
| `manuscript_fig3_anxiety_wordcloud.png` | Word cloud | Negative lexicon from r/antiai only (copper colormap, custom stopwords including "ai", "art", "reddit") |
| `manuscript_fig4_model_confidence.png` | Boxplot | DistilBERT confidence distribution across communities (y-axis: 0.4-1.05) |
| `manuscript_fig5_aspect_salience_volume.png` | Stacked bar chart | Relative percentage of aspect mentions per community (normalized by row sums) |

### Subreddit Targeting
- Specifically filters for three communities (order maintained for consistent visualization):
  1. `r/DefendingAIArt` - Pro-AI art advocates
  2. `r/aiArt` - Neutral/enthusiast community
  3. `r/antiai` - Anti-AI art critics

---

## Data Flow Between Scripts

```
Raw Reddit Data (Apify CSV)
         │
         ▼
reddit_preprocess.py ────────► preprocessed_manuscript_data.csv
                                         │
                                         ▼
                              reddit_distilbert.py
                                         │
                                         ▼
                         manuscript_analyzed_results.csv
                         + 5 PNG figures

Raw Instagram Data (2 CSVs)
         │
         ▼
    insta_sentiment.py ────────► FINAL_instagram_roberta_dataset.csv
                                + 3 PNG figures
```

---

## Technical Summary Table

| Feature | insta_sentiment.py | reddit_preprocess.py | reddit_distilbert.py |
|---------|-------------------|---------------------|----------------------|
| **Primary Model** | RoBERTa (Twitter-trained) | N/A (cleaning only) | VADER + DistilBERT |
| **Output Type** | Sentiment + engagement correlation | Cleaned CSV | Comparative + ABSA |
| **Visualizations** | 3 figures | None | 5 figures |
| **Key Metric** | Compound score (-1 to +1) | Text length filter (≥3 words) | Confidence + aspect mapping |
| **Hardware** | GPU/CPU auto-detect | CPU only | GPU/CPU auto-detect |
| **Platform Focus** | Instagram comments | Reddit (general) | Reddit (3 subreddits) |

---

## Academic Manuscript Relevance

These scripts were designed to produce **camera-ready figures**.
- High DPI (300) output
- Consistent color palettes (green=positive, gray=neutral, red=negative)
- Professional typography (matplotlib rcParams configured)
- Tight layouts with `bbox_inches='tight'`
- Descriptive titles with subfigure labels (A:, B:, etc.)

The multi-model approach (lexicon + transformer) allows for **methodological triangulation**, while ABSA provides **granular, domain-specific insights** beyond overall sentiment polarity.
