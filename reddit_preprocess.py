import pandas as pd
import html
import re

def rigorous_clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Decode HTML escape strings (e.g., &#39; -> ', &quot; -> ")
    cleaned = html.unescape(text)
    
    # 2. Strip out Web URLs and hyperlink anchors
    cleaned = re.sub(r'http\s+://\S+|https://\S+|www\.\S+', '', cleaned)
    
    # 3. Strip out Reddit specific syntax tags (e.g., /u/username, r/subreddit)
    cleaned = re.sub(r'/?u/[\w-]+', '', cleaned)
    cleaned = re.sub(r'/?r/[\w-]+', '', cleaned)
    
    # 4. Remove system markdown image blocks or preview markers left by Apify
    cleaned = re.sub(r'\[link\]|\[comments\]|submitted by', '', cleaned)
    
    # 5. Collapse excessive whitespace and line breaks into standard sentences
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

# Ingest raw dataset
raw_df = pd.read_csv("dataset_reddit-scraper-lite_2026-06-05_16-09-06-559.csv")

# Execute rigorous cleaning
print("Beginning corpus normalization loops...")
raw_df = raw_df.dropna(subset=['body'])
raw_df['processed_body'] = raw_df['body'].apply(rigorous_clean_text)

# Eliminate post-cleaning remnants that drop below viable text density
raw_df = raw_df[raw_df['processed_body'].apply(lambda x: len(x.split()) > 3)]

# Persist to disk
raw_df.to_csv("preprocessed_manuscript_data.csv", index=False)
print(f"Success. Compiled {len(raw_df)} normalized observations ready for Experiment Iteration 2!")
