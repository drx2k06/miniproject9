"""Dataset loading, validation, and metadata utilities for NewsLens AI.

Dataset: AG News Classification Benchmark
Categories:
  0: World
  1: Sports
  2: Business
  3: Sci/Tech

Author: Daniel Richard Xavier
"""

import os
import requests
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

CATEGORY_MAP_ID_TO_NAME: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

CATEGORY_MAP_NAME_TO_ID: Dict[str, int] = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3
}

CATEGORY_COLORS: Dict[str, str] = {
    "World": "#1f77b4",     # Blue
    "Sports": "#2ca02c",    # Green
    "Business": "#ff7f0e",  # Orange
    "Sci/Tech": "#9467bd"   # Purple
}

DATASET_METADATA: Dict[str, Any] = {
    "name": "AG News Topic Classification Benchmark",
    "citation": "Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS 2015.",
    "source_url": "https://huggingface.co/datasets/ag_news",
    "num_classes": 4,
    "classes": ["World", "Sports", "Business", "Sci/Tech"],
    "full_train_samples": 120000,
    "full_test_samples": 7600
}


def get_category_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return category mapping dictionaries."""
    return CATEGORY_MAP_ID_TO_NAME.copy(), CATEGORY_MAP_NAME_TO_ID.copy()


def download_ag_news(data_dir: str = "data", full: bool = False) -> None:
    """Download AG News dataset parquet files from verified upstream repository if not present.
    
    Args:
        data_dir: Directory to save the dataset.
        full: If True, downloads the full 120k training dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    test_path = os.path.join(data_dir, "ag_news_test.parquet")
    train_path = os.path.join(data_dir, "ag_news_train.parquet" if full else "ag_news_train_sample.parquet")

    test_url = "https://huggingface.co/datasets/ag_news/resolve/main/data/test-00000-of-00001.parquet"
    train_url = "https://huggingface.co/datasets/ag_news/resolve/main/data/train-00000-of-00001.parquet"

    if not os.path.exists(test_path):
        print(f"Downloading AG News test set from {test_url}...")
        resp = requests.get(test_url, timeout=60)
        resp.raise_for_status()
        with open(test_path, "wb") as f:
            f.write(resp.content)
        print("Test set saved successfully.")

    if not os.path.exists(train_path):
        if full:
            print(f"Downloading full AG News train set from {train_url}...")
            resp = requests.get(train_url, timeout=120)
            resp.raise_for_status()
            with open(train_path, "wb") as f:
                f.write(resp.content)
            print("Full train set saved successfully.")
        else:
            print(f"Downloading AG News train set for sampling...")
            resp = requests.get(train_url, timeout=120)
            resp.raise_for_status()
            temp_full = os.path.join(data_dir, "temp_train.parquet")
            with open(temp_full, "wb") as f:
                f.write(resp.content)
            df_full = pd.read_parquet(temp_full)
            sub, _ = train_test_split(df_full, train_size=24000, stratify=df_full['label'], random_state=42)
            sub.reset_index(drop=True).to_parquet(train_path, index=False)
            if os.path.exists(temp_full):
                os.remove(temp_full)
            print(f"Stratified sample of 24,000 records saved to {train_path}.")


def validate_dataset(df: pd.DataFrame, split_name: str = "Dataset") -> Dict[str, Any]:
    """Validate dataset structure, schema, missing values, duplicates, and labels.
    
    Args:
        df: DataFrame to validate.
        split_name: Name of the split (e.g., 'Train', 'Test').

    Returns:
        Dictionary with validation results.

    Raises:
        ValueError: If schema or essential constraints are violated.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(f"{split_name} must be a valid pandas DataFrame.")

    required_columns = {"text", "label"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"{split_name} is missing required columns: {missing_cols}")

    num_rows = len(df)
    if num_rows == 0:
        raise ValueError(f"{split_name} contains 0 rows.")

    null_text = int(df["text"].isnull().sum())
    null_label = int(df["label"].isnull().sum())
    empty_text = int((df["text"].astype(str).str.strip() == "").sum())

    unique_labels = sorted(df["label"].dropna().unique().tolist())
    expected_labels = {0, 1, 2, 3}
    invalid_labels = set(unique_labels) - expected_labels
    if invalid_labels:
        raise ValueError(f"{split_name} contains invalid labels: {invalid_labels}. Expected: {expected_labels}")

    class_dist = df["label"].value_counts().to_dict()
    named_dist = {CATEGORY_MAP_ID_TO_NAME.get(k, str(k)): v for k, v in class_dist.items()}

    char_lengths = df["text"].astype(str).str.len()
    word_lengths = df["text"].astype(str).apply(lambda x: len(x.split()))

    duplicate_texts = int(df["text"].duplicated().sum())

    summary = {
        "split_name": split_name,
        "num_rows": num_rows,
        "null_text_count": null_text,
        "null_label_count": null_label,
        "empty_text_count": empty_text,
        "duplicate_text_count": duplicate_texts,
        "unique_labels": unique_labels,
        "class_distribution": named_dist,
        "min_char_len": int(char_lengths.min()),
        "mean_char_len": float(round(char_lengths.mean(), 2)),
        "max_char_len": int(char_lengths.max()),
        "min_word_len": int(word_lengths.min()),
        "mean_word_len": float(round(word_lengths.mean(), 2)),
        "max_word_len": int(word_lengths.max())
    }
    return summary


def load_data(
    data_dir: str = "data",
    use_sample: bool = True,
    sample_size: int = 24000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets for AG News classification.
    
    Args:
        data_dir: Directory containing parquet files.
        use_sample: If True, uses stratified sample for train; if False and full exists, loads full.
        sample_size: Sample size if extracting from full train set.
        random_state: Seed for reproducible sampling.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_sample_path = os.path.join(data_dir, "ag_news_train_sample.parquet")
    train_full_path = os.path.join(data_dir, "ag_news_train.parquet")
    test_path = os.path.join(data_dir, "ag_news_test.parquet")

    # If test file is missing, trigger download
    if not os.path.exists(test_path):
        download_ag_news(data_dir=data_dir, full=False)

    test_df = pd.read_parquet(test_path)

    if use_sample:
        if os.path.exists(train_sample_path):
            train_df = pd.read_parquet(train_sample_path)
        elif os.path.exists(train_full_path):
            full_df = pd.read_parquet(train_full_path)
            train_df, _ = train_test_split(
                full_df,
                train_size=sample_size,
                stratify=full_df["label"],
                random_state=random_state
            )
            train_df = train_df.reset_index(drop=True)
        else:
            download_ag_news(data_dir=data_dir, full=False)
            train_df = pd.read_parquet(train_sample_path)
    else:
        if os.path.exists(train_full_path):
            train_df = pd.read_parquet(train_full_path)
        elif os.path.exists(train_sample_path):
            print("Full train dataset not found locally. Using existing train sample.")
            train_df = pd.read_parquet(train_sample_path)
        else:
            download_ag_news(data_dir=data_dir, full=True)
            train_df = pd.read_parquet(train_full_path)

    # Validate both splits
    validate_dataset(train_df, "Train Set")
    validate_dataset(test_df, "Test Set")

    return train_df, test_df
