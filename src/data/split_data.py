import pandas as pd
import os
from sklearn.model_selection import train_test_split
from load_data import load_kaggle, load_liar
from preprocess import preprocess_dataframe

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def create_splits(remove_stopwords=False, random_seed=42):
    """
    Load, preprocess, and split datasets into train/val/test.
    
    Args:
        remove_stopwords (bool): Whether to remove stopwords during preprocessing
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing all dataset splits
    """
    print("=" * 60)
    print("STEP 1: Loading datasets...")
    print("=" * 60)
    
    kaggle_df = load_kaggle()
    liar_df = load_liar()
    
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing text...")
    print("=" * 60)
    
    print(f"Preprocessing Kaggle dataset (remove_stopwords={remove_stopwords})...")
    kaggle_clean = preprocess_dataframe(kaggle_df, text_column='text', 
                                        remove_stopwords=remove_stopwords)
    
    print(f"Preprocessing LIAR dataset (remove_stopwords={remove_stopwords})...")
    liar_clean = preprocess_dataframe(liar_df, text_column='text', 
                                      remove_stopwords=remove_stopwords)
    
    print(f"\nAfter preprocessing:")
    print(f"  Kaggle: {len(kaggle_clean)} articles (removed {len(kaggle_df) - len(kaggle_clean)} empty texts)")
    print(f"  LIAR: {len(liar_clean)} statements (removed {len(liar_df) - len(liar_clean)} empty texts)")
    
    print("\n" + "=" * 60)
    print("STEP 3: Splitting Kaggle dataset (70/15/15)...")
    print("=" * 60)
    
    train_df, temp_df = train_test_split(
        kaggle_clean,
        test_size=0.30,
        random_state=random_seed,
        stratify=kaggle_clean['label']
    )
    
    val_df, test_a_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_seed,
        stratify=temp_df['label']
    )
    
    print(f"Train set: {len(train_df)} articles")
    print(f"  - Fake: {(train_df['label'] == 1).sum()}")
    print(f"  - Real: {(train_df['label'] == 0).sum()}")
    
    print(f"Validation set: {len(val_df)} articles")
    print(f"  - Fake: {(val_df['label'] == 1).sum()}")
    print(f"  - Real: {(val_df['label'] == 0).sum()}")
    
    print(f"Test A (in-distribution): {len(test_a_df)} articles")
    print(f"  - Fake: {(test_a_df['label'] == 1).sum()}")
    print(f"  - Real: {(test_a_df['label'] == 0).sum()}")
    
    print("\n" + "=" * 60)
    print("STEP 4: Using LIAR as external test set (Test B)...")
    print("=" * 60)
    
    test_b_df = liar_clean
    
    print(f"Test B (out-of-distribution): {len(test_b_df)} statements")
    print(f"  - Fake: {(test_b_df['label'] == 1).sum()}")
    print(f"  - Real: {(test_b_df['label'] == 0).sum()}")
    
    print("\n" + "=" * 60)
    print("STEP 5: Saving processed datasets...")
    print("=" * 60)
    
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
    test_a_path = os.path.join(PROCESSED_DATA_DIR, 'test_a.csv')
    test_b_path = os.path.join(PROCESSED_DATA_DIR, 'test_b.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_a_df.to_csv(test_a_path, index=False)
    test_b_df.to_csv(test_b_path, index=False)
    
    print(f"Saved to {PROCESSED_DATA_DIR}/")
    print(f"  - train.csv: {len(train_df)} rows")
    print(f"  - val.csv: {len(val_df)} rows")
    print(f"  - test_a.csv: {len(test_a_df)} rows")
    print(f"  - test_b.csv: {len(test_b_df)} rows")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    
    return {
        'train': train_df,
        'val': val_df,
        'test_a': test_a_df,
        'test_b': test_b_df
    }


if __name__ == '__main__':
    print("\nStarting data preparation pipeline...\n")
    splits = create_splits(remove_stopwords=False, random_seed=42)
    print("\nSample from training set:")
    print(splits['train'].head())
