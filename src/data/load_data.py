import pandas as pd
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')


def load_kaggle():
    fake_path = os.path.join(RAW_DATA_DIR, 'Fake.csv')
    true_path = os.path.join(RAW_DATA_DIR, 'True.csv')
    
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 1
    true_df['label'] = 0
    
    fake_df['text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
    true_df['text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
    
    fake_df = fake_df[['text', 'label']]
    true_df = true_df[['text', 'label']]
    
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Kaggle dataset loaded: {len(combined_df)} articles")
    print(f"  - Fake: {len(fake_df)}")
    print(f"  - Real: {len(true_df)}")
    
    return combined_df


def load_liar():
    liar_path = os.path.join(RAW_DATA_DIR, 'test.tsv')
    
    column_names = ['id', 'label', 'statement', 'subject', 'speaker', 
                    'job', 'state', 'party', 'barely_true_counts', 
                    'false_counts', 'half_true_counts', 'mostly_true_counts', 
                    'pants_on_fire_counts', 'context']
    
    liar_df = pd.read_csv(liar_path, sep='\t', header=None, names=column_names)
    
    label_mapping = {
        'pants-fire': 1,
        'false': 1,
        'barely-true': 1,
        'half-true': 0,
        'mostly-true': 0,
        'true': 0
    }
    
    liar_df['label'] = liar_df['label'].map(label_mapping)
    liar_df['text'] = liar_df['statement']
    liar_df = liar_df[['text', 'label']]
    liar_df = liar_df.dropna(subset=['label'])
    
    print(f"LIAR dataset loaded: {len(liar_df)} statements")
    print(f"  - Fake: {(liar_df['label'] == 1).sum()}")
    print(f"  - Real: {(liar_df['label'] == 0).sum()}")
    
    return liar_df


if __name__ == '__main__':
    print("Loading Kaggle dataset...")
    kaggle_df = load_kaggle()
    print(kaggle_df.head())
    print()
    
    print("Loading LIAR dataset...")
    liar_df = load_liar()
    print(liar_df.head())
