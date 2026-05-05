import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_processed_data():
    print("Loading processed datasets...")
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'))
    test_a_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_a.csv'))
    test_b_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_b.csv'))
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test A: {len(test_a_df)} samples")
    print(f"Test B: {len(test_b_df)} samples")
    
    return {
        'train': train_df,
        'val': val_df,
        'test_a': test_a_df,
        'test_b': test_b_df
    }


def train_advanced_model(train_df, val_df, max_features=15000, n_estimators=150):
    print("\n" + "="*60)
    print("TRAINING ADVANCED MODEL: TF-IDF + Random Forest")
    print("="*60)
    
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    print(f"\nCreating TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    print(f"\nTraining Random Forest classifier ({n_estimators} trees)...")
    print("This may take 2-3 minutes...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=1
    )
    
    model.fit(X_train_tfidf, y_train)
    
    print("\nEvaluating on validation set...")
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    print("\nTop 10 Most Important Features:")
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {importances[idx]:.6f}")
    
    return vectorizer, model, val_accuracy


def evaluate_model(vectorizer, model, test_df, dataset_name):
    print("\n" + "="*60)
    print(f"EVALUATING ON {dataset_name}")
    print("="*60)
    
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    predictions = model.predict(X_test_tfidf)
    probabilities = model.predict_proba(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Real', 'Fake']))
    
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Real  Fake")
    print(f"Actual Real     {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"       Fake     {cm[1,0]:5d} {cm[1,1]:5d}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, f'advanced_confusion_matrix_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {plot_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def save_model(vectorizer, model, model_name='advanced'):
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    vectorizer_path = os.path.join(MODELS_DIR, f'{model_name}_vectorizer.pkl')
    model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Model saved to: {model_path}")


def compare_with_baseline():
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE MODEL")
    print("="*60)
    
    print("\nModel Comparison Table:")
    print("-" * 70)
    print(f"{'Model':<20} {'Test A Acc':<15} {'Test B Acc':<15} {'Features':<10}")
    print("-" * 70)
    print(f"{'Baseline (LR)':<20} {'98.84%':<15} {'46.25%':<15} {'10,000':<10}")
    print(f"{'Advanced (RF)':<20} {'[Running...]':<15} {'[Running...]':<15} {'15,000':<10}")
    print("-" * 70)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    compare_with_baseline()
    
    data = load_processed_data()
    
    vectorizer, model, val_acc = train_advanced_model(
        data['train'], 
        data['val'],
        max_features=15000,
        n_estimators=150
    )
    
    val_metrics = evaluate_model(vectorizer, model, data['val'], "Validation Set")
    test_a_metrics = evaluate_model(vectorizer, model, data['test_a'], "Test A (In-Distribution)")
    test_b_metrics = evaluate_model(vectorizer, model, data['test_b'], "Test B (Out-of-Distribution)")
    
    save_model(vectorizer, model, 'advanced')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nValidation Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"Test A Accuracy:      {test_a_metrics['accuracy']:.4f}")
    print(f"Test B Accuracy:      {test_b_metrics['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(f"\n{'Model':<20} {'Test A Acc':<15} {'Test B Acc':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Baseline (LR)':<20} {'98.84%':<15} {'46.25%':<15} {'-':<15}")
    print(f"{'Advanced (RF)':<20} {f'{test_a_metrics["accuracy"]*100:.2f}%':<15} {f'{test_b_metrics["accuracy"]*100:.2f}%':<15} {f'+{(test_a_metrics["accuracy"]-0.9884)*100:.2f}%':<15}")
    print("-" * 70)
    
    print("\nModels and results saved successfully!")
