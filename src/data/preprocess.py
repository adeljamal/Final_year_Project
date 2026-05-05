import re
from nltk.corpus import stopwords
import nltk

# Try to load stopwords, download if not available
try:
    stopwords.words('english')
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')


def clean_text(text, remove_stopwords=False):
    """
    Clean text data.
    
    Args:
        text (str): Raw text to clean
        remove_stopwords (bool): Whether to remove English stopwords
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optionally remove stopwords (using simple split instead of punkt tokeniser)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Simple word splitting (works well for cleaned text)
        tokens = text.split()
        text = ' '.join([word for word in tokens if word not in stop_words])
    
    return text


def preprocess_dataframe(df, text_column='text', remove_stopwords=False):
    """
    Apply text cleaning to a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with text data
        text_column (str): Name of column containing text
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    df = df.copy()
    df[text_column] = df[text_column].apply(lambda x: clean_text(x, remove_stopwords))
    
    # Remove rows where text is empty after cleaning
    df = df[df[text_column].str.strip() != '']
    
    return df


if __name__ == '__main__':
    """
    Test the preprocessing functions.
    """
    sample_text = """
    BREAKING NEWS!!! Trump announces new policy: https://example.com/article123
    Contact us at fake@news.com for more info! #FakeNews2026
    This is a TEST of the preprocessing system!!!
    """
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text (without stopword removal):")
    print(clean_text(sample_text, remove_stopwords=False))
    print("\nCleaned text (with stopword removal):")
    print(clean_text(sample_text, remove_stopwords=True))
