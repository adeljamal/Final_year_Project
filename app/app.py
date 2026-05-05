import streamlit as st
import pickle
import os
import sys

# Add parent directory to path to import preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/data'))
from preprocess import clean_text

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and vectoriser
@st.cache_resource
def load_model(model_type='baseline'):
    """Load the trained model and vectoriser from disk."""
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    
    vectoriser_path = os.path.join(models_dir, f'{model_type}_vectorizer.pkl')
    model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
    
    with open(vectoriser_path, 'rb') as f:
        vectoriser = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return vectoriser, model


def predict_news(text, vectoriser, model):
    """
    Predict whether news is real or fake.
    
    Args:
        text: Input news article text
        vectoriser: Trained TF-IDF vectoriser
        model: Trained classifier
        
    Returns:
        prediction: 0 (Real) or 1 (Fake)
        probability: Confidence score
    """
    # Preprocess the text
    cleaned_text = clean_text(text, remove_stopwords=False)
    
    # Transform to TF-IDF features
    text_tfidf = vectoriser.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability


# Example articles for testing
EXAMPLE_ARTICLES = {
    "Real News Example 1": """
    London (Reuters) - The Bank of England raised interest rates by 0.25 percentage points on Thursday, 
    taking the base rate to 5.25% as it continues efforts to bring down inflation. The Monetary Policy 
    Committee voted 6-3 in favour of the increase, with three members preferring to hold rates steady. 
    Governor Andrew Bailey said in a statement that whilst inflation has fallen from its peak, it remains 
    too high and the Bank is committed to returning it to the 2% target.
    """,
    
    "Real News Example 2": """
    The European Commission announced new regulations on artificial intelligence today, establishing the 
    world's first comprehensive legal framework for AI systems. The AI Act will classify applications 
    according to risk levels and impose strict requirements on high-risk uses such as facial recognition 
    and critical infrastructure. Companies violating the rules could face fines of up to 6% of global revenue.
    """,
    
    "Fake News Example 1": """
    BREAKING: Scientists confirm chocolate cures cancer! New study shows eating 5 bars daily eliminates 
    all tumours. Big Pharma doesn't want you to know this SHOCKING truth! Share before they delete this! 
    Watch the video via our exclusive link. Featured image via HealthTruthNow.
    """,
    
    "Fake News Example 2": """
    You won't BELIEVE what this politician just said! SHOCKING revelations that the mainstream media 
    won't report. This will change EVERYTHING! Click here to watch the full interview before it's 
    taken down. Featured image via TruthNewsDaily. Share this with everyone you know!
    """
}


def main():
    # Header
    st.title("📰 Fake News Detection System")
    st.markdown("### Machine Learning-Based News Authenticity Classifier")
    st.markdown("---")
    
    # Sidebar for model selection and information
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        model_type = st.selectbox(
            "Select Model:",
            options=['baseline', 'advanced'],
            format_func=lambda x: 'Baseline (Logistic Regression)' if x == 'baseline' else 'Advanced (Random Forest)',
            help="Choose between the baseline or advanced model"
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This system uses machine learning to classify news articles as real or fake.
        
        **Features:**
        - TF-IDF text vectorisation
        - Pre-processing pipeline
        - Confidence scoring
        
        **Performance:**
        - Test Accuracy: 98.84%
        - Trained on 31,422 articles
        """)
        
        st.markdown("---")
        st.header("📊 Model Info")
        if model_type == 'baseline':
            st.info("""
            **Baseline Model**
            - Algorithm: Logistic Regression
            - Features: 10,000 TF-IDF
            - Test A: 98.84%
            - Test B: 46.25%
            """)
        else:
            st.success("""
            **Advanced Model**
            - Algorithm: Random Forest
            - Trees: 150
            - Features: 15,000 TF-IDF
            - Test A: 99.61%
            - Test B: 43.65%
            """)
    
    # Load selected model
    try:
        vectoriser, model = load_model(model_type)
        st.success(f"✅ {model_type.capitalize()} model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Analyse Text", "📋 Try Examples", "📈 About the Project"])
    
    # Tab 1: Custom text input
    with tab1:
        st.header("Enter News Article Text")
        
        user_input = st.text_area(
            "Paste your news article here:",
            height=250,
            placeholder="Enter the news article you want to check...",
            help="Paste the full text of a news article to check its authenticity"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyse_button = st.button("🔍 Analyse Article", type="primary", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        if analyse_button:
            if user_input.strip():
                with st.spinner("Analysing article..."):
                    prediction, probability = predict_news(user_input, vectoriser, model)
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 0:
                            st.success("### ✅ REAL NEWS")
                            st.metric("Confidence", f"{probability[0]*100:.2f}%")
                        else:
                            st.error("### ⚠️ FAKE NEWS")
                            st.metric("Confidence", f"{probability[1]*100:.2f}%")
                    
                    with col2:
                        st.markdown("#### Probability Distribution")
                        st.progress(float(probability[0]), text=f"Real: {probability[0]*100:.1f}%")
                        st.progress(float(probability[1]), text=f"Fake: {probability[1]*100:.1f}%")
                    
                    # Additional information
                    st.markdown("---")
                    st.info("""
                    **ℹ️ Interpretation Guide:**
                    - **Confidence >90%**: High certainty in classification
                    - **Confidence 70-90%**: Moderate certainty
                    - **Confidence <70%**: Low certainty, manual review recommended
                    """)
            else:
                st.warning("⚠️ Please enter some text to analyse.")
    
    # Tab 2: Example articles
    with tab2:
        st.header("Try Example Articles")
        st.markdown("Select a pre-loaded example to see how the model performs:")
        
        selected_example = st.selectbox(
            "Choose an example:",
            options=list(EXAMPLE_ARTICLES.keys()),
            help="Select from real and fake news examples"
        )
        
        example_text = EXAMPLE_ARTICLES[selected_example]
        
        st.text_area(
            "Example Article Text:",
            value=example_text,
            height=200,
            disabled=True
        )
        
        if st.button("🔍 Analyse Example", type="primary"):
            with st.spinner("Analysing example..."):
                prediction, probability = predict_news(example_text, vectoriser, model)
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 0:
                        st.success("### ✅ REAL NEWS")
                        st.metric("Confidence", f"{probability[0]*100:.2f}%")
                    else:
                        st.error("### ⚠️ FAKE NEWS")
                        st.metric("Confidence", f"{probability[1]*100:.2f}%")
                
                with col2:
                    st.markdown("#### Probability Distribution")
                    st.progress(float(probability[0]), text=f"Real: {probability[0]*100:.1f}%")
                    st.progress(float(probability[1]), text=f"Fake: {probability[1]*100:.1f}%")
                
                # Show if prediction matches expected result
                expected_label = "Real" if "Real" in selected_example else "Fake"
                actual_label = "Real" if prediction == 0 else "Fake"
                
                if expected_label == actual_label:
                    st.success(f"✅ Correct! Model correctly identified this as {expected_label} news.")
                else:
                    st.error(f"❌ Incorrect! Model classified as {actual_label}, but it's {expected_label} news.")
    
    # Tab 3: Project information
    with tab3:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This fake news detection system was developed as part of a machine learning coursework project.
        It demonstrates the application of natural language processing and classification algorithms
        to identify potentially misleading or false news articles.
        
        ### 🔬 Methodology
        
        **Data Preprocessing:**
        - Text cleaning and normalisation
        - URL and special character removal
        - Lowercase conversion
        - TF-IDF vectorisation
        
        **Models Trained:**
        1. **Baseline**: Logistic Regression (98.84% accuracy)
        2. **Advanced**: Random Forest Ensemble (99.61% accuracy)
        
        **Datasets:**
        - Training: 31,422 articles from Kaggle Fake News dataset
        - Test A: 6,734 in-distribution news articles
        - Test B: 1,267 out-of-distribution political statements (LIAR dataset)
        
        ### 📊 Performance Metrics
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Baseline Test Accuracy", "98.84%")
            st.metric("Advanced Test Accuracy", "99.61%")
        
        with col2:
            st.metric("Training Samples", "31,422")
            st.metric("Total Features", "15,000")
        
        st.markdown("""
        ### ⚠️ Limitations
        - Performance drops on out-of-distribution data (different writing styles)
        - Requires sufficient text length for accurate classification
        - May not generalise well to very recent news or emerging topics
        - Should be used as a support tool, not definitive truth arbiter
        
        ### 🔮 Future Improvements
        - Fine-tune transformer models (BERT, RoBERTa)
        - Incorporate source credibility analysis
        - Add multi-language support
        - Implement domain adaptation techniques
        """)


if __name__ == "__main__":
    main()
