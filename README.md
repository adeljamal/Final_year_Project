# Fake News Detection System

**Author:** Adel  Jamal
**Course:** 6Z0019 - Machine Learning Module  
**Submission:** Creative Piece  

---

## Overview

This project implements a machine learning-based fake news detection system using Natural Language Processing (NLP) techniques. The system classifies news articles as real or fake with 99.61% accuracy using a Random Forest ensemble model.

---

## System Requirements

- **Operating System:** macOS, Linux, or Windows
- **Python Version:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** ~500MB for project files

---

## Installation Instructions

### Step 1: Extract the Project Files

```bash
unzip fake-news-detection.zip
cd fake-news-detection
```

### Step 2: Create Python Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\\Scripts\\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This may take 2-3 minutes to install all packages.

---

## Running the Application

 Launch the Streamlit Web Interface.

This is the main Creative Piece demonstration.

```bash
streamlit run app/app.py
```

The application will open automatically in your web browser at `http://localhost:8501`

**Features:**
- Real-time news article classification
- Confidence scoring
- Pre-loaded example articles
- Model comparison (Baseline vs Advanced)


## Using the Web Application

### Main Features:

1. **Analyse Text Tab**
   - Paste any news article text
   - Click "Analyse Article"
   - View prediction (Real/Fake) with confidence score

2. **Try Examples Tab**
   - Select from pre-loaded real and fake news examples
   - See model performance on known cases

3. **About the Project Tab**
   - View methodology, datasets, and performance metrics

4. **Model Selection (Sidebar)**
   - Toggle between Baseline and Advanced models
   - Compare performance
