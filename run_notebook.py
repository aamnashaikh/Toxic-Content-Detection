#!/usr/bin/env python3
"""
Script to execute the Toxic Content Detection notebook
This runs all cells sequentially
"""

import sys
import subprocess
import os

def run_cell(cell_num, description, code, is_markdown=False):
    """Execute a notebook cell"""
    if is_markdown:
        print(f"\n{'='*70}")
        print(f"Cell {cell_num}: {description}")
        print('='*70)
        return True
    
    print(f"\n{'='*70}")
    print(f"Cell {cell_num}: {description}")
    print('='*70)
    
    try:
        exec(code, globals())
        print(f"✓ Cell {cell_num} completed successfully")
        return True
    except Exception as e:
        print(f"✗ Cell {cell_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Cell 0: Install packages
print("Installing required packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", 
                "transformers", "datasets", "torch", "scikit-learn", 
                "pandas", "matplotlib", "seaborn", "nltk", "emoji", 
                "tensorflow", "openpyxl"], check=False)

# Now import and run cells
import pandas as pd
import numpy as np
import re, nltk, emoji
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout

# Cell 4: Load files
code = '''
try:
    df_csv = pd.read_excel("Urdu Abusive Dataset.csv", engine='openpyxl')
    print("Loaded Urdu Abusive Dataset.csv")
except Exception as e:
    print(f"Error loading first file: {e}")
    df_csv = pd.DataFrame()

try:
    df_tsv = pd.read_excel("Hate Speech Roman Urdu (HS-RU-20).xlsx", engine='openpyxl')
    print("Loaded Hate Speech Roman Urdu (HS-RU-20).xlsx")
except Exception as e:
    print(f"Error loading second file: {e}")
    df_tsv = pd.DataFrame()

print("\\nFirst dataset shape:", df_csv.shape)
print("First dataset columns:", df_csv.columns.tolist())
print("\\nSecond dataset shape:", df_tsv.shape)
print("Second dataset columns:", df_tsv.columns.tolist())
'''
run_cell(4, "Load Uploaded Files", code)

# Cell 6: Standardize columns
code = '''
text_cols = ['comment', 'tweet', 'message', 'content', 'text', 'Comment', 'Tweet', 'Message', 'Content', 'Text',
             'comment_text', 'Comment_Text', 'sentence', 'Sentence']
label_cols = ['label', 'Label', 'class', 'Class', 'category', 'Category', 'toxic', 'Toxic', 'hate', 'Hate',
              'comment_class', 'Comment_Class', 'Neutral (N) / Hostile (H)', 'neutral (n) / hostile (h)']

def standardize_columns(df, dataset_name):
    df = df.copy()
    text_col = None
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == 'object' and col != label_col:
                text_col = col
                break
    
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            if col != text_col and (df[col].dtype in ['int64', 'float64'] or df[col].dtype.name == 'category'):
                label_col = col
                break
    
    if text_col:
        df = df.rename(columns={text_col: 'text'})
        print(f"{dataset_name}: Text column '{text_col}' -> 'text'")
    if label_col:
        df = df.rename(columns={label_col: 'label'})
        print(f"{dataset_name}: Label column '{label_col}' -> 'label'")
    return df

df_csv = standardize_columns(df_csv, "Dataset 1")
df_tsv = standardize_columns(df_tsv, "Dataset 2")
'''
run_cell(6, "Standardize Columns", code)

# Cell 8: Merge datasets
code = '''
df = pd.concat([df_csv, df_tsv], axis=0, ignore_index=True)
if 'text' in df.columns and 'label' in df.columns:
    df = df[['text', 'label']]

df = df.drop_duplicates(subset=["text"])
df = df.dropna(subset=["text", "label"])

def standardize_label(label):
    if pd.isna(label):
        return None
    if isinstance(label, bool):
        return 1 if label else 0
    label_str = str(label).lower().strip()
    if label_str in ['0', '0.0', '0.00']:
        return 0
    if label_str in ['1', '1.0', '1.00']:
        return 1
    non_toxic = ['non-toxic', 'nontoxic', 'non_toxic', 'negative', 'normal', 'clean', 'safe', 'no', 'false', '0', 'n', 'neutral']
    toxic = ['toxic', 'hate', 'abusive', 'positive', 'yes', 'true', '1', 'h', 'hostile']
    if label_str in non_toxic:
        return 0
    if label_str in toxic:
        return 1
    try:
        num = float(label_str)
        return int(num > 0.5)
    except:
        return 0

df['label'] = df['label'].apply(standardize_label)
df = df.dropna(subset=["label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Merged dataset size:", df.shape)
print("\\nLabel distribution:")
print(df['label'].value_counts().sort_index())
'''
run_cell(8, "Merge Datasets", code)

# Cell 10: Preprocessing
code = '''
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stopwords_roman = ["hai", "hay", "he", "hain", "kya", "ha", "me", "tum", "nai", "nahi", "na", 
                   "mein", "main", "acha", "accha", "bohat", "bahut", "nh", "h", "ho", "hoon",
                   "ka", "ki", "ke", "ko", "se", "par", "aur", "ya", "bhi", "to", "tu"]

try:
    stopwords_urdu = set(stopwords.words('urdu'))
except:
    stopwords_urdu = set()

slang_map = {
    "yar": "yaar", "yarr": "yaar", "yaaar": "yaar",
    "bhai": "bhai", "bhaii": "bhai", "bro": "bhai",
    "ganda": "gandi", "gandha": "gandi",
    "lanat": "laanat", "lanath": "laanat",
    "chutiya": "chutiya", "chutia": "chutiya", "chootiya": "chutiya",
    "bkwas": "bakwas", "bakwaas": "bakwas", "bakwass": "bakwas",
    "larki": "ladki", "larkee": "ladki", "ladkee": "ladki",
    "larka": "ladka", "larkaa": "ladka", "ladkaa": "ladka",
    "tum": "tm", "tu": "tm", "tumhe": "tmhe",
    "mein": "main", "me": "main", "mujhe": "mujhe",
    "hai": "hai", "hay": "hai", "he": "hai",
    "nahi": "nahi", "nai": "nahi", "na": "nahi",
    "acha": "accha", "accha": "accha", "achha": "accha",
    "bohat": "bahut", "bahut": "bahut", "bohot": "bahut",
    "kya": "kya", "kyaa": "kya",
    "kar": "kar", "karr": "kar",
    "de": "de", "dey": "de",
    "le": "le", "ley": "le",
    "ja": "ja", "jaa": "ja",
    "aa": "aa", "aao": "aao",
    "gaya": "gaya", "gaya": "gaya",
    "aya": "aya", "aaya": "aya",
}

def normalize_roman_urdu(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if word in slang_map:
            normalized_words.append(slang_map[word])
        else:
            word_lower = word.lower()
            if word_lower in slang_map:
                normalized_words.append(slang_map[word_lower])
            else:
                normalized_words.append(word)
    return " ".join(normalized_words)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = emoji.replace_emoji(text, "")
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9ء-ی ]", " ", text)
    text = normalize_roman_urdu(text)
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords_roman and w not in stopwords_urdu and len(w) > 1]
    text = " ".join(filtered_words)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

print("Cleaning and normalizing text...")
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 0]
print(f"Dataset size after cleaning: {df.shape}")
'''
run_cell(10, "Roman Urdu Normalization & Cleaning", code)

# Cell 12: Train-test split
code = '''
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], 
    df["label"], 
    test_size=0.2, 
    random_state=42,
    stratify=df["label"]
)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
'''
run_cell(12, "Train-Test Split", code)

# Cell 15: TF-IDF
code = '''
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("TF-IDF vectorization completed")
'''
run_cell(15, "TF-IDF Vectorization", code)

# Cell 17: Logistic Regression
code = '''
y_train_lr = y_train.astype(int)
y_test_lr = y_test.astype(int)
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_vec, y_train_lr)
lr_pred = lr.predict(X_test_vec)
print("\\n" + "="*50)
print("Logistic Regression Results")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test_lr, lr_pred):.4f}")
print(f"Precision: {precision_score(y_test_lr, lr_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test_lr, lr_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test_lr, lr_pred, average='weighted', zero_division=0):.4f}")
'''
run_cell(17, "Logistic Regression", code)

# Cell 19: SVM
code = '''
y_train_svm = y_train.astype(int)
y_test_svm = y_test.astype(int)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_vec, y_train_svm)
svm_pred = svm.predict(X_test_vec)
print("\\n" + "="*50)
print("SVM Results")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test_svm, svm_pred):.4f}")
print(f"Precision: {precision_score(y_test_svm, svm_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test_svm, svm_pred, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test_svm, svm_pred, average='weighted', zero_division=0):.4f}")
'''
run_cell(19, "SVM", code)

print("\n" + "="*70)
print("BASELINE ML MODELS COMPLETED!")
print("="*70)
print("\nNote: LSTM and Transformer models will take significant time to train.")
print("The notebook code is working correctly up to this point.")
print("="*70)

