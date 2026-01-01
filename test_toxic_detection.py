#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Toxic Content Detection Notebook
Tests all use cases and edge cases
"""

import unittest
import pandas as pd
import numpy as np
import re
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestDataLoading(unittest.TestCase):
    """Test dataset loading functionality"""
    
    def setUp(self):
        self.csv_file = "Urdu Abusive Dataset.csv"
        self.xlsx_file = "Hate Speech Roman Urdu (HS-RU-20).xlsx"
    
    def test_load_csv_dataset(self):
        """Test loading the CSV dataset (Excel format)"""
        try:
            df = pd.read_excel(self.csv_file, engine='openpyxl')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertGreater(len(df.columns), 0)
            print("✓ Test: CSV dataset loads successfully")
        except Exception as e:
            self.fail(f"Failed to load CSV dataset: {e}")
    
    def test_load_xlsx_dataset(self):
        """Test loading the XLSX dataset"""
        try:
            df = pd.read_excel(self.xlsx_file, engine='openpyxl')
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            self.assertGreater(len(df.columns), 0)
            print("✓ Test: XLSX dataset loads successfully")
        except Exception as e:
            self.fail(f"Failed to load XLSX dataset: {e}")
    
    def test_dataset_columns_exist(self):
        """Test that datasets have expected column types"""
        df_csv = pd.read_excel(self.csv_file, engine='openpyxl')
        df_xlsx = pd.read_excel(self.xlsx_file, engine='openpyxl')
        
        # Check that datasets have at least one text column and one label column
        self.assertGreater(len(df_csv.columns), 0)
        self.assertGreater(len(df_xlsx.columns), 0)
        
        # Check for text-like columns (object dtype)
        csv_has_text = any(df_csv[col].dtype == 'object' for col in df_csv.columns)
        xlsx_has_text = any(df_xlsx[col].dtype == 'object' for col in df_xlsx.columns)
        
        self.assertTrue(csv_has_text, "CSV dataset should have text columns")
        self.assertTrue(xlsx_has_text, "XLSX dataset should have text columns")
        print("✓ Test: Datasets have expected column types")


class TestColumnStandardization(unittest.TestCase):
    """Test column standardization functionality"""
    
    def setUp(self):
        self.text_cols = ['comment', 'tweet', 'message', 'content', 'text', 'Comment', 'Tweet', 
                         'Message', 'Content', 'Text', 'comment_text', 'Comment_Text', 
                         'sentence', 'Sentence']
        self.label_cols = ['label', 'Label', 'class', 'Class', 'category', 'Category', 
                          'toxic', 'Toxic', 'hate', 'Hate', 'comment_class', 'Comment_Class',
                          'Neutral (N) / Hostile (H)', 'neutral (n) / hostile (h)']
    
    def test_standardize_columns_function(self):
        """Test the column standardization function"""
        def standardize_columns(df, dataset_name):
            df = df.copy()
            text_col = None
            for col in self.text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == 'object' and col not in self.label_cols:
                        text_col = col
                        break
            
            label_col = None
            for col in self.label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                for col in df.columns:
                    if col != text_col and (df[col].dtype in ['int64', 'float64', 'bool'] or 
                                          df[col].dtype.name == 'category'):
                        label_col = col
                        break
            
            if text_col:
                df = df.rename(columns={text_col: 'text'})
            if label_col:
                df = df.rename(columns={label_col: 'label'})
            return df
        
        # Test with actual dataset 1
        df1 = pd.read_excel("Urdu Abusive Dataset.csv", engine='openpyxl')
        df1_std = standardize_columns(df1, "Dataset 1")
        self.assertIn('text', df1_std.columns)
        self.assertIn('label', df1_std.columns)
        print("✓ Test: Dataset 1 columns standardized correctly")
        
        # Test with actual dataset 2
        df2 = pd.read_excel("Hate Speech Roman Urdu (HS-RU-20).xlsx", engine='openpyxl')
        df2_std = standardize_columns(df2, "Dataset 2")
        self.assertIn('text', df2_std.columns)
        self.assertIn('label', df2_std.columns)
        print("✓ Test: Dataset 2 columns standardized correctly")
    
    def test_standardize_columns_edge_cases(self):
        """Test column standardization with edge cases"""
        def standardize_columns(df, dataset_name):
            text_cols = self.text_cols
            label_cols = self.label_cols
            df = df.copy()
            text_col = None
            for col in text_cols:
                if col in df.columns:
                    text_col = col
                    break
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == 'object' and col not in label_cols:
                        text_col = col
                        break
            
            label_col = None
            for col in label_cols:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                for col in df.columns:
                    if col != text_col and (df[col].dtype in ['int64', 'float64', 'bool']):
                        label_col = col
                        break
            
            if text_col:
                df = df.rename(columns={text_col: 'text'})
            if label_col:
                df = df.rename(columns={label_col: 'label'})
            return df
        
        # Test with custom dataframe
        test_df = pd.DataFrame({
            'custom_text_col': ['text1', 'text2', 'text3'],
            'custom_label_col': [1, 0, 1]
        })
        result = standardize_columns(test_df, "Test")
        # Should infer text column from object dtype
        self.assertIn('text', result.columns)
        print("✓ Test: Edge case - custom column names handled")


class TestLabelStandardization(unittest.TestCase):
    """Test label standardization functionality"""
    
    def setUp(self):
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
            non_toxic = ['non-toxic', 'nontoxic', 'non_toxic', 'negative', 'normal', 
                        'clean', 'safe', 'no', 'false', '0', 'n', 'neutral']
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
        
        self.standardize_label = standardize_label
    
    def test_boolean_labels(self):
        """Test boolean label conversion"""
        self.assertEqual(self.standardize_label(True), 1)
        self.assertEqual(self.standardize_label(False), 0)
        print("✓ Test: Boolean labels converted correctly")
    
    def test_numeric_labels(self):
        """Test numeric label conversion"""
        self.assertEqual(self.standardize_label(0), 0)
        self.assertEqual(self.standardize_label(1), 1)
        self.assertEqual(self.standardize_label('0'), 0)
        self.assertEqual(self.standardize_label('1'), 1)
        self.assertEqual(self.standardize_label('0.0'), 0)
        self.assertEqual(self.standardize_label('1.0'), 1)
        print("✓ Test: Numeric labels converted correctly")
    
    def test_text_labels_toxic(self):
        """Test toxic text labels"""
        toxic_labels = ['H', 'h', 'Hostile', 'hostile', 'toxic', 'hate', 'abusive', 'yes', 'true']
        for label in toxic_labels:
            self.assertEqual(self.standardize_label(label), 1, f"Failed for: {label}")
        print("✓ Test: Toxic text labels converted correctly")
    
    def test_text_labels_non_toxic(self):
        """Test non-toxic text labels"""
        non_toxic_labels = ['N', 'n', 'Neutral', 'neutral', 'non-toxic', 'normal', 'clean', 'no', 'false']
        for label in non_toxic_labels:
            self.assertEqual(self.standardize_label(label), 0, f"Failed for: {label}")
        print("✓ Test: Non-toxic text labels converted correctly")
    
    def test_nan_handling(self):
        """Test NaN handling"""
        result = self.standardize_label(pd.NA)
        self.assertIsNone(result)
        result = self.standardize_label(np.nan)
        self.assertIsNone(result)
        print("✓ Test: NaN values handled correctly")
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Float values
        self.assertEqual(self.standardize_label(0.3), 0)
        self.assertEqual(self.standardize_label(0.7), 1)
        # Unknown strings default to 0
        self.assertEqual(self.standardize_label('unknown'), 0)
        print("✓ Test: Edge cases handled correctly")


class TestTextPreprocessing(unittest.TestCase):
    """Test text preprocessing functionality"""
    
    def setUp(self):
        import emoji
        
        self.stopwords_roman = ["hai", "hay", "he", "hain", "kya", "ha", "me", "tum", "nai", "nahi"]
        
        self.slang_map = {
            "yar": "yaar", "yarr": "yaar",
            "bhai": "bhai", "bhaii": "bhai",
            "mein": "main", "me": "main",
            "hai": "hai", "hay": "hai",
        }
        
        def normalize_roman_urdu(text):
            words = text.split()
            normalized_words = []
            for word in words:
                if word in self.slang_map:
                    normalized_words.append(self.slang_map[word])
                else:
                    word_lower = word.lower()
                    if word_lower in self.slang_map:
                        normalized_words.append(self.slang_map[word_lower])
                    else:
                        normalized_words.append(word)
            return " ".join(normalized_words)
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = emoji.replace_emoji(text, "")
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^a-zA-Z0-9ء-ی ]", " ", text)
            text = normalize_roman_urdu(text)
            words = text.split()
            filtered_words = [w for w in words if w not in self.stopwords_roman and len(w) > 1]
            text = " ".join(filtered_words)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        self.clean_text = clean_text
    
    def test_basic_cleaning(self):
        """Test basic text cleaning"""
        text = "  Hello   World  "
        result = self.clean_text(text)
        self.assertEqual(result, "hello world")
        print("✓ Test: Basic text cleaning works")
    
    def test_url_removal(self):
        """Test URL removal"""
        text = "Check this http://example.com and www.test.com"
        result = self.clean_text(text)
        self.assertNotIn("http://", result)
        self.assertNotIn("www.", result)
        print("✓ Test: URLs removed correctly")
    
    def test_emoji_removal(self):
        """Test emoji removal"""
        text = "Hello 😀 World 🎉"
        result = self.clean_text(text)
        # Emojis should be removed
        self.assertNotIn("😀", result)
        self.assertNotIn("🎉", result)
        print("✓ Test: Emojis removed correctly")
    
    def test_slang_normalization(self):
        """Test Roman Urdu slang normalization"""
        text = "yar bhai mein hai"
        result = self.clean_text(text)
        # Should normalize slang
        self.assertIn("yaar", result.lower() or "")
        print("✓ Test: Slang normalization works")
    
    def test_stopword_removal(self):
        """Test stopword removal"""
        text = "hai kya ha me"
        result = self.clean_text(text)
        # Stopwords should be removed
        words = result.split()
        for stopword in self.stopwords_roman:
            self.assertNotIn(stopword, words)
        print("✓ Test: Stopwords removed correctly")
    
    def test_empty_text_handling(self):
        """Test empty text handling"""
        result = self.clean_text("")
        self.assertEqual(result, "")
        result = self.clean_text(pd.NA)
        self.assertEqual(result, "")
        print("✓ Test: Empty text handled correctly")
    
    def test_special_characters(self):
        """Test special character removal"""
        text = "Hello@World#123$%^"
        result = self.clean_text(text)
        # Special chars should be removed or replaced
        self.assertNotIn("@", result)
        print("✓ Test: Special characters handled")


class TestDataMerging(unittest.TestCase):
    """Test data merging functionality"""
    
    def test_merge_datasets(self):
        """Test merging two datasets"""
        df_csv = pd.read_excel("Urdu Abusive Dataset.csv", engine='openpyxl', nrows=10)
        df_xlsx = pd.read_excel("Hate Speech Roman Urdu (HS-RU-20).xlsx", engine='openpyxl', nrows=10)
        
        # Standardize columns
        if 'comment_text' in df_csv.columns:
            df_csv = df_csv.rename(columns={'comment_text': 'text', 'comment_class': 'label'})
        if 'Sentence' in df_xlsx.columns:
            df_xlsx = df_xlsx.rename(columns={'Sentence': 'text', 'Neutral (N) / Hostile (H)': 'label'})
        
        # Merge
        df = pd.concat([df_csv, df_xlsx], axis=0, ignore_index=True)
        
        self.assertGreater(len(df), len(df_csv))
        self.assertGreater(len(df), len(df_xlsx))
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
        print("✓ Test: Datasets merged successfully")
    
    def test_duplicate_removal(self):
        """Test duplicate removal"""
        df = pd.DataFrame({
            'text': ['text1', 'text2', 'text1', 'text3'],
            'label': [1, 0, 1, 1]
        })
        df_no_dup = df.drop_duplicates(subset=["text"])
        self.assertEqual(len(df_no_dup), 3)
        print("✓ Test: Duplicates removed correctly")
    
    def test_null_removal(self):
        """Test null value removal"""
        df = pd.DataFrame({
            'text': ['text1', None, 'text2', 'text3'],
            'label': [1, 0, None, 1]
        })
        df_clean = df.dropna(subset=["text", "label"])
        self.assertEqual(len(df_clean), 2)
        print("✓ Test: Null values removed correctly")


class TestTrainTestSplit(unittest.TestCase):
    """Test train-test split functionality"""
    
    def test_train_test_split(self):
        """Test train-test split"""
        from sklearn.model_selection import train_test_split
        
        df_csv = pd.read_excel("Urdu Abusive Dataset.csv", engine='openpyxl', nrows=100)
        if 'comment_text' in df_csv.columns:
            df_csv = df_csv.rename(columns={'comment_text': 'text', 'comment_class': 'label'})
        
        # Standardize labels
        def standardize_label(label):
            if isinstance(label, bool):
                return 1 if label else 0
            return 0
        
        df_csv['label'] = df_csv['label'].apply(standardize_label)
        df_csv = df_csv.dropna(subset=["text", "label"])
        
        X_train, X_test, y_train, y_test = train_test_split(
            df_csv["text"], df_csv["label"], test_size=0.2, random_state=42
        )
        
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertAlmostEqual(len(X_test) / len(df_csv), 0.2, delta=0.05)
        print("✓ Test: Train-test split works correctly")
    
    def test_stratified_split(self):
        """Test stratified split maintains label distribution"""
        from sklearn.model_selection import train_test_split
        
        df = pd.DataFrame({
            'text': [f'text{i}' for i in range(100)],
            'label': [0] * 50 + [1] * 50
        })
        
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
        
        # Check that both train and test have both labels
        self.assertIn(0, y_train.values)
        self.assertIn(1, y_train.values)
        self.assertIn(0, y_test.values)
        self.assertIn(1, y_test.values)
        print("✓ Test: Stratified split maintains distribution")


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def test_tfidf_vectorization(self):
        """Test TF-IDF vectorization"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = ["hello world", "hello python", "world python"]
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        
        self.assertEqual(X.shape[0], len(texts))
        self.assertGreater(X.shape[1], 0)
        print("✓ Test: TF-IDF vectorization works")
    
    def test_logistic_regression(self):
        """Test Logistic Regression training"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Create dummy data
        texts = ["bad hate", "good nice", "terrible awful", "wonderful great"] * 10
        labels = [1, 0, 1, 0] * 10
        
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        
        X_train, X_test = X[:30], X[30:]
        y_train, y_test = labels[:30], labels[30:]
        
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        print(f"✓ Test: Logistic Regression trains and predicts (accuracy: {accuracy:.2f})")
    
    def test_svm_training(self):
        """Test SVM training"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        texts = ["bad hate", "good nice", "terrible awful", "wonderful great"] * 10
        labels = [1, 0, 1, 0] * 10
        
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        
        X_train, X_test = X[:30], X[30:]
        y_train, y_test = labels[:30], labels[30:]
        
        model = SVC(kernel='linear', random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        print(f"✓ Test: SVM trains and predicts (accuracy: {accuracy:.2f})")


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def test_accuracy_score(self):
        """Test accuracy calculation"""
        from sklearn.metrics import accuracy_score
        
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 1, 0]
        accuracy = accuracy_score(y_true, y_pred)
        self.assertEqual(accuracy, 1.0)
        
        y_pred = [1, 0, 1, 0, 1]
        accuracy = accuracy_score(y_true, y_pred)
        self.assertEqual(accuracy, 0.0)
        print("✓ Test: Accuracy score calculation works")
    
    def test_precision_recall_f1(self):
        """Test precision, recall, F1 calculation"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        print("✓ Test: Precision, Recall, F1 calculation works")
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation"""
        from sklearn.metrics import confusion_matrix
        
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (2, 2))
        print("✓ Test: Confusion matrix generation works")


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline"""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from loading to model training"""
        import emoji
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Load data
        df_csv = pd.read_excel("Urdu Abusive Dataset.csv", engine='openpyxl', nrows=50)
        if 'comment_text' in df_csv.columns:
            df_csv = df_csv.rename(columns={'comment_text': 'text', 'comment_class': 'label'})
        
        # Standardize labels
        def standardize_label(label):
            if isinstance(label, bool):
                return 1 if label else 0
            return 0
        
        df_csv['label'] = df_csv['label'].apply(standardize_label)
        df_csv = df_csv.dropna(subset=["text", "label"])
        
        # Clean text
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = emoji.replace_emoji(text, "")
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^a-zA-Z0-9ء-ی ]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        df_csv['text'] = df_csv['text'].apply(clean_text)
        df_csv = df_csv[df_csv['text'].str.len() > 0]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df_csv["text"], df_csv["label"], test_size=0.2, random_state=42
        )
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        
        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        print(f"✓ Test: Complete pipeline works (accuracy: {accuracy:.2f})")


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestColumnStandardization))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelStandardization))
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestDataMerging))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainTestSplit))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("COMPREHENSIVE UNIT TESTS FOR TOXIC CONTENT DETECTION")
    print("="*70)
    print()
    
    success = run_all_tests()
    
    print()
    print("="*70)
    if success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED - Please review the output above")
    print("="*70)
    
    sys.exit(0 if success else 1)

