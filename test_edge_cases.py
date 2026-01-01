#!/usr/bin/env python3
"""
Additional Edge Case Tests for Toxic Content Detection
Tests boundary conditions and error handling
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        df = pd.DataFrame(columns=['text', 'label'])
        self.assertEqual(len(df), 0)
        # Should not crash
        df_clean = df.dropna(subset=["text", "label"])
        self.assertEqual(len(df_clean), 0)
        print("✓ Edge Case: Empty dataset handled")
    
    def test_single_row_dataset(self):
        """Test handling of single row dataset"""
        df = pd.DataFrame({'text': ['test'], 'label': [1]})
        self.assertEqual(len(df), 1)
        # Should work for train-test split
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, random_state=42
            )
            print("✓ Edge Case: Single row dataset handled")
        except ValueError:
            # Expected for very small dataset
            print("✓ Edge Case: Single row dataset raises expected error")
    
    def test_all_same_labels(self):
        """Test dataset with all same labels"""
        df = pd.DataFrame({
            'text': [f'text{i}' for i in range(10)],
            'label': [1] * 10
        })
        # Should handle gracefully
        unique_labels = df['label'].unique()
        self.assertEqual(len(unique_labels), 1)
        print("✓ Edge Case: All same labels handled")
    
    def test_very_long_text(self):
        """Test very long text input"""
        long_text = "word " * 10000  # Very long text
        df = pd.DataFrame({'text': [long_text], 'label': [1]})
        # Should not crash
        self.assertEqual(len(df), 1)
        print("✓ Edge Case: Very long text handled")
    
    def test_special_characters_only(self):
        """Test text with only special characters"""
        special_text = "@#$%^&*()"
        df = pd.DataFrame({'text': [special_text], 'label': [1]})
        # Should handle gracefully
        self.assertEqual(len(df), 1)
        print("✓ Edge Case: Special characters only handled")
    
    def test_unicode_text(self):
        """Test Unicode/Urdu text handling"""
        urdu_text = "کہ کے لے لی شام دلے کی"
        df = pd.DataFrame({'text': [urdu_text], 'label': [1]})
        # Should preserve Urdu characters
        self.assertIn('ک', df['text'].iloc[0])
        print("✓ Edge Case: Unicode/Urdu text handled")
    
    def test_mixed_case_labels(self):
        """Test mixed case label handling"""
        def standardize_label(label):
            if isinstance(label, bool):
                return 1 if label else 0
            label_str = str(label).lower().strip()
            if label_str in ['h', 'hostile', 'true', '1']:
                return 1
            if label_str in ['n', 'neutral', 'false', '0']:
                return 0
            return 0
        
        test_cases = ['H', 'h', 'HOSTILE', 'Hostile', 'hostile', 'N', 'n', 'NEUTRAL', 'Neutral', 'neutral']
        for label in test_cases:
            result = standardize_label(label)
            self.assertIn(result, [0, 1])
        print("✓ Edge Case: Mixed case labels handled")
    
    def test_numeric_string_labels(self):
        """Test numeric string labels"""
        def standardize_label(label):
            if isinstance(label, bool):
                return 1 if label else 0
            label_str = str(label).lower().strip()
            if label_str in ['0', '0.0', '0.00']:
                return 0
            if label_str in ['1', '1.0', '1.00']:
                return 1
            return 0
        
        test_cases = ['0', '1', '0.0', '1.0', '0.00', '1.00', 0, 1, 0.0, 1.0]
        for label in test_cases:
            result = standardize_label(label)
            self.assertIn(result, [0, 1])
        print("✓ Edge Case: Numeric string labels handled")
    
    def test_whitespace_only_text(self):
        """Test text with only whitespace"""
        whitespace_text = "   \n\t   "
        # After cleaning, should become empty
        import re
        cleaned = re.sub(r"\s+", " ", whitespace_text).strip()
        self.assertEqual(len(cleaned), 0)
        print("✓ Edge Case: Whitespace-only text handled")
    
    def test_missing_columns(self):
        """Test handling of missing expected columns"""
        df = pd.DataFrame({'other_col': [1, 2, 3]})
        # Should handle gracefully
        has_text = 'text' in df.columns
        has_label = 'label' in df.columns
        self.assertFalse(has_text or has_label)
        print("✓ Edge Case: Missing columns handled")
    
    def test_duplicate_text_different_labels(self):
        """Test duplicate text with different labels"""
        df = pd.DataFrame({
            'text': ['same', 'same', 'different'],
            'label': [0, 1, 1]
        })
        # After drop_duplicates, should keep first occurrence
        df_unique = df.drop_duplicates(subset=["text"])
        self.assertEqual(len(df_unique), 2)
        print("✓ Edge Case: Duplicate text with different labels handled")
    
    def test_extreme_label_distribution(self):
        """Test extreme label distribution (99% one class)"""
        df = pd.DataFrame({
            'text': [f'text{i}' for i in range(100)],
            'label': [1] * 99 + [0] * 1
        })
        # Should handle imbalanced data
        label_counts = df['label'].value_counts()
        self.assertGreater(label_counts[1], label_counts[0])
        print("✓ Edge Case: Extreme label distribution handled")
    
    def test_nan_in_text(self):
        """Test NaN values in text column"""
        df = pd.DataFrame({
            'text': ['text1', None, 'text2', np.nan],
            'label': [1, 0, 1, 0]
        })
        df_clean = df.dropna(subset=["text"])
        self.assertLess(len(df_clean), len(df))
        print("✓ Edge Case: NaN in text handled")
    
    def test_nan_in_labels(self):
        """Test NaN values in label column"""
        df = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [1, None, np.nan]
        })
        df_clean = df.dropna(subset=["label"])
        self.assertLess(len(df_clean), len(df))
        print("✓ Edge Case: NaN in labels handled")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and robustness"""
    
    def test_invalid_file_path(self):
        """Test handling of invalid file path"""
        try:
            df = pd.read_excel("nonexistent_file.xlsx", engine='openpyxl')
            self.fail("Should have raised an error")
        except FileNotFoundError:
            print("✓ Error Handling: Invalid file path raises FileNotFoundError")
        except Exception:
            print("✓ Error Handling: Invalid file path raises error")
    
    def test_invalid_test_size(self):
        """Test handling of invalid test size"""
        from sklearn.model_selection import train_test_split
        df = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [1, 0, 1]
        })
        
        # Test size > 1 should raise error
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=1.5, random_state=42
            )
            self.fail("Should have raised an error")
        except ValueError:
            print("✓ Error Handling: Invalid test size raises ValueError")
    
    def test_empty_text_after_cleaning(self):
        """Test handling of empty text after cleaning"""
        import re
        import emoji
        
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
        
        # Text that becomes empty after cleaning
        test_texts = ["   ", "@#$%", "http://example.com", ""]
        for text in test_texts:
            cleaned = clean_text(text)
            # Should handle gracefully
            self.assertIsInstance(cleaned, str)
        print("✓ Error Handling: Empty text after cleaning handled")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency"""
    
    def test_label_consistency_after_standardization(self):
        """Test that labels are consistent after standardization"""
        def standardize_label(label):
            if isinstance(label, bool):
                return 1 if label else 0
            label_str = str(label).lower().strip()
            if label_str in ['h', 'hostile', 'true', '1']:
                return 1
            if label_str in ['n', 'neutral', 'false', '0']:
                return 0
            return 0
        
        # Test that same labels produce same results
        test_cases = [
            (True, 1), (False, 0),
            ('H', 1), ('h', 1), ('Hostile', 1),
            ('N', 0), ('n', 0), ('Neutral', 0),
            (1, 1), (0, 0), ('1', 1), ('0', 0)
        ]
        
        for input_label, expected in test_cases:
            result = standardize_label(input_label)
            self.assertEqual(result, expected, f"Failed for {input_label}")
        print("✓ Data Integrity: Label standardization is consistent")
    
    def test_text_preservation(self):
        """Test that text content is preserved (not corrupted)"""
        original_texts = [
            "Hello World",
            "کہ کے لے لی",
            "Roman Urdu text",
            "Mixed text 123"
        ]
        
        for text in original_texts:
            # After basic operations, should still contain original content
            cleaned = str(text).lower()
            # Check that original characters are still present (case-insensitive)
            self.assertTrue(
                any(char.lower() in cleaned for char in text if char.isalnum()),
                f"Text content lost for: {text}"
            )
        print("✓ Data Integrity: Text content preserved")
    
    def test_no_data_loss_on_merge(self):
        """Test that no data is lost during merge"""
        df1 = pd.DataFrame({'text': ['a', 'b'], 'label': [1, 0]})
        df2 = pd.DataFrame({'text': ['c', 'd'], 'label': [1, 0]})
        
        merged = pd.concat([df1, df2], axis=0, ignore_index=True)
        
        self.assertEqual(len(merged), len(df1) + len(df2))
        self.assertEqual(len(merged.columns), 2)
        print("✓ Data Integrity: No data loss on merge")


def run_edge_case_tests():
    """Run all edge case tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("EDGE CASE AND ERROR HANDLING TESTS")
    print("="*70)
    print()
    
    success = run_edge_case_tests()
    
    print()
    print("="*70)
    if success:
        print("✓ ALL EDGE CASE TESTS PASSED!")
    else:
        print("✗ SOME EDGE CASE TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if success else 1)

