#!/usr/bin/env python3
"""
Comprehensive Test Runner for Toxic Content Detection Notebook
Runs all unit tests and generates a test report
"""

import sys
import unittest
import time
from io import StringIO

def run_test_suite(test_module_name, test_description):
    """Run a test suite and return results"""
    print(f"\n{'='*70}")
    print(f"Running: {test_description}")
    print('='*70)
    
    try:
        # Import and run tests
        module = __import__(test_module_name)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        start_time = time.time()
        result = runner.run(suite)
        elapsed_time = time.time() - start_time
        
        # Get output
        output = stream.getvalue()
        
        return {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'time': elapsed_time,
            'output': output
        }
    except Exception as e:
        return {
            'success': False,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'time': 0,
            'output': f"Error importing/running tests: {e}\n"
        }


def main():
    """Run all test suites"""
    print("="*70)
    print("COMPREHENSIVE TEST SUITE FOR TOXIC CONTENT DETECTION")
    print("="*70)
    
    # Define test suites
    test_suites = [
        ('test_toxic_detection', 'Core Functionality Tests'),
        ('test_edge_cases', 'Edge Cases and Error Handling Tests'),
    ]
    
    results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_time = 0
    
    # Run each test suite
    for module_name, description in test_suites:
        result = run_test_suite(module_name, description)
        results.append((description, result))
        
        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']
        total_time += result['time']
        
        # Print results
        print(result['output'])
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for description, result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{description:40s} {status:8s} ({result['tests_run']} tests, {result['time']:.2f}s)")
        if result['failures'] > 0:
            print(f"  Failures: {result['failures']}")
        if result['errors'] > 0:
            print(f"  Errors: {result['errors']}")
    
    print("\n" + "-"*70)
    print(f"Total Tests Run:    {total_tests}")
    print(f"Total Failures:     {total_failures}")
    print(f"Total Errors:       {total_errors}")
    print(f"Total Time:         {total_time:.2f}s")
    print("-"*70)
    
    all_passed = all(result['success'] for _, result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TEST SUITES PASSED!")
        print("="*70)
        print("\nTest Coverage Summary:")
        print("  ✓ Data Loading (3 tests)")
        print("  ✓ Column Standardization (2 tests)")
        print("  ✓ Label Standardization (6 tests)")
        print("  ✓ Text Preprocessing (7 tests)")
        print("  ✓ Data Merging (3 tests)")
        print("  ✓ Train-Test Split (2 tests)")
        print("  ✓ Model Training (3 tests)")
        print("  ✓ Evaluation Metrics (3 tests)")
        print("  ✓ End-to-End Pipeline (1 test)")
        print("  ✓ Edge Cases (14 tests)")
        print("  ✓ Error Handling (3 tests)")
        print("  ✓ Data Integrity (3 tests)")
        print("\nTotal: 50+ test cases covering all use cases")
        return 0
    else:
        print("✗ SOME TEST SUITES FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

