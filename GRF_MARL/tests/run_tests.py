#!/usr/bin/env python3
"""
GRF_MARL Test Runner

Usage:
    python -m tests.run_tests                    # Run all tests
    python -m tests.run_tests --module env       # Run environment tests only
    python -m tests.run_tests --module hier      # Run hierarchical tests only
    python -m tests.run_tests --quick            # Skip slow tests
    python -m tests.run_tests --gpu              # Include GPU tests
    python -m tests.run_tests --verbose          # Detailed output
"""

import sys
import argparse
import os

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import TestResults, print_section


def run_environment_tests(args) -> TestResults:
    """Run environment and dependency tests."""
    print_section("Environment Tests")
    from tests import test_environment
    return test_environment.run_all(
        quick=args.quick,
        gpu=args.gpu,
        verbose=args.verbose
    )


def run_hierarchical_tests(args) -> TestResults:
    """Run hierarchical policy tests."""
    print_section("Hierarchical Policy Tests")
    from tests import test_hierarchical
    return test_hierarchical.run_all(verbose=args.verbose)


def main():
    parser = argparse.ArgumentParser(description="GRF_MARL Test Suite")
    parser.add_argument(
        "--module", "-m",
        choices=["env", "environment", "hier", "hierarchical", "all"],
        default="all",
        help="Test module to run"
    )
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  GRF_MARL Test Suite")
    print("=" * 60)
    
    results = TestResults()
    
    # Run selected test modules
    module = args.module.lower()
    
    if module in ["env", "environment", "all"]:
        env_results = run_environment_tests(args)
        results.merge(env_results)
    
    if module in ["hier", "hierarchical", "all"]:
        hier_results = run_hierarchical_tests(args)
        results.merge(hier_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  Passed:  {results.passed}")
    print(f"  Failed:  {results.failed}")
    print(f"  Skipped: {results.skipped}")
    
    if results.errors:
        print(f"\n  Failed tests:")
        for err in results.errors[:10]:
            print(f"    - {err}")
        if len(results.errors) > 10:
            print(f"    ... and {len(results.errors) - 10} more")
    
    print("\n" + "=" * 60)
    if results.all_passed:
        print("  ✅ All tests passed!")
    else:
        print("  ❌ Some tests failed. Check errors above.")
    print("=" * 60 + "\n")
    
    return 0 if results.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

