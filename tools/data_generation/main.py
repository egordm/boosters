"""Main entry point for test data generation.

Usage:
    uv run python -m data_generation.main             # Generate all test cases
    uv run python -m data_generation.main xgboost     # Generate XGBoost only (inference + training)
"""
import sys


def main():
    args = sys.argv[1:]
    
    if not args or "xgboost" in args or "all" in args:
        from scripts.generate_xgboost import generate_all
        generate_all()
    
    # Future: add lightgbm, catboost, etc.
    # if not args or "lightgbm" in args or "all" in args:
    #     from scripts.generate_lightgbm import generate_all
    #     generate_all()


if __name__ == "__main__":
    main()
