"""Entry point for running dataset_compression as a module.

Usage:
    python -m dataset_compression --predictions predictions/ --output outputs/ --language en
"""
from .pipeline import main

if __name__ == "__main__":
    main()
