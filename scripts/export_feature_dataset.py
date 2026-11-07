#!/usr/bin/env python3
"""Export the durable labeled feature database to training CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from ml_artifacts import FeatureDatasetStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=config.ML_DATASET_PATH)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = FeatureDatasetStore(args.database).export_csv(args.output)
    print(f"Exported {count} labeled samples to {args.output}")


if __name__ == "__main__":
    main()
