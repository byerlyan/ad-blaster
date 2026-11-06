#!/usr/bin/env python3
"""Compare legacy and hybrid decisions against timestamped labels."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import compare_modes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV: timestamp,label,legacy,hybrid")
    parser.add_argument("--data-kind", required=True, choices=("captured", "simulated"))
    args = parser.parse_args()
    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(json.dumps(compare_modes(rows, data_kind=args.data_kind), indent=2))


if __name__ == "__main__":
    main()
