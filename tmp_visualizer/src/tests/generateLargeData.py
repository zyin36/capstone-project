#!/usr/bin/env python3

import json
import random
import argparse
from pathlib import Path

# ---- Argument Parsing ----

parser = argparse.ArgumentParser(
    description="Generate a large JSON chart file of specified size (in MB)."
)
parser.add_argument(
    "mb",
    nargs="?",
    type=float,
    default=1,
    help="Target size in megabytes (default: 1)"
)
args = parser.parse_args()

TARGET_SIZE_BYTES = int(args.mb * 1_000_000)

# ---- Paths ----

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = "data"
OUTPUT_FILE = SCRIPT_DIR / DATA_DIR / f"large_chart_{args.mb}mb.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---- Chart Template ----

legend = [
    {"name": "GPU-only",        "color": [0.1, 0.8, 1.0, 1.0]},
    {"name": "CPU-only",        "color": [1.0, 0.6, 0.1, 1.0]},
    {"name": "GPU/CPU overlap", "color": [0.6, 0.4, 1.0, 1.0]},
    {"name": "Data wait",       "color": [0.5, 0.5, 0.5, 1.0]},
]

def encode(obj):
    return len(json.dumps(obj, separators=(",", ":")).encode())

# ---- Measure overhead ----

base_chart = {"legend": legend, "bars": []}
base_size = encode(base_chart)

# Measure size of one bar
sample_bar = {
    "x": 0,
    "y": 50,
    "w": 80,
    "h": 120,
    "label": "Run 0",
    "segments": [50, 50, 50, 50],
}

one_bar_size = encode({"legend": legend, "bars": [sample_bar]}) - base_size

if one_bar_size <= 0:
    raise RuntimeError("Bar size measurement failed.")

# ---- Compute number of bars ----

bars_needed = max((TARGET_SIZE_BYTES - base_size) // one_bar_size, 0)

print(f"Target size: {TARGET_SIZE_BYTES/1024:.1f} KB")
print(f"Base size: {base_size} bytes")
print(f"Per-bar size (approx): {one_bar_size} bytes")
print(f"Bars needed: {bars_needed}")

# ---- Generate bars ----

bars = []
x_start = -5000
x_spacing = 90

for i in range(int(bars_needed)):
    bars.append({
        "x": x_start + i * x_spacing,
        "y": 50,
        "w": 80,
        "h": 120,
        "label": f"Run {i}",
        "segments": [random.randint(5, 100) for _ in range(4)],
    })

# ---- Final encode ----

final_json = json.dumps(
    {"legend": legend, "bars": bars},
    separators=(",", ":")
).encode()

with open(OUTPUT_FILE, "wb") as f:
    f.write(final_json)

print(f"Final size: {len(final_json)/1024:.2f} KB")
print(f"Bars generated: {len(bars)}")
print(f"Output: {OUTPUT_FILE}")