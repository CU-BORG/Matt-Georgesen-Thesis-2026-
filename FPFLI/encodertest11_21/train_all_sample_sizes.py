#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train models on all available sample sizes sequentially
Wrapper script that calls Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py
for each sample size: 1, 10, 100, 200, 500, 1000

@author: mg
"""

import subprocess
import time
import sys
import os

SAMPLE_SIZES = [1, 10, 100, 200, 500, 1000]

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("BATCH TRAINING - ALL SAMPLE SIZES")
print("=" * 80)
print(f"Will train models on sample sizes: {SAMPLE_SIZES}")
print(f"Total: {len(SAMPLE_SIZES)} training runs")
print("=" * 80 + "\n")

overall_start = time.time()
results = []

for idx, size in enumerate(SAMPLE_SIZES):
    print("\n" + "#" * 80)
    print(f"# TRAINING {idx + 1}/{len(SAMPLE_SIZES)}: n={size} images")
    print("#" * 80 + "\n")

    start_time = time.time()

    # Call the training script with this sample size
    training_script = os.path.join(SCRIPT_DIR, "Training_LLE_BiExp_Autoencoder_FullyFlatRandom_Batch.py")

    try:
        result = subprocess.run(
            [sys.executable,
             training_script,
             str(size)],
            check=True,
            capture_output=False,  # Show output in real-time
            cwd=SCRIPT_DIR  # Run in the script directory
        )

        elapsed = time.time() - start_time
        status = "SUCCESS"
        print(f"\n[OK] Training completed for n={size} in {elapsed/60:.1f} minutes")

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        status = "FAILED"
        print(f"\n[ERROR] Training failed for n={size}")
        print(f"Error: {e}")

    results.append({
        'size': size,
        'status': status,
        'time': elapsed
    })

# Final summary
total_time = time.time() - overall_start

print("\n" + "=" * 80)
print("BATCH TRAINING COMPLETE")
print("=" * 80)
print("\nResults:")
print(f"{'Size':<10}{'Status':<15}{'Time (min)':<15}")
print("-" * 40)

for r in results:
    print(f"{r['size']:<10}{r['status']:<15}{r['time']/60:.1f}")

print("-" * 40)
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

successes = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"Successful: {successes}/{len(results)}")
print("=" * 80)
