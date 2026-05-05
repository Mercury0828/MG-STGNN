#!/bin/bash
# Run CBAM scenario analysis
for gamma in 0.25 0.50 0.75 1.0; do
    echo "Running gamma=$gamma"
    python src/cbam_rollout.py --gamma $gamma --checkpoint checkpoints/best_model.pt
done
