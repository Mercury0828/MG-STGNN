"""Training script for MG-STGNN."""

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train MG-STGNN')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # TODO: Implement training loop
    # 1. Load data
    # 2. Initialize model
    # 3. 3-phase training: joint (7 ep.) -> CI (5 ep.) -> joint (2 ep.)
    # 4. Best-seed selection from 5 seeds
    raise NotImplementedError("Training implementation available upon publication.")


if __name__ == '__main__':
    main()
