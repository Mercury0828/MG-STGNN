"""Ablation-based mechanism attribution for MG-STGNN."""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Mechanism Attribution')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    # TODO: Implement mechanism attribution
    # 1. Load trained model
    # 2. For each mechanism (MO, TA, CC): zero out channel via forward hooks
    # 3. Compute attribution fraction per country
    raise NotImplementedError("Attribution implementation available upon publication.")


if __name__ == '__main__':
    main()
