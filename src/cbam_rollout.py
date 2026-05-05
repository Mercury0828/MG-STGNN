"""CBAM counterfactual scenario rollout."""

import argparse


def cbam_cost(ci, tau=50.0, gamma=1.0, ets_price=85.0):
    """Compute CBAM cost signal (Eq. 2 in paper)."""
    return max(0, ci - tau) * gamma * ets_price / 1000.0


def main():
    parser = argparse.ArgumentParser(description='CBAM Scenario Rollout')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tau', type=float, default=50.0)
    args = parser.parse_args()

    # TODO: Implement CBAM rollout
    # 1. Load trained model
    # 2. For each gamma level, inject CBAM cost into input features
    # 3. Run forward pass to get counterfactual predictions
    # 4. Compute price and CI deltas
    raise NotImplementedError("CBAM rollout implementation available upon publication.")


if __name__ == '__main__':
    main()
