"""
MG-STGNN model definition.

Mechanism-Guided Spatiotemporal Graph Neural Network for CBAM policy evaluation
in interconnected European electricity markets.
"""

import torch
import torch.nn as nn
from .layers import MeritOrderLayer, TradeArbitrageLayer, CarbonCostLayer, TCNBlock


class MGSTGNN(nn.Module):
    """Mechanism-Guided Spatiotemporal Graph Neural Network.

    Architecture:
        1. Mechanism-typed heterogeneous message passing (MO, TA, CC channels)
        2. Temporal convolutional network (TCN) encoding
        3. Dual-target prediction heads (price + carbon intensity)
    """

    def __init__(self, input_dim, hidden_dim=64, num_nodes=15, num_layers=2,
                 tcn_layers=4, dropout=0.15, num_gen_features=7):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Mechanism-typed message passing layers
        self.mo_layers = nn.ModuleList([
            MeritOrderLayer(hidden_dim, num_gen_features) for _ in range(num_layers)
        ])
        self.ta_layers = nn.ModuleList([
            TradeArbitrageLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.cc_layers = nn.ModuleList([
            CarbonCostLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Aggregation
        self.agg_linear = nn.Linear(3 * hidden_dim, hidden_dim)

        # Temporal encoding (TCN)
        self.tcn = TCNBlock(hidden_dim, hidden_dim, tcn_layers)

        # Dual-target prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.ci_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, gen_features, cbam_cost, ets_price, flows):
        """Forward pass.

        Args:
            x: Node features [batch, time, nodes, features]
            adj: Dynamic adjacency matrix [batch, time, nodes, nodes]
            gen_features: Generation mix features [batch, time, nodes, gen_dim]
            cbam_cost: CBAM cost signal [batch, time, nodes]
            ets_price: ETS price [batch, time]
            flows: Cross-border flows [batch, time, nodes, nodes]

        Returns:
            price_pred: Price predictions [batch, nodes]
            ci_pred: Carbon intensity predictions [batch, nodes]
        """
        # TODO: Implement full forward pass
        # 1. Project input features
        # 2. For each GNN layer: compute MO, TA, CC messages, aggregate
        # 3. TCN temporal encoding
        # 4. Dual-target prediction heads
        raise NotImplementedError("Full implementation available upon publication.")
