"""Thresholds that decide whether a retrained classifier replaces the deployed one.

Kept free of ML imports so summaries can explain the rules without loading torch.
"""

PROMOTION_BOOTSTRAP_SAMPLES = 1000
PROMOTION_F1_NONINFERIORITY_MARGIN = 0.02
PROMOTION_MAX_CLASS_RECALL_DROP = 0.10
PROMOTION_MIN_CLASS_SUPPORT = 10
