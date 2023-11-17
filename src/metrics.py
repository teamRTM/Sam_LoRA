import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auroc_from_logits(y_true, logits):
    
    y_scores = 1 / (1 + np.exp(-logits))
    
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()

    auroc = roc_auc_score(y_true_flat, y_scores_flat)
    
    return auroc
