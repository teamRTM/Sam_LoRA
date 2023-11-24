import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

def calculate_pixel_auroc_from_logits(y_true, logits):
    y_scores = 1 / (1 + np.exp(-logits))
    
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()

    auroc = roc_auc_score(y_true_flat, y_scores_flat)
    
    return auroc

def calculate_best_pixel_f1_score(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_scores.flatten())

    numerator = 2 * (precision * recall)
    denominator = precision + recall
    denominator[denominator == 0] = 1

    f1_scores = numerator / denominator

    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_precision = precision[best_threshold_index]
    best_recall = recall[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    return best_f1_score, best_precision, best_recall, best_threshold

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    
    return iou