import numpy as np

def histogram_intersection_over_union(p, q):
    
    # Normalize distributions to make sure they are probability distributions
    p = np.array(p) / np.sum(p) if np.sum(p) != 0 else 0.0000001
    q = np.array(q) / np.sum(q) if np.sum(q) != 0 else 0.0000001

    # Calculate the intersection and union
    intersection = np.minimum(p, q)
    union = np.maximum(p, q)

    # Calculate IoU score
    iou_score = np.sum(intersection) / np.sum(union)
    if np.sum(union) == 0 or np.sum(intersection) == 0:
      iou_score = 0.0000001

    return iou_score