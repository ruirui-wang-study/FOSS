import numpy as np
import math
import random
from collections import Counter

# ---- 工具函数 ----

def max_min_normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)

def compute_weighted_entropy(values, epsilon=1e-10):
    counts = Counter(values)
    total = len(values)
    unique_vals = np.array(list(counts.keys()))
    probs = np.array([counts[v] / total for v in unique_vals])
    norm_vals = max_min_normalize(unique_vals)
    expectation = np.sum(norm_vals * probs)
    entropy = 0.0
    for p, v in zip(probs, norm_vals):
        if p > 0:
            entropy += (-p * np.log2(p)) / (abs(v - expectation) + epsilon)
    return entropy

def get_dimension(S, nsam):
    d = S.shape[1]
    F_sam = random.sample(range(d), min(nsam, d))
    entropies = []
    for q in F_sam:
        column = S[:, q]
        H_q = compute_weighted_entropy(column)
        entropies.append((q, H_q))
    return min(entropies, key=lambda x: x[1])[0]
