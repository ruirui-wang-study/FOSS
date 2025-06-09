import numpy as np
import random
import math
from utils import get_dimension, compute_weighted_entropy

def FOSSTree(S, h, hmax, Smin=1, nsam=None):
    # 停止条件：深度、样本数量、或所有样本相同
    if h >= hmax or len(S) <= Smin or np.all(S == S[0]):
        return {"type": "leaf", "size": len(S)}

    # 默认 nsam = sqrt(d)
    if nsam is None:
        nsam = max(1, int(math.sqrt(S.shape[1])))

    # Step 1: 选择最优维度
    q = get_dimension(S, nsam)

    # Step 2: 随机选择分裂值 p ∈ [min, max]
    col = S[:, q]
    min_val, max_val = np.min(col), np.max(col)
    if min_val == max_val:
        return {"type": "leaf", "size": len(S)}  # 无法继续分裂

    p = random.uniform(min_val, max_val)

    # Step 3: 分割数据集
    Sl = S[col < p]
    Sr = S[col >= p]

    # 若有一个子集为空，则不继续分裂
    if len(Sl) == 0 or len(Sr) == 0:
        return {"type": "leaf", "size": len(S)}

    # Step 4: 递归构造子树
    return {
        "type": "node",
        "splitDim": q,
        "splitVal": p,
        "left": FOSSTree(Sl, h + 1, hmax, Smin, nsam),
        "right": FOSSTree(Sr, h + 1, hmax, Smin, nsam)
    }
