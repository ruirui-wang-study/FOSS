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

# ---- FOSSTree 构建 ----

def FOSSTree(S, h, hmax, Smin=1, nsam=None):
    if h >= hmax or len(S) <= Smin or np.all(S == S[0]):
        return {"type": "leaf", "size": len(S)}
    if nsam is None:
        nsam = max(1, int(math.sqrt(S.shape[1])))

    q = get_dimension(S, nsam)
    col = S[:, q]
    min_val, max_val = np.min(col), np.max(col)
    if min_val == max_val:
        return {"type": "leaf", "size": len(S)}

    p = random.uniform(min_val, max_val)
    Sl = S[col < p]
    Sr = S[col >= p]

    if len(Sl) == 0 or len(Sr) == 0:
        return {"type": "leaf", "size": len(S)}

    return {
        "type": "node",
        "splitDim": q,
        "splitVal": p,
        "left": FOSSTree(Sl, h + 1, hmax, Smin, nsam),
        "right": FOSSTree(Sr, h + 1, hmax, Smin, nsam)
    }

# ---- FOSSForest 类 ----

class FOSSForest:
    def __init__(self, n_trees=10, subset_size=64, Smin=1):
        self.n_trees = n_trees
        self.subset_size = subset_size
        self.Smin = Smin
        self.trees = []

    def fit(self, X):
        n_samples = X.shape[0]
        hmax = math.ceil(math.log2(self.subset_size))
        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, self.subset_size, replace=True)
            subset = X[indices]
            tree = FOSSTree(subset, h=0, hmax=hmax, Smin=self.Smin)
            self.trees.append(tree)

    def get_trees(self):
        return self.trees

    # ---- 异常检测评分：平均路径长度 ----
    def path_length(self, x, node, depth=0):
        if node["type"] == "leaf":
            return depth
        if x[node["splitDim"]] < node["splitVal"]:
            return self.path_length(x, node["left"], depth + 1)
        else:
            return self.path_length(x, node["right"], depth + 1)

    def score(self, x):
        lengths = [self.path_length(x, tree) for tree in self.trees]
        return np.mean(lengths)

    def anomaly_score(self, x):
        # 转换为 numpy 数组
        if isinstance(x, list):
            x = np.array(x)
        return self.score(x)

# ---- 示例使用 ----

if __name__ == "__main__":
    # 生成数据：二维数据 + 一些离群点
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=(200, 2))
    outliers = np.random.normal(loc=8, scale=1, size=(5, 2))
    X = np.vstack([normal_data, outliers])

    # 训练森林
    forest = FOSSForest(n_trees=25, subset_size=64)
    forest.fit(X)

    # 输出一棵树结构（可选）
    import pprint
    print("\nSample Tree Structure:")
    pprint.pprint(forest.get_trees()[0])

    # 测试评分
    print("\nAnomaly Scores:")
    for i, x in enumerate(X[-5:]):  # 查看后5个异常点
        score = forest.anomaly_score(x)
        print(f"Point {i+1} = {x}, Score = {score:.3f}")

