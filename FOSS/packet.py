import numpy as np
from collections import Counter, defaultdict
import random
import math
# 简易版，能跑起来
# ------------------ 特征提取 --------------------

def extract_features(flow):
    """
    flow: dict,包含如下字段示例
    {
        'src_ip': str,
        'src_port': int,
        'dst_ip': str,
        'dst_port': int,
        'protocol': int,
        'flags_forward': [int...],  # 33 flags统计
        'flags_backward': [int...],
        'flags_bidirectional': [int...],
        'stats_forward': [float...], # 72维统计 max,min,mean,std
        'stats_backward': [float...],
        'stats_bidirectional': [float...]
    }
    输出一个数值特征向量 np.array
    """
    features = []

    # (i) 协议编码（1维）
    features.append(flow['protocol'])

    # (ii) flags统计 (33*3=99维)
    features.extend(flow['flags_forward'])
    features.extend(flow['flags_backward'])
    features.extend(flow['flags_bidirectional'])

    # (iii) 72维统计 *3 = 216维
    features.extend(flow['stats_forward'])
    features.extend(flow['stats_backward'])
    features.extend(flow['stats_bidirectional'])

    return np.array(features, dtype=float)


# ------------------ FOSSTree & FOSSForest --------------------

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def weighted_entropy(Sq):
    """
    计算单个维度q的加权熵
    Sq: np.array,维度q的样本值
    """
    # 统计唯一值及概率分布
    values, counts = np.unique(Sq, return_counts=True)
    probs = counts / counts.sum()

    # 归一化values
    norm_vals = min_max_normalize(values)

    # 期望
    E = np.sum(norm_vals * probs)

    # 计算加权熵
    H = 0
    for pi, xi in zip(probs, norm_vals):
        if pi > 0:
            denom = abs(xi - E)
            if denom == 0:
                denom = 1e-6  # 避免除0
            H -= (pi * math.log2(pi)) / denom
    return H

def get_dimension(S, nsam=None):
    """
    选择最优分割维度
    S: np.array (样本数 * 特征数)
    nsam: 随机采样维度数 <= d
    """
    n_samples, d = S.shape
    if nsam is None:
        nsam = d
    dims = random.sample(range(d), nsam)

    Hqs = []
    for q in dims:
        Sq = S[:, q]
        Hq = weighted_entropy(Sq)
        Hqs.append((q, Hq))
    Hqs.sort(key=lambda x: x[1])
    return Hqs[0][0]

class FOSSTreeNode:
    def __init__(self, left=None, right=None, split_dim=None, split_val=None, leaf_data=None):
        self.left = left
        self.right = right
        self.split_dim = split_dim
        self.split_val = split_val
        self.leaf_data = leaf_data  # np.array of samples at leaf

def build_FOSSTree(S, h, hmax, Smin=10):
    """
    递归构建FOSSTree
    """
    n_samples = S.shape[0]
    if h >= hmax or n_samples <= Smin or np.all(np.std(S, axis=0) == 0):
        return FOSSTreeNode(leaf_data=S)
    q = get_dimension(S, nsam=int(np.sqrt(S.shape[1])))
    split_vals = S[:, q]
    p = random.uniform(np.min(split_vals), np.max(split_vals))
    Sl = S[S[:, q] < p]
    Sr = S[S[:, q] >= p]
    # 递归
    left = build_FOSSTree(Sl, h + 1, hmax, Smin)
    right = build_FOSSTree(Sr, h + 1, hmax, Smin)
    return FOSSTreeNode(left, right, q, p)

def path_length(x, node, current_length=0):
    """
    样本x在树中的路径长度
    """
    if node.leaf_data is not None:
        return current_length
    if x[node.split_dim] < node.split_val:
        return path_length(x, node.left, current_length + 1)
    else:
        return path_length(x, node.right, current_length + 1)

def leaf_node_centroid_radius(node):
    """
    计算叶节点质心和半径
    """
    if node.leaf_data is None or node.leaf_data.shape[0] == 0:
        return None, None
    centroid = np.mean(node.leaf_data, axis=0)
    dists = np.linalg.norm(node.leaf_data - centroid, axis=1)
    radius = np.max(dists)
    return centroid, radius

def find_leaf_node(x, node):
    if node.leaf_data is not None:
        return node
    if x[node.split_dim] < node.split_val:
        return find_leaf_node(x, node.left)
    else:
        return find_leaf_node(x, node.right)

class FOSSForest:
    def __init__(self, Ntree=10, subset_size=100, hmax=None, Smin=10):
        self.Ntree = Ntree
        self.subset_size = subset_size
        self.hmax = hmax
        self.Smin = Smin
        self.trees = []

    def fit(self, X):
        n_samples, d = X.shape
        if self.hmax is None:
            self.hmax = int(np.ceil(np.log2(self.subset_size)))
        self.trees = []
        for _ in range(self.Ntree):
            idx = np.random.choice(n_samples, self.subset_size, replace=True)
            subset = X[idx]
            tree = build_FOSSTree(subset, 1, self.hmax, self.Smin)
            self.trees.append(tree)

    def predict(self, X, path_thresh_ratio=0.1, deviation_thresh_ratio=0.5):
        """
        预测是否异常（UnknownClass）或已知类（返回叶节点标签）
        这里叶节点暂时假设一个标签,模拟已知类统一标为"KnownClass"
        """
        results = []
        # 1. 收集所有树中该样本路径长度
        for x in X:
            path_lengths = []
            centroids = []
            radii = []
            votes = []
            for tree in self.trees:
                length = path_length(x, tree)
                path_lengths.append(length)
                leaf = find_leaf_node(x, tree)
                c, r = leaf_node_centroid_radius(leaf)
                centroids.append(c)
                radii.append(r)
                # 这里默认叶节点标签为"KnownClass"
                votes.append("KnownClass")

            # 2. Isolation Path Anomaly判断：路径长度低于阈值
            path_lengths = np.array(path_lengths)
            thresh = np.quantile(path_lengths, path_thresh_ratio)  # 取路径长度前10%为异常阈值
            path_anomaly = path_lengths < thresh

            # 3. Instance Deviation Anomaly判断：距离质心 > 半径 * 阈值
            deviation_anomaly = []
            for i, x_i in enumerate(X):
                local_dev = []
                for j in range(len(self.trees)):
                    c = centroids[j]
                    r = radii[j]
                    if c is None or r is None:
                        local_dev.append(False)
                        continue
                    dist = np.linalg.norm(x_i - c)
                    local_dev.append(dist > r * deviation_thresh_ratio)
                deviation_anomaly.append(local_dev)
            deviation_anomaly = np.array(deviation_anomaly)

            # 4. 多树投票
            for i in range(len(X)):
                votes_new = []
                for j in range(self.Ntree):
                    if path_anomaly[j] and deviation_anomaly[i][j]:
                        votes_new.append("UnknownClass")
                    else:
                        votes_new.append("KnownClass")
                count = Counter(votes_new)
                label = count.most_common(1)[0][0]
                results.append(label)
            break  # 这里X是多个样本,简单处理成一个batch,返回第一个样本结果,演示用

        return results[0] if len(results) == 1 else results


# ------------------ 模拟数据生成及测试 --------------------

def simulate_network_flow(n_samples=500):
    flows = []
    for _ in range(n_samples):
        flow = {
            'src_ip': '192.168.1.' + str(random.randint(1, 254)),
            'src_port': random.randint(1024, 65535),
            'dst_ip': '10.0.0.' + str(random.randint(1, 254)),
            'dst_port': random.randint(80, 8080),
            'protocol': random.choice([6, 17]),  # TCP=6, UDP=17
            'flags_forward': [random.randint(0, 1) for _ in range(33)],
            'flags_backward': [random.randint(0, 1) for _ in range(33)],
            'flags_bidirectional': [random.randint(0, 1) for _ in range(33)],
            'stats_forward': [random.uniform(0, 100) for _ in range(72)],
            'stats_backward': [random.uniform(0, 100) for _ in range(72)],
            'stats_bidirectional': [random.uniform(0, 100) for _ in range(72)],
        }
        flows.append(flow)
    return flows

def main():
    # 1. 模拟训练数据,提取特征
    train_flows = simulate_network_flow(1000)
    X_train = np.array([extract_features(f) for f in train_flows])

    # 2. 训练FOSSForest
    forest = FOSSForest(Ntree=10, subset_size=200)
    forest.fit(X_train)

    # 3. 模拟测试样本,包含正常和异常
    test_flows = simulate_network_flow(10)
    X_test = np.array([extract_features(f) for f in test_flows])

    # 手动制造异常样本（偏移特征）
    X_test[0] += 50  # 异常偏移
    X_test[1] += 40  # 异常偏移

    # 4. 预测
    results = forest.predict(X_test)
    print("检测结果:", results)
    # 结果是 "KnownClass" 或 "UnknownClass"

if __name__ == '__main__':
    main()
