import numpy as np
import math
from collections import Counter, defaultdict
import random
import pandas as pd
from datetime import datetime
# 还需要优化,FOSSForest参数对不上
# ----- Part 1: 特征提取 -----
class Packet:
    def __init__(self, timestamp, direction, ip_frag, tcp_flags, ttl, window_size, length=0):
        self.timestamp = timestamp
        self.direction = direction
        self.ip_frag = ip_frag
        self.tcp_flags = tcp_flags
        self.ttl = ttl
        self.window_size = window_size
        self.length = length

class FlowSession:
    def __init__(self, src_ip, src_port, dst_ip, dst_port, protocol, label=None):
        self.key = (src_ip, src_port, dst_ip, dst_port, protocol)
        self.protocol = protocol
        self.packets = []
        self.label = label
        self.flow_duration = 0
        self.start_time = None
        self.end_time = None
        
        # 基本流量统计
        self.total_fwd_packets = 0
        self.total_bwd_packets = 0
        self.total_fwd_bytes = 0
        self.total_bwd_bytes = 0
        
        # 包长度统计
        self.fwd_packet_lengths = []
        self.bwd_packet_lengths = []
        
        # 时间间隔统计
        self.fwd_iat = []  # 前向包间隔时间
        self.bwd_iat = []  # 后向包间隔时间
        
        # TCP标志统计
        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0
        self.fwd_header_length = 0
        self.bwd_header_length = 0
        self.fin_flag_count = 0
        self.syn_flag_count = 0
        self.rst_flag_count = 0
        self.psh_flag_count = 0
        self.ack_flag_count = 0
        self.urg_flag_count = 0
        
        # 窗口大小统计
        self.init_win_bytes_forward = 0
        self.init_win_bytes_backward = 0
        self.fwd_win_bytes = []
        self.bwd_win_bytes = []
        
        # TTL统计
        self.fwd_ttl = []
        self.bwd_ttl = []
        
        # 新增：IP分片统计
        self.fwd_ip_fragments = 0
        self.bwd_ip_fragments = 0
        
        # 新增：协议特定统计
        self.tcp_retransmissions = 0
        self.tcp_duplicate_acks = 0
        self.tcp_zero_window = 0

    def add_packet(self, pkt):
        self.packets.append(pkt)
        
        # 更新流量统计
        if pkt.direction == 'forward':
            self.total_fwd_packets += 1
            self.total_fwd_bytes += pkt.length
            self.fwd_packet_lengths.append(pkt.length)
            self.fwd_ttl.append(pkt.ttl)
            self.fwd_win_bytes.append(pkt.window_size)
            if pkt.ip_frag:
                self.fwd_ip_fragments += 1
            if len(self.packets) > 1 and self.packets[-2].direction == 'forward':
                self.fwd_iat.append(pkt.timestamp - self.packets[-2].timestamp)
        else:  # backward
            self.total_bwd_packets += 1
            self.total_bwd_bytes += pkt.length
            self.bwd_packet_lengths.append(pkt.length)
            self.bwd_ttl.append(pkt.ttl)
            self.bwd_win_bytes.append(pkt.window_size)
            if pkt.ip_frag:
                self.bwd_ip_fragments += 1
            if len(self.packets) > 1 and self.packets[-2].direction == 'backward':
                self.bwd_iat.append(pkt.timestamp - self.packets[-2].timestamp)
        
        # 更新TCP标志
        if pkt.tcp_flags & 0x01:  # FIN
            self.fin_flag_count += 1
        if pkt.tcp_flags & 0x02:  # SYN
            self.syn_flag_count += 1
        if pkt.tcp_flags & 0x04:  # RST
            self.rst_flag_count += 1
        if pkt.tcp_flags & 0x08:  # PSH
            self.psh_flag_count += 1
            if pkt.direction == 'forward':
                self.fwd_psh_flags += 1
            else:
                self.bwd_psh_flags += 1
        if pkt.tcp_flags & 0x10:  # ACK
            self.ack_flag_count += 1
        if pkt.tcp_flags & 0x20:  # URG
            self.urg_flag_count += 1
            if pkt.direction == 'forward':
                self.fwd_urg_flags += 1
            else:
                self.bwd_urg_flags += 1
        
        # 更新窗口大小
        if pkt.direction == 'forward' and self.init_win_bytes_forward == 0:
            self.init_win_bytes_forward = pkt.window_size
        elif pkt.direction == 'backward' and self.init_win_bytes_backward == 0:
            self.init_win_bytes_backward = pkt.window_size
        
        # 更新流持续时间
        if self.start_time is None:
            self.start_time = pkt.timestamp
        self.end_time = pkt.timestamp
        self.flow_duration = self.end_time - self.start_time

    def extract_features(self):
        """提取增强的流量特征"""
        def get_stats(arr):
            if len(arr) == 0:
                return [0, 0, 0, 0]
            return [np.max(arr), np.min(arr), np.mean(arr), np.std(arr)]
        
        # 1. 协议编码 (1维)
        protocol_feat = np.array([self.protocol], dtype=float)
        
        # 2. 基本流量特征
        # 2.1 流持续时间 (1维)
        duration_feat = np.array([self.flow_duration], dtype=float)
        
        # 2.2 包数量特征 (3维)
        packet_count_feat = np.array([
            self.total_fwd_packets,
            self.total_bwd_packets,
            self.total_fwd_packets + self.total_bwd_packets
        ], dtype=float)
        
        # 2.3 字节数特征 (3维)
        byte_count_feat = np.array([
            self.total_fwd_bytes,
            self.total_bwd_bytes,
            self.total_fwd_bytes + self.total_bwd_bytes
        ], dtype=float)
        
        # 3. 统计特征
        # 3.1 包长度统计 (12维)
        fwd_pkt_len_stats = get_stats(self.fwd_packet_lengths)
        bwd_pkt_len_stats = get_stats(self.bwd_packet_lengths)
        all_pkt_len_stats = get_stats(self.fwd_packet_lengths + self.bwd_packet_lengths)
        
        # 3.2 流速率特征 (4维)
        if self.flow_duration > 0:
            flow_bytes_per_sec = (self.total_fwd_bytes + self.total_bwd_bytes) / self.flow_duration
            flow_packets_per_sec = (self.total_fwd_packets + self.total_bwd_packets) / self.flow_duration
            fwd_packets_per_sec = self.total_fwd_packets / self.flow_duration
            bwd_packets_per_sec = self.total_bwd_packets / self.flow_duration
        else:
            flow_bytes_per_sec = 0
            flow_packets_per_sec = 0
            fwd_packets_per_sec = 0
            bwd_packets_per_sec = 0
        
        rate_feat = np.array([
            flow_bytes_per_sec,
            flow_packets_per_sec,
            fwd_packets_per_sec,
            bwd_packets_per_sec
        ], dtype=float)
        
        # 3.3 包间隔时间统计 (12维)
        fwd_iat_stats = get_stats(self.fwd_iat)
        bwd_iat_stats = get_stats(self.bwd_iat)
        all_iat_stats = get_stats(self.fwd_iat + self.bwd_iat)
        
        # 3.4 TTL统计 (12维)
        fwd_ttl_stats = get_stats(self.fwd_ttl)
        bwd_ttl_stats = get_stats(self.bwd_ttl)
        all_ttl_stats = get_stats(self.fwd_ttl + self.bwd_ttl)
        
        # 3.5 窗口大小统计 (12维)
        fwd_win_stats = get_stats(self.fwd_win_bytes)
        bwd_win_stats = get_stats(self.bwd_win_bytes)
        all_win_stats = get_stats(self.fwd_win_bytes + self.bwd_win_bytes)
        
        # 4. TCP标志特征 (10维)
        flag_feat = np.array([
            self.fin_flag_count,
            self.syn_flag_count,
            self.rst_flag_count,
            self.psh_flag_count,
            self.ack_flag_count,
            self.urg_flag_count,
            self.fwd_psh_flags,
            self.bwd_psh_flags,
            self.fwd_urg_flags,
            self.bwd_urg_flags
        ], dtype=float)
        
        # 5. IP分片特征 (2维)
        ip_frag_feat = np.array([
            self.fwd_ip_fragments,
            self.bwd_ip_fragments
        ], dtype=float)
        
        # 6. 协议特定特征 (3维)
        protocol_specific_feat = np.array([
            self.tcp_retransmissions,
            self.tcp_duplicate_acks,
            self.tcp_zero_window
        ], dtype=float)
        
        # 合并所有特征
        all_features = np.concatenate([
            protocol_feat,          # 1维
            duration_feat,          # 1维
            packet_count_feat,      # 3维
            byte_count_feat,        # 3维
            fwd_pkt_len_stats,      # 4维
            bwd_pkt_len_stats,      # 4维
            all_pkt_len_stats,      # 4维
            rate_feat,              # 4维
            fwd_iat_stats,          # 4维
            bwd_iat_stats,          # 4维
            all_iat_stats,          # 4维
            fwd_ttl_stats,          # 4维
            bwd_ttl_stats,          # 4维
            all_ttl_stats,          # 4维
            fwd_win_stats,          # 4维
            bwd_win_stats,          # 4维
            all_win_stats,          # 4维
            flag_feat,              # 10维
            ip_frag_feat,           # 2维
            protocol_specific_feat  # 3维
        ])
        
        return all_features

    @staticmethod
    def from_csv_row(row):
        """从CSV行创建FlowSession对象"""
        try:
            # 解析Flow ID获取IP和端口信息
            flow_id_parts = row['Flow ID'].split('-')
            src_ip = flow_id_parts[0]
            dst_ip = flow_id_parts[1]
            src_port = int(flow_id_parts[2])
            dst_port = int(flow_id_parts[3])
            protocol = int(flow_id_parts[4])
            
            # 创建FlowSession对象
            flow = FlowSession(src_ip, src_port, dst_ip, dst_port, protocol, row['Label'])
            
            # 设置流持续时间
            flow.flow_duration = float(row['Flow Duration'])
            
            # 设置包统计信息
            flow.total_fwd_packets = int(row['Total Fwd Packets'])
            flow.total_bwd_packets = int(row['Total Backward Packets'])
            flow.total_fwd_bytes = int(row['Total Length of Fwd Packets'])
            flow.total_bwd_bytes = int(row['Total Length of Bwd Packets'])
            
            # 设置TCP标志
            flow.fin_flag_count = int(row['FIN Flag Count'])
            flow.syn_flag_count = int(row['SYN Flag Count'])
            flow.rst_flag_count = int(row['RST Flag Count'])
            flow.psh_flag_count = int(row['PSH Flag Count'])
            flow.ack_flag_count = int(row['ACK Flag Count'])
            flow.urg_flag_count = int(row['URG Flag Count'])
            flow.fwd_psh_flags = int(row['Fwd PSH Flags'])
            flow.bwd_psh_flags = int(row['Bwd PSH Flags'])
            flow.fwd_urg_flags = int(row['Fwd URG Flags'])
            flow.bwd_urg_flags = int(row['Bwd URG Flags'])
            
            # 设置窗口大小
            flow.init_win_bytes_forward = int(row['Init_Win_bytes_forward'])
            flow.init_win_bytes_backward = int(row['Init_Win_bytes_backward'])
            
            # 设置时间戳
            timestamp_str = row['Timestamp']
            try:
                timestamp = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M').timestamp()
                flow.start_time = timestamp
                flow.end_time = timestamp + flow.flow_duration
            except:
                # 如果时间戳解析失败，使用默认值
                flow.start_time = 0
                flow.end_time = flow.flow_duration
            
            return flow
        except Exception as e:
            print(f"处理行时出错: {e}")
            print(f"行数据: {row}")
            raise

class FOSSTreeNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.split_dim = None
        self.split_val = None
        self.left = None
        self.right = None
        self.samples = None  # 样本索引
        self.centroid = None
        self.radius = None
        self.label = None  # 叶节点类别或None

class FOSSTree:
    def __init__(self, hmax, smin, nsam, p_norm=2):
        self.hmax = hmax
        self.smin = smin
        self.nsam = nsam
        self.p_norm = p_norm
        self.root = None
        self.labels = None  # 添加标签数组
        self.feature_names = None  # 添加特征名称

    def fit(self, samples, sample_indices, labels=None, feature_names=None):
        """
        训练FOSS树
        
        Args:
            samples: 样本矩阵
            sample_indices: 样本索引列表
            labels: 样本标签数组
            feature_names: 特征名称列表
        """
        self.samples = samples
        self.labels = labels  # 保存标签数组
        self.feature_names = feature_names  # 保存特征名称
        self.root = self._build_tree(sample_indices, 0)

    def _build_tree(self, sample_indices, depth):
        node = FOSSTreeNode(depth)
        node.samples = sample_indices

        # 叶子判断：超过最大深度或样本数太少,或样本全等
        if depth >= self.hmax or len(sample_indices) <= self.smin or self._all_samples_same(sample_indices):
            node.is_leaf = True
            node.centroid, node.radius = self._compute_centroid_radius(sample_indices)
            
            # 使用实际标签
            if self.labels is not None and len(sample_indices) > 0:
                # 获取该节点所有样本的标签
                node_labels = [self.labels[i] for i in sample_indices]
                # 使用最常见的标签作为节点标签
                most_common = Counter(node_labels).most_common(1)
                if most_common:  # 确保有标签
                    node.label = most_common[0][0]
                else:
                    node.label = 'Unknown'
            else:
                node.label = 'Unknown'  # 如果没有标签信息或没有样本，标记为未知
                
            return node

        # 选维度
        q = self._get_dimension(sample_indices)
        node.split_dim = q

        # 选分割点
        values = self.samples[sample_indices, q]
        if len(values) == 0:  # 处理空值情况
            node.is_leaf = True
            node.centroid, node.radius = self._compute_centroid_radius(sample_indices)
            node.label = 'Unknown'
            return node
            
        p = np.random.uniform(np.min(values), np.max(values))
        node.split_val = p

        # 分割样本
        left_idx = [i for i in sample_indices if self.samples[i, q] < p]
        right_idx = [i for i in sample_indices if self.samples[i, q] >= p]

        # 递归构建
        node.left = self._build_tree(left_idx, depth + 1)
        node.right = self._build_tree(right_idx, depth + 1)
        return node

    def _all_samples_same(self, sample_indices):
        # 判断样本是否都一样(所有特征完全相同)
        if len(sample_indices) <= 1:
            return True
        first = self.samples[sample_indices[0]]
        for idx in sample_indices[1:]:
            if not np.array_equal(first, self.samples[idx]):
                return False
        return True

    def _compute_centroid_radius(self, sample_indices):
        """计算样本集的质心和半径"""
        if len(sample_indices) == 0:
            return np.zeros(self.samples.shape[1]), 0.0
            
        data = self.samples[sample_indices]
        centroid = np.mean(data, axis=0)
        distances = np.linalg.norm(data - centroid, ord=self.p_norm, axis=1)
        radius = np.max(distances) if len(distances) > 0 else 0
        return centroid, radius

    def _get_dimension(self, sample_indices):
        """选择最佳分割维度"""
        if len(sample_indices) == 0:
            return 0
            
        d = self.samples.shape[1]
        # Monte Carlo采样nsam个维度
        nsam = min(self.nsam, d)
        dims = np.random.choice(d, nsam, replace=False)

        best_dim = None
        best_entropy = float('inf')

        for q in dims:
            values = self.samples[sample_indices, q]
            if len(values) > 0:  # 确保有值
                entropy = self._weighted_entropy(values)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_dim = q
                    
        return best_dim if best_dim is not None else 0  # 如果没有找到合适的维度，返回0

    def _weighted_entropy(self, values):
        # max-min归一化
        min_v, max_v = np.min(values), np.max(values)
        if max_v == min_v:
            return 0.0
        norm_values = (values - min_v) / (max_v - min_v)

        counts = Counter(norm_values)
        total = len(norm_values)
        pis = np.array([c / total for c in counts.values()])
        xis = np.array(list(counts.keys()))

        E = np.sum(xis * pis)

        # 计算加权熵
        entropy = 0
        for pi, xi in zip(pis, xis):
            if pi > 0:
                entropy -= (pi * math.log2(pi)) / abs(xi - E + 1e-10)  # +1e-10防止除零
        return entropy

    def path_length(self, x):
        # 计算样本x在树中的路径长度(从根到叶节点)
        node = self.root
        length = 0
        while not node.is_leaf:
            q, p = node.split_dim, node.split_val
            if x[q] < p:
                node = node.left
            else:
                node = node.right
            length += 1
        return length, node

    def get_leaf_info(self, x):
        # 返回样本落叶节点及其质心、半径信息
        length, leaf = self.path_length(x)
        return length, leaf.centroid, leaf.radius, leaf.label

    def visualize_tree(self, filename='tree_visualization.png', max_depth=None):
        """
        可视化树结构
        
        Args:
            filename: 输出文件名
            max_depth: 最大显示深度,None表示显示全部
        """
        import matplotlib.pyplot as plt
        
        def get_tree_depth(node):
            if node.is_leaf:
                return 0
            left_depth = get_tree_depth(node.left) if node.left else 0
            right_depth = get_tree_depth(node.right) if node.right else 0
            return max(left_depth, right_depth) + 1
        
        def plot_node(node, x, y, dx, dy, ax):
            if max_depth is not None and y > max_depth:
                return
            
            # 创建节点标签
            if node.is_leaf:
                label = f"Leaf\nLabel: {node.label}\nSamples: {len(node.samples)}"
            else:
                # 安全地获取特征名称
                if self.feature_names and 0 <= node.split_dim < len(self.feature_names):
                    feature_name = self.feature_names[node.split_dim]
                else:
                    feature_name = f"Feature {node.split_dim}"
                label = f"{feature_name}\n< {node.split_val:.2f}"
            
            # 绘制节点
            circle = plt.Circle((x, y), 0.1, color='lightblue', alpha=0.7)
            ax.add_patch(circle)
            
            # 添加标签
            ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
            
            if not node.is_leaf:
                # 绘制左子树
                if node.left:
                    ax.plot([x, x-dx], [y, y-dy], 'gray', alpha=0.7)
                    plot_node(node.left, x-dx, y-dy, dx/2, dy, ax)
                
                # 绘制右子树
                if node.right:
                    ax.plot([x, x+dx], [y, y-dy], 'gray', alpha=0.7)
                    plot_node(node.right, x+dx, y-dy, dx/2, dy, ax)
        
        # 创建图形
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        
        # 计算树的深度
        tree_depth = get_tree_depth(self.root)
        if max_depth is not None:
            tree_depth = min(tree_depth, max_depth)
        
        # 计算初始间距
        dx = 1.0
        dy = 1.0
        
        # 绘制树
        plot_node(self.root, 0, 0, dx, dy, ax)
        
        # 设置图形属性
        plt.title('FOSS Tree Structure')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

class FOSSForest:
    def __init__(self, Ntree=200, psi=2000, smin=10, nsam=15, p_norm=2):
        """
        初始化FOSS森林
        
        Args:
            Ntree: 树的数量，增加到200以提高模型稳定性
            psi: 每棵树的样本数，增加到2000以捕获更多模式
            smin: 叶节点最小样本数，设置为10以平衡过拟合
            nsam: 特征采样数，增加到15以增加多样性
            p_norm: 距离度量范数，使用2范数（欧氏距离）
        """
        self.Ntree = Ntree
        self.psi = psi
        self.smin = smin
        self.nsam = nsam
        self.p_norm = p_norm
        self.trees = []
        self.hmax = math.ceil(math.log2(psi))  # 最大树深度
        self.X = None  # 训练数据
        self.y = None  # 训练标签
        self.feature_means = None  # 特征均值
        self.feature_stds = None   # 特征标准差
        self.class_weights = None  # 类别权重
        self.feature_names = None  # 特征名称
        self.feature_importance = None  # 特征重要性

    def _normalize_features(self, X):
        """特征归一化，使用稳健的归一化方法"""
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.median(X, axis=0)  # 使用中位数而不是均值
            self.feature_stds = np.percentile(np.abs(X - self.feature_means), 75, axis=0)  # 使用IQR
            # 避免除零
            self.feature_stds[self.feature_stds == 0] = 1
        return (X - self.feature_means) / self.feature_stds

    def _compute_class_weights(self, y):
        """计算类别权重，使用改进的权重计算方法"""
        class_counts = Counter(y)
        total_samples = len(y)
        # 使用平滑的权重计算
        self.class_weights = {
            label: (total_samples / (len(class_counts) * count)) ** 0.5
            for label, count in class_counts.items()
        }

    def _compute_feature_importance(self):
        """计算特征重要性"""
        importance = np.zeros(self.X.shape[1])
        for tree in self.trees:
            importance += self._get_tree_importance(tree)
        self.feature_importance = importance / self.Ntree

    def _get_tree_importance(self, tree):
        """计算单棵树的特征重要性"""
        importance = np.zeros(self.X.shape[1])
        def traverse(node):
            if not node.is_leaf:
                importance[node.split_dim] += len(node.samples)
                traverse(node.left)
                traverse(node.right)
        traverse(tree.root)
        return importance

    def fit(self, X, y=None, feature_names=None):
        """
        训练FOSS森林
        
        Args:
            X: 训练数据矩阵
            y: 训练标签数组
            feature_names: 特征名称列表
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        # 特征归一化
        X_normalized = self._normalize_features(X)
        
        # 计算类别权重
        if y is not None:
            self._compute_class_weights(y)
        
        n_samples = X.shape[0]
        self.trees = []
        
        # 构建每棵树
        for i in range(self.Ntree):
            # 使用分层采样确保每棵树的样本分布更均衡
            if y is not None:
                sample_indices = []
                for label in set(y):
                    label_indices = np.where(y == label)[0]
                    n_samples_per_class = max(1, self.psi // len(set(y)))
                    sampled_indices = np.random.choice(label_indices, 
                                                     size=min(n_samples_per_class, len(label_indices)),
                                                     replace=True)
                    sample_indices.extend(sampled_indices)
            else:
                sample_indices = np.random.choice(n_samples, self.psi, replace=True)
            
            # 创建并训练树
            tree = FOSSTree(self.hmax, self.smin, self.nsam, self.p_norm)
            tree.fit(X_normalized, sample_indices, y, self.feature_names)
            self.trees.append(tree)
            
            # 每10棵树保存一次可视化
            if (i + 1) % 10 == 0:
                tree.visualize_tree(f'tree_{i+1}.png', max_depth=5)
        
        # 计算特征重要性
        self._compute_feature_importance()

    def predict(self, X_test, path_thresholds=None, deviation_threshold=1.0):
        """
        对测试样本进行预测，使用改进的投票机制
        
        Args:
            X_test: 测试数据矩阵
            path_thresholds: 每棵树的路径长度阈值字典
            deviation_threshold: 实例偏差异常的距离阈值
            
        Returns:
            y_pred: 预测标签列表
            unknown_samples: 未知样本索引列表
        """
        # 归一化测试数据
        X_test_normalized = (X_test - self.feature_means) / self.feature_stds
        
        y_pred = []
        unknown_samples = []
        prediction_scores = defaultdict(list)  # 存储每个类别的得分

        for i, x in enumerate(X_test_normalized):
            votes = defaultdict(float)  # 使用浮点数存储加权投票
            unknown_votes = 0
            
            # 每棵树投票
            for t_idx, tree in enumerate(self.trees):
                length, centroid, radius, label = tree.get_leaf_info(x)

                # 判断是否为隔离路径异常
                threshold = path_thresholds.get(t_idx, None) if path_thresholds else None
                is_isolation_anomaly = False
                if threshold is not None:
                    is_isolation_anomaly = length <= threshold

                # 判断是否为实例偏差异常
                dist = np.linalg.norm(x - centroid, ord=self.p_norm)
                is_deviation_anomaly = dist > deviation_threshold * radius

                # 计算投票权重
                if is_isolation_anomaly and is_deviation_anomaly:
                    votes['UnknownClass'] += 1
                    unknown_votes += 1
                else:
                    # 使用改进的权重计算
                    distance_weight = 1.0 / (1.0 + dist/radius)  # 距离越近权重越大
                    class_weight = self.class_weights.get(label, 1.0) if self.class_weights else 1.0
                    tree_weight = 1.0 / (1.0 + length/self.hmax)  # 路径越短权重越大
                    votes[label] += distance_weight * class_weight * tree_weight

            # 记录每个类别的得分
            for label, score in votes.items():
                prediction_scores[label].append(score)

            # 选择得分最高的类别
            if votes:
                final_label = max(votes.items(), key=lambda kv: kv[1])[0]
            else:
                final_label = 'UnknownClass'
                
            y_pred.append(final_label)
            if final_label == 'UnknownClass':
                unknown_samples.append(i)

        return y_pred, unknown_samples

    def get_feature_importance(self):
        """返回特征重要性"""
        if self.feature_importance is None:
            return None
        if self.feature_names is None:
            return self.feature_importance
        return dict(zip(self.feature_names, self.feature_importance))

# ----- Part 3: 构造模拟数据并训练 -----
def generate_sample_flow(label):
    flow = FlowSession("192.168.0.1", 1234, "10.0.0.1", 80, 6, label)
    for _ in range(random.randint(5,15)):
        pkt = Packet(
            timestamp=random.random(),
            direction=random.choice(['forward', 'backward']),
            ip_frag=random.choice([0,1]),
            tcp_flags=random.choice([0x02,0x10,0x01,0x00]),
            ttl=random.randint(30, 64),
            window_size=random.randint(500,1500)
        )
        flow.add_packet(pkt)
    return flow

# ----- Part 3: 数据集处理 -----
def load_traffic_labelling_dataset(file_path):
    """
    加载TrafficLabelling数据集
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        X: 特征矩阵
        y: 标签数组
        feature_names: 特征名称列表
    """
    # 读取CSV文件，处理列名中的空格
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 去除列名中的空格
    
    # 创建FlowSession对象列表
    flows = []
    for _, row in df.iterrows():
        try:
            flow = FlowSession.from_csv_row(row)
            flows.append(flow)
        except Exception as e:
            print(f"处理行时出错: {e}")
            print(f"行数据: {row}")
            continue
    
    # 提取特征
    X = np.array([flow.extract_features() for flow in flows])
    y = np.array([flow.label for flow in flows])
    
    # 生成特征名称
    feature_names = []
    
    # 1. 协议编码 (1维)
    feature_names.append('Protocol')
    
    # 2. 基本流量特征
    # 2.1 流持续时间 (1维)
    feature_names.append('Flow Duration')
    
    # 2.2 包数量特征 (3维)
    feature_names.extend(['Total Fwd Packets', 'Total Bwd Packets', 'Total Packets'])
    
    # 2.3 字节数特征 (3维)
    feature_names.extend(['Total Fwd Bytes', 'Total Bwd Bytes', 'Total Bytes'])
    
    # 3. 统计特征
    # 3.1 包长度统计 (12维)
    for prefix in ['Fwd', 'Bwd', 'All']:
        feature_names.extend([f'{prefix} Packet Length {stat}' for stat in ['Max', 'Min', 'Mean', 'Std']])
    
    # 3.2 流速率特征 (4维)
    feature_names.extend(['Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s'])
    
    # 3.3 包间隔时间统计 (12维)
    for prefix in ['Fwd', 'Bwd', 'All']:
        feature_names.extend([f'{prefix} IAT {stat}' for stat in ['Max', 'Min', 'Mean', 'Std']])
    
    # 3.4 TTL统计 (12维)
    for prefix in ['Fwd', 'Bwd', 'All']:
        feature_names.extend([f'{prefix} TTL {stat}' for stat in ['Max', 'Min', 'Mean', 'Std']])
    
    # 3.5 窗口大小统计 (12维)
    for prefix in ['Fwd', 'Bwd', 'All']:
        feature_names.extend([f'{prefix} Window Size {stat}' for stat in ['Max', 'Min', 'Mean', 'Std']])
    
    # 4. TCP标志特征 (10维)
    feature_names.extend([
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags'
    ])
    
    # 5. IP分片特征 (2维)
    feature_names.extend(['Fwd IP Fragments', 'Bwd IP Fragments'])
    
    # 6. 协议特定特征 (3维)
    feature_names.extend(['TCP Retransmissions', 'TCP Duplicate ACKs', 'TCP Zero Window'])
    
    # 验证特征维度匹配
    assert len(feature_names) == X.shape[1], f"特征名称数量 ({len(feature_names)}) 与特征维度 ({X.shape[1]}) 不匹配"
    
    return X, y, feature_names

def load_friday_afternoon_data():
    """
    加载周五下午的数据集(DDoS和PortScan)
    
    Returns:
        X: 特征矩阵
        y: 标签数组
        feature_names: 特征名称列表
    """
    # 加载DDoS数据
    ddos_file = "FOSS/data/TrafficLabelling /Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    X_ddos, y_ddos, feature_names = load_traffic_labelling_dataset(ddos_file)
    
    # 加载PortScan数据
    portscan_file = "FOSS/data/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    X_portscan, y_portscan, _ = load_traffic_labelling_dataset(portscan_file)
    
    # 合并数据
    X = np.vstack([X_ddos, X_portscan])
    y = np.concatenate([y_ddos, y_portscan])
    
    return X, y, feature_names

if __name__ == "__main__":
    # 加载周五下午的数据
    X, y, feature_names = load_friday_afternoon_data()
    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"标签分布: {Counter(y)}")
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"训练集标签分布: {Counter(y_train)}")
    print(f"测试集标签分布: {Counter(y_test)}")
    
    # 训练FOSSForest
    forest = FOSSForest(Ntree=100, psi=1000, smin=5, nsam=10, p_norm=2)
    forest.fit(X_train, y_train, feature_names)
    
    # 可视化第一棵树（完整结构）
    forest.trees[0].visualize_tree('first_tree_full.png')
    
    # 可视化第一棵树（限制深度为5）
    forest.trees[0].visualize_tree('first_tree_depth5.png', max_depth=5)
    
    # 预测
    y_pred, unknown_samples = forest.predict(X_test)
    
    # 计算评估指标
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_test)),
                yticklabels=sorted(set(y_test)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 计算总体准确率
    accuracy = sum(1 for true, pred in zip(y_test, y_pred) if true == pred) / len(y_test)
    print(f"\n测试集准确率: {accuracy:.2%}")
    
    # 绘制各类别的准确率条形图
    class_accuracies = {}
    for label in set(y_test):
        # 修复索引错误
        correct = sum(1 for true, pred in zip(y_test, y_pred) if true == label and pred == label)
        total = sum(1 for true in y_test if true == label)
        class_accuracies[label] = correct / total if total > 0 else 0
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.title('Accuracy by Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.close()
    
    # 打印一些预测示例
    print("\n预测示例 (前10个样本):")
    for i, (true, pred) in enumerate(zip(y_test[:10], y_pred[:10])):
        print(f"样本 {i}: 真实标签 = {true}, 预测标签 = {pred}")
    
    if unknown_samples:
        print(f"\n未知样本数量: {len(unknown_samples)}")
        # 对未知样本进行聚类
        clusters = forest.cluster_unknowns(unknown_samples, isolation_threshold=0.7, Mc=2)
        print(f"聚类结果: {len(clusters)} 个簇")
        
        # 合并相似簇
        merged_clusters = forest.merge_clusters(clusters, merge_threshold=0.5)
        print(f"合并后: {len(merged_clusters)} 个簇")
        
        # 分析未知样本的真实标签分布
        unknown_true_labels = [y_test[i] for i in unknown_samples]
        print("\n未知样本的真实标签分布:")
        print(Counter(unknown_true_labels))
        
        # 绘制未知样本的真实标签分布饼图
        plt.figure(figsize=(8, 8))
        plt.pie([count for count in Counter(unknown_true_labels).values()],
                labels=[label for label in Counter(unknown_true_labels).keys()],
                autopct='%1.1f%%')
        plt.title('Unknown Samples True Label Distribution')
        plt.tight_layout()
        plt.savefig('unknown_samples_distribution.png')
        plt.close()
