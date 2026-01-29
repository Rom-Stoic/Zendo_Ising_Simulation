import torch
import numpy as np
import itertools  # [FIX] 引入 itertools 用于计算全排列
from torch_geometric.data import Data, Dataset
from collections import defaultdict
from config import Config

class ZendoGraphDataset(Dataset):
    """
    将 World 中的 KoanAtlas 桥接到 PyTorch Geometric。
    负责将 numpy tensor 转换为 Graph Data 对象。
    """
    def __init__(self, atlas):
        super().__init__()
        self.atlas = atlas
        self.num_koans = atlas.num_koans

    def len(self):
        return self.num_koans

    def get(self, idx):
        # 1. 提取节点特征 (3, 7) -> (N_active, 7)
        # 我们需要移除 padding 的节点 (即全0的行)
        # 利用 mass_tensor 来判断哪些是真实节点
        feature_raw = self.atlas.feature_tensor[idx] # (3, 7)
        mass = self.atlas.mass_tensor[idx]           # (3,)
        
        # 找出真实存在的 Block 索引
        active_indices = np.where(mass > 0)[0]
        if len(active_indices) == 0:
            # 极少数情况下的空图保护
            x = torch.zeros((1, Config.DIM_FEATURE), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)

        # 截取有效特征
        x = torch.tensor(feature_raw[active_indices], dtype=torch.float)

        # 2. 提取邻接矩阵并转为 Edge Index
        adj_raw = self.atlas.structure_tensor[idx] # (3, 3)
        # 只取有效部分的子矩阵
        adj_active = adj_raw[np.ix_(active_indices, active_indices)]
        
        # 转换为 COO 格式 (Source, Target)
        rows, cols = np.where(adj_active > 0)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # 返回 PyG 数据对象
        return Data(x=x, edge_index=edge_index, idx=idx)

class KoanSampler:
    """
    智能采样器：基于倒排索引实现 O(1) 的样本挖掘。
    """
    def __init__(self, atlas):
        self.atlas = atlas
        self.num_koans = atlas.num_koans
        
        print("🔍 [Sampler] 正在构建倒排索引 (Mining Indices)...")
        self.indices = {
            'color': self._build_index(self._get_color_sig),
            'size': self._build_index(self._get_size_sig),
            'ground': self._build_index(self._get_ground_sig),
            'structure': self._build_index(self._get_struct_sig)
        }
        print("✅ [Sampler] 索引构建完成。")

    def _get_color_sig(self, idx):
        # 提取颜色指纹: 排序后的 RGB 组合
        mass = self.atlas.mass_tensor[idx]
        active = np.where(mass > 0)[0]
        feats = self.atlas.feature_tensor[idx][active]
        # 取 RGB 部分 (前3列)
        colors = [tuple(x) for x in feats[:, Config.IDX_COLOR].tolist()]
        # 排序以保证集合无序性 ({红,蓝} == {蓝,红})
        return tuple(sorted(colors))

    def _get_size_sig(self, idx):
        # 提取大小指纹
        mass = self.atlas.mass_tensor[idx]
        active = np.where(mass > 0)[0]
        feats = self.atlas.feature_tensor[idx][active]
        sizes = [tuple(x) for x in feats[:, Config.IDX_SIZE].tolist()]
        return tuple(sorted(sizes))

    def _get_ground_sig(self, idx):
        # 提取接地指纹 (接地的数量)
        mass = self.atlas.mass_tensor[idx]
        active = np.where(mass > 0)[0]
        feats = self.atlas.feature_tensor[idx][active]
        # sum of ground values
        ground_val = np.sum(feats[:, Config.IDX_GROUND])
        return float(ground_val)

    def _get_struct_sig(self, idx):
        """
        [CRITICAL FIX] 纯拓扑结构签名提取
        原问题：world.py 的规范化依赖特征排序，导致相同拓扑结构因节点颜色不同而产生不同的邻接矩阵。
        修复：忽略节点特征，暴力搜索所有节点排列，找到字典序最小的邻接矩阵作为签名。
        """
        mass = self.atlas.mass_tensor[idx]
        # 找出真实节点数量 n
        active = np.where(mass > 0)[0]
        n = len(active)
        
        # 提取当前存储的邻接矩阵 (可能受特征排序影响)
        # 注意：world.py 已经把有效节点放在了左上角，所以直接取 [:n, :n] 即可
        # 无需再用 active索引，因为 tensorization 时已经紧凑排列了
        raw_adj = self.atlas.structure_tensor[idx][:n, :n]
        
        # 暴力尝试所有排列 (n <= 3，计算量极小)
        permutations = list(itertools.permutations(range(n)))
        best_adj_tuple = None
        
        for p in permutations:
            # 根据排列 p 重排矩阵行列
            # p 是一个 tuple, e.g., (2, 0, 1)
            # permuted_adj[i, j] = raw_adj[p[i], p[j]]
            # 使用 numpy 的高级索引 np.ix_
            permuted_adj = raw_adj[np.ix_(p, p)]
            
            # 展平并转为元组以便比较
            # (int, int, ...)
            current_adj_tuple = tuple(permuted_adj.flatten().tolist())
            
            # 寻找字典序最小的形式
            if best_adj_tuple is None or current_adj_tuple < best_adj_tuple:
                best_adj_tuple = current_adj_tuple
                
        return best_adj_tuple

    def _build_index(self, sig_func):
        index = defaultdict(list)
        for i in range(self.num_koans):
            sig = sig_func(i)
            index[sig].append(i)
        return index

    def get_triplet_batch(self, attribute, batch_size=32):
        """
        生成三元组索引: (Anchor, Positive, Negative)
        """
        anchors, positives, negatives = [], [], []
        target_index = self.indices[attribute]
        all_ids = np.arange(self.num_koans)
        
        # 获取所有可能的 key，用于随机选择
        available_sigs = list(target_index.keys())
        
        count = 0
        while count < batch_size:
            # 1. 随机选一个特征签名 (Signature)
            sig = available_sigs[np.random.randint(len(available_sigs))]
            candidates = target_index[sig]
            
            # 必须至少有2个样本才能组成 Anchor-Positive 对
            if len(candidates) < 2:
                continue
                
            # 2. 从该签名中随机选两个不同的 ID
            a_idx, p_idx = np.random.choice(candidates, 2, replace=False)
            
            # 3. 负样本挖掘 (Negative Mining)
            n_idx = np.random.choice(all_ids)
            # 确保负样本的属性确实不同 (Rejection Sampling)
            while n_idx in candidates:
                n_idx = np.random.choice(all_ids)
            
            anchors.append(a_idx)
            positives.append(p_idx)
            negatives.append(n_idx)
            count += 1
            
        return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)