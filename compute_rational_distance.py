"""
计算基于匈牙利算法的理性距离 (Rational Distance)

核心思路:
对于两个公案A和B，每个公案包含最多3个积木（不足的用"幽灵积木"填充）。
我们计算一个3×3的成本矩阵，其中每个元素表示A的第i个积木与B的第j个积木
之间的特征差异。然后使用匈牙利算法找到最优匹配，使得总成本最小。

这模拟了人类意识层面对"哪个积木对应哪个积木"的理性判断过程。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import os
from config import Config
from world import KoanAtlas


class RationalDistanceCalculator:
    """
    基于匈牙利算法的理性距离计算器
    """
    
    def __init__(self, atlas):
        """
        Args:
            atlas: KoanAtlas实例，包含所有公案的特征数据
        """
        self.atlas = atlas
        self.num_koans = atlas.num_koans
        
    def compute_block_distance(self, feat_a, feat_b):
        """
        计算两个积木之间的特征距离
        
        Args:
            feat_a: 积木A的特征向量 (7维)
            feat_b: 积木B的特征向量 (7维)
            
        Returns:
            distances: 包含4个维度距离的数组 [d_color, d_size, d_ground, d_touch]
                      注意: touch维度在这里暂时为0，因为touch是关系属性而非节点属性
        """
        # Color距离 (one-hot编码，使用欧氏距离)
        color_a = feat_a[Config.IDX_COLOR]
        color_b = feat_b[Config.IDX_COLOR]
        d_color = np.sqrt(np.sum((color_a - color_b) ** 2))
        
        # Size距离 (one-hot编码)
        size_a = feat_a[Config.IDX_SIZE]
        size_b = feat_b[Config.IDX_SIZE]
        d_size = np.sqrt(np.sum((size_a - size_b) ** 2))
        
        # Ground距离 (标量)
        ground_a = feat_a[Config.IDX_GROUND]
        ground_b = feat_b[Config.IDX_GROUND]
        d_ground = np.abs(ground_a - ground_b).item()
        
        # Touch距离 (在节点层面暂时为0，稍后会通过结构矩阵计算)
        d_touch = 0.0
        
        return np.array([d_color, d_size, d_ground, d_touch])
    
    def compute_cost_matrix(self, idx_a, idx_b):
        """
        计算两个公案之间的3×3成本矩阵
        
        Args:
            idx_a, idx_b: 公案索引
            
        Returns:
            cost_matrix: (3, 3) 矩阵，cost[i, j]表示A的第i个积木与B的第j个积木的总距离
        """
        feat_a = self.atlas.feature_tensor[idx_a]  # (3, 7)
        feat_b = self.atlas.feature_tensor[idx_b]  # (3, 7)
        
        # 获取实际积木数量（通过mass_tensor判断）
        mass_a = self.atlas.mass_tensor[idx_a]  # (3,)
        mass_b = self.atlas.mass_tensor[idx_b]  # (3,)
        
        cost_matrix = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                # 如果是幽灵积木对，设置一个基准成本
                if mass_a[i] == 0 and mass_b[j] == 0:
                    cost_matrix[i, j] = 0.0  # 两个幽灵积木匹配成本为0
                elif mass_a[i] == 0 or mass_b[j] == 0:
                    # 一个真实积木与一个幽灵积木匹配，成本较高
                    cost_matrix[i, j] = 10.0  # 惩罚项
                else:
                    # 两个真实积木之间的距离（仅考虑Color, Size, Ground）
                    dist_vec = self.compute_block_distance(feat_a[i], feat_b[j])
                    # 总成本 = 前三个维度的加权和（Touch在这里不计入）
                    cost_matrix[i, j] = np.sum(dist_vec[:3])
        
        return cost_matrix
    
    def compute_structure_distance(self, idx_a, idx_b, assignment):
        """
        基于匹配结果计算结构距离（Touch维度）
        
        结构距离定义：比较两个公案的邻接矩阵在最优匹配下的差异
        
        Args:
            idx_a, idx_b: 公案索引
            assignment: 匈牙利算法的匹配结果 (row_ind, col_ind)
            
        Returns:
            d_touch: 结构距离（标量）
        """
        adj_a = self.atlas.structure_tensor[idx_a]  # (3, 3)
        adj_b = self.atlas.structure_tensor[idx_b]  # (3, 3)
        
        row_ind, col_ind = assignment
        
        # 根据匹配重排B的邻接矩阵
        # 创建置换矩阵P，使得P^T @ adj_b @ P 对应匹配后的B
        perm_matrix = np.zeros((3, 3))
        for i, j in zip(row_ind, col_ind):
            perm_matrix[i, j] = 1
        
        # 重排B的邻接矩阵
        adj_b_reordered = perm_matrix.T @ adj_b @ perm_matrix
        
        # 计算Frobenius范数距离
        d_touch = np.linalg.norm(adj_a - adj_b_reordered, ord='fro')
        
        return d_touch
    
    def compute_pairwise_distance(self, idx_a, idx_b):
        """
        计算两个公案之间的4维距离向量
        
        Args:
            idx_a, idx_b: 公案索引
            
        Returns:
            distances: (4,) 数组，包含 [d_color, d_size, d_ground, d_touch]
        """
        # 1. 计算成本矩阵
        cost_matrix = self.compute_cost_matrix(idx_a, idx_b)
        
        # 2. 匈牙利算法求解最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 3. 基于匹配计算各维度距离
        feat_a = self.atlas.feature_tensor[idx_a]
        feat_b = self.atlas.feature_tensor[idx_b]
        mass_a = self.atlas.mass_tensor[idx_a]
        mass_b = self.atlas.mass_tensor[idx_b]
        
        # 初始化各维度距离累加器
        distances = np.zeros(4)
        count = 0
        
        for i, j in zip(row_ind, col_ind):
            # 只统计真实积木之间的距离
            if mass_a[i] > 0 and mass_b[j] > 0:
                dist_vec = self.compute_block_distance(feat_a[i], feat_b[j])
                distances[:3] += dist_vec[:3]  # Color, Size, Ground
                count += 1
        
        # 平均化（避免积木数量影响）
        if count > 0:
            distances[:3] /= count
        
        # 4. 计算结构距离（Touch维度）
        distances[3] = self.compute_structure_distance(idx_a, idx_b, (row_ind, col_ind))
        
        return distances
    
    def compute_all_distances(self):
        """
        计算所有公案两两之间的距离，返回(N, N, 4)张量
        
        Returns:
            dist_rational: (N, N, 4) 距离张量
        """
        N = self.num_koans
        dist_rational = np.zeros((N, N, 4), dtype=np.float32)
        
        print(f"🧮 开始计算理性距离 (Rational Distance)...")
        print(f"   总计算量: {N * (N - 1) // 2} 对公案")
        
        # 使用进度条
        with tqdm(total=N, desc="计算进度") as pbar:
            for i in range(N):
                for j in range(i, N):  # 利用对称性，只计算上三角
                    if i == j:
                        # 自己与自己的距离为0
                        dist_rational[i, j, :] = 0.0
                    else:
                        distances = self.compute_pairwise_distance(i, j)
                        dist_rational[i, j, :] = distances
                        dist_rational[j, i, :] = distances  # 对称
                
                pbar.update(1)
        
        print(f"✅ 理性距离计算完成！")
        return dist_rational


def main():
    """
    主函数：计算并保存理性距离
    """
    print("="*60)
    print("🧠 理性距离计算器 (Rational Distance Calculator)")
    print("="*60)
    
    # 1. 初始化公案图册
    print("\n📚 加载公案图册...")
    atlas = KoanAtlas(load_distances=False)
    
    # 2. 创建计算器
    calculator = RationalDistanceCalculator(atlas)
    
    # 3. 计算距离
    dist_rational = calculator.compute_all_distances()
    
    # 4. 保存结果
    save_path = Config.DIST_RATIONAL_FILE
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.save(save_path, dist_rational)
    print(f"\n💾 理性距离已保存至: {save_path}")
    print(f"📦 文件大小: {os.path.getsize(save_path) / (1024**2):.2f} MB")
    print(f"📊 张量形状: {dist_rational.shape}")
    
    # 5. 统计信息
    print(f"\n📈 距离统计:")
    for dim, name in enumerate(['Color', 'Size', 'Ground', 'Touch']):
        dim_data = dist_rational[:, :, dim]
        # 排除对角线（自己与自己的距离）
        mask = ~np.eye(atlas.num_koans, dtype=bool)
        dim_data_off_diag = dim_data[mask]
        
        print(f"   {name:8s}: mean={np.mean(dim_data_off_diag):.4f}, "
              f"std={np.std(dim_data_off_diag):.4f}, "
              f"max={np.max(dim_data_off_diag):.4f}")
    
    print("\n" + "="*60)
    print("✅ 全部完成！")
    print("="*60)


if __name__ == "__main__":
    main()
