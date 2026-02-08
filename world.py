import numpy as np
import itertools
import os
from collections import deque
from config import Config

class KoanAtlas:
    """
    公案图册 (Koan Atlas) - Zendo 宇宙的静态数据库
    
    功能:
    1. 生成并存储宇宙中所有 5127 个物理合法的公案。
    2. 将公案数据张量化 (Tensorization)，以便进行高速矩阵运算。
    3. 提供向量化的成本矩阵计算接口，供 Optimal Transport (FGW) 使用。
    """

    def __init__(self, load_distances=True):
        print("正在初始化 Zendo 公案图册 (Koan Atlas)...")
        
        # 1. 生成所有合法公案
        # self.koans 是一个列表，每个元素是 (feature_indices, adjacency) 的规范化元组
        # [优化] 使用特征索引而非字符串元组存储，以确保同构比较的稳定性
        self.koans, self.block_types = self._generate_universe()
        
        # 2. 初始化全局张量
        # N: 公案总数, M: 最大Block数 (3), D: 特征维度 (7)
        self.num_koans = len(self.koans)
        
        # 特征张量 F: (N, 3, 7)
        # 维度 7 = Color(3) + Size(3) + Grounded(1)
        self.feature_tensor = np.zeros((self.num_koans, Config.MAX_BLOCKS, Config.DIM_FEATURE), dtype=np.float32)
        
        # 结构张量 C: (N, 3, 3)
        # 邻接矩阵
        self.structure_tensor = np.zeros((self.num_koans, Config.MAX_BLOCKS, Config.MAX_BLOCKS), dtype=np.float32)
        
        # [CRITICAL FIX] 质量张量 mu: (N, 3)
        # 用于告诉 OT 求解器：哪些是真积木，哪些是 Padding
        # 只有真实积木才有质量，且单个公案的总质量归一化为 1
        self.mass_tensor = np.zeros((self.num_koans, Config.MAX_BLOCKS), dtype=np.float32)
        
        # 3. 填充张量
        self._tensorize_data()
        
        print(f"图册构建完成。已索引 {self.num_koans} 个公案。")
        print(f"特征张量形状: {self.feature_tensor.shape}")
        print(f"结构张量形状: {self.structure_tensor.shape}")
        print(f"质量张量形状: {self.mass_tensor.shape}")

        # 4. 可选加载预计算的距离基石 (Critical for Ising Model)
        if load_distances:
            self.load_precomputed_distances()
        else:
            print("⚠️  跳过距离张量加载 (仅用于距离预计算阶段)")
            self.dist_basis = None

    def get_feature_cost_matrix(self, idx_a, idx_b, weights):
        """
        计算两个公案之间的特征成本矩阵 M (3x3)。
        这是 FGW 距离中 Wasserstein 部分的基础。
        
        利用 NumPy Broadcasting 实现向量级加速，避免 Python 循环。
        
        参数:
        idx_a, idx_b: 公案在 feature_tensor 中的索引 (int)
        weights: 当前注意力权重 [w_color, w_size, w_ground, w_touch]
                 注意：这里只用到前三个权重计算 M，w_touch 用于结构项 L。
        
        返回:
        M: (3, 3) 矩阵，M[i, j] 表示公案 A 的第 i 个块与公案 B 的第 j 个块的加权特征距离。
        """
        # 1. 获取特征向量 (Shape: 3x7)
        # 即使公案只有 1 个块，NumPy 也会返回 (3,7)，后面填充的是 0
        F_a = self.feature_tensor[idx_a] 
        F_b = self.feature_tensor[idx_b]
        
        # 2. 升维以利用广播 (Broadcasting)
        # F_a: (3, 1, 7)
        # F_b: (1, 3, 7)
        # diff_sq: (3, 3, 7) -> 包含了 A 中每块与 B 中每块的所有特征维度的差的平方
        diff_sq = (F_a[:, np.newaxis, :] - F_b[np.newaxis, :, :]) ** 2
        
        # 3. 根据切片聚合特征距离
        # 使用 Config 中的切片索引
        dist_color = np.sum(diff_sq[:, :, Config.IDX_COLOR], axis=-1)  # (3, 3)
        dist_size  = np.sum(diff_sq[:, :, Config.IDX_SIZE], axis=-1)   # (3, 3)
        dist_ground= np.sum(diff_sq[:, :, Config.IDX_GROUND], axis=-1) # (3, 3)
        
        # 4. 加权求和
        # weights 顺序: [Color, Size, Ground, Touch]
        w_c, w_s, w_g = weights[0], weights[1], weights[2]
        
        M = (w_c * dist_color) + (w_s * dist_size) + (w_g * dist_ground)
        
        return M

    def get_structure_matrices(self, idx_a, idx_b):
        """
        获取两个公案的结构(邻接)矩阵，用于 FGW 的 Gromov 项计算。
        """
        C_a = self.structure_tensor[idx_a] # (3, 3)
        C_b = self.structure_tensor[idx_b] # (3, 3)
        return C_a, C_b
        
    def get_masses(self, idx_a, idx_b):
        """
        [NEW] 获取两个公案的质量分布向量，供 OT 求解器使用。
        确保填充的 Ghost Blocks 质量为 0，不参与运输规划。
        """
        mu_a = self.mass_tensor[idx_a]
        mu_b = self.mass_tensor[idx_b]
        return mu_a, mu_b

    def _tensorize_data(self):
        """将原始的公案列表转换为 NumPy 张量"""
        
        # 映射字典用于 One-Hot 编码 (辅助)
        # 注意：现在 self.koans 存储的是 block_types 的索引，所以我们需要解析
        
        for i, (feature_indices, adj_tuple) in enumerate(self.koans):
            n = len(feature_indices)
            
            # --- [CRITICAL FIX] 设置概率测度 mu ---
            # 只有存在的积木才有质量，且总和为 1 (归一化)
            if n > 0:
                self.mass_tensor[i, :n] = 1.0 / n
            
            # --- 填充特征张量 ---
            for b_idx, type_idx in enumerate(feature_indices):
                # 从 block_types 中恢复具体特征
                # block 格式: (Size, Color, Grounded) e.g., ('Small', 'R', 1)
                sz_str, col_str, g_val = self.block_types[type_idx]
                
                # Color One-Hot (前3位): R=0, G=1, B=2
                c_idx = {'R': 0, 'G': 1, 'B': 2}[col_str]
                self.feature_tensor[i, b_idx, c_idx] = 1.0
                
                # Size One-Hot (中间3位): Small=0, Medium=1, Large=2
                # Config.IDX_SIZE 是 slice(3, 6) -> 索引 3, 4, 5
                s_idx = {'Small': 0, 'Medium': 1, 'Large': 2}[sz_str]
                self.feature_tensor[i, b_idx, 3 + s_idx] = 1.0
                
                # Grounded Scalar (第7位, 索引6)
                self.feature_tensor[i, b_idx, 6] = float(g_val)
            
            # --- 填充结构张量 ---
            # adj_tuple 是扁平化的元组，需要重塑为 (n, n)
            adj_mat = np.array(adj_tuple).reshape(n, n)
            # 填入 (3, 3) 张量的左上角
            self.structure_tensor[i, :n, :n] = adj_mat

    # =========================================================
    #  生成逻辑 (移植自 calculate_koans.py)
    # =========================================================
    
    def _generate_universe(self):
        """生成符合物理定律的所有公案"""
        sizes = ['Small', 'Medium', 'Large']
        colors = ['R', 'G', 'B']
        grounded = [0, 1]
        
        # 所有的 Block 类型定义 (18种)
        block_types = list(itertools.product(sizes, colors, grounded))
        num_block_types = len(block_types)
        
        valid_koans = set()
        
        # 遍历 N = 1, 2, 3
        for n in range(1, Config.MAX_BLOCKS + 1):
            
            # 特征组合 (直接使用索引，方便后续稳定排序)
            feature_combinations_indices = itertools.product(range(num_block_types), repeat=n)
            
            # 边组合 (上三角)
            num_edges = n * (n - 1) // 2
            edge_combinations = list(itertools.product([0, 1], repeat=num_edges))
            
            # 排列 (用于同构检查)
            permutations = list(itertools.permutations(range(n)))
            
            for features_indices in feature_combinations_indices:
                # 临时恢复成特征值以进行物理检查
                current_features = [block_types[i] for i in features_indices]
                
                # 快速剪枝: 必须至少一个接地
                if sum(f[2] for f in current_features) == 0:
                    continue

                for edge_config in edge_combinations:
                    # 构造邻接矩阵
                    adj_matrix = np.zeros((n, n), dtype=int)
                    edge_idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            val = edge_config[edge_idx]
                            adj_matrix[i, j] = val
                            adj_matrix[j, i] = val
                            edge_idx += 1
                    
                    # 物理合法性检查
                    if not self._is_physically_valid(n, current_features, adj_matrix):
                        continue
                    
                    # 规范化 (Canonicalization)
                    canonical_forms = []
                    for p in permutations:
                        # [优化] 使用 indices 进行重排，保证比较的稳定性
                        # (数字 0,1,2 比字符串 'Small','Medium' 更符合逻辑大小顺序)
                        new_features_idx = tuple(features_indices[p[i]] for i in range(n))
                        
                        new_adj_flat = []
                        for i in range(n):
                            for j in range(n):
                                new_adj_flat.append(adj_matrix[p[i], p[j]])
                        new_adj_tuple = tuple(new_adj_flat)
                        
                        canonical_forms.append((new_features_idx, new_adj_tuple))
                    
                    # 取字典序最小的形式加入集合
                    best_form = min(canonical_forms)
                    valid_koans.add(best_form)
        
        # [优化] 先按 Block 数量排序 (len(x[0]))，再按特征字典序排序
        sorted_koans = sorted(list(valid_koans), key=lambda x: (len(x[0]), x))
        return sorted_koans, block_types

    def _is_physically_valid(self, n, features, adj_matrix):
        """检查物理规则: 1.重力 2.连通性"""
        grounded_status = [f[2] for f in features]
        if sum(grounded_status) == 0:
            return False
            
        visited = set()
        queue = deque()
        for idx, is_grounded in enumerate(grounded_status):
            if is_grounded == 1:
                queue.append(idx)
                visited.add(idx)
                
        while queue:
            curr = queue.popleft()
            for neighbor in range(n):
                if adj_matrix[curr, neighbor] == 1:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return len(visited) == n
    
    def load_precomputed_distances(self):
        """
        加载预计算的距离张量 (N, N, 4)。
        """
        path = Config.DIST_TENSOR_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到预计算文件: {path}\n"
                "请先运行 'python precompute.py --mode run' 生成数据。"
            )
        
        print(f"正在加载距离张量: {path} ...")
        # mmap_mode='r' 允许大文件内存映射，虽然几百M直接读入内存也没问题
        # 这里直接读入内存以获得最快查询速度
        self.dist_basis = np.load(path) 
        print(f"距离张量已加载。Shape: {self.dist_basis.shape}")

    def get_weighted_distance_matrix(self, attention_weights):
        """
        【高速接口】根据当前注意力计算全局距离矩阵 D_final。
        
        参数:
            attention_weights: np.array [w_c, w_s, w_g, w_t]
        
        返回:
            D_final: (N, N) 矩阵，表示当前视角下所有公案两两之间的距离。
        """
        # 张量点积: (N, N, 4) dot (4,) -> (N, N)
        # 相当于: D = w0*D0 + w1*D1 + w2*D2 + w3*D3
        # 这是一个极其高效的向量化操作
        return np.tensordot(self.dist_basis, attention_weights, axes=([2], [0]))

# 单例模式
if __name__ == "__main__":
    # 简单的测试代码
    atlas = KoanAtlas()
    
    # --- 测试 1: 检查是否成功加载了预计算数据 ---
    if hasattr(atlas, 'dist_basis'):
        print(f"\n✅ 成功加载距离基石。Shape: {atlas.dist_basis.shape}")
    else:
        print("\n❌ 未加载距离基石！请检查 __init__ 方法。")

    # --- 测试 2: 测试极速距离计算接口 ---
    print("\n正在测试加权距离矩阵计算...")
    weights = Config.INIT_ATTENTION # [0.25, 0.25, 0.25, 0.25]
    
    # 这步应该瞬间完成
    D = atlas.get_weighted_distance_matrix(weights)
    
    print(f"生成的全局距离矩阵 D_final Shape: {D.shape}")
    print(f"D[0, 100] (示例距离): {D[0, 100]:.4f}")
    
    # 简单的合理性检查：对角线应该是 0
    print(f"对角线元素平均值 (应为0): {np.mean(np.diag(D)):.6f}")