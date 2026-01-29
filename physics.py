import numpy as np
from config import Config

class IsingModel:
    """
    Zendo 物理引擎核心类
    
    实现基于 Ising Model 的假设生成动力学。
    """

    def __init__(self, num_koans):
        """
        初始化物理引擎
        
        Args:
            num_koans (int): 宇宙中公案的总数 (N)
        """
        self.num_koans = num_koans
        
        # 1. 耦合矩阵 J_ij (N x N)
        # 初始化为 0，需调用 update_couplings 计算
        self.J_matrix = np.zeros((num_koans, num_koans), dtype=np.float32)
        
        # 2. 外部场/钉扎场 h_i (N,)
        # 包含来自 Ground Truth 的反馈钉扎
        self.h_field = np.zeros(num_koans, dtype=np.float32)

    def update_couplings(self, distance_matrix):
        """
        根据环境计算出的距离矩阵更新耦合强度 J_{ij}
        
        对应公式: 
        J_{ij} = J_0 * exp( - D(K_i, K_j)^2 / (2 * sigma^2) )
        
        Args:
            distance_matrix (np.ndarray): (N, N) 的距离矩阵，元素为 D(K_i, K_j)
        """
        # 获取物理常数
        J0 = Config.J0
        sigma = Config.SIGMA
        
        # 防止除以零 (虽然 sigma 通常不为 0)
        if sigma <= 1e-6:
            sigma = 1e-6

        # 计算高斯衰减耦合
        # 注意：文档公式中指数部分是距离的平方 D^2
        decay_factor = - (distance_matrix ** 2) / (2 * (sigma ** 2))
        self.J_matrix = J0 * np.exp(decay_factor)
        
        # 物理约束：自旋不与自身发生交换作用 (J_ii = 0)
        # 虽然数学上包含自能项，但在 MCMC 动力学中通常置零以避免计算冗余
        np.fill_diagonal(self.J_matrix, 0.0)

    def set_pinning_field(self, positive_indices, negative_indices, round_num):
        """
        设置外部钉扎场 (Pinning Field) h_i
        
        对应 Pinning Field 定义:
        h_i = +B(t)  若 K_i 符合真理 (Known Positive)
        h_i = -B(t)  若 K_i 不符合真理 (Known Negative)
        h_i = 0      若 K_i 未知 (Unknown)
    
        Args:
            positive_indices (list/array): 已知正例的 ID 列表
            negative_indices (list/array): 已知反例的 ID 列表
            round_num (int): 当前游戏轮次 (用于计算记忆衰减)
        """
        # 1. 重置场
        self.h_field = np.zeros(self.num_koans, dtype=np.float32)
        
        # 2. 计算当前轮次的场强度 B(t)
        # [Audit Fix]: 修正了参数传递错误。Config方法不需要 is_positive_example 参数。
        # 依据 Config 注释：统一正负例的场强幅度。
        field_strength = Config.get_decayed_pinning_field(round_num)
        
        # 3. 施加钉扎
        if len(positive_indices) > 0:
            self.h_field[positive_indices] = field_strength
            
        if len(negative_indices) > 0:
            # 反例对应负场 -B(t)，但幅度与正例完全一致
            self.h_field[negative_indices] = -field_strength
            
        # 4. (已移除) 稀疏性偏差
        # self.h_field -= Config.SPARSITY_BIAS

    def compute_energy(self, spins):
        """
        计算系统当前的全局哈密顿量 (能量) H(s)
        
        H(s) = - 1/2 * sum_{i,j} J_{ij} s_i s_j - sum_i h_i s_i
        
        Args:
            spins (np.ndarray): 自旋状态向量 (N,), 取值为 {+1, -1}
            
        Returns:
            float: 系统的总能量
        """
        # 1. 交互项 (Interaction Term): - 1/2 * s^T * J * s
        # 使用矩阵乘法加速: (1, N) @ (N, N) @ (N, 1)
        interaction_energy = -0.5 * (spins.T @ self.J_matrix @ spins)
        
        # 2. 外部场项 (Field Term): - h^T * s
        field_energy = -np.dot(self.h_field, spins)
        
        return interaction_energy + field_energy

    def calculate_delta_energy(self, spins, flip_idx):
        """
        计算翻转第 k 个自旋产生的能量变化 (Delta H)
        用于 MCMC 快速采样，避免每次都重新计算全局能量。
        
        Delta H = H(s_new) - H(s_old)
        其中 s_new 仅在 flip_idx 处符号相反。
        
        推导:
        H = -1/2 sum J s s - sum h s
        只关注与 s_k 有关的项: E_k = - s_k ( sum_{j!=k} J_{kj} s_j + h_k )
        若 s_k 翻转 (s_k -> -s_k)，则能量变化为:
        Delta E = E_new - E_old = - (-E_old) - E_old = -2 * E_new = 2 * s_k_old * (局部场)
        
        Args:
            spins (np.ndarray): 当前自旋状态
            flip_idx (int): 尝试翻转的节点索引
            
        Returns:
            float: 能量变化量
        """
        s_i = spins[flip_idx]
        
        # 计算局部场 (Local Field) 作用于节点 i
        # J_matrix[flip_idx] 是第 i 行，表示 i 与所有 j 的耦合
        local_field = np.dot(self.J_matrix[flip_idx], spins) + self.h_field[flip_idx]
        
        # Delta H = 2 * s_i * (Local Field)
        # 注意：这里的 s_i 是翻转前的自旋值
        delta_H = 2.0 * s_i * local_field
        
        return delta_H