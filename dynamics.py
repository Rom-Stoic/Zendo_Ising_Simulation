import numpy as np
import itertools
from config import Config
from numba import jit

# 将核心逻辑剥离为静态函数，以便 JIT 优化
# nopython=True 确保完全编译为机器码，不回退到 Python 对象模式
@jit(nopython=True)
def _glauber_core(spins, J_matrix, h_field, steps, beta):
    n_koans = len(spins)
    # 显式复制，避免修改输入数组
    current_spins = spins.copy()
    
    for _ in range(steps):
        # 1. 随机选择一个公案节点尝试翻转
        idx = np.random.randint(0, n_koans)
        
        # 2. 内联计算能量差 (Inlined Delta Energy Calculation)
        # 避免调用 Python 函数带来的开销
        # Delta H = 2 * s_i * (Local Field)
        # Local Field = sum(J_ij * s_j) + h_i
        
        # 计算相互作用部分 (O(N) 复杂度，但被 JIT 极大加速)
        interaction = 0.0
        for j in range(n_koans):
            interaction += J_matrix[idx, j] * current_spins[j]
            
        # 加上外部钉扎场 (关键修正: 不能忽略外部反馈)
        local_field = interaction + h_field[idx]
        
        # 计算翻转该自旋带来的能量变化
        # 注意: current_spins[idx] 是翻转前的状态
        delta_E = 2.0 * current_spins[idx] * local_field
        
        # 3. Glauber Dynamics 接受准则
        # P(flip) = 1 / (1 + exp(beta * delta_E))
        prob = 1.0 / (1.0 + np.exp(beta * delta_E))
        
        if np.random.rand() < prob:
            current_spins[idx] *= -1
            
    return current_spins

class FastSolver:
    """
    快动力学求解器 (Fast Dynamics Solver)
    基于 Glauber Dynamics 算法，使用 Numba JIT 加速。
    
    功能:
    在固定的物理环境 (J, h) 下，演化自旋场以寻找低能态 (Local Minima)。
    这模拟了人类在观察到数据后，脑海中'瞬间'涌现出的直觉判断。
    """
    
    def __init__(self):
        pass

    def run_glauber(self, ising_model, initial_spins, steps=Config.MCMC_STEPS):
        """
        运行 Glauber 动力学模拟 (JIT 加速版)
        
        Args:
            ising_model: physics.py 中的 IsingModel 实例，提供能量计算接口
            initial_spins: 初始自旋配置 (N,), 取值 {-1, 1}
            steps: 蒙特卡洛迭代步数
            
        Returns:
            spins: 演化达到(准)稳态后的自旋配置 s*
        """
        # 调用加速后的内核
        # 必须传入纯 NumPy 数组，不能传入对象
        return _glauber_core(
            initial_spins, 
            ising_model.J_matrix, 
            ising_model.h_field, # [Critical] 必须传入外部场，否则无法学习
            steps, 
            Config.MCMC_BETA
        )


class DPP:
    """
    多假设选择过程 (Determinantal Point Process)
    
    功能:
    从 FastSolver 产生的多个平行宇宙(Chains)中，筛选出一组既“合理”又“多样”的假设。
    对应文档中的多假设维护机制 (k=1~3)。
    """
    
    def __init__(self):
        pass

    def select_subset(self, candidates, ising_model):
        """
        根据 DPP 核矩阵选择最佳子集
        
        Args:
            candidates: 候选自旋列表 [s^(1), s^(2), ..., s^(M)]
            ising_model: 用于计算每个假设的能量
            
        Returns:
            best_subset_indices: 选中的假设在 candidates 列表中的索引 (list)
            max_det: 对应子集的行列式值 (代表该组合的概率权重)
        """
        n_candidates = len(candidates)
        k = Config.NUM_HYPOTHESES_KEPT
        
        # 如果候选数量不足，直接返回所有
        if n_candidates <= k:
            return list(range(n_candidates)), 1.0
            
        # --- 1. 计算质量 Quality (q_i) ---
        # q_i = exp( - H(s^(i)) / T_quality )
        # 能量越低，质量越高
        energies = np.array([ising_model.compute_energy(s) for s in candidates])
        # 数值稳定性技巧: 减去最小能量防止溢出
        min_E = np.min(energies)
        qualities = np.exp(-(energies - min_E) / Config.DPP_QUALITY_TEMP)
        
        # --- 2. 计算相似度 Similarity (S_ij) ---
        # S_ij = (1/N) * <s^(i), s^(j)>
        # 也就是两个自旋构型的 Cosine Similarity (归一化点积)
        # 将候选列表转换为矩阵 (M, N)
        S_mat = np.stack(candidates)
        # 矩阵乘法: (M, N) @ (N, M) -> (M, M)
        similarity_matrix = (S_mat @ S_mat.T) / ising_model.num_koans
        
        # --- 3. 构建 L 核矩阵 ---
        # L_ij = q_i * S_ij * q_j
        # 利用广播机制: (M, 1) * (M, M) * (1, M)
        q_vec = qualities.reshape(-1, 1)
        L_matrix = q_vec * similarity_matrix * q_vec.T
        
        # [Audit Correction] 数值稳定性增强
        # 添加微小的 Ridge Term 防止矩阵奇异
        L_matrix += 1e-6 * np.eye(n_candidates)
        
        # --- 4. 采样/选择子集 Y ---
        # 遍历所有大小为 k 的子集组合，寻找 det(L_Y) 最大的组合
        # 由于 M (链数) 通常较小 (e.g. 20), C(20, 3) = 1140，计算量可以接受
        
        max_det = -1.0
        best_subset_indices = []
        
        for subset in itertools.combinations(range(n_candidates), k):
            indices = list(subset)
            # 提取子矩阵 L_Y (k x k)
            # 使用 numpy 的 ix_ 进行网格索引
            L_Y = L_matrix[np.ix_(indices, indices)]
            
            # 计算行列式
            current_det = np.linalg.det(L_Y)
            
            if current_det > max_det:
                max_det = current_det
                best_subset_indices = indices
                
        return best_subset_indices, max_det


class SlowLearner:
    """
    慢动力学学习器 (Slow Dynamics / Meta-Learning)
    
    功能:
    根据当前最佳假设的反馈，通过梯度下降更新注意力权重 (Omega)。
    这模拟了人类的反思过程：“我之前的关注点是不是错了？”
    """
    
    def __init__(self, atlas):
        """
        Args:
            atlas: KoanAtlas 实例，提供基础距离矩阵张量
        """
        self.atlas = atlas

    def update_attention(self, current_attention, best_spins, ising_model, learning_rate=Config.LEARNING_RATE):
        """
        执行一步梯度下降更新注意力参数
        
        Args:
            current_attention: 当前注意力权重 [w_c, w_s, w_g, w_t]
            best_spins: 当前轮次的最优假设 s* (N,)
            ising_model: 物理引擎 (包含当前的 J 矩阵)
            
        Returns:
            new_attention: 更新并归一化后的注意力权重
        """
        # ---------------------------------------------------------
        # 梯度推导 (Chain Rule):
        # 目标: 最小化能量 H(s*) 关于权重 w_k 的导数
        # 
        # dH/dw_k = sum_{i,j} (dH/dJ_ij) * (dJ_ij/dD_total) * (dD_total/dw_k)
        # 
        # 最终推导结果 (Audit Verified):
        # dH/dw_k = (0.5 / sigma^2) * sum_{i,j} (s_i * s_j * J_ij * D_total * D_k)
        # ---------------------------------------------------------
        
        # 1. 获取中间变量
        sigma_sq = Config.SIGMA ** 2
        
        # 当前的总距离矩阵 D_total (N, N)
        # D_total = sum(w_k * D_k)
        D_total = self.atlas.get_weighted_distance_matrix(current_attention)
        
        # 自旋的外积矩阵 S_outer[i,j] = s_i * s_j
        S_outer = np.outer(best_spins, best_spins)
        
        # 2. 计算公共梯度项 (Common Term)
        # Combine: (0.5 / sigma^2) * (S * J * D)
        # 注意：这里全部是元素级乘法 (Hadamard Product)
        common_term = (0.5 / sigma_sq) * (S_outer * ising_model.J_matrix * D_total)
        
        # 3. 对每个注意力维度计算梯度
        gradients = np.zeros_like(current_attention)
        
        for k in range(4): # 4个属性维度: Color, Size, Ground, Struct
            # 获取第 k 个维度的基础距离矩阵 D_k (N, N)
            # dist_basis shape: (N, N, 4)
            D_k = self.atlas.dist_basis[:, :, k]
            
            # [Audit Correction] 学习率归一化
            # 使用 np.mean 代替 np.sum，避免因 N^2 项导致的梯度数值爆炸
            gradients[k] = np.mean(common_term * D_k)
            
        # 4. 梯度下降更新
        # new_w = old_w - lr * grad
        new_attention = current_attention - learning_rate * gradients
        
        # 5. 投影与归一化 (Projected Gradient Descent)
        # 约束1: 权重必须非负 (给一个小的 epsilon 防止 0)
        new_attention = np.maximum(new_attention, 0.01)
        
        # 约束2: 权重之和为 1
        new_attention = new_attention / np.sum(new_attention)
        
        return new_attention