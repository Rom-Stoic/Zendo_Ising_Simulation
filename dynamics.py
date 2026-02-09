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
    慢动力学学习器 (Slow Dynamics / Meta-Learning) - 对比学习版
    
    功能:
    使用鲁棒的对比度量学习 (Robust Contrastive Metric Learning) 更新注意力权重。
    通过最小化类内距离 (Intra-class Distance) 和最大化类间距离 (Inter-class Distance) 
    来调整特征空间中的注意力分配。
    
    设计原则:
    - 支持冷启动 (Cold Start): 处理只有1个正例或0个反例的场景
    - 数值鲁棒: 防止 NaN/Inf，确保梯度有界
    - 物理可解释性: 梯度方向明确对应"拉近正例/推远反例"
    """
    
    def __init__(self, atlas):
        """
        Args:
            atlas: KoanAtlas 实例，必须包含预计算的 dist_basis (N, N, 4)
        """
        self.atlas = atlas
        # 确保距离基石已加载
        if not hasattr(self.atlas, 'dist_basis') or self.atlas.dist_basis is None:
            raise ValueError("SlowLearner 需要 KoanAtlas 加载 dist_basis。请确保运行过 precompute.py 并启用了 load_distances=True")

    def update_attention(self, current_attention, known_pos_indices, known_neg_indices, 
                        learning_rate=Config.LEARNING_RATE):
        """
        执行一步对比学习更新注意力参数
        
        Args:
            current_attention: 当前注意力权重 [w_c, w_s, w_g, w_t]
            known_pos_indices: 已知正例的索引列表 (list or array)
            known_neg_indices: 已知反例的索引列表 (list or array)
            learning_rate: 学习率
            
        Returns:
            new_attention: 更新并归一化后的注意力权重
        """
        # 转换为 numpy array 以便索引
        pos_idx = np.array(known_pos_indices, dtype=int)
        neg_idx = np.array(known_neg_indices, dtype=int)
        
        gradients = np.zeros(4)
        
        # 遍历 4 个特征维度 (Color, Size, Ground, Touch)
        for k in range(4):
            # 获取该维度的全距离矩阵 D_k (N, N)
            D_k = self.atlas.dist_basis[:, :, k]
            
            # --- 1. 计算类内距离 (Intra-class Pull) ---
            # 目标：让正例之间靠得更近
            if len(pos_idx) > 1:
                # 使用 np.ix_ 生成网格索引，提取正例两两之间的子矩阵
                intra_submatrix = D_k[np.ix_(pos_idx, pos_idx)]
                # 计算平均距离 (排除对角线其实影响不大，因为对角线为0且是对称的)
                mean_intra = np.mean(intra_submatrix)
            else:
                # 冷启动：只有1个或0个正例，无法计算聚集程度，梯度贡献为0
                mean_intra = 0.0
                
            # --- 2. 计算类间距离 (Inter-class Push) ---
            # 目标：让正例和反例离得更远
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                # 提取正例行、反例列构成的子矩阵
                inter_submatrix = D_k[np.ix_(pos_idx, neg_idx)]
                mean_inter = np.mean(inter_submatrix)
            else:
                # 冷启动：没有反例，无法对比，梯度贡献为0
                mean_inter = 0.0
            
            # --- 3. 计算梯度 ---
            # Loss = Intra - Inter
            # 我们希望 Intra 越小越好 (梯度正)，Inter 越大越好 (梯度负)
            # Grad = d(Loss)/dw = Intra_dist - Inter_dist
            # 逻辑：
            # 如果 Intra 很大(正例在该维度很散)，Grad > 0，w = w - lr*Grad (权重减小) -> 正确，因为该特征无效
            # 如果 Inter 很大(正负在该维度分很开)，Grad < 0，w = w - lr*Grad (权重增加) -> 正确，这是关键特征
            gradients[k] = mean_intra - mean_inter

        # --- 4. 更新权重 ---
        new_attention = current_attention - learning_rate * gradients
        
        # 检查 NaN (防止数值异常导致崩溃)
        if np.any(np.isnan(new_attention)):
            print("⚠️ 警告: Attention update 产生 NaN，回滚到上一步权重。")
            return current_attention

        # --- 5. 投影与归一化 ---
        # 约束1: 权重必须非负且保持最小活性 (Projected Gradient)
        new_attention = np.maximum(new_attention, 0.01)
        
        # 约束2: 权重之和为 1 (Simplex Projection)
        total_weight = np.sum(new_attention)
        if total_weight > 0:
            new_attention = new_attention / total_weight
        else:
            # 极端情况防御
            new_attention = np.array([0.25, 0.25, 0.25, 0.25])
            
        return new_attention