import numpy as np
import os

class Config:
    """
    Zendo 仿真系统的全局配置类 (Configuration Control Panel)
    存储所有物理常数、系统超参数和环境设定。
    对应文档: 《Zendo假设生成和检验初步模型》
    """

    # =========================================================================
    # 1. 空间与公案参数 (Koan Space & Features)
    #    对应文档 3.2 节: 公案空间定义
    # =========================================================================
    
    # 你的宇宙里总共有多少个公案 (Block组合，经计算得到)
    NUM_KOANS = 5127 
    
    # [NumPy Optimization] 最大块数 (Padding)
    # 显式定义最大块数，用于构建固定大小的 NumPy 张量 (N_samples, MAX_BLOCKS, DIM)
    # 这样可以进行批量的矩阵运算，而不需要写低效的 Python 循环
    MAX_BLOCKS = 3

    # --- 节点特征 (Node Features) ---
    # F \in R^{7 \times n}
    DIM_FEATURE = 7 
    
    # 特征切片索引 (用于从特征向量 F 中提取特定属性)
    # 方便代码中写 koan.features[IDX_COLOR] 而不是硬编码数字
    IDX_COLOR  = slice(0, 3)  # R3 one-hot -> shape (3,)
    IDX_SIZE   = slice(3, 6)  # R3 one-hot -> shape (3,)
    
    # [NumPy Optimization] 统一维度行为
    # 使用 slice(6, 7) 而不是 index 6，确保提取出来的是 shape (1,) 的向量而不是标量
    # 这样在计算距离 D = w * ||f1 - f2||^2 时，所有特征的维度一致，避免 Broadcasting 错误
    IDX_GROUND = slice(6, 7)  # scalar 0/1 -> shape (1,)

    # --- 结构特征 (Structure Features) ---
    # 接触矩阵 C \in {0,1}^{n \times n}
    # 注意：接触关系不存储在 DIM_FEATURE 中，而是作为独立的邻接矩阵存在
    
    # =========================================================================
    # 2. 物理引擎参数 (Physics Engine / Ising Model)
    #    对应文档 3.3 节 (距离) 和 3.4 节 (哈密顿量)
    # =========================================================================
    
    # [公式 286/PDF 3.4] 基础耦合强度 (J_0)
    # 决定了系统整体的“铁磁性”倾向，即假设之间倾向于一致的程度
    J0 = 1.0 
    
    # [公式 286/PDF 3.4] 相关长度 (Sigma, σ)
    # 控制两个公案在特征空间距离多远时，它们对应的自旋还会相互影响
    # sigma 越小，交互越局部；sigma 越大，交互越长程
    SIGMA = 0.5 
    
    # [公式 PDF 3.3] 感知温度 (T_1)
    # 用于计算 FGW 距离时的熵正则化项系数，控制视知觉的模糊程度
    T_PERCEPTION = 0.1 

    # [Deep Debug 1] 稀疏性偏差 (Sparsity Bias, μ)
    # 对应文档 3.5 节疑问 2: -μ * Σs_i
    # 倾向于让总自旋为 -1 (即大部分公案被判定为 False)。
    # 这是一个负偏置，用于解释 Positive Test Bias (人们倾向于认为规则是稀疏的)。
    # 设为 > 0.0 时，哈密顿量增加项: + mu * sum(s) (若 s=-1 能量更低)
    SPARSITY_BIAS = 0.0  # 初始关闭，可设为 0.1~0.5 进行测试

    # [Deep Debug 2] 非对称钉扎场 (Asymmetric Pinning Field, B)
    # 对应文档 1.7 节: 正反馈学习率高于负反馈
    # 正例 (+B) 提供强烈的对齐信号，反例 (-B) 提供的信号较弱
    PINNING_POS_MULTIPLIER = 5.0  # 正例给予强烈的确认
    PINNING_NEG_MULTIPLIER = 1.0  # 反例给予较弱的排除信息 (模拟人类难以利用负反馈)
    
    PINNING_FIELD_DECAY_RATE = 0.95 # 记忆衰减系数

    # =========================================================================
    # 3. 认知与注意力参数 (Cognition & Attention)
    #    对应文档 3.3 节: 距离定义与注意力权重
    # =========================================================================
    
    # [Deep Debug 3] 初始注意力权重 [ω_c, ω_s, ω_g, ω_t]
    # 对应: 1.颜色, 2.大小, 3.接地 (Node Features), 4.接触 (Edge/Structure Feature)
    # 这是一个归一化的分布，sum(INIT_ATTENTION) 应为 1.0
    # 索引映射:
    # 0 -> Color (IDX_COLOR)
    # 1 -> Size (IDX_SIZE)
    # 2 -> Ground (IDX_GROUND)
    # 3 -> Touch (Contact Matrix C)
    # [Optimization] 使用 np.array 替代 list，明确数据类型，防止后续运算出错
    INIT_ATTENTION = np.array([0.25, 0.25, 0.25, 0.25])

    # [补全] Softmax 温度 (Inverse Temperature for Attention Update)
    # 对应文档公式 (194) 的归一化过程: ω ~ exp(ω_tilde / T)
    # 温度越低 (e.g. 0.1)，更新后的注意力越倾向于只关注最显著的那一个维度 (Winner-take-all)
    # 温度越高 (e.g. 10.0)，注意力分布越平滑均匀
    ATTENTION_SOFTMAX_TEMP = 1.0 
    
    # 注意力更新的学习率 (Eta, η)
    # 对应文档 3.4 节慢动力学公式: omega_new = omega_old - eta * gradient
    LEARNING_RATE = 0.05

    # =========================================================================
    # 4. 动力学演化参数 (Dynamics / MCMC / DPP)
    #    对应文档 3.4 节: 快动力学与多假设选择
    # =========================================================================
    
    # [公式 3.4] 探索温度 (T_2 / Beta)
    # 控制 MCMC 采样时的随机性。
    MCMC_TEMP = 1.0 
    MCMC_BETA = 1.0 / MCMC_TEMP 

    # 快动力学 (Fast Dynamics) 的迭代步数
    MCMC_STEPS = 1000 
    
    # 并行链的数量 (Number of Chains)
    NUM_CHAINS = 20 

    # [DPP] 从候选集合 S 中选出的假设数量 (k)
    NUM_HYPOTHESES_KEPT = 3 

    # [DPP] 质量温度 (T for Quality)
    DPP_QUALITY_TEMP = 1.0

    # =========================================================================
    # 5. 实验流程控制 (Experiment Control)
    # =========================================================================
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 最大游戏轮数 (Rounds)
    MAX_ROUNDS = 10

    @staticmethod
    def get_decayed_pinning_field(round_num, is_positive_example=True):
        """
        计算当前轮次的钉扎场强度 B(t)
        
        参数:
            round_num (int): 当前轮数 (0-indexed)
            is_positive_example (bool): True 为正例(Green), False 为反例(Red)
            
        机制:
            1. 根据正负例选择初始强度 (模拟反馈不对称性)
            2. 应用时间衰减 (模拟记忆消退)
            B(t) = B_base * decay^t
        """
        base_strength = (Config.PINNING_POS_MULTIPLIER 
                         if is_positive_example 
                         else Config.PINNING_NEG_MULTIPLIER)
        
        return base_strength * (Config.PINNING_FIELD_DECAY_RATE ** round_num)
    
    # =========================================================================
    # 6. 预计算路径 (Precompute Paths)
    # =========================================================================
    
    # 预计算文件的存储路径
    PRECOMPUTED_DIR = "data"
    DIST_TENSOR_FILE = os.path.join(PRECOMPUTED_DIR, "dist_basis_5127.npy")
    
    # 并行计算核心数 (-1 表示使用所有可用核心)
    NUM_CORES = -1