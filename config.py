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
    MAX_BLOCKS = 3

    # --- 节点特征 (Node Features) ---
    DIM_FEATURE = 7 
    IDX_COLOR  = slice(0, 3)  # R3 one-hot -> shape (3,)
    IDX_SIZE   = slice(3, 6)  # R3 one-hot -> shape (3,)
    IDX_GROUND = slice(6, 7)  # scalar 0/1 -> shape (1,)

    # =========================================================================
    # 2. 物理引擎参数 (Physics Engine / Ising Model)
    #    对应文档 3.3 节 (距离) 和 3.4 节 (哈密顿量)
    # =========================================================================
    
    # [公式 286/PDF 3.4] 基础耦合强度 (J_0)
    J0 = 1.0 
    
    # [公式 286/PDF 3.4] 相关长度 (Sigma, σ)
    SIGMA = 0.5 
    
    # [公式 PDF 3.3] 感知温度 (T_1)
    T_PERCEPTION = 0.1 

    # [已移除] 稀疏性偏差 (Sparsity Bias)
    # 依据: 文档 3.5 疑问部分，非核心公理，且可能干扰动力学。
    SPARSITY_BIAS = 0.0  

    # [Deep Debug Corrected] 钉扎场强度 (Pinning Field Strength, B)
    # 依据: 文档 3.4 节 H(s) 定义。
    # 修正: 统一正负例的场强幅度。不再区分 Pos/Neg Multiplier。
    # 任何学习率的不对称性应由 Ising 动力学自发涌现 (Surface Tension)，而非硬编码。
    PINNING_FIELD_STRENGTH = 1000.0  
    
    PINNING_FIELD_DECAY_RATE = 0.95 # 记忆衰减系数

    # =========================================================================
    # 3. 认知与注意力参数 (Cognition & Attention)
    #    对应文档 3.3 节: 距离定义与注意力权重
    # =========================================================================
    
    # [Deep Debug 3] 初始注意力权重 [ω_c, ω_s, ω_g, ω_t]
    # sum(INIT_ATTENTION) 应为 1.0
    INIT_ATTENTION = np.array([0.25, 0.25, 0.25, 0.25])

    # [补全] Softmax 温度
    ATTENTION_SOFTMAX_TEMP = 1.0 
    
    # 注意力更新的学习率 (Eta, η)
    LEARNING_RATE = 0.05

    # =========================================================================
    # 4. 动力学演化参数 (Dynamics / MCMC / DPP)
    #    对应文档 3.4 节: 快动力学与多假设选择
    # =========================================================================
    
    # [公式 3.4] 探索温度 (T_2 / Beta)
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
    def get_decayed_pinning_field(round_num):
        """
        计算当前轮次的钉扎场强度 B(t)
        
        参数:
            round_num (int): 当前轮数 (0-indexed)
            
        机制:
            1. 使用统一的基础强度 (PINNING_FIELD_STRENGTH)
            2. 应用时间衰减 (模拟记忆消退)
        """
        base_strength = Config.PINNING_FIELD_STRENGTH
        return base_strength * (Config.PINNING_FIELD_DECAY_RATE ** round_num)
    
    # =========================================================================
    # 6. 预计算路径 (Precompute Paths)
    # =========================================================================
    
    PRECOMPUTED_DIR = "data"
    DIST_TENSOR_FILE = os.path.join(PRECOMPUTED_DIR, "dist_basis_5127.npy")
    NUM_CORES = -1