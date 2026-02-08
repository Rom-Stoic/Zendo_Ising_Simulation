"""
距离融合脚本 (Distance Fusion)

将GNN距离和理性距离按照认知偏好权重融合，得到最终的距离张量。

认知模型：
- GNN距离 (30%): 直觉层面，基于图神经网络嵌入的"模糊相似感"
- Rational距离 (70%): 理性层面，基于匈牙利匹配的"精确属性核对"

融合策略：
1. 分别归一化两个距离张量（按每个维度的全局均值）
2. 按权重加权求和
3. 保存融合后的距离供后续使用
"""

import numpy as np
import os
from config import Config


def normalize_distance_tensor(dist_tensor):
    """
    归一化距离张量
    
    对每个维度(Color, Size, Ground, Touch)分别归一化，
    使得每个维度的全局均值为1（排除对角线）
    
    Args:
        dist_tensor: (N, N, 4) 距离张量
        
    Returns:
        dist_normalized: 归一化后的距离张量
    """
    N = dist_tensor.shape[0]
    
    # 创建掩码，排除对角线
    mask = ~np.eye(N, dtype=bool)
    
    # 对每个维度分别归一化
    dist_normalized = dist_tensor.copy()
    
    for dim in range(4):
        # 计算非对角线元素的均值
        dim_data = dist_tensor[:, :, dim]
        mean_val = np.mean(dim_data[mask])
        
        # 归一化（避免除零）
        if mean_val > 1e-6:
            dist_normalized[:, :, dim] = dim_data / mean_val
        else:
            print(f"⚠️  警告: 维度 {dim} 的均值接近0，跳过归一化")
    
    return dist_normalized


def fuse_distances():
    """
    主函数：加载、归一化并融合两种距离
    """
    print("="*60)
    print("🔬 距离融合器 (Distance Fusion Engine)")
    print("="*60)
    
    # 1. 加载GNN距离
    print(f"\n📂 加载GNN距离: {Config.DIST_GNN_FILE}")
    if not os.path.exists(Config.DIST_GNN_FILE):
        print(f"❌ 错误: GNN距离文件不存在")
        print(f"   请先运行: python train_metric.py --mode run")
        print(f"   然后运行: python precompute.py")
        return
    
    dist_gnn = np.load(Config.DIST_GNN_FILE)
    print(f"   ✅ 形状: {dist_gnn.shape}, 大小: {dist_gnn.nbytes / (1024**2):.2f} MB")
    
    # 2. 加载理性距离
    print(f"\n📂 加载理性距离: {Config.DIST_RATIONAL_FILE}")
    if not os.path.exists(Config.DIST_RATIONAL_FILE):
        print(f"❌ 错误: 理性距离文件不存在")
        print(f"   请先运行: python compute_rational_distance.py")
        return
    
    dist_rational = np.load(Config.DIST_RATIONAL_FILE)
    print(f"   ✅ 形状: {dist_rational.shape}, 大小: {dist_rational.nbytes / (1024**2):.2f} MB")
    
    # 3. 验证形状一致性
    if dist_gnn.shape != dist_rational.shape:
        print(f"❌ 错误: 两个距离张量形状不匹配")
        print(f"   GNN: {dist_gnn.shape}, Rational: {dist_rational.shape}")
        return
    
    # 4. 归一化
    print(f"\n⚙️  正在归一化距离张量...")
    dist_gnn_norm = normalize_distance_tensor(dist_gnn)
    dist_rational_norm = normalize_distance_tensor(dist_rational)
    
    print(f"   GNN距离归一化后的统计:")
    for dim, name in enumerate(['Color', 'Size', 'Ground', 'Touch']):
        mask = ~np.eye(dist_gnn.shape[0], dtype=bool)
        mean_val = np.mean(dist_gnn_norm[:, :, dim][mask])
        print(f"      {name:8s}: mean={mean_val:.4f}")
    
    print(f"   Rational距离归一化后的统计:")
    for dim, name in enumerate(['Color', 'Size', 'Ground', 'Touch']):
        mask = ~np.eye(dist_rational.shape[0], dtype=bool)
        mean_val = np.mean(dist_rational_norm[:, :, dim][mask])
        print(f"      {name:8s}: mean={mean_val:.4f}")
    
    # 5. 加权融合
    print(f"\n🔗 融合权重配置:")
    print(f"   GNN (直觉):     {Config.GNN_WEIGHT*100:.1f}%")
    print(f"   Rational (理性): {Config.RATIONAL_WEIGHT*100:.1f}%")
    
    dist_fused = (Config.GNN_WEIGHT * dist_gnn_norm + 
                  Config.RATIONAL_WEIGHT * dist_rational_norm)
    
    # 6. 保存融合距离
    save_path = Config.DIST_FUSED_FILE
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.save(save_path, dist_fused)
    print(f"\n💾 融合距离已保存至: {save_path}")
    print(f"📦 文件大小: {os.path.getsize(save_path) / (1024**2):.2f} MB")
    
    # 7. 最终统计
    print(f"\n📊 融合距离统计:")
    for dim, name in enumerate(['Color', 'Size', 'Ground', 'Touch']):
        mask = ~np.eye(dist_fused.shape[0], dtype=bool)
        dim_data = dist_fused[:, :, dim][mask]
        print(f"   {name:8s}: mean={np.mean(dim_data):.4f}, "
              f"std={np.std(dim_data):.4f}, "
              f"max={np.max(dim_data):.4f}")
    
    print("\n" + "="*60)
    print("✅ 距离融合完成！")
    print("="*60)
    print(f"\n💡 提示: 系统现在将使用融合距离进行游戏")
    print(f"   文件路径: {Config.DIST_TENSOR_FILE}")


if __name__ == "__main__":
    fuse_distances()
