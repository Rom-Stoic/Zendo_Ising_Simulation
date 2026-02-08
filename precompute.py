import numpy as np
import os
from scipy.spatial.distance import cdist
from config import Config # 确保能读取到 Config 中的路径配置

def precompute_distances():
    print("⏳ [Precompute] 正在初始化全宇宙距离矩阵...")
    
    # 0: Color, 1: Size, 2: Ground, 3: Structure
    emb_files = [
        "data/emb_color.npy",
        "data/emb_size.npy",
        "data/emb_ground.npy",
        "data/emb_structure.npy"
    ]
    
    # 检查文件是否存在
    for f in emb_files:
        if not os.path.exists(f):
            print(f"❌ 错误: 找不到文件 {f}。请先运行 'python train_metric.py --mode run'")
            return

    # 初始化大张量 (N, N, 4)
    N = 5127
    print(f"📦 正在计算 5127x5127 的距离矩阵，这可能需要几秒钟...")
    dist_basis = np.zeros((N, N, 4), dtype=np.float32)
    
    for i, file_path in enumerate(emb_files):
        print(f"🔗 正在处理属性 {i} ...")
        emb = np.load(file_path)
        # 核心运算：计算两两欧氏距离
        dist_basis[:, :, i] = cdist(emb, emb, metric='euclidean')

    # 保存到 Config 指定的位置
    save_path = Config.DIST_GNN_FILE
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    np.save(save_path, dist_basis)
    print(f"✅ 成功！GNN 距离基石已保存至: {save_path} (等待与理性距离融合)")
    print(f"📦 文件大小约为: {os.path.getsize(save_path) / (1024**2):.2f} MB")

if __name__ == "__main__":
    precompute_distances()