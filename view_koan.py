#!/usr/bin/env python3
"""
公案查看器 (Koan Viewer)

用法:
    python view_koan.py [公案编号]
    或直接运行后输入编号
"""

import sys
import numpy as np
from world import KoanAtlas

def view_koan(atlas, koan_id):
    """
    显示指定编号公案的详细信息
    
    Args:
        atlas: KoanAtlas 实例
        koan_id: 公案编号 (0 到 5126)
    """
    # 检查编号有效性
    if koan_id < 0 or koan_id >= atlas.num_koans:
        print(f"❌ 错误: 公案编号必须在 0-{atlas.num_koans-1} 之间")
        return
    
    # 获取公案数据
    feature_indices, adj_tuple = atlas.koans[koan_id]
    n = len(feature_indices)
    
    # 重构邻接矩阵
    adj_matrix = np.array(adj_tuple).reshape(n, n)
    
    # 输出基本信息
    print(f"\n{'='*50}")
    print(f"公案编号: {koan_id}")
    print(f"{'='*50}")
    print(f"积木数量: {n}")
    print()
    
    # 输出每个积木的特征
    print("积木特征:")
    print(f"{'块号':<6} {'尺寸':<10} {'颜色':<6} {'接地':<6}")
    print("-" * 50)
    
    color_map = {'R': '红色', 'G': '绿色', 'B': '蓝色'}
    size_map = {'Small': '小', 'Medium': '中', 'Large': '大'}
    
    for i, type_idx in enumerate(feature_indices):
        size_str, color_str, grounded = atlas.block_types[type_idx]
        size_cn = size_map[size_str]
        color_cn = color_map[color_str]
        grounded_cn = '是' if grounded == 1 else '否'
        print(f"块{i:<5} {size_cn:<10} {color_cn:<6} {grounded_cn:<6}")
    
    print()
    
    # 输出接触关系
    print("接触关系 (邻接矩阵):")
    print("   ", end="")
    for i in range(n):
        print(f"块{i} ", end="")
    print()
    
    for i in range(n):
        print(f"块{i} ", end="")
        for j in range(n):
            if i == j:
                print(" -  ", end="")  # 对角线不显示
            else:
                print(f" {adj_matrix[i, j]}  ", end="")
        print()
    
    print("\n说明: 1 = 接触, 0 = 不接触")
    
    # 输出接触对列表（更直观）
    touching_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 1:
                touching_pairs.append((i, j))
    
    if touching_pairs:
        print(f"\n接触的积木对: ", end="")
        print(", ".join([f"(块{i}, 块{j})" for i, j in touching_pairs]))
    else:
        print("\n接触的积木对: 无 (所有积木相互独立)")
    
    print(f"{'='*50}\n")


def main():
    """主函数"""
    # 初始化图册
    print("正在加载 Zendo 宇宙...")
    atlas = KoanAtlas()
    print(f"✅ 已加载 {atlas.num_koans} 个公案\n")
    
    # 从命令行参数或交互式输入获取编号
    if len(sys.argv) > 1:
        try:
            koan_id = int(sys.argv[1])
            view_koan(atlas, koan_id)
        except ValueError:
            print("❌ 错误: 请输入有效的整数编号")
            sys.exit(1)
    else:
        # 交互式模式
        print("进入交互模式 (输入 'q' 退出)")
        while True:
            try:
                user_input = input(f"\n请输入公案编号 (0-{atlas.num_koans-1}): ").strip()
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("再见!")
                    break
                
                koan_id = int(user_input)
                view_koan(atlas, koan_id)
                
            except ValueError:
                print("❌ 错误: 请输入有效的整数编号")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break


if __name__ == "__main__":
    main()
