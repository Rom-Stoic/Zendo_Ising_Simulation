import numpy as np
import os
from world import KoanAtlas
from config import Config

# ==========================================
# 🎨 终端可视化工具
# ==========================================
def draw_bar(value, max_val=2.0, width=20, color_code=None):
    """绘制字符进度条"""
    if max_val == 0: max_val = 1.0
    ratio = min(max(value / max_val, 0), 1.0)
    fill_len = int(ratio * width)
    
    bar = "█" * fill_len + "░" * (width - fill_len)
    
    # 简单的 ANSI 颜色
    RESET = "\033[0m"
    if color_code:
        return f"{color_code}{bar} {value:.4f}{RESET}"
    return f"{bar} {value:.4f}"

def print_header(title):
    print(f"\n\033[1;36m{'='*60}\033[0m")
    print(f"\033[1;33m🧪 SCENARIO: {title}\033[0m")
    print(f"\033[1;36m{'='*60}\033[0m")

# ==========================================
# 🔍 增强型搜索引擎
# ==========================================
def describe_koan(atlas, idx):
    """生成公案的自然语言描述"""
    if idx is None: return "Unknown Koan"
    
    mass = atlas.mass_tensor[idx]
    active = np.where(mass > 0)[0]
    num_blocks = len(active)
    feats = atlas.feature_tensor[idx]
    struct = atlas.structure_tensor[idx]
    
    # 显式定义切片，防止 Config 差异
    SLICE_COLOR = slice(0, 3)
    SLICE_SIZE = slice(3, 6)
    IDX_GROUND = 6

    descriptions = []
    for i in active:
        c_vec = feats[i, SLICE_COLOR]
        color = ['R', 'G', 'B'][np.argmax(c_vec)]
        
        s_vec = feats[i, SLICE_SIZE]
        size = ['S', 'M', 'L'][np.argmax(s_vec)]
        
        ground = "GND" if feats[i, IDX_GROUND] > 0.5 else "FLY"
        descriptions.append(f"{size}{color}({ground})")
    
    adj_flat = struct[:num_blocks, :num_blocks].flatten()
    edges = int(np.sum(adj_flat) // 2)
    
    return f"ID {idx:04d}: [{', '.join(descriptions)}] (Edges: {edges})"

def find_koan_advanced(atlas, num_blocks=None, colors=None, sizes=None, 
                       min_ground=None, max_ground=None, num_edges=None):
    """
    高级搜索:
    - colors: list of strings, e.g. ['R', 'R'] (顺序不限)
    - sizes: list of strings, e.g. ['S', 'L']
    - num_edges: int, 指定连接边数 (区分直线和三角形)
    """
    # 显式定义切片
    SLICE_COLOR = slice(0, 3)
    SLICE_SIZE = slice(3, 6)
    IDX_GROUND = 6

    for idx in range(atlas.num_koans):
        mass = atlas.mass_tensor[idx]
        active = np.where(mass > 0)[0]
        
        # 1. Block Count
        if num_blocks is not None and len(active) != num_blocks:
            continue
            
        feats = atlas.feature_tensor[idx]
        
        # 2. Ground Count
        ground_count = np.sum(feats[active, IDX_GROUND] > 0.5)
        if min_ground is not None and ground_count < min_ground: continue
        if max_ground is not None and ground_count > max_ground: continue

        # 3. Edges (Structure)
        if num_edges is not None:
            struct = atlas.structure_tensor[idx]
            current_edges = np.sum(struct[:len(active), :len(active)]) // 2
            if current_edges != num_edges: continue

        # 4. Attribute Sets (Multiset comparison)
        if colors is not None:
            curr_colors = []
            for i in active:
                c_vec = feats[i, SLICE_COLOR]
                c_idx = np.argmax(c_vec)
                curr_colors.append(['R', 'G', 'B'][c_idx])
            if sorted(curr_colors) != sorted(colors): continue
            
        if sizes is not None:
            curr_sizes = []
            for i in active:
                s_vec = feats[i, SLICE_SIZE]
                s_idx = np.argmax(s_vec)
                curr_sizes.append(['S', 'M', 'L'][s_idx])
            if sorted(curr_sizes) != sorted(sizes): continue
            
        return idx
    return None

# ==========================================
# 📊 对比核心逻辑 (增强版：显示三种距离)
# ==========================================
def compare_pair(atlas, idx_a, idx_b, dist_gnn, dist_rational, expectation=""):
    print(f"🅰️  {describe_koan(atlas, idx_a)}")
    print(f"🅱️  {describe_koan(atlas, idx_b)}")
    print(f"\033[3mExpectation: {expectation}\033[0m\n")
    
    # 获取三种距离
    d_fused = atlas.dist_basis[idx_a, idx_b]      # 融合距离 (当前使用)
    d_gnn = dist_gnn[idx_a, idx_b]                # GNN距离 (直觉)
    d_rat = dist_rational[idx_a, idx_b]           # Rational距离 (理性)
    
    # 定义颜色变量
    C_RED = "\033[31m"
    C_YEL = "\033[33m"
    C_GRN = "\033[32m"
    C_CYN = "\033[36m"
    C_WHT = "\033[1;37m"
    C_MAG = "\033[35m"
    C_BLU = "\033[34m"
    
    dim_names = ["Color", "Size", "Ground", "Struct"]
    dim_colors = [C_RED, C_YEL, C_GRN, C_CYN]
    
    # 打印三路对比
    print("  ┌─ 📊 三路距离对比 ─────────────────────────────────────────┐")
    print(f"  │ {'维度':<8} │ {'GNN (直觉)':<25} │ {'Rational (理性)':<25} │ {'Fused (最终)':<25} │")
    print("  ├──────────┼───────────────────────────┼───────────────────────────┼───────────────────────────┤")
    
    for i, name in enumerate(dim_names):
        gnn_bar = draw_bar(d_gnn[i], color_code=dim_colors[i])
        rat_bar = draw_bar(d_rat[i], color_code=dim_colors[i])
        fus_bar = draw_bar(d_fused[i], color_code=dim_colors[i])
        
        # 去除ANSI颜色计算实际长度，统一对齐
        # 简化处理：直接固定宽度
        print(f"  │ {name:<8} │ {gnn_bar:<25} │ {rat_bar:<25} │ {fus_bar:<25} │")
    
    print("  ├──────────┴───────────────────────────┴───────────────────────────┴───────────────────────────┤")
    
    # 加权总和 (使用统一权重)
    weights = Config.INIT_ATTENTION
    total_gnn = np.dot(d_gnn, weights)
    total_rat = np.dot(d_rat, weights)
    total_fused = np.dot(d_fused, weights)
    
    print(f"  │ {'TOTAL':<8} │ {draw_bar(total_gnn, max_val=2.5, color_code=C_MAG):<25} │ "
          f"{draw_bar(total_rat, max_val=2.5, color_code=C_BLU):<25} │ "
          f"{draw_bar(total_fused, max_val=2.5, color_code=C_WHT):<25} │")
    print("  └────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # 显示融合权重提示
    print(f"\n  💡 融合公式: {Config.GNN_WEIGHT*100:.0f}% GNN + {Config.RATIONAL_WEIGHT*100:.0f}% Rational")
    print(f"  📈 差异分析:")
    
    # 计算各维度的最大差异来源
    for i, name in enumerate(dim_names):
        if abs(d_gnn[i] - d_rat[i]) > 0.1:  # 显著差异阈值
            if d_gnn[i] > d_rat[i]:
                print(f"     • {name}: GNN 高 {d_gnn[i] - d_rat[i]:.3f} (直觉感知更强)")
            else:
                print(f"     • {name}: Rational 高 {d_rat[i] - d_gnn[i]:.3f} (理性匹配更精确)")


# ==========================================
# 🚀 主程序
# ==========================================
def run_perception_test():
    print("🧠 Loading Atlas & Distance Tensors...")
    print("="*60)
    
    # 1. 加载公案图册 (默认加载融合距离到 dist_basis)
    atlas = KoanAtlas()
    
    # 2. 额外加载 GNN 和 Rational 距离用于对比
    print("📂 Loading GNN Distance (直觉层)...")
    if not os.path.exists(Config.DIST_GNN_FILE):
        print(f"\033[31m❌ GNN距离文件不存在: {Config.DIST_GNN_FILE}\033[0m")
        print("请先运行: python train_metric.py --mode run && python precompute.py")
        return
    dist_gnn = np.load(Config.DIST_GNN_FILE)
    print(f"   ✅ 形状: {dist_gnn.shape}")
    
    print("📂 Loading Rational Distance (理性层)...")
    if not os.path.exists(Config.DIST_RATIONAL_FILE):
        print(f"\033[31m❌ Rational距离文件不存在: {Config.DIST_RATIONAL_FILE}\033[0m")
        print("请先运行: python compute_rational_distance.py")
        return
    dist_rational = np.load(Config.DIST_RATIONAL_FILE)
    print(f"   ✅ 形状: {dist_rational.shape}")
    
    print(f"📂 Fused Distance (融合层) 已加载到 atlas.dist_basis")
    print(f"   ✅ 形状: {atlas.dist_basis.shape}")
    print(f"   ⚖️  权重: {Config.GNN_WEIGHT*100:.0f}% GNN + {Config.RATIONAL_WEIGHT*100:.0f}% Rational")
    print("="*60)
    
    # -------------------------------------------------
    # 1. 基础对照组 (Identity)
    # -------------------------------------------------
    print_header("1. Identity Test (Self vs Self)")
    # 之前失败是因为强制找colors=['R']。改为找任意一个单块。
    k1 = find_koan_advanced(atlas, num_blocks=1) 
    
    # 🛠️ [CRITICAL FIX] 使用 'is not None'，因为索引 0 是合法的但会被 if 判为 False
    if k1 is not None: 
        compare_pair(atlas, k1, k1, dist_gnn, dist_rational, "所有距离应严格为 0")
    else:
        print("\033[31m❌ 找不到单块样本 (Check data generation)\033[0m")

    # -------------------------------------------------
    # 2. 纯颜色差异 (Color)
    # -------------------------------------------------
    print_header("2. Color Discrepancy")
    # 放宽尺寸限制，只求颜色不同
    k_red = find_koan_advanced(atlas, num_blocks=1, colors=['R'])
    k_blue = find_koan_advanced(atlas, num_blocks=1, colors=['B'])
    
    # 如果找不到纯色，尝试更宽泛的搜索
    if k_red is None: k_red = find_koan_advanced(atlas, num_blocks=1) # 任意单块
    if k_blue is None and k_red is not None: 
        # 找一个颜色和k_red不一样的
        # 使用硬编码的切片 0:3 确保正确
        c_idx_red = np.argmax(atlas.feature_tensor[k_red, 0, 0:3])
        target_color = ['G', 'B'][0] if c_idx_red == 0 else 'R' # 简单变色逻辑
        k_blue = find_koan_advanced(atlas, num_blocks=1, colors=[target_color])

    if k_red is not None and k_blue is not None:
        compare_pair(atlas, k_red, k_blue, dist_gnn, dist_rational, "Color 距离应较高，Size 可能也有差异")
    else:
        print("\033[31m❌ 找不到两种不同颜色的单块样本\033[0m")

    # -------------------------------------------------
    # 3. 纯尺寸差异 (Size)
    # -------------------------------------------------
    print_header("3. Size Discrepancy")
    k_small = find_koan_advanced(atlas, num_blocks=1, sizes=['S'])
    k_large = find_koan_advanced(atlas, num_blocks=1, sizes=['L'])
    if k_small is not None and k_large is not None:
        compare_pair(atlas, k_small, k_large, dist_gnn, dist_rational, "Size 距离高，Color 可能会有杂讯")
    else:
        print("\033[31m❌ 找不到不同尺寸的单块样本\033[0m")

    # -------------------------------------------------
    # 4. 接地逻辑 (Grounding Logic)
    # -------------------------------------------------
    print_header("4. Grounding Distribution")
    # A: 2个都接地
    k_flat = find_koan_advanced(atlas, num_blocks=2, min_ground=2) 
    # B: 1个接地, 1个悬空
    k_stack = find_koan_advanced(atlas, num_blocks=2, max_ground=1)
    
    if k_flat is not None and k_stack is not None:
        compare_pair(atlas, k_flat, k_stack, dist_gnn, dist_rational, "Ground 距离应显著，Struct 也会有差异")
    else:
        print("\033[31m❌ 找不到接地/悬空对比样本\033[0m")

    # -------------------------------------------------
    # 5. 拓扑结构：直线 vs 三角形
    # -------------------------------------------------
    print_header("5. Topology: Line vs Triangle")
    # 尝试放宽颜色要求，只关注边数
    k_line = find_koan_advanced(atlas, num_blocks=3, num_edges=2)
    k_tri = find_koan_advanced(atlas, num_blocks=3, num_edges=3)
    
    if k_line is not None and k_tri is not None:
        compare_pair(atlas, k_line, k_tri, dist_gnn, dist_rational, "Structure 距离应占主导")
    else:
        print("\033[31m⚠️ 未找到完美的直线/三角形样本，可能受物理限制\033[0m")

    # -------------------------------------------------
    # 6. 基数感知：多 vs 少
    # -------------------------------------------------
    print_header("6. Cardinality: 1 vs 3 Blocks")
    k_one = find_koan_advanced(atlas, num_blocks=1)
    k_three = find_koan_advanced(atlas, num_blocks=3)
    
    if k_one is not None and k_three is not None:
        compare_pair(atlas, k_one, k_three, dist_gnn, dist_rational, "Struct 应该有巨大差异")
    else:
        print("\033[31m❌ 找不到 1 vs 3 块的样本\033[0m")

    # -------------------------------------------------
    # 7.1 绑定问题 (Mixed Structure)
    # -------------------------------------------------
    print_header("7.1. The Binding Problem (Same Ground, Mixed Structure)")
    # 寻找 {红大, 蓝小} vs {红小, 蓝大}，且都接地 (避免 Ground 距离干扰)
    idx_bind_a, idx_bind_b = None, None
    
    SLICE_COLOR = slice(0, 3)
    SLICE_SIZE = slice(3, 6)
    IDX_GROUND = 6

    for i in range(atlas.num_koans):
        mass = atlas.mass_tensor[i]
        if np.sum(mass > 0) != 2: continue
        feats = atlas.feature_tensor[i]
        active = np.where(mass > 0)[0]
        
        # 强制要求全接地，排除重力干扰
        if np.sum(feats[active, IDX_GROUND] > 0.5) != 2: continue

        attrs = set()
        for b in active:
            c = np.argmax(feats[b, SLICE_COLOR]) # 0,1,2
            s = np.argmax(feats[b, SLICE_SIZE])  # 0,1,2
            attrs.add((c, s))
        
        # 0=Red, 2=Blue; 0=Small, 2=Large
        if attrs == {(0, 2), (2, 0)}: idx_bind_a = i # R-L, B-S
        if attrs == {(0, 0), (2, 2)}: idx_bind_b = i # R-S, B-L
        
        if idx_bind_a is not None and idx_bind_b is not None: break
    
    if idx_bind_a is not None and idx_bind_b is not None:
        compare_pair(atlas, idx_bind_a, idx_bind_b, dist_gnn, dist_rational, "距离应 > 0 (但可能受结构差异干扰)")
    else:
        print("\033[31m⚠️ 未找到绑定测试样本\033[0m")

    # -------------------------------------------------
    # 7.2 绑定问题 (Strict: Same Structure)
    # -------------------------------------------------
    print_header("7.2. The Binding Problem (Strict: Same Structure)")
    
    k_strict_a, k_strict_b = None, None
    
    # 我们不仅要找绑定属性，还要保证 edge 数量一致
    # 优先找 Edges=0 (分离)，如果找不到再找 Edges=1
    for target_edge in [0, 1]:
        cand_a, cand_b = None, None
        
        for i in range(atlas.num_koans):
            mass = atlas.mass_tensor[i]
            if np.sum(mass > 0) != 2: continue
            
            # 1. 结构检查 (Struct)
            struct = atlas.structure_tensor[i]
            active_struct = struct[:2, :2]
            n_edges = int(np.sum(active_struct) // 2)
            if n_edges != target_edge: continue

            # 2. 接地检查 (Ground)
            feats = atlas.feature_tensor[i]
            active = np.where(mass > 0)[0]
            if np.sum(feats[active, IDX_GROUND] > 0.5) != 2: continue

            # 3. 属性检查 (Attribute Binding)
            attrs = set()
            for b in active:
                c = np.argmax(feats[b, SLICE_COLOR])
                s = np.argmax(feats[b, SLICE_SIZE])
                attrs.add((c, s))
            
            if attrs == {(0, 2), (2, 0)}: cand_a = i
            if attrs == {(0, 0), (2, 2)}: cand_b = i
            
            if cand_a is not None and cand_b is not None: break
        
        if cand_a is not None and cand_b is not None:
            k_strict_a, k_strict_b = cand_a, cand_b
            break

    if k_strict_a is not None and k_strict_b is not None:
        compare_pair(atlas, k_strict_a, k_strict_b, dist_gnn, dist_rational, "严格控制变量：Edge 相同。Struct 距离应极小。")
    else:
        print("\033[31m⚠️ 未找到严格结构的绑定测试样本 (需要 R-L/B-S 且同结构)\033[0m")

    # -------------------------------------------------
    # 8. 集合重叠 (Set Overlap)
    # -------------------------------------------------
    print_header("8. Set Overlap")
    k_rg = find_koan_advanced(atlas, num_blocks=2, colors=['R', 'G'])
    k_rb = find_koan_advanced(atlas, num_blocks=2, colors=['R', 'B'])
    
    if k_rg is not None and k_rb is not None:
        compare_pair(atlas, k_rg, k_rb, dist_gnn, dist_rational, "Color 距离应中等 (部分重叠)")
    else:
        print("\033[31m❌ 找不到红绿 vs 红蓝样本\033[0m")

    # -------------------------------------------------
    # 9. 杂乱 vs 纯净 (Entropy)
    # -------------------------------------------------
    print_header("9. Complexity: Monochrome vs Rainbow")
    k_mono = find_koan_advanced(atlas, num_blocks=3, colors=['R', 'R', 'R'])
    if k_mono is None: k_mono = find_koan_advanced(atlas, num_blocks=3, colors=['B', 'B', 'B']) # 备选
    k_rain = find_koan_advanced(atlas, num_blocks=3, colors=['R', 'G', 'B'])
    
    if k_mono is not None and k_rain is not None:
        compare_pair(atlas, k_mono, k_rain, dist_gnn, dist_rational, "Color 距离应很高")
    else:
        print("\033[31m❌ 找不到纯色 vs 彩虹色样本\033[0m")

    # -------------------------------------------------
    # 10. 最大对比 (Maximal Contrast)
    # -------------------------------------------------
    print_header("10. Maximal Contrast")
    # A: 1个，小 (放宽颜色)
    k_min = find_koan_advanced(atlas, num_blocks=1, sizes=['S'])
    # B: 3个，大 (放宽颜色)
    k_max = find_koan_advanced(atlas, num_blocks=3, sizes=['L','L','L'])
    
    if k_min is not None and k_max is not None:
        compare_pair(atlas, k_min, k_max, dist_gnn, dist_rational, "所有指标爆炸。Total Distance 最高。")
    else:
        print("\033[31m❌ 找不到最大对比样本 (1小 vs 3大)\033[0m")

if __name__ == "__main__":
    run_perception_test()