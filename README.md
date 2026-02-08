# Zendo_Ising_Simulation

以下文件基本按照运行顺序排列，顺着跑即可。

## calculate_koans.py

单纯计算所有可能的合法公案数量，和代码其他部分完全独立。

## config.py

配置文件，包含所有的超参数和常量。

## world.py

定义了所有可能的公案，构造了一个公案宇宙。

## dataset.py

把公案的物理数据转化为图数据；实现训练中正反例的挖掘。

## model.py

4 heads（颜色、大小、接地、接触） Graph Isomorphism Network

## train_metric.py

Train模式: 执行 Triplet Loss 训练。
python train_metric.py --mode train

Run模式: 加载模型，嵌入所有 5127 个公案，保存 .npy 文件供后续使用。
python train_metric.py --mode run

## precompute.py

算好公案两两之间在各属性维度上的距离保存备用（基于GNN嵌入）。

## compute_rational_distance.py

基于匈牙利算法计算理性距离：对每对公案找到最佳积木映射，然后计算各属性维度的距离。

使用方式：
```bash
python compute_rational_distance.py
```

## fuse_distances.py

融合GNN距离和理性距离，按照认知偏好权重（默认30% GNN + 70% Rational）生成最终距离。

使用方式：
```bash
python fuse_distances.py
```

## test_distance.py

测试一下图神经网络给出的嵌入和匈牙利匹配算出的距离是否符合直觉，尤其是是否符合人类前注意阶段和特征整合阶段的加工特征，和项目其它代码完全独立。

## physics.py

Ising model求解器。

## dynamics.py

快慢动力学过程+DPP子集选择。

## game.py

运行游戏。

## visualization.ipynb

可视化每一局游戏的玩家数据。

## view_koan.py

输出单个公案的物理数据方便查看。

使用方式：

- 方式1：python view_koan.py 0（0为公案编号，可替换为其他编号）
- 方式2：python view_koan.py 然后输入公案编号（可连续交互），输入 q 退出