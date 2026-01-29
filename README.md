# Zendo_Ising_Simulation

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

算好公案两两之间在各属性维度上的距离保存备用。

## test_distance_perception.py

测试一下图神经网络给出的嵌入算出的距离是否符合直觉，尤其是是否符合人类前注意阶段的加工特征，和项目其它代码完全独立。