import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from world import KoanAtlas
from dataset import ZendoGraphDataset, KoanSampler
from model import ZendoNet
from config import Config

def train(epochs=50, batch_size=64):
    print("🚀 [Train] 初始化训练环境...")
    
    # 1. 准备数据
    atlas = KoanAtlas()
    dataset = ZendoGraphDataset(atlas)
    sampler = KoanSampler(atlas) # 构建索引
    
    # 2. 模型与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZendoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Triplet Margin Loss
    # 使得 D(A, P) + margin < D(A, N)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    attributes = ['color', 'size', 'ground', 'structure']
    
    print(f"🔥 [Train] 开始训练 (Device: {device})...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        # 每个 Epoch 随机挖掘 100 个 Batch
        num_batches = 100 
        
        for _ in range(num_batches):
            batch_loss = 0
            optimizer.zero_grad()
            
            # 对每个属性头分别进行训练
            for attr in attributes:
                # A. 挖掘三元组索引
                a_idx, p_idx, n_idx = sampler.get_triplet_batch(attr, batch_size)
                
                # B. 构建 Batch 图数据
                # 将索引转换为 Dataset 中的 Data 对象列表，再 collate 成 Batch
                batch_a = Batch.from_data_list([dataset.get(i) for i in a_idx]).to(device)
                batch_p = Batch.from_data_list([dataset.get(i) for i in p_idx]).to(device)
                batch_n = Batch.from_data_list([dataset.get(i) for i in n_idx]).to(device)
                
                # C. 前向传播
                out_a = model(batch_a)[attr]
                out_p = model(batch_p)[attr]
                out_n = model(batch_n)[attr]
                
                # D. 计算损失
                loss = criterion(out_a, out_p, out_n)
                batch_loss += loss
            
            # E. 反向传播 (累积了4个头的 Loss)
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            steps += 1
            
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    # 保存模型
    if not os.path.exists("models"): os.makedirs("models")
    torch.save(model.state_dict(), "models/zendonet_metric.pth")
    print("💾 模型已保存至 models/zendonet_metric.pth")

def run_inference():
    """
    加载训练好的模型，对宇宙中所有公案进行编码，并保存为 .npy 文件
    """
    print("🔮 [Inference] 开始全量推理...")
    
    atlas = KoanAtlas()
    dataset = ZendoGraphDataset(atlas)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZendoNet().to(device)
    
    model_path = "models/zendonet_metric.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 已加载预训练模型")
    else:
        print("⚠️ 未找到模型文件，使用随机初始化模型进行测试...")
    
    model.eval()
    
    # 构造全量 Batch
    # 注意：如果显存不够，这里需要分批次 (DataLoader)
    print("📦 正在打包所有数据...")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    emb_c, emb_s, emb_g, emb_t = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            emb_c.append(out['color'].detach().cpu().numpy())
            emb_s.append(out['size'].detach().cpu().numpy())
            emb_g.append(out['ground'].detach().cpu().numpy())
            emb_t.append(out['structure'].detach().cpu().numpy())
            
    # 拼接
    final_c = np.concatenate(emb_c, axis=0)
    final_s = np.concatenate(emb_s, axis=0)
    final_g = np.concatenate(emb_g, axis=0)
    final_t = np.concatenate(emb_t, axis=0)
    
    # 保存
    if not os.path.exists("data"): os.makedirs("data")
    np.save("data/emb_color.npy", final_c)
    np.save("data/emb_size.npy", final_s)
    np.save("data/emb_ground.npy", final_g)
    np.save("data/emb_structure.npy", final_t)
    
    print(f"✅ Embedding 已保存至 data/ 目录")
    print(f"   Color Shape: {final_c.shape}")
    print(f"   Struct Shape: {final_t.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'run'], help='Mode: train or run (inference)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    else:
        run_inference()