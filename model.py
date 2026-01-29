import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from config import Config

class MLP(nn.Module):
    """辅助 MLP 模块，供 GIN 使用"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class ZendoNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        input_dim = Config.DIM_FEATURE # 7
        
        # --- 1. Backbone (特征提取器) ---
        # GIN (Graph Isomorphism Network) 理论上具有区分图同构的最强能力
        
        # Layer 1
        self.conv1 = GINConv(MLP(input_dim, hidden_dim, hidden_dim))
        
        # Layer 2
        self.conv2 = GINConv(MLP(hidden_dim, hidden_dim, hidden_dim))
        
        # Layer 3 (Deep GNN helps with complex structures)
        self.conv3 = GINConv(MLP(hidden_dim, hidden_dim, hidden_dim))

        # --- 2. Disentangled Heads (解耦头) ---
        # 每个 Head 负责将图特征映射到特定的语义空间
        # 输出维度参考 Markdown 设置
        
        # Color Head: (N, 16)
        self.head_color = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 16) 
        )
        
        # Size Head: (N, 16)
        self.head_size = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Ground Head: (N, 8) - 维度较小，因为信息简单
        self.head_ground = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Structure Head: (N, 32) - 维度最大，因为拓扑结构最复杂
        self.head_struct = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, data):
        """
        Args:
            data: PyG Batch Data 对象
        Returns:
            Dictionary of embeddings {attr_name: tensor}
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- Message Passing ---
        x = self.conv1(x, edge_index)
        x = x + self.conv2(x, edge_index) # Residual connection
        x = x + self.conv3(x, edge_index)
        
        # --- Readout (Pooling) ---
        # Sum Pooling 具有置换不变性，适合计数任务 (e.g., "有两个红色")
        # Global graph feature
        g = global_add_pool(x, batch) 
        
        # --- Projection Heads ---
        # 并在最后进行 L2 Normalization，使得 Embedding 落在单位球面上
        # 这样 Euclidean Distance 更加稳定
        
        z_c = F.normalize(self.head_color(g), p=2, dim=1)
        z_s = F.normalize(self.head_size(g),  p=2, dim=1)
        z_g = F.normalize(self.head_ground(g),p=2, dim=1)
        z_t = F.normalize(self.head_struct(g),p=2, dim=1)
        
        return {
            'color': z_c,
            'size': z_s,
            'ground': z_g,
            'structure': z_t
        }