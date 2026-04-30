import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from linformer import LinformerSelfAttention
from nystrom_attention import NystromAttention
from fast_transformers.attention.linear_attention import LinearAttention


path = './data/ZINC-PE'
# ZINC dataset is of molecular graphs, saves 20d vectors under attribute pe giving unique structure signature
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform) # split the dataset into train, validate, and test %)
val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)


# Dataloader looks at the data in chunks, setting train to 32 for prediciton and error calculations along with backpropagation, val & test are only performing forward
# passes so there is no need thus can be set to 64 to maximize GPU usage
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 64)
test_loader = DataLoader(test_dataset, batch_size = 64)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args(args=[]) # Fix: Pass an empty list to parse_args in Colab notebooks

# positional encodings help to answer 'Where are we'
'''
local PE(node) - distance to cluster center, sum of non-diagonal elements in m-step RW
global PE(node) - eigenvectors of A, L, or distance matrix, distance to graphs centroid, unique ID for each CC
relative PE(edge) - pairwise dists from heat kernels, random walks, green's function, graph geodesic, gradient of eigen vectors
'''

'''
structural encodings - what does my neighborhood look like
local (node) - node degree, RW diagonals, Ricci curvature, Enumerate substructures (traingles,rings)
global (graph) - eigvenvecotrs of A,L. Graph diameter, girth, degree, #CC
relative (edge) - Gradient of any local SE, gradient of sub-structure enumeration
'''

'''
linear transformer, to bypass the computation of the full attention matrix and approximate it instead with "math tricks", such as low
rank decomposition in Linformer, or softmax kernel approx in Performer.

in normal cases the Transformer Layer would materialize the attention matrix, "Materialize attention matrix" refers to the act of explicitly calculating and storing the full
matrix (being the sequence length) that represents how much each token in a sequence attends to every other token.

But with linear transformer no materialization happens, and instead an approximation of full attention, thus meanning cant take edge features directly. This results in
a TC of O(N) where normal Transformer Layer would be O(N^2)

attention matrix is a table of every single relationship
input is a feature vector with different values which is then mult by random projection matrix shortening to the important features it projects and captures,
so we just use a different random projection matrix each time focusing on important features and over time we should get a learned model,
that also be able to handle overfitting better because its percieving different matrices
'''
class RedrawProjection:
    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self): # creates random projection matrices
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(Linear(channels, channels),
                            ReLU(),
                            Linear(channels, channels))
            conv = GPSConv(channels, GINEConv(nn), heads=4, attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1)
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)


attn_kwargs = {'dropout': 0.5}
# Fix: Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=args.attn_type,
            attn_kwargs=attn_kwargs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                            min_lr=0.00001)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    scheduler.step(val_mae)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')