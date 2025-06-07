import torch
import numpy as np
import torch.nn as nn


def sinusoidal_encoding(indices, dim):
    """
    indices: Tensor of shape (L,) or scalar, e.g., residue indices or time steps
    dim: embedding dimension
    returns: Tensor of shape (L, dim)
    """
    if isinstance(indices, int):
        indices = torch.tensor([indices], dtype=torch.float)
    elif isinstance(indices, float):
        indices = torch.tensor([indices], dtype=torch.float)

    indices = indices.float().unsqueeze(-1)  # (L, 1)
    half_dim = dim // 2

    # torch.exp(-np.log(...) * ...) done for numerical stability
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, device=indices.device).float() / half_dim)  # (dim/2,)
    angles = indices * freqs  # (N, dim/2)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (L, dim)


def orthogonalize_time_encoding(time_enc, pos_enc):
    """
    Projects time_enc orthogonal to pos_enc per node
    time_enc, pos_enc: (L, D)
    returns: (L, D)
    """
    dot = (time_enc * pos_enc).sum(dim=-1, keepdim=True)  # (L, 1)
    norm_sq = (pos_enc * pos_enc).sum(dim=-1, keepdim=True) + 1e-6  # avoid div by zero
    proj = dot / norm_sq * pos_enc  # (L, D)
    return time_enc - proj

def encode_node_features(x_t, t, dim):
    """
    x_t: (N_copies, L, 3) - noised coordinates
    t:   (N_copies, 1) - diffusion times
    dim: int - dimension for each encoding (output will be 2*dim)

    Returns:
        node_feats: (N_copies, L, 2*dim)
    """
    N_copies, L, _ = x_t.shape
    device = x_t.device

    residue_idx = torch.arange(L, device=device)  # (L,)
    residue_enc = sinusoidal_encoding(residue_idx, dim)  # (L, dim), shared across batch

    # Broadcast residue encoding to each protein copy
    residue_enc = residue_enc.unsqueeze(0).expand(N_copies, L, dim)  # (N_copies, L, dim)

    # Encode each t (N_copies, 1) as (N_copies, L, dim)
    t_enc = sinusoidal_encoding(t, dim).expand(-1, L, -1)  # (N_copies, L, dim)

    t_enc_ortho = orthogonalize_time_encoding(t_enc, residue_enc)  # (N_copies, L, dim)

    return torch.cat([residue_enc, t_enc_ortho], dim=-1)  # (N_copies, L, 2*dim)

def positional_encoding(offsets, dim=16):
    """
    Sinusoidal positional encoding for edge positional offsets using consistent
    scaling with node sinusoidal encoding.
    
    Args:
        offsets: Tensor of shape (L, L) with absolute residue offsets
        dim: number of frequency terms (output will be 2 * dim)
    
    Returns:
        pos_enc: Tensor of shape (L, L, 2 * dim)
    """
    offsets = offsets.unsqueeze(-1).float()  # (L, L, 1)

    freqs = torch.exp(-np.log(10000.0) * torch.arange(dim, device=offsets.device).float() / dim)  # (dim,)
    angles = offsets * freqs  # (L, L, dim)

    pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (L, L, 2 * dim)
    return pos_enc


def encode_edge_features(L, dim, device='cpu'):
    """
    L: int - number of residues in the protein
    dim: int - embedding dimension (output will be 2*dim)

    Returns:
        edge_feats: (L, L, 2*dim) - positional offset encoding between residue pairs
    """
    offset_matrix = torch.arange(L, device=device).view(-1, 1) - torch.arange(L, device=device).view(1, -1)
    offsets = offset_matrix.abs()  # (L, L)
    edge_feats = positional_encoding(offsets, dim=dim)  # (L, L, 2*dim)
    return edge_feats


class EGNNLayer(nn.Module):
    def __init__(self, h_dim, edge_dim, hidden_dim=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * h_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(h_dim + hidden_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim)
        )

        self.norm = nn.LayerNorm(h_dim)

    def forward(self, x, h, edge_attr):
        """
        x: (N_copies, L, 3) - coordinates
        h: (N_copies, L, h_dim) - node features
        edge_attr: (N_copies, L, L, edge_dim) - edge features
        """
        N, L, _ = x.shape

        # Compute pairwise differences and distances
        dx = x[:, :, None, :] - x[:, None, :, :]  # (N_copies, L, L, 3)
        d2 = (dx ** 2).sum(dim=-1, keepdim=True)  # (N_copies, L, L, 1)

        # Expand features for all pairs
        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)  # (N_copies, L, L, h_dim)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)  # (N_copies, L, L, h_dim)

        edge_input = torch.cat([h_i, h_j, edge_attr, d2], dim=-1)  # (N_copies, L, L, 2h + e + 1)
        m_ij = self.edge_mlp(edge_input)  # (N_copies, L, L, hidden_dim)

        # Coordinate update
        coord_weights = self.coord_mlp(m_ij)  # (N_copies, L, L, 1)
        dx_update = (dx * coord_weights).sum(dim=2)  # (N_copies, L, 3)
        x = x + dx_update

        # Feature update
        m_sum = m_ij.sum(dim=2)  # (N_copies, L, hidden_dim)
        h = self.node_mlp(torch.cat([h, m_sum], dim=-1))  # (N_copies, L, h_dim)

        h = self.norm(h)
        
        return x, h # (N_copies, L, 3), (N_copies, L, h_dim)

class EGNN(nn.Module):
    def __init__(self, h_dim, edge_dim, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(h_dim, edge_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, x, h, edge_attr):
        """
        Args:
            x: Tensor of shape (N_copies, L, 3)
            h: Tensor of shape (N_copies, L, h_dim)
            edge_attr: Tensor of shape (N_copies, L, L, edge_dim)
        Returns:
            updated coordinates and node features
        """
        for layer in self.layers:
            x, h = layer(x, h, edge_attr)
        return x, h # (N_copies, L, 3), (N_copies, L, h_dim)

class EGNNScoreModel(nn.Module):
    def __init__(self, h_dim, edge_dim, hidden_dim, n_layers):
        super().__init__()
        self.egnn = EGNN(h_dim, edge_dim, hidden_dim, n_layers)
        self.output_proj = nn.Linear(h_dim, 3)  # To predict (dx/dt) ∈ R³

    def forward(self, x_t, t, node_features, edge_features):
        x, h = self.egnn(x_t, node_features, edge_features)
        score_pred = self.output_proj(h)  # (N_copies, L, 3)
        return score_pred