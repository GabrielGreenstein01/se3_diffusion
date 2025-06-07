from torch.utils.data import Dataset
import torch
from utils.pdb_utils import pdb_to_frames

class SE3FrameDataset(Dataset):
    def __init__(self, frames_dict, max_time = 5, min_time = 1e-4, N_max_pairs=100000):
        self.ids = list(frames_dict.keys())
        self.R_list = [frames_dict[pid][0] for pid in self.ids]
        self.x_list = [frames_dict[pid][1] for pid in self.ids]
        self.max_time = max_time
        self.min_time = min_time
        self.N_max_pairs = N_max_pairs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        R0 = self.R_list[idx] # (L, 3, 3); L = # of residues
        x0 = self.x_list[idx] # (L, 3)
        protein_id = self.ids[idx]
        L = R0.shape[0]

        # scale and remove center of mass of x0
        x0_center = x0.mean(dim=0, keepdim=True) # (L, 1)
        x0 = x0 - x0_center  # (L,3); subtract center of mass per copy
        
        # Determine number of copies
        num_copies = max(1, self.N_max_pairs // (L * L))

        # Create copies of R0, x0
        R0 = R0.expand(num_copies, *R0.shape).clone()  # (N_copies, L, 3, 3)
        x0 = x0.expand(num_copies, *x0.shape).clone()      # (N_copies, L, 3)

        # Sample different timesteps for each copy
        timesteps = torch.rand((num_copies, 1), dtype=torch.float64) * (self.max_time - self.min_time) + self.min_time
            
        # Shuffle copies (done to avoid overfitting in training)
        perm = torch.randperm(num_copies)
        R0 = R0[perm]
        x0 = x0[perm]
        timesteps = timesteps[perm]

        # Return copies; noising, denoising, and loss calculations are done in training loop
        return {
            "protein_id": protein_id,
            "rotations": R0,         # (N_copies, L, 3, 3)
            "translations": x0,      # (N_copies, L, 3)
            "timesteps": timesteps   # (N_copies, 1)
        }

def squeeze_batch(batch_list):
    assert len(batch_list) == 1  # batch_size=1 expected
    batch = batch_list[0]
    return {
        k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 1 else v
        for k, v in batch.items()
    }

def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}