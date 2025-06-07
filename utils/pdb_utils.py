import torch
import numpy as np
from Bio.PDB import PDBParser

def normalize(v, axis=-1, tol=1e-6):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + tol)

def pdb_to_frames(pdb_path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", pdb_path)
    chain = next(struct.get_chains())

    coords_N, coords_CA, coords_C = [], [], []

    # Read N-terminus to the C-terminus
    for res in chain:
        if all(atom in res for atom in ("N", "CA", "C")):
            coords_N.append(res["N"].get_vector().get_array())
            coords_CA.append(res["CA"].get_vector().get_array())
            coords_C.append(res["C"].get_vector().get_array())

    # Stack and convert to float64 for stability
    N = np.stack(coords_N).astype(np.float64)
    CA = np.stack(coords_CA).astype(np.float64)
    C = np.stack(coords_C).astype(np.float64)

    e1 = normalize(C - CA) # (L, 3)
    v = N - CA # (L, 3)
    u2 = v - e1 * np.einsum('ij,ij->i', e1, v)[..., None] # einsum is dot product ==> (L, 1)
    e2 = normalize(u2) # (L,3)
    e3 = normalize(np.cross(e1, e2)) # (L,3)

    # Stack into rotation matrices
    R = np.stack([e1, e2, e3], axis=-1)  # (L, 3, 3)

    # Convert to torch.float64 tensors
    R = torch.from_numpy(R).to(torch.float64)
    x = torch.from_numpy(CA).to(torch.float64)

    return R, x

def write_ca_to_pdb(ca_tensor: torch.Tensor, output_file: str, chain_id: str = "A"):
    """
    Write a minimal PDB file from Cα coordinates stored in a tensor (L, 3).

    Args:
        ca_tensor (torch.Tensor): Tensor of shape (L, 3)
        output_file (str): Path to save the PDB file
        chain_id (str): Single-letter chain ID
    """
    assert ca_tensor.ndim == 2 and ca_tensor.shape[1] == 3, "Tensor must be of shape (L, 3)"

    ca_tensor = ca_tensor.detach().cpu().numpy()
    with open(output_file, "w") as f:
        for i, coord in enumerate(ca_tensor):
            x, y, z = coord
            atom_line = (
                f"ATOM  {i+1:5d}  CA  ALA {chain_id}{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            f.write(atom_line)
        f.write("END\n")
    print(f"Saved Cα PDB to: {output_file}")

def write_frames_to_pdb(x, R, out_path, include_frame_atoms=True, frame_scale=2.0):
    """
    Write a full noised protein backbone (N, CA, C, O) to a PDB file,
    optionally including dummy atoms in the e1, e2, e3 directions for visualization.
    PyMOL will render a cartoon from this file.
    """
    def fmt3(vec):
        return f"{float(vec[0]):8.3f}{float(vec[1]):8.3f}{float(vec[2]):8.3f}"

    x = x.detach().cpu().numpy()
    R = R.detach().cpu().numpy()

    with open(out_path, "w") as f:
        atom_id = 1
        for i in range(len(x)):
            ca = x[i]
            e1 = R[i][:, 0]
            e2 = R[i][:, 1]
            e3 = R[i][:, 2]

            # Estimate positions of backbone atoms
            N = ca - 1.5 * e2
            C = ca + 1.5 * e1
            O = C + 1.2 * e3  # approximated peptide plane geometry

            res_id = i + 1

            f.write(f"ATOM  {atom_id:5d}  N   ALA A{res_id:4d}    {fmt3(N)}  1.00  0.00           N\n")
            atom_id += 1
            f.write(f"ATOM  {atom_id:5d}  CA  ALA A{res_id:4d}    {fmt3(ca)}  1.00  0.00           C\n")
            atom_id += 1
            f.write(f"ATOM  {atom_id:5d}  C   ALA A{res_id:4d}    {fmt3(C)}  1.00  0.00           C\n")
            atom_id += 1
            f.write(f"ATOM  {atom_id:5d}  O   ALA A{res_id:4d}    {fmt3(O)}  1.00  0.00           O\n")
            atom_id += 1

            if include_frame_atoms:
                for axis, label in zip([e1, e2, e3], ["E1", "E2", "E3"]):
                    vec = ca + frame_scale * axis
                    f.write(f"ATOM  {atom_id:5d} {label:<4} ALA A{res_id:4d}    {fmt3(vec)}  1.00  0.00           X\n")
                    atom_id += 1

        f.write("END\n")

