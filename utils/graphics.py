from pymol import cmd, cgo
import numpy as np
from Bio.PDB import PDBParser

def cgo_arrow(start, end, radius=0.1, color=[1,1,1]):
    arrow = [
        cgo.CYLINDER, *start, *end, radius,
        *color, *color,
        cgo.CONE, *end,
        *(end + (start-end)*0.1),
        radius*1.5, 0.0,
        *color, *color,
        1.0, 0.0
    ]
    return arrow

def cgo_sphere(center, radius=0.5, color=[1,1,0]):
    return [cgo.SPHERE, *center, radius, *color]

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-8, None)

def pdb_to_frames(pdb_path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", pdb_path)
    chain = next(struct.get_chains())

    coords_N, coords_CA, coords_C = [], [], []
    for res in chain:
        if all(atom in res for atom in ("N", "CA", "C")):
            coords_N.append(res["N"].get_vector().get_array())
            coords_CA.append(res["CA"].get_vector().get_array())
            coords_C.append(res["C"].get_vector().get_array())

    N = np.stack(coords_N).astype(np.float64)
    CA = np.stack(coords_CA).astype(np.float64)
    C = np.stack(coords_C).astype(np.float64)

    e1 = normalize(C - CA)
    v = N - CA
    u2 = v - e1 * np.einsum('ij,ij->i', e1, v)[..., None]
    e2 = normalize(u2)
    e3 = normalize(np.cross(e1, e2))

    return CA, e1, e2, e3

def add_frames(pdb_fn, scale=5.0, sphere_radius=0.5):
    """
    Load pdb_fn, show cartoon, overlay a chemically-informed local frame (e1,e2,e3) at each C-alpha,
    and draw a sphere around each C-alpha.

    ARGS:
      pdb_fn       : path to PDB file
      scale        : arrow length scaling factor (Angstroms)
      sphere_radius: radius for the C-alpha sphere (Angstroms)
    """
    # Load protein
    cmd.load(pdb_fn, 'prot')
    cmd.hide('everything', 'prot')
    cmd.show('cartoon', 'prot')
    cmd.color('slate', 'prot')

    # Get frames and CA coords
    CA, e1, e2, e3 = pdb_to_frames(pdb_fn)
    n = len(CA)

    for i in range(n):
        O = CA[i]
        sph = cgo_sphere(O, radius=sphere_radius, color=[1,1,0])
        cmd.load_cgo(sph, f'sphere_CA_{i}')

        arr = []
        arr += cgo_arrow(O, O + e1[i]*scale, color=[1,0,0])  # red = e1 = CA → C
        arr += cgo_arrow(O, O + e2[i]*scale, color=[0,1,0])  # green = e2 = CA → N, orthogonalized
        arr += cgo_arrow(O, O + e3[i]*scale, color=[0,0,1])  # blue = e3 = e1 × e2
        cmd.load_cgo(arr, f'frame_CA_{i}')

    cmd.zoom('prot')

# Register in PyMOL
cmd.extend('add_frames', add_frames)
