import os
import re
import numpy as np
from . import protein as protein

def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    chain_idx: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_idx_arrs = []
    _, unique_indices, unique_counts = np.unique(
        chain_idx, return_index=True, return_counts=True
    )
    # Sort by first occurance
    for chain_counts in unique_counts[np.argsort(unique_indices)]:
        residue_idx_arrs.append(np.arange(chain_counts))
    residue_index = np.concatenate(residue_idx_arrs)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_idx,
        b_factors=b_factors,
    )

def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    chain_idx: np.ndarray,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, chain_idx, aatype=aatype, b_factors=b_factors
                )
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos,
                atom37_mask,
                chain_idx=chain_idx,
                aatype=aatype,
                b_factors=b_factors,
            )
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path

def save_traj(
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        chain_idx: np.ndarray,
        plm_embed: np.ndarray,
        output_dir: str,
):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        chain_idx: which chain each residue belongs to.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, "sample")
    prot_traj_path = os.path.join(output_dir, "bb_traj")
    x0_traj_path = os.path.join(output_dir, "x0_traj")

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    sample_path = write_prot_to_pdb(
        bb_prot_traj[0],
        sample_path,
        b_factors=b_factors,
        chain_idx=chain_idx,
    )
    prot_traj_path = write_prot_to_pdb(
        bb_prot_traj,
        prot_traj_path,
        b_factors=b_factors,
        chain_idx=chain_idx,
    )
    x0_traj_path = write_prot_to_pdb(
        x0_traj, x0_traj_path, b_factors=b_factors, chain_idx=chain_idx
    )
    np.save(output_dir + '/sample_1.npy', plm_embed)
    return {
        "sample_path": sample_path,
        "traj_path": prot_traj_path,
        "x0_traj_path": x0_traj_path,
    }