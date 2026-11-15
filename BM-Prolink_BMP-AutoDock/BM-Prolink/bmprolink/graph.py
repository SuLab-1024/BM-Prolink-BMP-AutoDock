import numpy as np


def _atom_features(mol) -> np.ndarray:
    """
    Extract 78 features per atom covering atomic number, degree, valence, hybridization,
    formal charge, hydrogen count, aromaticity, ring membership, and chirality.
    """
    atoms = mol.GetAtoms()
    feats = []

    for a in atoms:
        f = []

        an = min(a.GetAtomicNum(), 85)
        bins = [0]*44
        if 0 < an <= 85:
            bin_idx = (an - 1) // 2
            if bin_idx < 44:
                bins[bin_idx] = 1
        f.extend(bins)

        d = min(a.GetDegree(), 5)
        deg = [0]*6; deg[d] = 1
        f.extend(deg)

        v = min(a.GetTotalValence(), 5)
        val = [0]*6; val[v] = 1
        f.extend(val)

        hyb = str(a.GetHybridization())
        hyb_map = {'SP':0,'SP2':1,'SP3':2,'SP3D':3,'SP3D2':4}
        hyb_onehot = [0]*5
        hyb_onehot[hyb_map.get(hyb, 2)] = 1
        f.extend(hyb_onehot)

        fc = min(max(int(a.GetFormalCharge()) + 2, 0), 4)
        fc_onehot = [0]*5; fc_onehot[fc] = 1
        f.extend(fc_onehot)

        nh = min(int(a.GetTotalNumHs()), 4)
        nh_onehot = [0]*5; nh_onehot[nh] = 1
        f.extend(nh_onehot)

        f.append(float(a.GetIsAromatic()))

        f.append(float(a.IsInRing()))

        chi = str(a.GetChiralTag())
        chi_map = {'CHI_TETRAHEDRAL_CW':0,'CHI_TETRAHEDRAL_CCW':1,
                    'CHI_OTHER':2,'CHI_UNSPECIFIED':3}
        chi_onehot = [0]*5
        chi_onehot[chi_map.get(chi, 4)] = 1
        f.extend(chi_onehot)

        feats.append(f)

    if not feats:
        return np.zeros((1, 78), dtype=np.float32)
    return np.array(feats, dtype=np.float32)


def build_drug_graph(smiles: str, max_nodes: int = 200):
    """
    Convert one SMILES to (atom_feats, adj, mask).

    Returns:
        atom_feats: (max_nodes, 78)
        adj:        (max_nodes, max_nodes)
        mask:       (max_nodes,)  1=real atom, 0=pad
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    feats = _atom_features(mol)
    n = len(feats)

    # Build an undirected adjacency matrix with self-loops, then truncate or pad to max_nodes.
    adj_mat = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_mat[i, j] = 1.0
        adj_mat[j, i] = 1.0
    np.fill_diagonal(adj_mat, 1.0)

    if n > max_nodes:
        n = max_nodes
        feats = feats[:n]
        adj_mat = adj_mat[:n, :n]

    padded_feats = np.zeros((max_nodes, 78), dtype=np.float32)
    padded_feats[:n] = feats

    padded_adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    padded_adj[:n, :n] = adj_mat

    mask = np.zeros(max_nodes, dtype=np.float32)
    mask[:n] = 1.0

    return padded_feats, padded_adj, mask
