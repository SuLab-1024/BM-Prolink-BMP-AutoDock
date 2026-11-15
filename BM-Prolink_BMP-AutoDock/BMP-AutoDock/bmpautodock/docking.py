"""Prepare ligands and receptors and invoke AutoDock Vina."""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np



# Prepare ligand PDBQT from SMILES with RDKit and Meeko.

def _parse_smiles(smiles: str):
    from rdkit import Chem

    if not smiles or not smiles.strip():
        raise ValueError("SMILES is empty.")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles[:80]}")

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f"RDKit could not sanitize the molecule for SMILES: {smiles[:80]}") from e

    return mol


def _generate_and_optimize_conformer(mol, random_seed: int = 42):
    from rdkit.Chem import AllChem

    mol = AllChem.AddHs(mol)
    mol = type(mol)(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 1
    params.pruneRmsThresh = -1.0

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    if len(conf_ids) == 0:
        params.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    if len(conf_ids) == 0:
        raise ValueError("Could not generate a 3D ligand conformer.")

    props = AllChem.MMFFGetMoleculeProperties(mol)
    try:
        if props is not None:
            result = AllChem.MMFFOptimizeMolecule(mol, confId=0, mmffVariant="MMFF94")
            if result != 0:
                AllChem.MMFFOptimizeMolecule(mol, confId=0, mmffVariant="MMFF94s")
        else:
            AllChem.UFFOptimizeMolecule(mol, confId=0)
    except Exception:
        pass  # Use the unoptimized conformer if optimization fails.

    return mol


def _mol_to_pdbqt_string(mol) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    mol_setup = preparator.prepare(mol)
    if not mol_setup:
        raise ValueError("Meeko did not produce a ligand setup.")

    pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup[0])[0]
    if not pdbqt_string.strip():
        raise ValueError("Meeko returned an empty ligand PDBQT.")

    return pdbqt_string


def smiles_to_pdbqt_file(smiles: str, output_path: str) -> bool:
    """Write ligand PDBQT from SMILES. Return True on success or False on failure."""
    try:
        mol = _parse_smiles(smiles)
        mol = _generate_and_optimize_conformer(mol)
        pdbqt_string = _mol_to_pdbqt_string(mol)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(pdbqt_string)
        return True
    except Exception as e:
        print(f"Ligand preparation failed: {e}")
        return False



_BOX_SKIP_RESIDUES = {
    'HOH', 'WAT', 'H2O', 'DOD',
    'EDO', 'PEG', 'PG4', 'PGE', 'PE8', '1PE', '2PE',
    'ACT', 'ACN', 'DMF', 'DMS', 'DMSO', 'EOH', 'ETH',
    'GOL', 'MPD', 'MRD', 'SO4', 'PO4', 'TLA',
}

_STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL',
}

_RECEPTOR_PREP_VERSION = 2

_FLEX_KEYWORD_PATTERN = re.compile(
    r'^(ROOT|ENDROOT|BRANCH|ENDBRANCH|TORSDOF)(?:\s|\d|$)', re.IGNORECASE
)


def _clean_receptor_pdb(pdb_path: Path, output_path: Path) -> None:
    """Write a PDB containing only standard amino acid residues."""
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBIO import PDBIO, Select

    class ProteinSelect(Select):
        def __init__(self):
            self.atom_count = 0

        def accept_residue(self, residue):
            return residue.get_resname().strip() in _STANDARD_AMINO_ACIDS

        def accept_atom(self, atom):
            self.atom_count += 1
            return True

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    selector = ProteinSelect()
    io = PDBIO()
    io.set_structure(structure)

    io.save(str(output_path), select=selector)
    if selector.atom_count == 0:
        raise ValueError("No receptor atoms remained after PDB cleaning.")


def _find_obabel() -> Optional[str]:
    found = shutil.which('obabel') or shutil.which('babel')
    if found:
        return found

    # Look for OpenBabel in the current Python environment on Windows.
    import sys
    conda_env_dir = os.path.dirname(sys.executable)
    candidate = os.path.join(conda_env_dir, "Library", "bin", "obabel.exe")
    if os.path.exists(candidate):
        return candidate

    return None


def _pdb_to_pdbqt_receptor(cleaned_pdb_path: str, output_path: Path, obabel: str) -> None:
    """
    Convert a cleaned receptor PDB to rigid PDBQT with added hydrogens and Gasteiger
    charges, then remove flexible-residue directives.
    """
    cmd = [obabel, cleaned_pdb_path, '-O', str(output_path), '-h', '-xr',
           '--partialcharge', 'gasteiger']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel receptor conversion failed: {result.stderr}")

    # Remove flexible-residue directives from the rigid receptor PDBQT.
    rigid_lines = []
    with open(output_path, 'r') as f:
        for line in f:
            if _FLEX_KEYWORD_PATTERN.match(line.strip()):
                continue
            rigid_lines.append(line)
    if not any(line.startswith(('ATOM  ', 'HETATM')) for line in rigid_lines):
        raise ValueError("OpenBabel did not produce any receptor atoms.")
    with output_path.open('w') as f:
        f.writelines(rigid_lines)

def prepare_receptor(pdb_path: Path, cache_dir: Path) -> Path:
    """Return a cached rigid PDBQT for the input PDB and converter."""
    from Bio import __version__ as biopython_version

    obabel = _find_obabel()
    if obabel is None:
        raise FileNotFoundError("OpenBabel is required to prepare a PDB receptor.")
    converter = Path(obabel).resolve()
    converter_stat = converter.stat()
    source = pdb_path.read_bytes()
    key = hashlib.sha256(source)
    key.update(repr((_RECEPTOR_PREP_VERSION, biopython_version, str(converter),
                     converter_stat.st_size, converter_stat.st_mtime_ns)).encode())
    cached = cache_dir.resolve() / f"{key.hexdigest()}.pdbqt"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=cached.parent) as folder:
        snapshot = Path(folder) / 'receptor.pdb'
        snapshot.write_bytes(source)
        cleaned = Path(folder) / 'cleaned.pdb'
        pending = snapshot.with_suffix('.pdbqt')
        _clean_receptor_pdb(snapshot, cleaned)
        _pdb_to_pdbqt_receptor(str(cleaned), pending, obabel)
        try:
            pending.replace(cached)
        except PermissionError:
            # Reuse the prepared receptor if another process has populated the cache.
            if not cached.is_file() or cached.stat().st_size == 0:
                raise
    return cached



def detect_box_from_ligand(pdb_path: str, padding: float = 5.0,
                            min_box_size: float = 15.0) -> dict:
    """Set the box center and size from the largest eligible individual HETATM residue."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]

    eligible = [residue for residue in model.get_residues()
                if residue.id[0].startswith('H_') and len(residue) > 3
                and residue.get_resname().strip() not in _BOX_SKIP_RESIDUES | _STANDARD_AMINO_ACIDS]
    if not eligible:
        raise ValueError("No eligible co-crystallized ligand found in the PDB for automatic box placement.")

    ligand = max(eligible, key=len)
    positions = np.array([atom.get_coord() for atom in ligand])

    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    center = (min_coords + max_coords) / 2
    size = np.maximum(max_coords - min_coords + 2 * padding, min_box_size)

    return {
        'center_x': float(center[0]), 'center_y': float(center[1]), 'center_z': float(center[2]),
        'size_x': float(size[0]), 'size_y': float(size[1]), 'size_z': float(size[2]),
    }


# Run AutoDock Vina and parse its output.

_AFFINITY_PATTERN = re.compile(
    r'^\s*(\d+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', re.MULTILINE
)


def _run_vina(vina_bin: str, receptor_pdbqt: str, ligand_pdbqt: str,
              output_pdbqt: str, box: dict, exhaustiveness: int, num_modes: int,
              energy_range: float = 3.0, timeout: int = 300) -> Optional[float]:
    cmd = [
        vina_bin,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--out', output_pdbqt,
        '--center_x', str(box['center_x']), '--center_y', str(box['center_y']),
        '--center_z', str(box['center_z']),
        '--size_x', str(box['size_x']), '--size_y', str(box['size_y']),
        '--size_z', str(box['size_z']),
        '--exhaustiveness', str(exhaustiveness),
        '--num_modes', str(num_modes),
        '--energy_range', str(energy_range),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    matches = list(_AFFINITY_PATTERN.finditer(result.stdout))
    if not matches:
        return None

    return float(matches[0].group(2))  # Return the Vina score of the top-ranked pose.


class AutoDockWrapper:

    def __init__(
        self, receptor_pdbqt: str, box: dict, vina_executable: str = "vina",
        exhaustiveness: int = 8, num_modes: int = 1,
    ):
        executable = shutil.which(vina_executable)
        if executable is None:
            raise FileNotFoundError(f"AutoDock Vina executable not found: {vina_executable}")
        self.vina_executable = executable
        self.receptor_pdbqt = receptor_pdbqt
        self.box_params = box
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

    def dock_single(self, smiles: str, ligand_id: str) -> Tuple[str, Optional[float]]:
        """Dock one ligand.

        Args:
            smiles: Ligand SMILES.
            ligand_id: Ligand identifier.

        Returns:
            (ligand_id, docking_score) on success or (ligand_id, None) on failure.
        """
        temp_dir = tempfile.mkdtemp()

        try:
            ligand_pdbqt = os.path.join(temp_dir, "ligand.pdbqt")
            if not smiles_to_pdbqt_file(smiles, ligand_pdbqt):
                return (ligand_id, None)

            output_pdbqt = os.path.join(temp_dir, "pose.pdbqt")

            score = _run_vina(
                self.vina_executable, self.receptor_pdbqt, ligand_pdbqt,
                output_pdbqt, self.box_params, self.exhaustiveness, self.num_modes
            )
            return (ligand_id, score)

        except Exception as e:
            print(f"Docking failed for ligand {ligand_id}: {e}")
            return (ligand_id, None)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def dock_batch(
        self,
        compounds: List[Tuple[str, str]],
        parallel: int = 4,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> List[Tuple[str, Optional[float]]]:
        """Dock a batch of ligands.

        Args:
            compounds: List of (smiles, ligand_id) pairs.
            parallel: Number of docking workers.
            progress_callback: Optional (completed, total, success_count) -> None
                callback called after each docking job.

        Returns:
            List of (ligand_id, score) pairs.
        """
        total = len(compounds)
        results = []
        success_count = 0

        if parallel > 1:
            with Pool(processes=parallel) as pool:
                for result in pool.imap_unordered(self._dock_single_star, compounds):
                    results.append(result)
                    if result[1] is not None:
                        success_count += 1
                    if progress_callback:
                        progress_callback(len(results), total, success_count)
        else:
            for smiles, lid in compounds:
                result = self.dock_single(smiles, lid)
                results.append(result)
                if result[1] is not None:
                    success_count += 1
                if progress_callback:
                    progress_callback(len(results), total, success_count)

        return results

    def _dock_single_star(self, args: Tuple[str, str]) -> Tuple[str, Optional[float]]:
        smiles, ligand_id = args
        return self.dock_single(smiles, ligand_id)
