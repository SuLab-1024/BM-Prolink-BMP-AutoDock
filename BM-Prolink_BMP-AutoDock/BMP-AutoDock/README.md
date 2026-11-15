# BMP-AutoDock

BMP-AutoDock uses BM-Prolink to score a compound library against one target.
It prescreens candidates by DTI score, prepares the ligands and receptor, and
runs AutoDock Vina. Successfully docked candidates are ranked by the Vina score
of their top-ranked pose, from lowest to highest.

## Requirements and setup

This is the `bmp-autodock` 0.1.0 source release. It requires Python 3.10 or later
and the accompanying BM-Prolink 0.1.0 source in the sibling `BM-Prolink` folder.

| Python dependency | Version | Role |
| --- | --- | --- |
| `bm-prolink` | `0.1.0` (included source) | DTI prescreening |
| `biopython` | `>=1.81` | Receptor PDB parsing |
| `meeko` | `>=0.7,<0.8` | Ligand PDBQT preparation |
| `gemmi` | `>=0.6` | Meeko support |

Install the BM-Prolink dependencies listed in its [README](../BM-Prolink/README.md).
In the same Python environment, install the additional docking dependencies.

```sh
python -m pip install "biopython>=1.81" "meeko>=0.7,<0.8" "gemmi>=0.6"
```

From the directory that contains both project folders, enter `BMP-AutoDock`
and add the sibling `BM-Prolink` folder to the Python module search path for
the current terminal.

**PowerShell**

```powershell
Set-Location .\BMP-AutoDock
$env:PYTHONPATH = (Resolve-Path ..\BM-Prolink).Path
```

**macOS or Linux**

```sh
cd BMP-AutoDock
export PYTHONPATH="$(cd ../BM-Prolink && pwd)"
```

Install AutoDock Vina as a separate executable. Put it on PATH or pass its executable
path with `--vina`. Install OpenBabel and put `obabel` on PATH if the receptor
input is a PDB file. For BM-Prolink weights and encoder setup, see the
[BM-Prolink README](../BM-Prolink/README.md).

## Input and usage

Provide a UTF-8 CSV with a `smiles` column and optionally other fields such as
compound identifiers, a single target protein sequence in FASTA or plain text,
and the receptor structure. The automatic box uses the largest eligible individual
HETATM residue in the first model of the original PDB. Water, common solvents,
standard amino acids and residues with fewer than four atoms are excluded. The box
extends 5 Å beyond the selected residue on each side and has a minimum side length
of 15 Å. Pass `--box CX CY CZ SX SY SZ` to specify the box in angstroms. A prepared
PDBQT receptor requires `--box` and bypasses receptor preparation.

PDB receptor preparation keeps standard amino acids. OpenBabel adds hydrogens,
assigns Gasteiger partial charges, and writes a rigid PDBQT. By default,
prepared receptors are cached in `.bmpautodock-cache`
next to the output CSV and reused for unchanged inputs and preparation tools.
Use `--receptor-cache-dir PATH` to share a cache across output directories.

```sh
python -m bmpautodock.cli --input compounds.csv --sequence-file target.fasta --receptor target.pdb --weights model.pt --output ranked.csv
```

By default, compounds with a DTI score of 0.5 or higher are sent to docking. Use
`--threshold` to change the cutoff or `--retention-fraction` to keep the highest
scoring fraction, such as `--retention-fraction 0.10` for the top 10%. The retained
count is rounded up to the next whole compound. Ties at the cutoff use input order.
Use at most one of the two options.

## Output

The output keeps every input row, including failed or rejected entries.

| Column | Meaning |
| --- | --- |
| `dti_score` | BM-Prolink prediction used for prescreening |
| `vina_score` | Best Vina score in kcal/mol, when docking succeeds |
| `status` | `docked`, `prescreen_rejected`, or `docking_failed` |
| `vina_rank` | Assigned only to successful dockings, from the lowest Vina score upward. Exact ties are listed in input order. |

Run `python -m bmpautodock.cli --help` for device, batch size and docking options.
Use a new output path for each run.
