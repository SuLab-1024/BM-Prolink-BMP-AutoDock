"""Command-line BMP-AutoDock cascade screening."""
import argparse
import math
from pathlib import Path

from bmprolink.csv_io import read_rows, write_rows

from .screen import screen

OUTPUT_FIELDS = ("dti_score", "vina_score", "status", "vina_rank")
BOX_KEYS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")


def main():
    parser = argparse.ArgumentParser(description="Screen a compound library against one target.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with a smiles column")
    parser.add_argument("--output", type=Path, required=True, help="new CSV output path")
    parser.add_argument("--weights", type=Path, required=True, help="full BM-Prolink checkpoint (.pt)")
    parser.add_argument("--sequence-file", type=Path, required=True, help="one FASTA or plain-text sequence")
    parser.add_argument("--receptor", type=Path, required=True, help="PDB or prepared PDBQT")
    parser.add_argument("--vina", default="vina", help="Vina name on PATH or executable path")
    parser.add_argument("--box", nargs=6, type=float, metavar="VALUE",
                        help="center x y z and size x y z in angstroms")
    prescreen = parser.add_mutually_exclusive_group()
    prescreen.add_argument("--threshold", type=float, help="DTI probability cutoff (default: 0.5)")
    prescreen.add_argument("--retention-fraction", type=float,
                           help="fraction of highest-scoring DTI candidates to dock, in (0, 1]")
    parser.add_argument("--receptor-cache-dir", type=Path,
                        help="PDBQT cache (default: .bmpautodock-cache next to the output CSV)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--workers", type=int, default=1, help="parallel Vina jobs")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=1)
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.retention_fraction is not None and not 0 < args.retention_fraction <= 1:
        parser.error("--retention-fraction must be greater than 0 and at most 1")
    if min(args.batch_size, args.workers, args.exhaustiveness, args.num_modes) < 1:
        parser.error("batch size, workers, exhaustiveness and num-modes must be positive")
    suffix = args.receptor.suffix.lower()
    if suffix not in {".pdb", ".pdbqt"}:
        parser.error("--receptor must be a PDB or PDBQT file")
    if suffix == ".pdbqt" and args.box is None:
        parser.error("--box is required for PDBQT receptors")
    if args.box and (not all(map(math.isfinite, args.box)) or min(args.box[3:]) <= 0):
        parser.error("--box must contain finite coordinates and positive sizes")

    from bmprolink.predict import Predictor
    from .docking import AutoDockWrapper, detect_box_from_ligand, prepare_receptor

    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if not args.receptor.is_file():
            raise FileNotFoundError(f"Receptor not found: {args.receptor}")
        lines = args.sequence_file.read_text(encoding="utf-8-sig").splitlines()
        if sum(line.startswith(">") for line in lines) > 1:
            raise ValueError("Provide exactly one target sequence.")
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
        if not sequence.strip():
            raise ValueError("Target sequence is empty.")
        fields, rows = read_rows(args.input, ("smiles",), OUTPUT_FIELDS)
        if not rows:
            write_rows(args.output, [*fields, *OUTPUT_FIELDS], [])
            return
        cache_dir = args.receptor_cache_dir or args.output.parent / ".bmpautodock-cache"

        def create_docker():
            receptor = args.receptor.resolve()
            box = dict(zip(BOX_KEYS, args.box)) if args.box else detect_box_from_ligand(str(receptor))
            if suffix == ".pdb":
                receptor = prepare_receptor(receptor, cache_dir)
            return AutoDockWrapper(str(receptor), box, args.vina, args.exhaustiveness, args.num_modes)

        results = screen(
            rows, sequence, Predictor(args.weights, args.device), create_docker,
            threshold=args.threshold, retention_fraction=args.retention_fraction,
            batch_size=args.batch_size, workers=args.workers,
        )
        write_rows(args.output, [*fields, *OUTPUT_FIELDS], results)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"BMP-AutoDock: {error}\n")


if __name__ == "__main__":
    main()
