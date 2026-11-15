"""Command-line BM-Prolink inference."""
import argparse
from pathlib import Path

from .csv_io import read_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description="Score SMILES–protein pairs with BM-Prolink.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with smiles and sequence columns")
    parser.add_argument("--output", type=Path, required=True, help="new CSV output path")
    parser.add_argument("--weights", type=Path, required=True, help="full BM-Prolink checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    from .predict import Predictor

    try:
        fields, rows = read_rows(args.input, ("smiles", "sequence"), ("dti_score",))
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if rows:
            predictor = Predictor(args.weights, args.device)
            scores = predictor.predict([(row["smiles"], row["sequence"]) for row in rows], args.batch_size)
            for row, score in zip(rows, scores, strict=True):
                row["dti_score"] = score
        write_rows(args.output, [*fields, "dti_score"], rows)
    except (OSError, ValueError) as error:
        parser.exit(2, f"BM-Prolink: {error}\n")


if __name__ == "__main__":
    main()
