"""Read and write the CSV inputs shared by the two command-line tools."""
import csv
from pathlib import Path


def read_rows(path: Path, required: tuple[str, ...], added: tuple[str, ...]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Input CSV contains duplicate column names.")
        if missing := set(required).difference(fields):
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        if repeated := set(fields).intersection(added):
            raise ValueError(f"Input CSV already has output columns: {', '.join(sorted(repeated))}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record at line {reader.line_num}.")
            if any(not row[field].strip() for field in required):
                raise ValueError(f"Empty required field at line {reader.line_num}.")
            rows.append(row)
    return fields, rows


def write_rows(path: Path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
