"""Prescreen with BM-Prolink, dock retained candidates, and rank by Vina score."""
import math
from decimal import Decimal


def screen(rows, sequence, predictor, create_docker, *, threshold=None,
           retention_fraction=None, batch_size=8, workers=1):
    if threshold is not None and retention_fraction is not None:
        raise ValueError("Choose either a DTI threshold or a retention fraction.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("The DTI threshold must be between 0 and 1.")
    if retention_fraction is not None and not 0 < retention_fraction <= 1:
        raise ValueError("The retention fraction must be greater than 0 and at most 1.")
    if batch_size < 1 or workers < 1:
        raise ValueError("Batch size and workers must be positive.")
    if not rows:
        return []

    scores = predictor.predict([(row["smiles"], sequence) for row in rows], batch_size)
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("BM-Prolink returned a nonfinite DTI score.")

    results = [
        dict(row, dti_score=score, vina_score=None, status="prescreen_rejected", vina_rank=None)
        for row, score in zip(rows, scores, strict=True)
    ]
    if retention_fraction is None:
        cutoff = 0.5 if threshold is None else threshold
        selected_indices = [index for index, score in enumerate(scores) if score >= cutoff]
    else:
        count = math.ceil(len(results) * Decimal(str(retention_fraction)))
        selected_indices = sorted(range(len(results)), key=scores.__getitem__, reverse=True)[:count]

    selected = []
    for index in selected_indices:
        row = results[index]
        row["status"] = "docking_failed"
        selected.append((row["smiles"].strip(), str(index)))
    if selected:
        for ligand_id, score in create_docker().dock_batch(selected, parallel=workers):
            if score is not None and math.isfinite(score):
                results[int(ligand_id)].update(vina_score=score, status="docked")

    ranked = sorted(
        (row for row in results if row["status"] == "docked"), key=lambda row: row["vina_score"]
    )
    for rank, row in enumerate(ranked, start=1):
        row["vina_rank"] = rank
    return ranked + [row for row in results if row["status"] != "docked"]
