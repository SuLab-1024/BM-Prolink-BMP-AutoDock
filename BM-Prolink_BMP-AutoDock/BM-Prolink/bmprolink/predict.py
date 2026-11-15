"""Load a full-model checkpoint and score compound–protein pairs."""
from pathlib import Path

import numpy as np
import torch

from .config import MAX_DRUG_LEN, MAX_PROT_LEN
from .graph import build_drug_graph


class Predictor:
    def __init__(self, weights: str | Path, device: str | None = None):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"BM-Prolink checkpoint not found: {weights}")
        from .model import CPIModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = CPIModel()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        """Return DTI scores in input order. Each pair is (SMILES, protein sequence)."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            smiles, sequences = zip(*batch)
            sequences = ["".join(sequence.split()).upper() for sequence in sequences]
            if not all(sequences):
                raise ValueError("Protein sequences must be nonempty.")
            graphs = [build_drug_graph(smile.strip()) for smile in smiles]
            features, adjacency, nodes = (
                torch.from_numpy(np.stack(values)).to(self.device) for values in zip(*graphs)
            )
            drug = self.model.drug_encoder.tokenizer(
                [smile.strip() for smile in smiles], return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_DRUG_LEN,
            )
            protein = self.model.prot_encoder.tokenizer(
                sequences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROT_LEN,
            )
            scores.extend(self.model(
                drug["input_ids"].to(self.device), drug["attention_mask"].to(self.device),
                protein["input_ids"].to(self.device), protein["attention_mask"].to(self.device),
                features, adjacency, nodes,
            ).cpu().tolist())
        return scores
