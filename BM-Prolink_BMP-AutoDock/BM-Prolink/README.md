# BM-Prolink

BM-Prolink scores drug–target pairs from SMILES, molecular graphs and protein sequences.

## Model workflow

BM-Prolink combines ChemBERTa token features with a GIN molecular graph
representation and encodes the protein sequence with ESM-2. It applies
bidirectional differential cross-attention, then uses three dilated convolution
branches with learned weights to refine each sequence. Masked mean pooling
produces the drug and protein vectors used by the gated prediction head.

## Requirements and setup

This is the `bm-prolink` 0.1.0 source release. It requires Python 3.10 or later.
Install the third-party dependencies in your Python environment.

```sh
python -m pip install "torch>=2.1" "transformers>=4.40,<5" "numpy>=1.24" "rdkit>=2023.9"
```

Open a terminal in the `BM-Prolink` folder. Run the source directly with
`python -m bmprolink.cli`.

## Input and usage

Use `--weights` to specify the path to a compatible BM-Prolink state dictionary.
The ESM-2 and ChemBERTa encoders and tokenizers are resolved through the
Hugging Face cache. With an empty cache, the first run downloads the required
artifacts. For an offline run, populate the cache first, or use an existing cache
via `HF_HOME` and set `HF_HUB_OFFLINE=1`.

Create a UTF-8 CSV with one drug–target pair per row. Other columns, such as a
compound identifier, are carried through to the output.

```csv
compound_id,smiles,sequence
example_1,CCO,MKTLLILAV
example_2,CC(=O)O,MKTLLILAV
```

```sh
python -m bmprolink.cli --input pairs.csv --weights model.pt --output predictions.csv
```

The output contains one `dti_score` per input row, in input order. An invalid SMILES
or an empty sequence interrupts the run without creating a result file. Each
SMILES field should contain only the SMILES string. You can set `--device cpu`
or `--device cuda:0`, and change `--batch-size` from its default of 8. Keep the
input order and batch size fixed when comparing repeated runs because padded
sequence positions enter the model's attention output normalization.

Python API

```python
from bmprolink.predict import Predictor

predictor = Predictor("model.pt", device="cpu")
scores = predictor.predict([("CCO", "MKTLLILAV")])
```
