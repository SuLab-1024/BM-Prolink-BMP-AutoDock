"""BM-Prolink inference model, including its input encoders and prediction head."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import EsmModel, EsmTokenizer, RobertaModel, RobertaTokenizer

from .config import (
    CHEMBERTA_MODEL,
    CHEMBERTA_OUTPUT_DIM,
    D_MODEL,
    DIFF_ATTN_HEADS,
    DROPOUT,
    ESM2_MODEL,
    ESM2_OUTPUT_DIM,
    MRCL_LAYERS,
)
from .diff_attention import BidirectionalDiffCrossAttention
from .drug_gnn import DrugGNN
from .multiscale_conv_refiner import MultiScaleConvRefiner


class ProteinEncoder(nn.Module):
    """Encode protein sequences with frozen ESM-2."""

    def __init__(self):
        super().__init__()
        self.esm = EsmModel.from_pretrained(
            ESM2_MODEL, add_pooling_layer=False, use_safetensors=True, attn_implementation="eager"
        )
        self.esm.requires_grad_(False)
        self.tokenizer = EsmTokenizer.from_pretrained(ESM2_MODEL)
        self.proj = nn.Sequential(nn.LayerNorm(ESM2_OUTPUT_DIM), nn.Linear(ESM2_OUTPUT_DIM, D_MODEL))

    def forward(self, input_ids, attention_mask):
        hidden = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(hidden), attention_mask == 0


class DrugEncoder(nn.Module):
    """Combine frozen ChemBERTa token features with a GIN graph representation."""

    def __init__(self):
        super().__init__()
        self.chem = RobertaModel.from_pretrained(
            CHEMBERTA_MODEL, add_pooling_layer=False, use_safetensors=True, attn_implementation="eager"
        )
        self.chem.requires_grad_(False)
        self.tokenizer = RobertaTokenizer.from_pretrained(CHEMBERTA_MODEL)
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(CHEMBERTA_OUTPUT_DIM), nn.Linear(CHEMBERTA_OUTPUT_DIM, D_MODEL)
        )
        self.gnn = DrugGNN(atom_dim=78, hidden_dim=128, out_dim=D_MODEL, num_layers=3)
        self.gnn_project = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, input_ids, attention_mask, atom_features, adjacency, node_mask):
        with torch.no_grad():
            hidden = self.chem(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        graph = self.gnn_project(self.gnn(atom_features, adjacency, node_mask))
        return self.seq_proj(hidden) + graph.unsqueeze(1), attention_mask == 0


class GatedMLPHead(nn.Module):
    """Apply sigmoid gating and map the joint representation to a single logit."""

    HIDDEN = 256

    def __init__(self, in_dim: int = 3 * D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        h = self.HIDDEN
        self.gate_proj = nn.Linear(in_dim, h * 2, bias=False)
        self.norm = nn.LayerNorm(h)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h, h // 2)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(h // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_proj(x).chunk(2, dim=-1)
        h = self.drop1(self.norm(torch.sigmoid(gate) * value))
        h = self.drop2(F.gelu(self.fc2(h)))
        return self.out(h)


class CPIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugEncoder()
        self.prot_encoder = ProteinEncoder()
        self.fusion = BidirectionalDiffCrossAttention(D_MODEL, DIFF_ATTN_HEADS, DROPOUT)
        self.drug_refiner = MultiScaleConvRefiner(D_MODEL, num_layers=MRCL_LAYERS, dropout=DROPOUT)
        self.prot_refiner = MultiScaleConvRefiner(D_MODEL, num_layers=MRCL_LAYERS, dropout=DROPOUT)
        self.head = GatedMLPHead(in_dim=3 * D_MODEL)

    def forward(self, drug_input_ids, drug_attn_mask, prot_input_ids, prot_attn_mask,
                drug_atom_feats, drug_adj, drug_node_mask):
        drug, drug_padding = self.drug_encoder(
            drug_input_ids, drug_attn_mask, drug_atom_feats, drug_adj, drug_node_mask
        )
        protein, protein_padding = self.prot_encoder(prot_input_ids, prot_attn_mask)
        drug, protein, _, _ = self.fusion(drug, protein, drug_mask=drug_padding, prot_mask=protein_padding)
        drug_vector = self._masked_mean(self.drug_refiner(drug), drug_attn_mask)
        protein_vector = self._masked_mean(self.prot_refiner(protein), prot_attn_mask)
        joint = torch.cat([drug_vector, protein_vector, drug_vector * protein_vector], dim=-1)
        return torch.sigmoid(self.head(joint)).squeeze(-1)

    @staticmethod
    def _masked_mean(hidden, mask):
        valid = mask.float().unsqueeze(-1)
        return (hidden * valid).sum(1) / valid.sum(1).clamp(min=1e-9)
