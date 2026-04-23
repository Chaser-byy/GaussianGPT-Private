"""Causal GPT-2 style transformer for autoregressive 3D Gaussian generation.

Architecture based on GPT-2 medium (scenes) / small (objects) with:
- 3D RoPE instead of learned positional embeddings
- Query-key normalization
- Per-layer residual scaling
- Separate position and feature vocabulary heads
- Muon + AdamW optimizer support

Paper specs:
  - Scene: GPT-2 medium, context=16384 tokens
  - Object: GPT-2 small, context=8192 tokens
  - Temperature=0.9, Nucleus sampling p=0.9
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope3d import RoPE3D


# GPT-2 size configs
GPT2_CONFIGS = {
    "small":  dict(n_layer=12, n_head=12, n_embd=768),
    "medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "large":  dict(n_layer=36, n_head=20, n_embd=1280),
}


class QKNorm(nn.Module):
    """Per-head query-key normalization (RMSNorm)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return (x.float() / norm * self.scale).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with 3D RoPE and QK-norm."""

    def __init__(self, n_embd: int, n_head: int, context_len: int, rope: RoPE3D):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.rope = rope

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.q_norm = QKNorm(self.head_dim)
        self.k_norm = QKNorm(self.head_dim)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_len, context_len, dtype=torch.bool)).view(
                1, 1, context_len, context_len
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        token_type: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 3D RoPE
        q, k = self.rope(q, k, coords, token_type)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """GPT-2 transformer block with per-layer residual scaling (nanochat style).

    Residual formula: x = input_skip * x + res_scale * sublayer(ln(x))
    res_scale initialized to 1.0, input_skip initialized to 0.1 (paper Appendix D).
    """

    def __init__(self, n_embd: int, n_head: int, context_len: int, rope: RoPE3D):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, context_len, rope)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        # Per-layer residual scaling: initialized to 1.0 (paper Appendix D)
        self.res_scale_attn = nn.Parameter(torch.ones(1))
        self.res_scale_mlp = nn.Parameter(torch.ones(1))
        # Input skip weights: initialized to 0.1 (paper Appendix D)
        self.input_skip_attn = nn.Parameter(torch.full((1,), 0.1))
        self.input_skip_mlp = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        token_type: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_skip_attn * x + self.res_scale_attn * self.attn(self.ln1(x), coords, token_type)
        x = self.input_skip_mlp * x + self.res_scale_mlp * self.mlp(self.ln2(x))
        return x


class GaussianGPT(nn.Module):
    """Autoregressive transformer for 3D Gaussian scene generation.

    Vocabulary:
      - Position tokens: voxel indices within a chunk (0..chunk_vol-1) + BOS/EOS
      - Feature tokens: LFQ codebook indices (0..codebook_size-1)
      - Sequence alternates: [BOS, pos_0, feat_0, pos_1, feat_1, ..., EOS]

    Special tokens:
      - BOS: begin-of-sequence (appended to position vocab)
      - EOS: end-of-sequence (appended to position vocab)
    """

    def __init__(
        self,
        size: str = "medium",
        context_len: int = 16384,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        codebook_size: int = 4096,
    ):
        super().__init__()
        cfg = GPT2_CONFIGS[size]
        self.n_layer = cfg["n_layer"]
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.context_len = context_len
        self.codebook_size = codebook_size

        # Chunk vocabulary: number of voxels in a chunk + BOS + EOS
        cx, cy, cz = chunk_size
        self.chunk_vol = cx * cy * cz
        self.chunk_size = chunk_size
        self.pos_vocab_size = self.chunk_vol + 2  # +BOS, +EOS
        self.BOS = self.chunk_vol
        self.EOS = self.chunk_vol + 1

        # Separate embeddings for position and feature tokens
        self.pos_emb = nn.Embedding(self.pos_vocab_size, self.n_embd)
        self.feat_emb = nn.Embedding(codebook_size, self.n_embd)

        # 3D RoPE (shared across all layers)
        self.rope = RoPE3D(head_dim=self.n_embd // self.n_head)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.n_embd, self.n_head, context_len, self.rope)
            for _ in range(self.n_layer)
        ])
        self.ln_f = nn.LayerNorm(self.n_embd)

        # Separate output heads for position and feature prediction
        self.pos_head = nn.Linear(self.n_embd, self.pos_vocab_size, bias=False)
        self.feat_head = nn.Linear(self.n_embd, codebook_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Token embeddings: N(0, 1)
        nn.init.normal_(self.pos_emb.weight, std=1.0)
        nn.init.normal_(self.feat_emb.weight, std=1.0)
        # Output heads: N(0, 1e-3)
        nn.init.normal_(self.pos_head.weight, std=1e-3)
        nn.init.normal_(self.feat_head.weight, std=1e-3)
        # Attention/MLP input projections: uniform ±sqrt(3/d)
        # Attention/MLP output projections: zero (already done in submodule __init__)
        d = self.n_embd
        bound = math.sqrt(3.0 / d)
        for block in self.blocks:
            for name, p in block.attn.named_parameters():
                if "qkv" in name and p.dim() >= 2:
                    nn.init.uniform_(p, -bound, bound)
                # out_proj already zero-initialized in CausalSelfAttention.__init__
            for name, p in block.mlp.named_parameters():
                if "fc1" in name and p.dim() >= 2:
                    nn.init.uniform_(p, -bound, bound)
                # fc2 already zero-initialized in MLP.__init__
            # Residual scale weights: 1.0 (already set by torch.ones)

    def _embed_tokens(
        self,
        tokens: torch.Tensor,
        is_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a mixed sequence of position and feature tokens.

        Args:
            tokens: (B, T) token indices
            is_feature: (B, T) bool mask, True for feature tokens
        Returns:
            embeddings: (B, T, n_embd)
        """
        pos_emb = self.pos_emb(tokens.clamp(0, self.pos_vocab_size - 1))
        feat_emb = self.feat_emb(tokens.clamp(0, self.codebook_size - 1))
        # Select based on token type
        return torch.where(is_feature.unsqueeze(-1), feat_emb, pos_emb)

    def forward(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        token_type: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: (B, T) input token indices
            coords: (B, T, 3) voxel coordinates for each token
            token_type: (B, T) 0=position token, 1=feature token
            targets: (B, T) target tokens for loss computation
        Returns:
            pos_logits: (B, T, pos_vocab_size) position head logits
            feat_logits: (B, T, codebook_size) feature head logits
            loss: cross-entropy loss (if targets provided)
        """
        B, T = tokens.shape
        is_feature = token_type.bool()

        x = self._embed_tokens(tokens, is_feature)

        for block in self.blocks:
            x = block(x, coords, token_type)

        x = self.ln_f(x)
        pos_logits = self.pos_head(x)
        feat_logits = self.feat_head(x)

        loss = None
        if targets is not None:
            loss = self._compute_loss(pos_logits, feat_logits, targets, token_type)

        return pos_logits, feat_logits, loss

    def _compute_loss(
        self,
        pos_logits: torch.Tensor,
        feat_logits: torch.Tensor,
        targets: torch.Tensor,
        token_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with separate vocabularies.

        Position tokens predict from pos_vocab, feature tokens from feat_vocab.
        Invalid entries are masked.
        """
        B, T, _ = pos_logits.shape
        is_feature = token_type.bool().view(-1)

        # Flatten
        pos_logits_flat = pos_logits.view(B * T, -1)
        feat_logits_flat = feat_logits.view(B * T, -1)
        targets_flat = targets.view(-1)

        # Position loss (at position token steps)
        pos_mask = ~is_feature
        pos_loss = F.cross_entropy(
            pos_logits_flat[pos_mask], targets_flat[pos_mask], ignore_index=-1
        ) if pos_mask.any() else torch.tensor(0.0, device=pos_logits.device)

        # Feature loss (at feature token steps)
        feat_loss = F.cross_entropy(
            feat_logits_flat[is_feature], targets_flat[is_feature], ignore_index=-1
        ) if is_feature.any() else torch.tensor(0.0, device=feat_logits.device)

        return pos_loss + feat_loss

    # Public alias used by training scripts
    compute_loss = _compute_loss

    @torch.no_grad()
    def generate(
        self,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_coords: Optional[torch.Tensor] = None,
        prefix_token_type: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive generation with nucleus sampling.

        Args:
            prefix_*: optional conditioning prefix (for scene completion)
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
        Returns:
            tokens, coords, token_type for the generated sequence
        """
        if prefix_tokens is None:
            # Start with BOS
            tokens = torch.tensor([[self.BOS]], device=device)
            coords = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
            token_type = torch.zeros(1, 1, dtype=torch.long, device=device)
        else:
            tokens = prefix_tokens.to(device)
            coords = prefix_coords.to(device)
            token_type = prefix_token_type.to(device)

        # Track occupied voxel indices to enforce ordering constraints (paper Sec 3.2)
        occupied_indices: set = set()
        # Sequence alternates: BOS -> pos_0 -> feat_0 -> pos_1 -> feat_1 -> ...
        # After BOS or a feature token: next is position. After a position token: next is feature.
        last_tok = tokens[0, -1].item()
        last_type = token_type[0, -1].item()  # 0=position, 1=feature
        if last_tok == self.BOS:
            next_is_feature = False  # BOS -> predict first position token
        else:
            next_is_feature = (last_type == 0)  # after pos -> feat, after feat -> pos
        for _ in range(max_new_tokens):
            # Truncate to context window
            t = tokens[:, -self.context_len:]
            c = coords[:, -self.context_len:]
            tt = token_type[:, -self.context_len:]

            pos_logits, feat_logits, _ = self.forward(t, c, tt)

            # Get logits for the last position
            if next_is_feature:
                logits = feat_logits[:, -1, :]  # (1, codebook_size)
            else:
                logits = pos_logits[:, -1, :]   # (1, pos_vocab_size)
                # Mask already-generated voxel positions to enforce ordering
                if occupied_indices:
                    mask = torch.zeros(self.pos_vocab_size, dtype=torch.bool, device=device)
                    for idx in occupied_indices:
                        mask[idx] = True
                    logits[0][mask] = float("-inf")

            # Temperature scaling + nucleus sampling
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = _nucleus_sample(probs, top_p)  # (1, 1)

            # Check for EOS
            if not next_is_feature and next_token.item() == self.EOS:
                break

            # Determine coordinates for new token
            if not next_is_feature:
                # Position token: decode voxel index to 3D coords
                current_pos_idx = next_token.item()
                occupied_indices.add(current_pos_idx)
                new_coord = self._idx_to_coord(current_pos_idx)
                new_coord_t = torch.tensor([[new_coord]], dtype=torch.long, device=device)  # (1, 1, 3)
                new_type = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                # Feature token: same coord as preceding position token
                new_coord_t = coords[:, -1:, :]  # (1, 1, 3) — reuse last coord
                new_type = torch.ones(1, 1, dtype=torch.long, device=device)

            tokens = torch.cat([tokens, next_token], dim=1)
            coords = torch.cat([coords, new_coord_t], dim=1)
            token_type = torch.cat([token_type, new_type], dim=1)

            next_is_feature = not next_is_feature

        return tokens, coords, token_type

    def _idx_to_coord(self, idx: int) -> list:
        """Convert flat voxel index to (x, y, z) coordinates (xyz ordering)."""
        cx, cy, cz = self.chunk_size
        z = idx % cz
        y = (idx // cz) % cy
        x = idx // (cy * cz)
        return [x, y, z]


def _nucleus_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Remove tokens with cumulative probability above threshold
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    # Guard against all-zero (e.g. extreme temperature); fall back to greedy
    total = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / total.clamp(min=1e-8)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, sampled)
