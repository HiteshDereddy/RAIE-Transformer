import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    """
    Novel Recency-Augmented Input Embeddings (RAIE) Transformer:
    - Augments (x, y) inputs with a normalized recency feature (t/(T-1)) to emphasize recent frames.
    - Uses standard transformer encoder-decoder with learned PE, matching baseline capacity.
    - Tailored for cricket ball trajectories, prioritizing recent kinematics for better rollout.
    """
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=512, max_len=500):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)  # Input: (x, y, recency)
        self.pos_encoder = nn.Embedding(max_len, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                       batch_first=True, norm_first=True),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                       batch_first=True, norm_first=True),
            num_decoder_layers
        )
        self.pred_head = nn.Linear(d_model, 2)

        # Initialize weights narrowly for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, tgt):
        # Add recency feature: t/(T-1), normalized to [0, 1]
        src_T = src.size(1)
        src_pos = torch.arange(src_T, device=src.device, dtype=torch.float) / (src_T - 1 + 1e-6)
        src_pos = src_pos.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        src_aug = torch.cat([src, src_pos], dim=-1)  # (B, T, 3)

        tgt_T = tgt.size(1)
        tgt_pos = torch.arange(tgt_T, device=tgt.device, dtype=torch.float) / (tgt_T - 1 + 1e-6)
        tgt_pos = tgt_pos.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        tgt_aug = torch.cat([tgt, tgt_pos], dim=-1)  # (B, T, 3)

        # Positional encoding
        src_pe = torch.arange(src_T, device=src.device).unsqueeze(0)
        tgt_pe = torch.arange(tgt_T, device=tgt.device).unsqueeze(0)

        src_emb = self.input_proj(src_aug) + self.pos_encoder(src_pe)
        memory = self.encoder(src_emb)

        tgt_emb = self.input_proj(tgt_aug) + self.pos_encoder(tgt_pe)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt.device)

        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.pred_head(out)