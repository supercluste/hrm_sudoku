
import torch
import torch.nn as nn
import math

from models.layers import HRMBlock, RotaryEmbedding


class ReasoningModule(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            HRMBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, input_injection, cos_sin):
        # Inject information from the other module
        hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)

        return hidden_states


class HRMCore(nn.Module):
    def __init__(
        self,
        vocab_size=11,
        hidden_size=256,
        num_heads=4,
        H_layers=2,
        L_layers=2,
        H_cycles=2,
        L_cycles=2,
        seq_len=81,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Token embedding
        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size
        )

        self.embed_scale = math.sqrt(hidden_size)

        # Positional encoding
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_seq_len=seq_len
        )

        # High-level reasoning module
        self.H_level = ReasoningModule(
            hidden_size,
            num_heads,
            H_layers
        )

        # Low-level reasoning module
        self.L_level = ReasoningModule(
            hidden_size,
            num_heads,
            L_layers
        )

        # Initial latent states
        self.H_init = nn.Parameter(
            torch.randn(hidden_size) * 0.02
        )

        self.L_init = nn.Parameter(
            torch.randn(hidden_size) * 0.02
        )

        # Output projection
        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

    def forward(self, inputs):

        B, S = inputs.shape

        # Embed Sudoku tokens
        input_embeddings = (
            self.embed_tokens(inputs) * self.embed_scale
        )

        cos_sin = self.rotary_emb()

        # Initialize recurrent states
        z_H = self.H_init.view(1, 1, -1).expand(B, S, -1)
        z_L = self.L_init.view(1, 1, -1).expand(B, S, -1)

        # Recurrent computation without gradients
        with torch.no_grad():

            for h in range(self.H_cycles):

                for l in range(self.L_cycles):

                    # Skip the final L update
                    if not (
                        h == self.H_cycles - 1
                        and l == self.L_cycles - 1
                    ):
                        z_L = self.L_level(
                            z_L,
                            z_H + input_embeddings,
                            cos_sin
                        )

                # Skip the final H update
                if h != self.H_cycles - 1:
                    z_H = self.H_level(
                        z_H,
                        z_L,
                        cos_sin
                    )

        # Final recurrent step with gradients
        z_L = self.L_level(
            z_L,
            z_H + input_embeddings,
            cos_sin
        )

        z_H = self.H_level(
            z_H,
            z_L,
            cos_sin
        )

        # Predict token for every Sudoku cell
        logits = self.lm_head(z_H)

        return logits
