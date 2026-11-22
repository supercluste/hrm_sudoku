import torch
from models.layers import (
    HRMBlock,
    RotaryEmbedding
)

device = "cuda" if torch.cuda.is_available() else "cpu"

B = 16
S = 81
D = 256
H = 4

x = torch.randn(B, S, D, device=device)

rope = RotaryEmbedding(
    dim=D // H,
    max_seq_len=S
).to(device)

block = HRMBlock(
    hidden_size=D,
    num_heads=H
).to(device)

cos_sin = rope()

output = block(x, cos_sin)

print("Device:", device)
print("Input:", x.shape)
print("Output:", output.shape)

assert output.shape == x.shape

loss = output.mean()
loss.backward()

print("Backward pass: SUCCESS")
print("Parameters:", sum(
    p.numel() for p in block.parameters()
))
