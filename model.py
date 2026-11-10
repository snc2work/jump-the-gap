"""Causal action-conditioned Transformer for binary video."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 32
PIXELS = IMAGE_SIZE * IMAGE_SIZE
CONTEXT = 32
MODES = 4


class WorldModel(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        width = 96
        self.image = nn.Linear(PIXELS, width)
        self.action = nn.Embedding(4, width)
        self.time = nn.Parameter(torch.randn(CONTEXT, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 3,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.frame_head = nn.Linear(width, MODES * PIXELS)
        self.mode_head = nn.Linear(width, MODES)
        self.register_buffer("causal_mask", torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1))

    def forward(self, frames: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Pair each observed frame with the action that follows it."""
        frames = frames.flatten(start_dim=2).float()
        length = frames.size(1)
        tokens = self.image(frames) + self.action(actions) + self.time[:length]
        hidden = self.transformer(tokens, mask=self.causal_mask[:length, :length], is_causal=True)
        # Start from a strong persistence prior and learn where it should change.
        residual = self.frame_head(hidden).unflatten(-1, (MODES, PIXELS))
        prior = (2 * frames - 1).unsqueeze(-2) * 5
        return self.mode_head(hidden), prior + residual

    @torch.no_grad()
    def rollout(self, prefix: Tensor, actions: Tensor, *, sample: bool = True) -> Tensor:
        """Continue a batch of observed prefixes from their aligned action plans."""
        frames = prefix
        for step in range(prefix.size(1) - 1, actions.size(1)):
            start = max(0, step + 1 - CONTEXT)
            weights, logits = self(frames[:, start:], actions[:, start:step + 1])
            weights, logits = weights[:, -1], logits[:, -1]
            if sample:
                mode = torch.multinomial(weights.softmax(-1), 1).squeeze(-1)
            else:
                mode = weights.argmax(-1)
            # Display a coherent component image using its per-pixel modes.
            next_frame = logits[torch.arange(len(mode), device=mode.device), mode] > 0
            frames = torch.cat((frames, next_frame.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)), dim=1)
        return frames


def loss_for_frames(weights: Tensor, logits: Tensor, targets: Tensor) -> Tensor:
    targets = targets.flatten(start_dim=2).float().unsqueeze(-2)
    log_likelihood = (
        targets * F.logsigmoid(logits) + (1 - targets) * F.logsigmoid(-logits)
    ).sum(dim=-1)
    return -(weights.log_softmax(-1) + log_likelihood).logsumexp(-1).mean()
