"""Group-relative policy optimisation (GRPO), one update per batch of samples."""

from __future__ import annotations

import math

import torch


def advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages: the reward less the group mean, over the group's standard
    deviation. A group with one reward throughout has none, and gives no gradient."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    if var < 1e-12:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(r - mean) / sd for r in rewards]


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B, G, V) predicting targets (B, G): the log-probability of each target, one row
    at a time so the float32 softmax never holds the whole batch."""
    out = []
    for b in range(logits.shape[0]):
        row = logits[b].float()
        out.append(row.gather(-1, targets[b, :, None]).squeeze(-1) - torch.logsumexp(row, -1))
    return torch.stack(out)


def pg_loss(
    logp: torch.Tensor,
    mask: torch.Tensor,
    adv: torch.Tensor,
    ref_logp: torch.Tensor | None = None,
    kl: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Minus the advantage times the token-mean log-probability of each sample (the importance
    ratio is one and the clip inactive), plus `kl` times the k3 estimate of the KL divergence
    to the reference policy on the same tokens."""
    tok = mask.sum(1).clamp(min=1)
    seq = (logp * mask).sum(1) / tok
    loss = -(adv * seq).mean()
    stats = {"pg": float(loss.detach())}
    if ref_logp is not None and kl > 0:
        d = (ref_logp.detach() - logp) * mask
        k3 = (torch.exp(d) - d - 1) * mask
        klv = (k3.sum(1) / tok).mean()
        loss = loss + kl * klv
        stats["kl"] = float(klv.detach())
    return loss, stats
