"""A ticket's group of sampled replies as one packed sequence, so the prompt is encoded once a
ticket and not once a reply. This is what makes a training step cheap.

A packed sequence is the prompt and then the group's replies one after another. Every reply
attends to the prompt and to itself and never to another reply, and every token keeps the
position it would have had in a sequence of its own (the prompt's tokens 0 to P-1, each reply's
tokens from P). The prompt's hidden states do not depend on what follows it, so each reply's
outputs are the ones a separate pass over prompt and reply would give. Gemma 3 takes an attention
mask per layer type; the sliding-window layers get the same mask cut at the window, by position.

`pack` and `packed_logprobs` are the training pass; `sample_packed` is the rollout, a decode loop
that feeds every reply's next token at once against one cache a ticket. `tests/test_packed.py`
checks both against the plain pass on a small random Gemma 3 whose window is shorter than the
prompt.
"""

from __future__ import annotations

import torch


def tree_masks(
    path: torch.Tensor,
    pos: torch.Tensor,
    window: int,
    dtype: torch.dtype,
    q_path: torch.Tensor | None = None,
    q_pos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Additive attention masks (B, 1, Q, K) for packed sequences, one per layer type.

    `path` (B, K) is -1 on padding, 0 on the prompt and k on the k-th reply; `pos` (B, K) is
    the position each token would have had alone. A query sees a key when the key belongs to
    the prompt or to the query's own reply and does not come after it; the sliding layers also
    need the key within the window. The queries are the keys themselves (a whole pass) unless
    a decode step gives the new tokens' own path and positions.
    """
    whole = q_path is None
    qp = (path if whole else q_path)[:, :, None]
    qs = (pos if q_pos is None else q_pos)[:, :, None]
    kp, ks = path[:, None, :], pos[:, None, :]
    ok = (kp >= 0) & ((kp == 0) | (kp == qp)) & (ks <= qs)
    if whole:  # a padded query sees itself, so no row of the mask is empty
        ok = ok | torch.eye(path.shape[1], dtype=torch.bool, device=path.device)
    low = torch.finfo(dtype).min

    def additive(m: torch.Tensor) -> torch.Tensor:
        return torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, low)[:, None]

    return {"full_attention": additive(ok),
            "sliding_attention": additive(ok & ((qs - ks) < window))}  # fmt: skip


def pack(items: list[tuple[list[int], list[list[int]]]], pad: int) -> dict[str, torch.Tensor]:
    """Packed sequences for a training pass: one (prompt, replies) a ticket, every reply with
    at least one token. Rows are padded on the left, so the replies end together and the model
    need only return its last `keep` positions. For each reply, in the order given: `row` (S),
    and per token `src` (S, G), the kept position whose output predicts it (the prompt's last
    token for a reply's first, the token before it otherwise), `tgt` and `mask`."""
    sizes = [(len(p), sum(len(g) for g in gs)) for p, gs in items]
    assert all(g for _p, gs in items for g in gs), "an empty reply"
    width = max(p + r for p, r in sizes)
    keep = max(r for _p, r in sizes) + 1
    ids = torch.full((len(items), width), pad, dtype=torch.long)
    path = torch.full((len(items), width), -1, dtype=torch.long)
    pos = torch.zeros((len(items), width), dtype=torch.long)
    samples = []
    for i, (p, gs) in enumerate(items):
        n = len(p)
        off = width - sizes[i][0] - sizes[i][1]
        ids[i, off : off + n] = torch.tensor(p)
        path[i, off : off + n] = 0
        pos[i, off : off + n] = torch.arange(n)
        x = off + n
        for k, g in enumerate(gs, 1):
            ids[i, x : x + len(g)] = torch.tensor(g)
            path[i, x : x + len(g)] = k
            pos[i, x : x + len(g)] = n + torch.arange(len(g))
            src = [off + n - 1, *range(x, x + len(g) - 1)]
            samples.append((i, [s - (width - keep) for s in src], g))
            x += len(g)
    longest = max(len(g) for _i, _s, g in samples)
    row = torch.tensor([i for i, _s, _g in samples])
    src = torch.zeros((len(samples), longest), dtype=torch.long)
    tgt = torch.zeros((len(samples), longest), dtype=torch.long)
    mask = torch.zeros((len(samples), longest))
    for s, (_i, sr, g) in enumerate(samples):
        src[s, : len(g)] = torch.tensor(sr)
        tgt[s, : len(g)] = torch.tensor(g)
        mask[s, : len(g)] = 1.0
    return {"ids": ids, "path": path, "pos": pos, "keep": torch.tensor(keep), "row": row,
            "src": src, "tgt": tgt, "mask": mask}  # fmt: skip


def packed_logprobs(logits: torch.Tensor, row: torch.Tensor, src: torch.Tensor,
                    tgt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:  # fmt: skip
    """logits (B, keep, V) of a packed pass: the log-probability of each reply's tokens (S, G)
    at the sampling temperature, one reply at a time so the float32 softmax stays small."""
    out = []
    for s in range(row.shape[0]):
        x = logits[row[s], src[s]].float() / temperature
        out.append(x.gather(-1, tgt[s][:, None]).squeeze(-1) - torch.logsumexp(x, -1))
    return torch.stack(out)


@torch.no_grad()
def sample_packed(model, prompts: list[list[int]], n: int, *, temperature: float, max_new: int,
                  eos_ids: set[int], pad: int, window: int, dtype: torch.dtype,
                  device) -> list[list[tuple[list[int], list[float]]]]:  # fmt: skip
    """`n` sampled replies a prompt from the full distribution at `temperature`: the prompts
    encoded once, then every reply's next token fed together against the one cache. For each
    prompt, each reply's token ids (cut after the first end token) and the log-probability
    each was drawn with."""
    from transformers import DynamicCache

    assert max_new < window, "a reply longer than the window needs its own keys cut"
    count, longest = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((count, longest), pad, dtype=torch.long)
    path = torch.full((count, longest), -1, dtype=torch.long)
    pos = torch.zeros((count, longest), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, longest - len(p) :] = torch.tensor(p)
        path[i, longest - len(p) :] = 0
        pos[i, longest - len(p) :] = torch.arange(len(p))
    ids, path, pos = ids.to(device), path.to(device), pos.to(device)
    start = torch.tensor([len(p) for p in prompts], device=device)[:, None].expand(count, n)
    kid = torch.arange(1, n + 1, device=device)[None, :].expand(count, n)
    stop = torch.tensor(sorted(eos_ids), device=device)
    cache = DynamicCache()  # no config: no layer cuts its keys, the masks carry the window
    masks = tree_masks(path, pos, window, dtype)
    out = model(input_ids=ids, attention_mask=masks, position_ids=pos, past_key_values=cache,
                use_cache=True, logits_to_keep=1)  # fmt: skip
    logp = torch.log_softmax(out.logits[:, -1].float() / temperature, -1)
    tok = torch.multinomial(logp.exp(), n, replacement=True)
    toks, lps = [tok], [logp.gather(-1, tok)]
    done = torch.isin(tok, stop)
    for step in range(1, max_new):
        if bool(done.all()):
            break
        q_pos = start + (step - 1)
        path, pos = torch.cat([path, kid], 1), torch.cat([pos, q_pos], 1)
        masks = tree_masks(path, pos, window, dtype, q_path=kid, q_pos=q_pos)
        out = model(input_ids=tok, attention_mask=masks, position_ids=q_pos,
                    past_key_values=cache, use_cache=True)  # fmt: skip
        logp = torch.log_softmax(out.logits.float() / temperature, -1)
        nxt = torch.multinomial(logp.exp().view(count * n, -1), 1).view(count, n)
        lps.append(logp.gather(-1, nxt[..., None]).squeeze(-1))
        tok = torch.where(done, torch.full_like(nxt, pad), nxt)
        toks.append(tok)
        done = done | torch.isin(tok, stop)
    all_t, all_l = torch.stack(toks, -1).tolist(), torch.stack(lps, -1).tolist()
    res = []
    for i in range(count):
        group = []
        for k in range(n):
            g = all_t[i][k]
            cut = next((j + 1 for j, t in enumerate(g) if t in eos_ids), len(g))
            group.append((g[:cut], all_l[i][k][:cut]))
        res.append(group)
    return res


__all__ = ["pack", "packed_logprobs", "sample_packed", "tree_masks"]
