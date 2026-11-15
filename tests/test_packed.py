"""The packed group (`thornecliffe.packed`) gives the plain pass's numbers. The model is a
small random Gemma 3 with both layer types and a window shorter than the prompts, so the
sliding mask is exercised."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from thornecliffe.grpo import pg_loss, token_logprobs  # noqa: E402
from thornecliffe.packed import pack, packed_logprobs, sample_packed, tree_masks  # noqa: E402

VOCAB, PAD, EOS, WINDOW = 40, 0, 1, 6


@pytest.fixture(scope="module")
def model():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention"] * 2, max_position_embeddings=128,
        pad_token_id=PAD, eos_token_id=EOS, bos_token_id=2)  # fmt: skip
    m = Gemma3ForCausalLM(cfg).double().eval()
    assert m.config.layer_types.count("sliding_attention") == 2
    return m


def plain_logprobs(model, samples, temperature=1.0):
    """The trainer's plain pass: one left-padded row a (prompt, reply)."""
    G = max(len(g) for _p, g in samples)
    T = max(len(p) + len(g) for p, g in samples)
    ids = torch.full((len(samples), T), PAD, dtype=torch.long)
    att = torch.zeros((len(samples), T), dtype=torch.long)
    for i, (p, g) in enumerate(samples):
        seq = p + g
        ids[i, T - len(seq) :] = torch.tensor(seq)
        att[i, T - len(seq) :] = 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos,
                   logits_to_keep=G + 1).logits[:, :-1]  # fmt: skip
    lp = token_logprobs(logits / temperature, ids[:, -G:])
    return [lp[i, G - len(g) :] for i, (_p, g) in enumerate(samples)]


def tickets(rng, n, replies):
    out = []
    for _ in range(n):
        p = [rng.randrange(3, VOCAB) for _ in range(rng.randrange(8, 15))]
        out.append((p, [[rng.randrange(3, VOCAB) for _ in range(rng.randrange(1, 9))]
                        for _ in range(rng.randrange(1, replies + 1))]))  # fmt: skip
    return out


def test_the_tree_mask_lets_a_reply_see_the_prompt_and_itself_alone():
    # one row: a pad, a prompt of three, reply 1 of two tokens, reply 2 of one
    path = torch.tensor([[-1, 0, 0, 0, 1, 1, 2]])
    pos = torch.tensor([[0, 0, 1, 2, 3, 4, 3]])
    m = tree_masks(path, pos, 3, torch.float32)
    full, slide = (m[k][0, 0] == 0 for k in ("full_attention", "sliding_attention"))
    assert full.tolist() == [
        [1, 1, 0, 0, 0, 0, 0],   # the pad: itself (and the first token, which nothing reads)
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],   # the prompt never sees a reply
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],   # reply 2 never sees reply 1
    ]
    # the window is by position: a key three or more positions back is cut
    assert slide[5].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert slide[6].tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert m["full_attention"].min() == torch.finfo(torch.float32).min
    # a decode step: the new tokens as queries against every key so far
    step = tree_masks(path, pos, 3, torch.float32, q_path=torch.tensor([[1, 2]]),
                      q_pos=torch.tensor([[4, 3]]))  # fmt: skip
    assert (step["full_attention"][0, 0] == 0).tolist() == [full[5].tolist(), full[6].tolist()]


def test_pack_points_each_token_at_the_output_that_predicts_it():
    b = pack([([5, 6, 7], [[8, 9], [10]]), ([5, 6], [[11, 12, 13]])], PAD)
    assert b["ids"].tolist() == [[5, 6, 7, 8, 9, 10], [PAD, 5, 6, 11, 12, 13]]
    assert b["path"].tolist() == [[0, 0, 0, 1, 1, 2], [-1, 0, 0, 1, 1, 1]]
    assert b["pos"].tolist() == [[0, 1, 2, 3, 4, 3], [0, 0, 1, 2, 3, 4]]
    assert int(b["keep"]) == 4 and b["row"].tolist() == [0, 0, 1]
    # kept positions are the last four of six: the prompt's last token is kept index 0
    assert b["src"].tolist() == [[0, 1, 0], [0, 0, 0], [0, 1, 2]]
    assert b["tgt"].tolist() == [[8, 9, 0], [10, 0, 0], [11, 12, 13]]
    assert b["mask"].tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
    with pytest.raises(AssertionError):
        pack([([5], [[]])], PAD)


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_the_packed_pass_gives_the_plain_log_probabilities(model, temperature):
    rng = random.Random(3)
    items = tickets(rng, 4, 5)
    assert max(len(p) for p, _ in items) > WINDOW  # the window cuts inside the prompt
    b = pack(items, PAD)
    masks = tree_masks(b["path"], b["pos"], WINDOW, torch.float64)
    with torch.no_grad():
        logits = model(input_ids=b["ids"], attention_mask=masks, position_ids=b["pos"],
                       logits_to_keep=int(b["keep"])).logits  # fmt: skip
        got = packed_logprobs(logits, b["row"], b["src"], b["tgt"], temperature)
        want = plain_logprobs(model, [(p, g) for p, gs in items for g in gs], temperature)
    assert got.shape[0] == len(want) == sum(len(gs) for _p, gs in items)
    for s, w in enumerate(want):
        assert torch.allclose(got[s, : len(w)].float(), w, atol=1e-5), s
        assert b["mask"][s].sum() == len(w)
    # and the loss the trainer takes from them is the same
    G = b["mask"].shape[1]
    plain = torch.zeros_like(got)
    for s, w in enumerate(want):
        plain[s, : len(w)] = w
    adv = torch.linspace(-1, 1, got.shape[0], dtype=got.dtype)
    a, _ = pg_loss(got, b["mask"].to(got.dtype), adv)
    c, _ = pg_loss(plain, b["mask"].to(got.dtype), adv)
    assert abs(float(a) - float(c)) < 1e-5 and G >= 1


def test_the_packed_pass_would_differ_if_a_reply_saw_another(model):
    # the test above has teeth: with every reply on one path the numbers move
    rng = random.Random(4)
    items = tickets(rng, 2, 4)
    items = [(p, gs if len(gs) > 1 else [*gs, [3, 4, 5]]) for p, gs in items]
    b = pack(items, PAD)
    merged = torch.where(b["path"] > 0, torch.ones_like(b["path"]), b["path"])
    with torch.no_grad():
        outs = []
        for path in (b["path"], merged):
            masks = tree_masks(path, b["pos"], WINDOW, torch.float64)
            logits = model(input_ids=b["ids"], attention_mask=masks, position_ids=b["pos"],
                           logits_to_keep=int(b["keep"])).logits  # fmt: skip
            outs.append(packed_logprobs(logits, b["row"], b["src"], b["tgt"]))
    assert (outs[0] - outs[1]).abs().max() > 1e-3


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_the_packed_rollout_draws_from_the_plain_distribution(model, temperature):
    rng = random.Random(5)
    prompts = [[rng.randrange(3, VOCAB) for _ in range(k)] for k in (9, 14, 11)]
    torch.manual_seed(1)
    groups = sample_packed(model, prompts, 4, temperature=temperature, max_new=5,
                           eos_ids={EOS}, pad=PAD, window=WINDOW, dtype=torch.float64,
                           device="cpu")  # fmt: skip
    assert len(groups) == 3 and all(len(g) == 4 for g in groups)
    samples = [(p, g) for p, group in zip(prompts, groups, strict=True) for g, _lp in group]
    for _p, g in samples:
        assert 1 <= len(g) <= 5 and EOS not in g[:-1]
    assert len({tuple(g) for _p, g in samples}) > 4  # the replies of a group differ
    with torch.no_grad():
        want = plain_logprobs(model, samples, temperature)
    drawn = [lp for group in groups for _g, lp in group]
    for w, d in zip(want, drawn, strict=True):
        assert torch.allclose(w.float(), torch.tensor(d), atol=1e-5)


def test_the_packed_rollout_stops_a_reply_at_its_end_token(model):
    # with the end token made likely, replies end early and at different steps
    prompts = [[5, 6, 7, 8, 9, 10, 11, 12], [9, 8, 7, 6, 5, 4, 3, 13, 14]]
    bias = model.lm_head.weight.data.clone()
    try:
        model.lm_head.weight.data[EOS] += 0.6 * model.lm_head.weight.data.abs().mean() * 20
        torch.manual_seed(2)
        groups = sample_packed(model, prompts, 6, temperature=1.0, max_new=5, eos_ids={EOS},
                               pad=PAD, window=WINDOW, dtype=torch.float64, device="cpu")
        samples = [(p, g) for p, group in zip(prompts, groups, strict=True) for g, _ in group]
        with torch.no_grad():
            want = plain_logprobs(model, samples, 1.0)
    finally:
        model.lm_head.weight.data.copy_(bias)
    lengths = {len(g) for _p, g in samples}
    assert all(EOS not in g[:-1] for _p, g in samples)
    drawn = [lp for group in groups for _g, lp in group]
    for w, d in zip(want, drawn, strict=True):
        assert torch.allclose(w.float(), torch.tensor(d), atol=1e-5)
    assert len(lengths) > 1 and min(lengths) < 5
