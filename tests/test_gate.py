import json

import pytest

from thornecliffe.gate import Stored, blocks, digest, same
from thornecliffe.policy import Decision, Ticket

DEFECT = Ticket(120, True, False, 10, "fault", cause="defect")


def test_the_gate_passes_only_the_untrained_model_s_whole_decision():
    own = Decision(2, 120.0, "firm")
    assert not blocks(DEFECT, Decision(2, 120.0, "firm"), own)
    assert blocks(DEFECT, Decision(2, 120.0, "courier"), own)
    assert blocks(DEFECT, Decision(3, 120.0, "firm"), own)
    assert blocks(DEFECT, Decision(2, 60.0, "firm"), own)
    assert blocks(DEFECT, Decision(None, None, None), own)


def test_where_the_untrained_model_is_wrong_the_gate_passes_its_error():
    own = Decision(2, 120.0, "courier")  # the untrained model's own false claim
    assert not blocks(DEFECT, Decision(2, 120.0, "courier"), own)
    assert blocks(DEFECT, Decision(2, 120.0, "firm"), own)


def test_no_decision_from_the_untrained_model_blocks_everything():
    assert blocks(DEFECT, Decision(2, 120.0, "firm"), Decision(None, None, None))
    assert blocks(DEFECT, Decision(2, 120.0, "firm"), None)


def test_the_party_is_not_compared_at_zero():
    assert same(Decision(7, 0.0, "firm"), Decision(7, 0.0, "courier"))
    assert not same(Decision(7, 0.0, "firm"), Decision(6, 0.0, "firm"))


def test_a_stored_decision_under_another_prompt_is_refused(tmp_path):
    p = tmp_path / "gate.jsonl"
    p.write_text(json.dumps({"step": 1, "slot": 0, "input": digest("prompt a"),
                             "decision": [2, 120.0, "firm"]}) + "\n", encoding="utf-8")
    store = Stored(p)
    assert store.decision(1, 0, "prompt a") == Decision(2, 120.0, "firm")
    with pytest.raises(ValueError):
        store.decision(1, 0, "prompt b")


def test_the_oracle_passes_only_the_required_decision():
    from thornecliffe.gate import oracle_blocks
    from thornecliffe.policy import grid, owed

    for t in grid():
        o = owed(t)
        assert not oracle_blocks(t, o)
        assert oracle_blocks(t, Decision(o.clause, o.paid, "courier")) == (
            o.paid > 0 and o.bearer == "firm")
        assert oracle_blocks(t, Decision(None, None, None))
