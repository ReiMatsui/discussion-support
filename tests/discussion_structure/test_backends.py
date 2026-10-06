import math
from types import SimpleNamespace

import pytest

from das.discussion_structure.config import Config
from das.discussion_structure.judgement import (
    QUESTIONS,
    BudgetExceededError,
    JevBackend,
    OpenAILogprobsBackend,
    OpenAITransport,
    ScriptedBackend,
    logprob_distribution,
    options,
)
from das.discussion_structure.labels import OpenAILabels, checked
from das.discussion_structure.models import Distribution, Tree, Turn
from das.discussion_structure.views import build_view


def context():
    turn = Turn(turn_id="t1", speaker="A", text="紙でいいと思う、でも高い", ms=0, end_ms=2000)
    return turn, build_view(Tree("学食の環境対策"), [], turn, Config())


def response(token="A", probs=None, content=None):
    values = probs or {token: 0.9, "Z": 0.1}
    top = [SimpleNamespace(token=k, logprob=math.log(p)) for k, p in values.items()]
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=1),
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token=token, logprob=math.log(values[token]), top_logprobs=top
                        )
                    ]
                ),
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_openai_network_stub_all_judgement_items_and_single_token_contract():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response()

    config = Config()
    transport = OpenAITransport(config, create)
    turn, view = context()
    result = OpenAILogprobsBackend(config, transport).judge(turn, view)
    assert len(calls) == len(QUESTIONS)
    assert result.target.probabilities["root"] == pytest.approx(0.9)
    assert result.target.probabilities["__other__"] == pytest.approx(0.1)
    assert all(c["max_tokens"] == 1 and c["logprobs"] and c["top_logprobs"] == 20 for c in calls)
    assert transport.spent_usd == pytest.approx(len(calls) * (100 * 0.4 + 1.6) / 1e6)
    assert transport.reserved_usd == 0


def test_logprob_truncation_does_not_inflate_confidence():
    result = logprob_distribution(
        [SimpleNamespace(token="A", logprob=math.log(0.5))], {"A": "support", "B": "concern"}
    )
    assert result.probabilities == {"support": 0.5, "concern": 0, "__other__": 0.5}


@pytest.mark.parametrize(
    "probabilities", [{"A": -0.1, "B": 1.1}, {"A": 0.3}, {"A": float("nan")}, {}]
)
def test_malformed_probability_distributions_fail(probabilities):
    with pytest.raises(ValueError):
        Distribution(probabilities=probabilities)


def test_budget_checks_before_call_and_shared_slow_layer_cost():
    calls = []
    transport = OpenAITransport(Config(budget_usd=0.00001), lambda **kw: calls.append(kw))
    with pytest.raises(BudgetExceededError):
        transport.complete(
            [{"role": "user", "content": "長い" * 100}], model="gpt-4.1-mini", max_tokens=1
        )
    assert not calls
    transport = OpenAITransport(
        Config(), lambda **kw: response(content='{"label":"費用は？","answer_type":"open"}')
    )
    label = OpenAILabels(Config(), transport).generate("issue", {"source_turns": []}, False)
    assert label.label == "費用は？"
    assert transport.spent_usd > 0


def test_network_failure_reservation_is_retained_and_message_has_no_key():
    def broken(**kw):
        raise ConnectionError("SECRET-API-KEY")

    transport = OpenAITransport(Config(), broken)
    with pytest.raises(RuntimeError, match="ConnectionError") as exc:
        transport.complete([], model="gpt-4.1-mini", max_tokens=1)
    assert "SECRET" not in str(exc.value)
    assert transport.reserved_usd > 0
    assert transport.calls == 1


def test_missing_logprobs_and_usage_fail_clearly():
    turn, view = context()
    transport = OpenAITransport(
        Config(),
        lambda **kw: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[SimpleNamespace(logprobs=None)],
        ),
    )
    with pytest.raises(RuntimeError, match="logprobs"):
        OpenAILogprobsBackend(Config(), transport).judge(turn, view)
    transport = OpenAITransport(Config(), lambda **kw: SimpleNamespace(usage=None))
    with pytest.raises(RuntimeError, match="usage"):
        transport.complete([], model="gpt-4.1-mini", max_tokens=1)
    assert transport.reserved_usd > 0


def test_missing_keys_fail_before_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevBackend(Config(), lambda **kw: pytest.fail("network must not run"))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAITransport(Config())


def test_jev_official_payload_and_stubbed_probability_results(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        questions = kwargs["json"]["questions"]
        answers = {
            name: {
                "type": "choice",
                "choice": next(iter(q["criteria"])),
                "probabilities": {value: float(i == 0) for i, value in enumerate(q["criteria"])},
                "confidence": 1.0,
            }
            for name, q in questions.items()
        }
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    turn, view = context()
    result = JevBackend(Config(), post).judge(turn, view)
    assert result.stance.choice == "support"
    url, kw = calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert kw["headers"]["Authorization"] == "Bearer test-not-a-real-key"
    assert kw["json"]["model"] == "jev-1.13.0"
    assert set(kw["json"]["questions"]) == set(QUESTIONS)
    assert kw["json"]["questions"]["alternative"]["criteria"] == {
        "alternative": None,
        "additional": None,
    }
    assert result.target.probabilities == dict.fromkeys(options(view)["target"], 0) | {"root": 1}


def test_jev_http_error_is_sanitized(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")

    def post(*args, **kwargs):
        raise RuntimeError("do not leak meeting text or keys")

    turn, view = context()
    with pytest.raises(RuntimeError, match=r"Jev request failed \(RuntimeError\)") as exc:
        JevBackend(Config(), post).judge(turn, view)
    assert "meeting" not in str(exc.value)


def test_label_regenerates_once_and_never_displays_invalid_names_or_lengths():
    class Generator:
        def __init__(self):
            self.attempts = []

        def generate(self, purpose, context, retry):
            self.attempts.append(retry)
            from das.discussion_structure.labels import Label

            return Label(label="高田さんはどうするか？" if not retry else "費用をどうするか？")

    gen = Generator()
    label = checked(gen, "issue", {}, Config(), ["高田"])
    assert label.label == "費用をどうするか？"
    assert gen.attempts == [False, True]
    assert checked(gen, "issue", {}, Config(issue_chars=2), ["高田"]) is None


def test_missing_scripted_turn_is_clear():
    turn, view = context()
    with pytest.raises(ValueError, match="missing scripted"):
        ScriptedBackend({}).judge(turn, view)


def test_large_candidate_set_is_grouped_without_losing_probabilities():
    transport = OpenAITransport(Config(), lambda **kw: response(probs={"A": 1.0}))
    backend = OpenAILogprobsBackend(Config(), transport)
    choices = [f"i{i}" for i in range(45)]
    result = backend.choose("target", choices, context()[1])
    assert set(result.probabilities) == set(choices)
    assert result.probabilities["i0"] == 1
    assert transport.calls == 4


def test_stubbed_openai_replay_writes_all_outputs(tmp_path):
    import json

    from das.discussion_structure.replay import run

    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps(
            {"turn_id": 1, "speaker": "A", "text": "費用をどう賄うか？", "ms": 0, "end_ms": 2000}
        )
    )
    expected_values = {
        "target": "new_issue",
        "stance": "none",
        "alternative": "additional",
        "shift": "no",
        "relevance": "on",
        "resolution": "none",
        "correction": "none",
        "trigger": "explicit",
        "parent": "root",
        "response_to": "none",
        "switch": "default",
        "decision_answer": "yes",
    }

    def create(**kw):
        if not kw.get("logprobs"):
            return response(content='{"label":"費用をどう賄うか？","answer_type":"open"}')
        mapping = json.loads(kw["messages"][0]["content"].splitlines()[-1])
        field = next(
            name for name, question in QUESTIONS.items() if question in kw["messages"][0]["content"]
        )
        expected = expected_values[field]
        token = next(k for k, v in mapping.items() if v == expected)
        return response(token=token, probs={token: 1.0})

    transport = OpenAITransport(Config(), create)
    tracker = run(
        path, "学食の環境対策", "openai_logprobs", out=tmp_path / "out", transport=transport
    )
    assert tracker.tree.nodes["i1"].label == "費用をどう賄うか？"
    assert len(list((tmp_path / "out").iterdir())) == 4
    assert transport.calls == 13
