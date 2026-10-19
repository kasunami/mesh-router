from copy import deepcopy

import pytest

from mesh_router import qualified_catalog as q


def policy():
    return dict(qualified=True, public_id="example-9b-q4-k-m", context_limit=16384,
                engine_context=65536, output_limit=4096, reasoning_limit=1024,
                safety_tokens=1024, parallel=1, artifact_path="/models/example.gguf",
                evidence="existing qualification", runtime="llama")


def row(**updates):
    r = dict(policy=policy(), model_name="internal-model", present=True, loadable=True,
             loaded=False, suspended=False, dispatch_ready=False)
    r.update(updates)
    return r


def test_policy_requires_explicit_qualification_and_valid_joint_limits():
    assert q.validate_policy(policy())
    for key, value in [("qualified", False), ("context_limit", True), ("output_limit", 0),
                       ("parallel", 2), ("engine_context", 1024), ("safety_tokens", 16384),
                       ("reasoning_limit", 4097), ("artifact_path", ""), ("evidence", "")]:
        p = policy(); p[key] = value
        assert q.validate_policy(p) is None


def test_catalog_keeps_quantization_and_aliases_are_not_catalog_duplicates():
    other = policy(); other["public_id"] = "example-9b-q8-0"
    models = q.catalog([row(), row(), row(policy=other)])["models"]
    assert [m["id"] for m in models] == ["example-9b-q4-k-m", "example-9b-q8-0"]
    assert "host" not in str(models)


def test_context_output_pair_is_supported_by_one_placement():
    small = policy(); small.update(context_limit=8192, output_limit=2048)
    model = q.catalog([row(policy=small), row()])["models"][0]
    assert (model["context_limit"], model["output_limit"]) == (16384, 4096)


def test_installed_loaded_suspended_and_ready_are_separate():
    model = q.catalog([row(loadable=False, loaded=True, suspended=True)])["models"][0]
    assert model["availability"] == dict(present=True, loadable=False, loaded=True,
                                           suspended=True, dispatch_ready=False)


def test_request_aliases_are_exact_not_family_quant_collapsing():
    r = row(); r["policy"]["request_aliases"] = ["old-explicit-alias"]
    assert q.matches(r, "old-explicit-alias")
    assert q.matches(r, "EXAMPLE-9B-Q4-K-M")
    assert not q.matches(r, "example-9b-q8-0")
    assert not q.matches(r, "example-9b")


def test_total_output_budget_never_gets_legacy_qwen_expansion():
    assert q.bound_payload({"max_tokens": 4096}, policy())["max_tokens"] == 4096
    assert q.bound_payload({"max_completion_tokens": 128}, policy())["max_tokens"] == 128
    for request in [{"max_tokens": 4097}, {"max_tokens": -1}, {"max_tokens": True},
                    {"max_tokens": 10, "thinking_tokens": 1025}]:
        with pytest.raises(ValueError):
            q.bound_payload(request, policy())


def test_prompt_bound_includes_tool_schemas_and_unicode_bytes():
    p = {"messages": [{"role": "user", "content": "hello"}]}
    before = q.conservative_prompt_bound(p)
    p["tools"] = [{"function": {"description": "\u4e16\u754c" * 200}}]
    assert q.conservative_prompt_bound(p) > before + 1200


def test_prompt_and_completion_share_context_with_reserve():
    assert q.fits({"max_tokens": 4096}, policy(), 11264)
    assert not q.fits({"max_tokens": 4096}, policy(), 11265)


def test_q27_uses_native_explicit_thinking_budget():
    p = policy(); p["runtime"] = "q27"
    request = q.bound_payload({"max_tokens": 4096}, p)
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_read_profiles_never_qualifies_stale_suspended_or_excluded_hosts(monkeypatch):
    from contextlib import contextmanager
    rows = []
    for state, reason in [("ready", None), ("offline", None), ("ready", "operator_disabled")]:
        rows.append(dict(policy=policy(), local_path="/models/example.gguf", host_name="worker-a",
                         model_name="internal-model", current_model_name="internal-model",
                         effective_status=state, suspension_reason=reason, present=True, eligible=True))
    class Cursor:
        def execute(self, *_): pass
        def fetchall(self): return deepcopy(rows)
        def __enter__(self): return self
        def __exit__(self, *_): pass
    class Conn:
        def cursor(self): return Cursor()
    class DB:
        @contextmanager
        def connect(self): yield Conn()
    monkeypatch.setattr(q, "db", DB())
    monkeypatch.setattr(q, "apply_mw_effective_status", lambda *_a, **_k: None)
    monkeypatch.setattr(q.settings, "route_allowed_hosts", "")
    monkeypatch.setattr(q.settings, "route_denied_hosts", "")
    result = q.read_profiles()
    assert [r["dispatch_ready"] for r in result] == [True, False, False]
    monkeypatch.setattr(q.settings, "route_denied_hosts", "worker-a")
    assert not any(r["loadable"] for r in q.read_profiles())


def test_native_count_conversion_preserves_tools_calls_and_results():
    payload={"messages":[{"role":"system","content":"rules"},
        {"role":"assistant","content":"","tool_calls":[{"id":"a","function":{"name":"lookup","arguments":"{\"key\":\"value\"}"}}]},
        {"role":"tool","tool_call_id":"a","content":"answer"}],
        "tools":[{"type":"function","function":{"name":"lookup","description":"find","parameters":{"type":"object"}}}]}
    counted=q.q27_count_body(payload)
    assert counted["messages"][1]["content"][0]["input"]=={"key":"value"}
    assert counted["messages"][2]["content"][0]["type"]=="tool_result"
    assert counted["tools"][0]["input_schema"]=={"type":"object"}


def test_native_count_conversion_fails_closed_for_unsupported_blocks():
    with pytest.raises(ValueError):
        q.q27_count_body({"messages":[{"role":"user","content":[{"type":"image_url"}]}]})
