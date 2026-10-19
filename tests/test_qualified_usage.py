import asyncio
import json
from types import SimpleNamespace

from mesh_router import app as api


def test_qualified_collection_preserves_total_budget_and_final_usage(monkeypatch):
    seen = []
    zero = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    class Client:
        async def stream_chat(self, **kwargs):
            seen.append(kwargs)
            yield SimpleNamespace(event_type="delta", usage=zero, finish_reason="",
                text_delta="", raw_backend_payload=json.dumps({
                    "choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":123,"completion_tokens":2,"total_tokens":125}
                }).encode())
            yield SimpleNamespace(event_type="completed", usage=zero, finish_reason="",
                                  text_delta="", raw_backend_payload=b"")
    monkeypatch.setattr(api,"MwGrpcClient",Client)
    monkeypatch.setattr(api.settings,"reasoning_budget_tokens",4096)
    monkeypatch.setattr(api.settings,"reasoning_min_visible_tokens",512)
    monkeypatch.setattr(api.settings,"reasoning_max_backend_tokens",8192)
    result=asyncio.run(api._collect_mw_chat_completion(target=SimpleNamespace(),
        request_id="request-a",model="qwen3.6-35b-a3b",request_payload={"messages":[],"max_tokens":4096},
        apply_reasoning_budget=False))
    assert seen[0]["max_tokens"]==4096
    assert seen[0]["request_payload"]["max_tokens"]==4096
    assert result["usage"]==dict(prompt_tokens=123,completion_tokens=2,total_tokens=125)
