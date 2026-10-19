"""Explicit artifact profiles for host-agnostic catalog and bounded dispatch.

Profiles live in model_tuning_profiles.settings.qualified_dispatch. No model is
qualified merely because it fits a file-size heuristic or appears in /models.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .config import settings
from .db import db, mw_state_db
from .mw_overlay import apply_mw_effective_status

CONTRACT = "mesh-router-selectable-logical-catalog"


def validate_policy(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("qualified") is not True:
        return None
    p = dict(value)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.:-]*", str(p.get("public_id", ""))):
        return None
    for key in ("context_limit", "output_limit", "engine_context", "safety_tokens"):
        if type(p.get(key)) is not int or p[key] <= 0:
            return None
    if type(p.get("reasoning_limit")) is not int or not 0 <= p["reasoning_limit"] <= p["output_limit"]:
        return None
    if p["output_limit"] + p["safety_tokens"] >= p["context_limit"]:
        return None
    if p["context_limit"] > p["engine_context"] or type(p.get("parallel")) is not int or p["parallel"] != 1:
        return None
    if not p.get("artifact_path") or not p.get("evidence"):
        return None
    return p


def read_profiles() -> list[dict[str, Any]]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT l.lane_id,l.host_id,l.lane_name,l.lane_type,l.backend_type,
                  l.base_url,l.status,l.suspension_reason,l.in_use,
                  l.current_model_name,l.proxy_auth_metadata,h.host_name,
                  m.model_id,m.model_name,hma.local_path,hma.size_bytes,
                  hma.present,hma.eligible,p.max_ctx AS policy_max_ctx,
                  t.settings->'qualified_dispatch' AS policy
                FROM model_tuning_profiles t
                JOIN lanes l ON l.lane_id=t.lane_id AND l.host_id=t.host_id
                JOIN hosts h ON h.host_id=l.host_id
                JOIN models m ON m.model_id=t.model_id
                JOIN host_model_artifacts hma ON hma.host_id=t.host_id
                  AND hma.model_id=t.model_id
                LEFT JOIN lane_model_policy p ON p.lane_id=l.lane_id AND p.model_id=m.model_id
                WHERE t.settings ? 'qualified_dispatch'
                  AND hma.storage_scope='local' AND p.allowed IS DISTINCT FROM false
                ORDER BY h.host_name,m.model_name
            """)
            rows = [dict(r) for r in cur.fetchall()]
    apply_mw_effective_status(rows, mw_state_db=mw_state_db,
                              stale_seconds=settings.default_lease_stale_seconds)
    result = []
    allowed = {x.strip().lower() for x in settings.route_allowed_hosts.split(",") if x.strip()}
    denied = {x.strip().lower() for x in settings.route_denied_hosts.split(",") if x.strip()}
    for row in rows:
        policy = validate_policy(row.get("policy"))
        if policy is None or row["local_path"] != policy["artifact_path"]:
            continue
        if row.get("policy_max_ctx") is not None:
            policy["context_limit"] = min(policy["context_limit"], int(row["policy_max_ctx"]))
            if validate_policy(policy) is None:
                continue
        row["policy"] = policy
        host = str(row["host_name"]).lower()
        row["excluded"] = host in denied or bool(allowed and host not in allowed)
        row["loaded"] = str(row.get("current_model_name") or "").lower() in {
            str(x).lower() for x in [row["model_name"], *policy.get("runtime_aliases", [])]
        }
        state = str(row.get("effective_status") or row.get("status") or "").lower()
        row["suspended"] = bool(row.get("suspension_reason")) or state == "suspended"
        row["operational"] = (state == "ready" and not row["suspended"] and not row["excluded"]
                              and not row.get("active_job") and not row.get("in_use"))
        row["loadable"] = bool(row.get("present") and row.get("eligible") and row["operational"])
        row["dispatch_ready"] = row["loadable"] and row["loaded"]
        result.append(row)
    return result


def catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["policy"]["public_id"], []).append(row)
    models = []
    for public_id, placements in sorted(groups.items()):
        # Advertise a context/output pair that one concrete placement supports.
        # Dispatch filters again by both values; smaller hosts are not forced to
        # honor a larger peer's promise.
        largest = max(placements, key=lambda r: (r["policy"]["context_limit"], r["policy"]["output_limit"]))
        policy = largest["policy"]
        models.append({"id": public_id, "name": policy.get("name", public_id), "tags": ["chat"],
                       "context_limit": policy["context_limit"], "output_limit": policy["output_limit"],
                       "availability": {key: any(bool(r.get(key)) for r in placements)
                                        for key in ["present", "loadable", "loaded", "suspended", "dispatch_ready"]}})
    return {"contract": CONTRACT, "scope": "selectable_logical_models", "models": models}


def matches(row: dict[str, Any], model: str) -> bool:
    # Exact explicit aliases only. Never strip quantization or use family-only
    # fuzzy matching to cross from one artifact to a different artifact.
    names = [row["policy"]["public_id"], row["model_name"], *row["policy"].get("request_aliases", [])]
    return model.lower() in {str(x).lower() for x in names}


def bound_payload(payload: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    requested = out.pop("max_completion_tokens", None)
    if requested is None:
        requested = out.get("max_tokens")
    if requested is None:
        requested = policy["output_limit"]
    if type(requested) is not int or requested <= 0 or requested > policy["output_limit"]:
        raise ValueError("completion budget exceeds the model's total output limit")
    out["max_tokens"] = requested
    # This is a TOTAL generation cap. Do not run the legacy Qwen expansion.
    for key in ("reasoning_budget", "reasoning_budget_tokens", "thinking_tokens", "thinking_budget_tokens"):
        if key in out and (type(out[key]) is not int or not 0 <= out[key] <= policy["reasoning_limit"]):
            raise ValueError("reasoning budget exceeds model policy")
    out["reasoning_budget"] = min(policy["reasoning_limit"], requested)
    out["thinking_budget_tokens"] = out["reasoning_budget"]
    if policy.get("runtime") == "q27":
        out["thinking"] = {"type": "enabled", "budget_tokens": out["reasoning_budget"]}
    return out


def conservative_prompt_bound(payload: dict[str, Any]) -> int:
    # UTF-8 byte bound includes tool schemas and all message/tool-call fields.
    # A separate configured reserve covers template/control tokens. Exact
    # backend formatting/tokenization is preferred when available.
    prompt = {k: v for k, v in payload.items() if k in ("messages", "tools", "tool_choice", "response_format")}
    return len(json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def fits(payload: dict[str, Any], policy: dict[str, Any], prompt_tokens: int) -> bool:
    return prompt_tokens + payload["max_tokens"] + policy["safety_tokens"] <= policy["context_limit"]


def q27_count_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Equivalent text/tool history for q27's CPU-only native count endpoint.

    Its Anthropic and OpenAI bridges share chatml_prompt, tool schemas and
    tool-call/result rendering. Unsupported blocks fall back to the byte bound.
    Native Anthropic tool-choice accounting can overcount extra instructions;
    dispatch also retains the profile's template/runtime safety reserve.
    """
    out = {k: payload[k] for k in ("model", "thinking", "reasoning_effort", "chat_template_kwargs") if k in payload}
    if payload.get("response_format"):
        raise ValueError("structured-response counting uses the conservative fallback")
    messages = []
    for message in payload.get("messages", []):
        role = message.get("role")
        content = message.get("content") or ""
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text" for p in content):
                raise ValueError("unsupported content block")
            content = "".join(p.get("text", "") for p in content)
        if not isinstance(content, str):
            raise ValueError("unsupported content")
        if role == "tool":
            messages.append({"role": "user", "content": [{"type": "tool_result",
                             "tool_use_id": message.get("tool_call_id", ""), "content": content}]})
            continue
        blocks = []
        if message.get("reasoning_content"):
            blocks.append({"type": "thinking", "thinking": message["reasoning_content"]})
        if content:
            blocks.append({"type": "text", "text": content})
        for call in message.get("tool_calls", []):
            function = call["function"]
            args = function.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            blocks.append({"type": "tool_use", "id": call.get("id", ""),
                           "name": function["name"], "input": args})
        messages.append({"role": role, "content": blocks or content})
    out["messages"] = messages
    out["tools"] = []
    for tool in payload.get("tools", []):
        if tool.get("type") != "function":
            raise ValueError("unsupported tool type")
        f = tool["function"]
        out["tools"].append({"name": f["name"], "description": f.get("description", ""),
                             "input_schema": f.get("parameters", {})})
    choice = payload.get("tool_choice", "auto")
    if isinstance(choice, str):
        out["tool_choice"] = {"type": {"required": "any"}.get(choice, choice)}
    else:
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    return out
