-- 016: cloud provider lanes (DeepSeek, OpenAI/codex, Anthropic/claude, Google/agy).
--
-- Cloud lanes are router-facing lanes without a physical worker. Pseudo-host rows
-- satisfy the lanes.host_id FK; lanes authenticate to their provider with API keys
-- read from env vars referenced by proxy_auth_metadata.api_key_env
-- (MESH_ROUTER_CLOUD_KEY_<PROVIDER>). probe.py probes them via authenticated
-- GET /models and never rewrites current_model_name.

INSERT INTO hosts (host_name, status, notes)
VALUES
  ('cloud-deepseek',  'ready', 'cloud provider pseudo-host (DeepSeek API)'),
  ('cloud-openai',    'ready', 'cloud provider pseudo-host (OpenAI API, codex)'),
  ('cloud-anthropic', 'ready', 'cloud provider pseudo-host (Anthropic API, claude)'),
  ('cloud-google',    'ready', 'cloud provider pseudo-host (Google Gemini API, agy)')
ON CONFLICT (host_name) DO NOTHING;

INSERT INTO lanes
  (host_id, lane_name, lane_type, backend_type, base_url, openai_path_prefix, status,
   current_model_name, proxy_auth_mode, proxy_auth_metadata)
SELECT h.host_id, v.lane_name, 'other', 'llama', v.base_url, '', 'ready',
       v.model, 'bearer', v.meta::jsonb
FROM (VALUES
  (
    'cloud-deepseek', 'cloud-deepseek-pro', 'https://api.deepseek.com',
    'deepseek-v4-pro',
    '{"cloud": true, "api_key_env": "MESH_ROUTER_CLOUD_KEY_DEEPSEEK", "declared_models": ["deepseek-v4-pro"], "declared_model_tags": {"deepseek-v4-pro": ["cloud", "deepseek", "chat"]}}'
  ),
  (
    'cloud-deepseek', 'cloud-deepseek-flash', 'https://api.deepseek.com/v1',
    'deepseek-v4-flash',
    '{"cloud": true, "api_key_env": "MESH_ROUTER_CLOUD_KEY_DEEPSEEK", "declared_models": ["deepseek-v4-flash"], "declared_model_tags": {"deepseek-v4-flash": ["cloud", "deepseek", "chat"]}}'
  ),
  (
    'cloud-openai', 'cloud-codex', 'https://api.openai.com/v1',
    'gpt-5.6-terra',
    '{"cloud": true, "api_key_env": "MESH_ROUTER_CLOUD_KEY_OPENAI", "declared_models": ["gpt-5.6-terra"], "declared_model_tags": {"gpt-5.6-terra": ["cloud", "openai", "chat", "codex"]}}'
  ),
  (
    'cloud-anthropic', 'cloud-claude', 'https://api.anthropic.com/v1',
    'claude-sonnet-4-6',
    '{"cloud": true, "api_key_env": "MESH_ROUTER_CLOUD_KEY_ANTHROPIC", "declared_models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"], "declared_model_tags": {"claude-opus-4-7": ["cloud", "anthropic", "chat"], "claude-sonnet-4-6": ["cloud", "anthropic", "chat"], "claude-haiku-4-5-20251001": ["cloud", "anthropic", "chat"]}}'
  ),
  (
    'cloud-google', 'cloud-agy', 'https://generativelanguage.googleapis.com/v1beta/openai',
    'gemini-3.1-pro-preview',
    '{"cloud": true, "api_key_env": "MESH_ROUTER_CLOUD_KEY_GEMINI", "declared_models": ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "antigravity-preview-05-2026", "gemma-4-31b-it", "gemma-4-26b-a4b-it"], "declared_model_tags": {"gemini-3.1-pro-preview": ["cloud", "google", "chat", "agy"], "gemini-3-flash-preview": ["cloud", "google", "chat", "agy"], "gemini-3.1-flash-lite-preview": ["cloud", "google", "chat", "agy"], "antigravity-preview-05-2026": ["cloud", "google", "chat", "agy"], "gemma-4-31b-it": ["cloud", "google", "chat"], "gemma-4-26b-a4b-it": ["cloud", "google", "chat"]}}'
  )
) AS v(host_name, lane_name, base_url, model, meta)
JOIN hosts h ON h.host_name = v.host_name
ON CONFLICT DO NOTHING;

INSERT INTO models (model_name, family, tags)
VALUES
  ('deepseek-v4-pro',          'deepseek',  ARRAY['cloud', 'deepseek', 'chat']),
  ('deepseek-v4-flash',        'deepseek',  ARRAY['cloud', 'deepseek', 'chat']),
  ('gpt-5.6-terra',            'openai',    ARRAY['cloud', 'openai', 'chat', 'codex']),
  ('claude-opus-4-7',          'anthropic', ARRAY['cloud', 'anthropic', 'chat']),
  ('claude-sonnet-4-6',        'anthropic', ARRAY['cloud', 'anthropic', 'chat']),
  ('claude-haiku-4-5-20251001','anthropic', ARRAY['cloud', 'anthropic', 'chat']),
  ('gemini-3.1-pro-preview',   'google',    ARRAY['cloud', 'google', 'chat', 'agy']),
  ('gemini-3-flash-preview',   'google',    ARRAY['cloud', 'google', 'chat', 'agy']),
  ('gemini-3.1-flash-lite-preview', 'google', ARRAY['cloud', 'google', 'chat', 'agy']),
  ('antigravity-preview-05-2026', 'google',  ARRAY['cloud', 'google', 'chat', 'agy']),
  ('gemma-4-31b-it',           'google',    ARRAY['cloud', 'google', 'chat']),
  ('gemma-4-26b-a4b-it',       'google',    ARRAY['cloud', 'google', 'chat'])
ON CONFLICT (model_name) DO NOTHING;
