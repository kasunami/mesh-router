# MR Live Certification

Use `scripts/certify_mr.py` to verify a deployed Mesh Router from the client side.

Dry run:

```bash
scripts/certify_mr.py --mode dry-run --checks all
```

Live example:

```bash
scripts/certify_mr.py \
  --mode live \
  --base-url http://mesh-router.example:4010 \
  --checks chat,chat-stream,inventory,mw-command \
  --chat-model qwen3.5:0.8B \
  --mw-host-id worker-a \
  --mw-lane-id gpu
```

All live host, lane, and model values must be supplied by the operator. The script intentionally uses public-safe defaults and does not embed private deployment details.
