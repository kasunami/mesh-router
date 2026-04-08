# Recert Baseline Summary 2026-04-08

- Deskix image gateway drift fixed:
  - `sd-gpu` on `21440`
  - `worker-gateway-11440` restored
  - MR `image-gpu` now `ready`
- Legacy MLX overlay fixed:
  - `tiffs-macbook` now publishes as `backend_type=mlx`
  - `effective_status=ready`
  - current model `/Users/kasunami/models/Falcon3-10B-Instruct-1.58bit`
- MW host health probe fixed:
  - Deskix, pupix1, and `tiffs-macbook` return `ok=true`
- Future benchmark duplicate scheduling fixed in MC
- Remaining live blocker:
  - `mesh-computer` unready due DB acquisition timeouts during final recert
- `tiffs-macbook` still needs full lane viability publication so MC can queue all local MLX models from MR inventory
