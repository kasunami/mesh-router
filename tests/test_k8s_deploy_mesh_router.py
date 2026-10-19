from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DeployHelperPatchTests(unittest.TestCase):
    def test_revision_update_preserves_existing_annotations_and_env(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts" / "k8s_deploy_mesh_router.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fixture_path = root / "deployment.json"
            patch_path = root / "patch.json"
            fixture = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": "old-time",
                                "mesh-router.openai.com/credential-revision": "credential-rev",
                                "mesh-router.openai.com/lease-secret-revision": "lease-rev",
                                "mesh-router.openai.com/deploy-revision": "old-revision",
                            }
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "mesh-router",
                                    "image": "registry.invalid/mesh-router@sha256:" + "a" * 64,
                                    "imagePullPolicy": "IfNotPresent",
                                    "env": [
                                        {"name": "OTHER_SETTING", "value": "preserve-me"},
                                        {"name": "MESH_ROUTER_DEPLOYMENT_REVISION", "value": "old-revision"},
                                    ],
                                }
                            ]
                        },
                    }
                }
            }
            fixture_path.write_text(json.dumps(fixture))

            kubectl = fake_bin / "kubectl"
            kubectl.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if 'get' in args and 'deploy' in args:\n"
                " print(pathlib.Path(os.environ['DEPLOYMENT_FIXTURE']).read_text())\n"
                "elif 'patch' in args:\n"
                " pathlib.Path(os.environ['PATCH_CAPTURE']).write_text(args[args.index('-p')+1])\n"
                "elif 'rollout' in args or 'get' in args:\n"
                " print('mock-ok')\n"
                "else:\n"
                " raise SystemExit('unexpected kubectl invocation: '+repr(args))\n"
            )
            kubectl.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{Path(os.sys.executable).parent}:{fake_bin}:{env['PATH']}",
                    "DEPLOYMENT_FIXTURE": str(fixture_path),
                    "PATCH_CAPTURE": str(patch_path),
                    "IMAGE": "registry.invalid/mesh-router@sha256:" + "b" * 64,
                    "REVISION": "f" * 40,
                }
            )
            result = subprocess.run(
                [str(script)],
                cwd=repo,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            operations = json.loads(patch_path.read_text())
            paths = {op["path"] for op in operations}
            self.assertNotIn("/spec/template/metadata/annotations", paths)
            self.assertIn(
                "/spec/template/metadata/annotations/mesh-router.openai.com~1deploy-revision",
                paths,
            )
            self.assertIn("/spec/template/spec/containers/0/env/1/value", paths)
            self.assertNotIn("/spec/template/spec/containers/0/env", paths)
            self.assertIn("/spec/template/spec/containers/0/image", paths)
            self.assertIn("/spec/template/spec/containers/0/imagePullPolicy", paths)
            self.assertEqual(
                next(op["value"] for op in operations if op["path"].endswith("/env/1/value")),
                "f" * 40,
            )


if __name__ == "__main__":
    unittest.main()
