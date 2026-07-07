# Security policy

## Supported versions

Security fixes target the latest commit on `main` and the most recent tagged
release, when a release exists. This personal-lab project does not currently
promise backports to older commits or deployment configurations.

## Reporting a vulnerability

Do not disclose credentials, private topology, exploit details, or other
sensitive evidence in a public issue. Prefer GitHub private vulnerability
reporting at:

https://github.com/kasunami/mesh-router/security/advisories/new

If private reporting is not enabled, contact the repository maintainer
privately through GitHub before sharing details. The maintainer will acknowledge
the report, assess affected versions, and coordinate disclosure when practical.

## Secret handling

- Supply credentials and internal tokens through environment variables or the
  deployment platform's secret manager.
- Never commit `.env`, plaintext Kubernetes `Secret` resources, bearer tokens,
  database credentials, or captured request payloads containing sensitive data.
- Keep example configuration limited to nonfunctional placeholders.
- Rotate any credential immediately if it appears in Git history or CI output;
  deleting the current file is not sufficient remediation.

## Public hygiene

Public source, tests, fixtures, documentation, and file paths must not contain
private hostnames, RFC1918 addresses, personal filesystem paths, or lab-specific
identifiers. Run the repository gate before publishing a change:

```bash
scripts/check_public_hygiene.sh
```

Live certification must use operator-provided targets. The documented dry-run
mode is safe for public evaluation because it does not contact worker lanes.
