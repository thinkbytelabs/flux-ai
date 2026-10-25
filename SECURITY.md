# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Report them
privately through GitHub's
[private vulnerability reporting](https://github.com/thinkbytelabs/flux-ai/security/advisories/new).
We aim to acknowledge reports within 3 business days and to agree a disclosure
timeline with you.

## Supported versions

Flux is pre-1.0. Security fixes land on `main` and in the latest release only.

## Deployment model — read before exposing `flux-server`

`flux-server` has **no built-in authentication or authorization**. Anyone who
can reach its port can:

- submit runs, spending your LLM budget and triggering tool side effects;
- approve human-approval gates, and cancel any run;
- read any run's events, and replay any conversation's memory by passing its
  `conversation_id`.

Run it only on a trusted network, behind a gateway or reverse proxy that
authenticates callers, rate-limits them, and caps request size. Note that the
default `config/base.yaml` listens on `0.0.0.0:8080`; set `server.listen` to
`127.0.0.1:8080` when the proxy runs on the same host.

Other properties worth knowing:

- **Tools:** an agent can only call the tools listed in its `tools:` config;
  a model-requested call to any other tool is refused before it reaches a
  tool server. MCP servers do not inherit Flux's environment: each gets only
  `PATH`, `HOME`, locale, cache-dir, CA-bundle and similar non-secret
  variables, plus the variables named in its `env` list.
- **Timeouts:** gRPC tool calls have a 120s deadline, MCP tool calls 120s, and
  an LLM attempt fails after 180s of silence from the provider.
- **Secrets in logs:** passwords in Redis/Postgres URLs are redacted from logs
  and errors.
