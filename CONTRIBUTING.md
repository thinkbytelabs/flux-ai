# Contributing to Flux AI

Thanks for helping. This guide covers what a pull request needs to be merged.
Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

- Rust is pinned in `rust-toolchain.toml` (currently 1.96.0); `rustup` picks
  it up automatically. That version is also the minimum supported Rust version.
- `protoc` (the Protocol Buffers compiler) must be on `PATH`: `flux-tool-client`
  compiles its gRPC definitions at build time. Install it with
  `brew install protobuf` or `apt-get install protobuf-compiler`.
- Most tests need nothing else. Backend tests marked `#[ignore]` need a local
  Redis and Postgres:

  ```bash
  docker run -d -p 6379:6379 redis:7
  docker run -d -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust \
    -e POSTGRES_DB=flux_memory_test postgres:16
  cargo test -p flux-session -p flux-memory -- --ignored
  ```

## Before you open a pull request

CI runs all of these; run them locally first:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo deny check            # cargo install cargo-deny
```

## Code conventions

- **No panics in library code.** Every library crate denies `unwrap`, `expect`
  and `panic!` outside `#[cfg(test)]`. Return an error instead.
- **Errors:** libraries use `thiserror` / `flux_core::error::FluxError`; only
  the binaries (`flux-cli`, the `flux-server` main) use `anyhow`.
- **Tests first** for bug fixes and features: a test that fails without your
  change and passes with it.
- **Comments** explain constraints and *why*, not what the next line does.
- **Dependencies:** a change to any `Cargo.toml` dependency must include the
  matching `Cargo.lock` change, or `--locked` CI fails. New dependencies must
  carry a license allowed by `deny.toml`.
- **Crate layering** is strictly downward and acyclic (see the README's
  architecture graph). Don't add an upward dependency.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/)
  (`fix(runner): …`, `feat(server): …`). One logical change per commit.
- Add a line under `## [Unreleased]` in `CHANGELOG.md` for user-visible changes.
- Describe *why* in the PR body, and how you verified the change.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the license of this project.
