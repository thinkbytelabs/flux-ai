# syntax=docker/dockerfile:1

# Build stage: protoc is needed by flux-tool-client's build script. The Rust
# version matches rust-toolchain.toml.
FROM rust:1.96-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p flux-server -p flux-cli \
    && mkdir /out \
    && cp target/release/flux-server target/release/run_agent target/release/run_workflow /out/

# Runtime stage: CA certificates are needed for HTTPS calls to LLM providers.
FROM debian:bookworm-slim
LABEL org.opencontainers.image.source="https://github.com/thinkbytelabs/flux-ai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Flux AI: orchestrate AI-agent workflows defined in YAML"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home flux
COPY --from=builder /out/ /usr/local/bin/
WORKDIR /app
COPY config/ /app/config/
COPY agents/ /app/agents/
COPY workflows/ /app/workflows/
ENV FLUX_CONFIG_DIR=/app/config
USER flux
EXPOSE 8080
ENTRYPOINT ["flux-server"]
CMD ["--agents-dir", "/app/agents", "--workflows-dir", "/app/workflows"]
