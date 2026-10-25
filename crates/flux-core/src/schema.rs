//! JSON-Schema validator construction, with external `$ref` resolution off.
//!
//! Every validator Flux builds goes through [`build_validator`]. The reason is
//! that JSON Schema lets a schema delegate its rules to another document:
//!
//! ```json
//! { "$ref": "https://example.com/shared.json" }
//! ```
//!
//! The `jsonschema` crate honours that by default — a blocking HTTP request for
//! `http`/`https`, a local file read for `file:`. That was harmless while every
//! schema came from a first-party gRPC tool server, but an MCP server supplies
//! its own tool schemas, so the document is no longer under Flux's control.
//!
//! The cost is mostly reliability rather than confidentiality (an MCP server
//! already runs as a child process with Flux's whole environment): the fetch is
//! blocking, it happens on every validation because validators are rebuilt per
//! call, and it fails confusingly wherever egress is restricted.
//!
//! Note that Flux declares `jsonschema` with `default-features = false`, which
//! would disable the resolvers — but a transitive dependency requests the
//! default features and Cargo unions features per crate-version across the
//! graph, so they stay compiled in regardless. Replacing the *retriever* is
//! what actually closes the behaviour, independent of any feature flag.

use crate::error::{FluxError, Result};

/// A retriever that refuses every external reference.
///
/// `jsonschema` calls this only when a schema references a document that is not
/// already in the registry — i.e. exactly the case Flux does not want resolved.
#[derive(Debug, Clone, Copy)]
pub struct NoExternalRefs;

impl jsonschema::Retrieve for NoExternalRefs {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<String>,
    ) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(format!(
            "external schema reference is not supported: {uri} \
             (schemas must be self-contained)"
        )
        .into())
    }
}

/// Build a validator for `schema`, refusing external `$ref`s.
///
/// Behaves like `jsonschema::validator_for` for any self-contained schema. A
/// schema that reaches outside itself fails here, at build time, instead of
/// causing a network or filesystem access.
pub fn build_validator(schema: &serde_json::Value) -> Result<jsonschema::Validator> {
    jsonschema::options()
        .with_retriever(NoExternalRefs)
        .build(schema)
        .map_err(|e| FluxError::Config(format!("invalid schema: {e}")))
}

/// Upper bound on distinct cached validators.
///
/// A real deployment caches one entry per configured tool schema plus one per
/// agent `output_schema` — tens, not hundreds. The cap exists so the bound holds
/// even if schemas start arriving dynamically (an MCP server that varies its
/// tool descriptions, say), rather than resting on an assumption about callers
/// that a later change could quietly invalidate.
const CACHE_CAP: usize = 512;

/// Process-wide validator cache.
///
/// Keyed by the schema's serialized form, deliberately **not** by tool name.
/// Names are unique within one routing table (`from_backends` rejects a
/// duplicate), but nothing stops two `Manifest`s in the same process from
/// disagreeing about what a given name means, and serving the wrong validator
/// would silently accept or reject the wrong arguments. Keying on the schema
/// makes that class of bug impossible rather than unlikely.
///
/// It lives here, next to [`build_validator`], rather than in one caller: both
/// tool-argument validation and agent `output_schema` checking compile the same
/// way, and a cache reachable from only one of them left the other rebuilding a
/// validator on every single response.
#[allow(clippy::type_complexity)]
fn validator_cache() -> &'static std::sync::RwLock<
    std::collections::HashMap<String, std::sync::Arc<jsonschema::Validator>>,
> {
    static CACHE: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<String, std::sync::Arc<jsonschema::Validator>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Fetch or build the validator for `schema`, reusing a compiled one when this
/// exact schema has been seen before.
///
/// A poisoned lock degrades to building uncached rather than failing the call: a
/// validation path should not start erroring because some unrelated thread
/// panicked while holding the lock.
///
/// The key is built by serializing `schema` on every call, including hits. That
/// is deliberate: the alternative — hashing to a fixed-width key — trades an
/// exact match for a collision that would silently serve the *wrong* validator,
/// which is the failure this cache is keyed to avoid. The serialization is on
/// the order of a microsecond against a tool call that costs milliseconds of
/// process or network I/O, so it is not worth that risk.
pub fn cached_validator(
    schema: &serde_json::Value,
) -> Result<std::sync::Arc<jsonschema::Validator>> {
    let key = schema.to_string();

    if let Ok(cache) = validator_cache().read() {
        if let Some(found) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(found));
        }
    }

    // Built outside the write lock. Compilation is the slow part, so two threads
    // racing on the same new schema is cheaper than serializing every miss —
    // whichever insert lands second is discarded, and both values are equivalent.
    let validator = std::sync::Arc::new(build_validator(schema)?);
    if let Ok(mut cache) = validator_cache().write() {
        // At the cap, stay correct and stop growing: callers still get a working
        // validator, just an uncached one. Dropping the whole cache instead
        // would penalise the hot entries for the newcomer.
        if cache.len() < CACHE_CAP || cache.contains_key(&key) {
            cache.insert(key, std::sync::Arc::clone(&validator));
        } else {
            tracing::warn!(
                cap = CACHE_CAP,
                "validator cache is full; compiling uncached from here. \
                 Schemas are probably being generated dynamically."
            );
        }
    }
    Ok(validator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn self_contained_schema_builds_and_validates() {
        let schema = json!({
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
        });
        let validator = build_validator(&schema).expect("self-contained schema builds");
        assert!(validator.validate(&json!({ "location": "Berlin" })).is_ok());
        assert!(validator.validate(&json!({})).is_err());
    }

    #[test]
    fn internal_ref_still_resolves() {
        // A `$ref` pointing inside the same document must keep working — only
        // references that leave the document are refused.
        let schema = json!({
            "type": "object",
            "properties": { "id": { "$ref": "#/$defs/id" } },
            "required": ["id"],
            "$defs": { "id": { "type": "string", "minLength": 3 } }
        });
        let validator = build_validator(&schema).expect("internal $ref builds");
        assert!(validator.validate(&json!({ "id": "abc" })).is_ok());
        assert!(validator.validate(&json!({ "id": "ab" })).is_err());
    }

    #[test]
    fn https_ref_is_refused_not_fetched() {
        let schema = json!({ "$ref": "https://example.com/shared.json" });
        match build_validator(&schema) {
            Err(FluxError::Config(m)) => {
                assert!(
                    m.contains("external schema reference is not supported"),
                    "our retriever must be the one that refused it, got: {m}"
                );
                assert!(m.contains("example.com"), "names the refused URI: {m}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn file_ref_is_refused_not_read() {
        let schema = json!({ "$ref": "file:///etc/passwd" });
        match build_validator(&schema) {
            Err(FluxError::Config(m)) => assert!(
                m.contains("external schema reference is not supported"),
                "got: {m}"
            ),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Whether a schema currently has a cached validator. Tests share one
    /// process and run in parallel, so they assert *presence* of their own
    /// unique schema rather than a total cache size, which would be racy.
    fn is_cached(schema: &serde_json::Value) -> bool {
        validator_cache()
            .read()
            .map(|c| c.contains_key(&schema.to_string()))
            .unwrap_or(false)
    }

    #[test]
    fn a_schema_is_compiled_once_and_then_reused() {
        // Unique to this test so no other test can have populated it.
        let schema = json!({
            "type": "object",
            "properties": { "core_cache_reuse": { "type": "string" } },
            "required": ["core_cache_reuse"]
        });
        assert!(!is_cached(&schema), "precondition: not yet compiled");

        let first = cached_validator(&schema).expect("compiles");
        assert!(is_cached(&schema), "first call must populate the cache");

        let second = cached_validator(&schema).expect("hits the cache");
        // Same allocation, so the compile genuinely did not happen twice.
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "a hit must hand back the same validator, not an equivalent one"
        );
        assert!(second.validate(&json!({ "core_cache_reuse": "x" })).is_ok());
        assert!(second.validate(&json!({})).is_err());
    }

    #[test]
    fn distinct_schemas_get_distinct_validators() {
        // The reason the cache is keyed by schema and not by name: two manifests
        // in one process can disagree about what a name means, and serving the
        // wrong validator would silently accept or reject the wrong arguments.
        let as_string = json!({
            "type": "object",
            "properties": { "core_cache_xc": { "type": "string" } },
            "required": ["core_cache_xc"]
        });
        let as_number = json!({
            "type": "object",
            "properties": { "core_cache_xc": { "type": "number" } },
            "required": ["core_cache_xc"]
        });
        let s = cached_validator(&as_string).expect("compiles");
        let n = cached_validator(&as_number).expect("compiles");
        assert!(!std::sync::Arc::ptr_eq(&s, &n));
        assert!(s.validate(&json!({ "core_cache_xc": "v" })).is_ok());
        assert!(s.validate(&json!({ "core_cache_xc": 7 })).is_err());
        assert!(n.validate(&json!({ "core_cache_xc": 7 })).is_ok());
        assert!(n.validate(&json!({ "core_cache_xc": "v" })).is_err());
    }

    #[test]
    fn an_unbuildable_schema_is_not_cached() {
        // Caching the failure would be pointless; caching a *validator* for it
        // would be wrong. Assert it stays absent so a later change cannot mask a
        // schema that must never compile.
        let schema = json!({ "$ref": "https://example.com/core-cache-bad.json" });
        assert!(cached_validator(&schema).is_err());
        assert!(
            !is_cached(&schema),
            "a schema that cannot compile is not cached"
        );
        assert!(cached_validator(&schema).is_err(), "and still fails");
    }

    #[test]
    fn a_cached_validator_matches_an_uncached_build() {
        // The cache must not be a behaviour fork: same verdicts either way.
        let schema = json!({
            "type": "object",
            "properties": { "core_cache_parity": { "type": "integer" } },
            "required": ["core_cache_parity"]
        });
        let direct = build_validator(&schema).expect("builds");
        let cached = cached_validator(&schema).expect("compiles");
        for probe in [
            json!({ "core_cache_parity": 1 }),
            json!({ "core_cache_parity": "no" }),
            json!({}),
        ] {
            assert_eq!(
                direct.validate(&probe).is_ok(),
                cached.validate(&probe).is_ok(),
                "verdicts diverged for {probe}"
            );
        }
    }

    #[test]
    fn nested_external_ref_is_refused() {
        // The reference is not at the root — confirm depth does not smuggle one
        // past the retriever.
        let schema = json!({
            "type": "object",
            "properties": { "inner": { "$ref": "https://example.com/inner.json" } }
        });
        assert!(build_validator(&schema).is_err());
    }
}
