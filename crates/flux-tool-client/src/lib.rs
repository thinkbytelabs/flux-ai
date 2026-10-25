#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod pb {
    tonic::include_proto!("flux.tool.v1");
}

pub mod mcp;

use flux_core::error::{FluxError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub tools: Vec<ToolSpec>,
}

impl Manifest {
    pub fn find(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|t| t.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    pub context: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolResult {
    /// The full result. This is what a tool *stage* exposes to workflow
    /// templates, so it must stay lossless.
    pub result: serde_json::Value,
    /// Optional compact projection of `result` for the agent path, where the
    /// value is serialized into a prompt on every turn.
    ///
    /// A transport sets this only when `result` carries redundancy that costs
    /// tokens without telling the model anything new. `flux-runner` prefers it
    /// when present and falls back to `result` otherwise, so a transport that
    /// leaves it `None` behaves exactly as before.
    ///
    /// It deliberately lives here rather than in `flux-runner`: the runner must
    /// not inspect a result's shape to guess which transport produced it.
    pub for_model: Option<serde_json::Value>,
}

impl ToolResult {
    /// A result with no separate model-facing projection.
    pub fn new(result: serde_json::Value) -> Self {
        ToolResult {
            result,
            for_model: None,
        }
    }
}

pub trait ToolTransport {
    fn list_tools(&self) -> impl std::future::Future<Output = Result<Manifest>> + Send;
    fn call_tool(
        &self,
        call: &ToolCall,
    ) -> impl std::future::Future<Output = Result<ToolResult>> + Send;
}

/// Process-wide cache of compiled validators, keyed by the schema itself.
///
/// A `Validator` is immutable once built, and a tool's schema is fixed for the
/// life of the process — the manifest is read once at boot and frozen — so
/// compiling one per tool *call* was repeated work on the dispatch path.
/// Validate a tool call's arguments against the tool's JSON-schema before dispatch.
///
/// The validator is compiled once per distinct schema and reused
/// (`flux_core::schema::cached_validator`) rather than rebuilt per call.
///
/// Built via `flux_core::schema`, not `jsonschema::validator_for`, so an external
/// `$ref` in the schema is refused rather than fetched over the network. That
/// matters here specifically: an MCP server supplies its own tool schemas, so
/// this document is not under Flux's control.
pub fn validate_args(spec: &ToolSpec, args: &serde_json::Value) -> Result<()> {
    let validator = flux_core::schema::cached_validator(&spec.parameters)
        .map_err(|e| FluxError::Config(format!("invalid tool schema for {}: {e}", spec.name)))?;
    // ValidationError<'i> borrows from args, so format to owned string before returning.
    validator.validate(args).map_err(|e| {
        FluxError::Config(format!(
            "arguments for {} failed validation: {e}",
            spec.name
        ))
    })
}

use pb::tool_service_client::ToolServiceClient;
use pb::{CallToolRequest, ListToolsRequest};
use std::time::Duration;

pub struct GrpcTransport {
    client: ToolServiceClient<tonic::transport::Channel>,
}

/// Budget for establishing the channel. tonic's default is no timeout at all,
/// so an endpoint that accepts TCP but never speaks HTTP/2 would hang boot.
const GRPC_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Budget for one RPC, sent as the `grpc-timeout` deadline. A backstop against
/// a wedged tool server, not a latency target — matches the MCP call budget.
const GRPC_CALL_TIMEOUT: Duration = Duration::from_secs(120);

impl GrpcTransport {
    pub async fn connect(endpoint: String) -> Result<Self> {
        Self::connect_with_budget(endpoint, GRPC_CALL_TIMEOUT).await
    }

    async fn connect_with_budget(endpoint: String, call: Duration) -> Result<Self> {
        let channel = tonic::transport::Endpoint::from_shared(endpoint)
            .map_err(|e| FluxError::Config(e.to_string()))?
            .connect_timeout(GRPC_CONNECT_TIMEOUT)
            .timeout(call)
            .connect()
            .await
            .map_err(|e| FluxError::Provider(e.to_string()))?;
        Ok(Self {
            client: ToolServiceClient::new(channel),
        })
    }
}

impl ToolTransport for GrpcTransport {
    async fn list_tools(&self) -> Result<Manifest> {
        let mut client = self.client.clone();
        let resp = client
            .list_tools(ListToolsRequest {})
            .await
            .map_err(|e| FluxError::Provider(e.to_string()))?
            .into_inner();
        let tools = resp
            .tools
            .into_iter()
            .map(|t| {
                let parameters = serde_json::from_str(&t.parameters_json)
                    .map_err(|e| FluxError::Decode(e.to_string()))?;
                Ok(ToolSpec {
                    name: t.name,
                    description: t.description,
                    parameters,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Manifest { tools })
    }

    #[tracing::instrument(skip_all, fields(tool = %call.name))]
    async fn call_tool(&self, call: &ToolCall) -> Result<ToolResult> {
        let mut client = self.client.clone();
        let resp = client
            .call_tool(CallToolRequest {
                name: call.name.clone(),
                args_json: call.args.to_string(),
                context_json: call
                    .context
                    .as_ref()
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            })
            .await
            .map_err(|e| FluxError::Provider(e.to_string()))?
            .into_inner();
        let result = serde_json::from_str(&resp.result_json)
            .map_err(|e| FluxError::Decode(e.to_string()))?;
        // gRPC results are whatever the tool server chose to return, with no
        // Flux-imposed envelope, so there is nothing redundant to project away.
        Ok(ToolResult::new(result))
    }
}

/// Fans `ToolTransport` calls out to several backends, routing by tool name.
///
/// The routing table is built once from each backend's manifest, so callers
/// declare `tools: [name]` without knowing which server answers. A tool name
/// exported by two backends is a configuration error, not a silent shadow.
#[derive(Debug)]
pub struct MultiTransport<T> {
    backends: Vec<T>,
    /// tool name -> index into `backends`
    routes: std::collections::HashMap<String, usize>,
    manifest: Manifest,
}

impl<T: ToolTransport> MultiTransport<T> {
    pub async fn from_backends(backends: Vec<T>) -> Result<MultiTransport<T>> {
        let mut routes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut tools: Vec<ToolSpec> = Vec::new();
        for (idx, backend) in backends.iter().enumerate() {
            let manifest = backend.list_tools().await?;
            for spec in manifest.tools {
                if routes.contains_key(&spec.name) {
                    return Err(FluxError::Config(format!(
                        "tool '{}' is exported by more than one tool endpoint",
                        spec.name
                    )));
                }
                routes.insert(spec.name.clone(), idx);
                tools.push(spec);
            }
        }
        Ok(MultiTransport {
            backends,
            routes,
            manifest: Manifest { tools },
        })
    }
}

/// One tool backend, of whichever kind.
///
/// `ToolTransport` uses RPITIT and so is not dyn-compatible; a closed enum is
/// how heterogeneous backends share one routing table. Transports live in this
/// crate, so a closed set is honest — this is not a plugin point. Deliberately
/// does not derive `Debug`: `GrpcTransport` has none either.
pub enum AnyTransport {
    Grpc(GrpcTransport),
    Mcp(mcp::McpTransport),
}

impl ToolTransport for AnyTransport {
    async fn list_tools(&self) -> Result<Manifest> {
        match self {
            AnyTransport::Grpc(t) => t.list_tools().await,
            AnyTransport::Mcp(t) => t.list_tools().await,
        }
    }

    async fn call_tool(&self, call: &ToolCall) -> Result<ToolResult> {
        match self {
            AnyTransport::Grpc(t) => t.call_tool(call).await,
            AnyTransport::Mcp(t) => t.call_tool(call).await,
        }
    }
}

/// Connect every gRPC endpoint, spawn every MCP server, and build one
/// name-routed transport over the union. A tool name exported by two backends
/// is a startup error regardless of their kinds.
pub async fn connect_all(
    endpoints: &[String],
    mcp_servers: &[mcp::McpServer],
) -> Result<MultiTransport<AnyTransport>> {
    let mut backends = Vec::with_capacity(endpoints.len() + mcp_servers.len());
    for ep in endpoints {
        backends.push(AnyTransport::Grpc(
            GrpcTransport::connect(ep.clone()).await?,
        ));
    }
    for server in mcp_servers {
        backends.push(AnyTransport::Mcp(mcp::McpTransport::spawn(server).await?));
    }
    MultiTransport::from_backends(backends).await
}

impl<T: ToolTransport + Sync> ToolTransport for MultiTransport<T> {
    async fn list_tools(&self) -> Result<Manifest> {
        Ok(self.manifest.clone())
    }

    async fn call_tool(&self, call: &ToolCall) -> Result<ToolResult> {
        let idx = self.routes.get(&call.name).copied().ok_or_else(|| {
            FluxError::Config(format!(
                "tool '{}' is not exported by any tool endpoint",
                call.name
            ))
        })?;
        let backend = self.backends.get(idx).ok_or_else(|| {
            FluxError::Config(format!(
                "routing table points at a missing backend for '{}'",
                call.name
            ))
        })?;
        backend.call_tool(call).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_spec() -> ToolSpec {
        ToolSpec {
            name: "get_weather".into(),
            description: "Get weather for a location".into(),
            parameters: json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }),
        }
    }

    #[test]
    fn manifest_find_locates_by_name() {
        let m = Manifest {
            tools: vec![weather_spec()],
        };
        assert!(m.find("get_weather").is_some());
        assert!(m.find("missing").is_none());
    }

    #[test]
    fn validate_args_accepts_valid() {
        let spec = weather_spec();
        assert!(validate_args(&spec, &json!({ "location": "San Francisco" })).is_ok());
    }

    #[test]
    fn validate_args_rejects_missing_required() {
        let spec = weather_spec();
        match validate_args(&spec, &json!({})) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    fn spec_with(name: &str, parameters: serde_json::Value) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "d".into(),
            parameters,
        }
    }

    #[test]
    fn repeated_validation_of_one_schema_stays_correct() {
        // Cache mechanics live with the cache, in flux_core::schema. What matters
        // here is that reuse does not change the verdict on any call after the
        // first — including one that must fail.
        let schema = json!({
            "type": "object",
            "properties": { "cache_probe_reuse": { "type": "string" } },
            "required": ["cache_probe_reuse"]
        });
        let spec = spec_with("cache_probe_reuse", schema);
        assert!(validate_args(&spec, &json!({ "cache_probe_reuse": "x" })).is_ok());
        assert!(validate_args(&spec, &json!({ "cache_probe_reuse": "y" })).is_ok());
        assert!(validate_args(&spec, &json!({})).is_err());
        assert!(validate_args(&spec, &json!({ "cache_probe_reuse": 7 })).is_err());
    }

    #[test]
    fn two_specs_sharing_one_schema_validate_identically() {
        let schema = json!({
            "type": "object",
            "properties": { "cache_probe_shared": { "type": "integer" } }
        });
        let a = spec_with("cache_probe_shared_a", schema.clone());
        let b = spec_with("cache_probe_shared_b", schema);
        assert!(validate_args(&a, &json!({ "cache_probe_shared": 1 })).is_ok());
        // b differs only by name; keying on the schema means it reuses a's entry
        // and must be judged by the same rules.
        assert!(validate_args(&b, &json!({ "cache_probe_shared": 2 })).is_ok());
        assert!(validate_args(&b, &json!({ "cache_probe_shared": "no" })).is_err());
    }

    #[test]
    fn same_tool_name_with_different_schemas_does_not_cross_contaminate() {
        // The reason the cache is keyed by schema and not by name. Two manifests
        // in one process can disagree about what a name means; serving the wrong
        // validator would silently accept or reject the wrong arguments.
        let string_schema = json!({
            "type": "object",
            "properties": { "cache_probe_xc": { "type": "string" } },
            "required": ["cache_probe_xc"]
        });
        let number_schema = json!({
            "type": "object",
            "properties": { "cache_probe_xc": { "type": "number" } },
            "required": ["cache_probe_xc"]
        });
        let as_string = spec_with("cache_probe_xc", string_schema);
        let as_number = spec_with("cache_probe_xc", number_schema);

        assert!(validate_args(&as_string, &json!({ "cache_probe_xc": "s" })).is_ok());
        assert!(validate_args(&as_string, &json!({ "cache_probe_xc": 7 })).is_err());
        // Same tool name, different schema — must be judged by its own rules.
        assert!(validate_args(&as_number, &json!({ "cache_probe_xc": 7 })).is_ok());
        assert!(validate_args(&as_number, &json!({ "cache_probe_xc": "s" })).is_err());
    }

    #[test]
    fn an_unbuildable_schema_keeps_failing_and_never_starts_passing() {
        // An external $ref cannot compile (the retriever refuses it). It must
        // fail every time — a cache that stored something for it would be worse
        // than one that stored nothing.
        let schema = json!({ "$ref": "https://example.com/cache-probe-bad.json" });
        let spec = spec_with("cache_probe_bad", schema);
        assert!(validate_args(&spec, &json!({})).is_err());
        assert!(validate_args(&spec, &json!({})).is_err());
    }

    #[test]
    fn validate_args_refuses_a_schema_with_an_external_ref() {
        // An MCP server supplies its own tool schemas, so this document is not
        // under Flux's control. It must fail fast rather than causing a blocking
        // HTTP fetch on every tool call.
        let spec = ToolSpec {
            name: "hostile".into(),
            description: "d".into(),
            parameters: json!({ "$ref": "https://example.com/schema.json" }),
        };
        match validate_args(&spec, &json!({ "anything": 1 })) {
            Err(FluxError::Config(m)) => {
                assert!(m.contains("hostile"), "names the tool: {m}");
                assert!(
                    m.contains("external schema reference is not supported"),
                    "our retriever refused it: {m}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn validate_args_still_accepts_an_internal_ref() {
        let spec = ToolSpec {
            name: "internal".into(),
            description: "d".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": { "$ref": "#/$defs/id" } },
                "required": ["id"],
                "$defs": { "id": { "type": "string" } }
            }),
        };
        assert!(validate_args(&spec, &json!({ "id": "x" })).is_ok());
        assert!(validate_args(&spec, &json!({ "id": 1 })).is_err());
    }

    #[test]
    fn validate_args_rejects_wrong_type() {
        let spec = weather_spec();
        match validate_args(&spec, &json!({ "location": 42 })) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[derive(Debug)]
    struct FakeBackend {
        tools: Vec<ToolSpec>,
        tag: &'static str,
    }

    impl ToolTransport for FakeBackend {
        async fn list_tools(&self) -> Result<Manifest> {
            Ok(Manifest {
                tools: self.tools.clone(),
            })
        }
        async fn call_tool(&self, call: &ToolCall) -> Result<ToolResult> {
            Ok(ToolResult::new(
                json!({ "served_by": self.tag, "tool": call.name }),
            ))
        }
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "d".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    #[tokio::test]
    async fn multi_transport_unions_manifests_and_routes_by_name() {
        let multi = MultiTransport::from_backends(vec![
            FakeBackend {
                tools: vec![spec("a_one")],
                tag: "A",
            },
            FakeBackend {
                tools: vec![spec("b_one")],
                tag: "B",
            },
        ])
        .await
        .unwrap();

        let manifest = multi.list_tools().await.unwrap();
        let mut names: Vec<&str> = manifest.tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a_one", "b_one"]);

        let served = |v: serde_json::Value| v["served_by"].as_str().unwrap_or("").to_string();
        let a = multi
            .call_tool(&ToolCall {
                name: "a_one".into(),
                args: json!({}),
                context: None,
            })
            .await
            .unwrap();
        let b = multi
            .call_tool(&ToolCall {
                name: "b_one".into(),
                args: json!({}),
                context: None,
            })
            .await
            .unwrap();
        assert_eq!(served(a.result), "A");
        assert_eq!(served(b.result), "B");
    }

    #[tokio::test]
    async fn multi_transport_rejects_duplicate_tool_name() {
        let err = MultiTransport::from_backends(vec![
            FakeBackend {
                tools: vec![spec("same")],
                tag: "A",
            },
            FakeBackend {
                tools: vec![spec("same")],
                tag: "B",
            },
        ])
        .await;
        match err {
            Err(FluxError::Config(m)) => assert!(m.contains("same")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn multi_transport_unknown_tool_is_config_error() {
        let multi = MultiTransport::from_backends(vec![FakeBackend {
            tools: vec![spec("a_one")],
            tag: "A",
        }])
        .await
        .unwrap();
        match multi
            .call_tool(&ToolCall {
                name: "ghost".into(),
                args: json!({}),
                context: None,
            })
            .await
        {
            Err(FluxError::Config(m)) => assert!(m.contains("ghost")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn connect_all_with_no_backends_yields_an_empty_manifest() {
        let multi = connect_all(&[], &[])
            .await
            .expect("no backends is not an error");
        assert!(multi.list_tools().await.expect("list").tools.is_empty());
    }

    /// A gRPC tool server whose CallTool accepts the request and never answers.
    struct WedgedToolServer;

    #[tonic::async_trait]
    impl pb::tool_service_server::ToolService for WedgedToolServer {
        async fn list_tools(
            &self,
            _req: tonic::Request<pb::ListToolsRequest>,
        ) -> std::result::Result<tonic::Response<pb::Manifest>, tonic::Status> {
            Ok(tonic::Response::new(pb::Manifest { tools: vec![] }))
        }
        async fn call_tool(
            &self,
            _req: tonic::Request<pb::CallToolRequest>,
        ) -> std::result::Result<tonic::Response<pb::CallToolResponse>, tonic::Status> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn a_grpc_call_to_a_wedged_server_fails_within_its_budget() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(pb::tool_service_server::ToolServiceServer::new(
                    WedgedToolServer,
                ))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let transport = GrpcTransport::connect_with_budget(
            format!("http://{addr}"),
            std::time::Duration::from_millis(200),
        )
        .await
        .expect("connect");
        let call = ToolCall {
            name: "anything".into(),
            args: json!({}),
            context: None,
        };
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.call_tool(&call),
        )
        .await
        .expect("call_tool must not hang past its budget");
        assert!(outcome.is_err());
    }
}
