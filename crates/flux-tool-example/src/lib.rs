#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_tool_client::pb::tool_service_server::{ToolService, ToolServiceServer};
use flux_tool_client::pb::{
    CallToolRequest, CallToolResponse, ListToolsRequest, Manifest, ToolSpec,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

/// Deterministic canned weather. Pure so it is unit-testable.
pub fn weather_for(location: &str) -> serde_json::Value {
    serde_json::json!({
        "location": location,
        "temperature_f": 68,
        "conditions": "Partly cloudy",
    })
}

pub const WEATHER_SCHEMA: &str = r#"{
  "type": "object",
  "properties": { "location": { "type": "string" } },
  "required": ["location"]
}"#;

#[derive(Default)]
pub struct WeatherTool;

#[tonic::async_trait]
impl ToolService for WeatherTool {
    async fn list_tools(
        &self,
        _request: Request<ListToolsRequest>,
    ) -> std::result::Result<Response<Manifest>, Status> {
        Ok(Response::new(Manifest {
            tools: vec![ToolSpec {
                name: "get_weather".into(),
                description: "Get the current weather for a location.".into(),
                parameters_json: WEATHER_SCHEMA.into(),
            }],
        }))
    }

    async fn call_tool(
        &self,
        request: Request<CallToolRequest>,
    ) -> std::result::Result<Response<CallToolResponse>, Status> {
        let req = request.into_inner();
        if req.name != "get_weather" {
            return Err(Status::not_found(format!("unknown tool: {}", req.name)));
        }
        let args: serde_json::Value = serde_json::from_str(&req.args_json)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let location = args
            .get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Status::invalid_argument("missing 'location'"))?;
        let result = weather_for(location);
        Ok(Response::new(CallToolResponse {
            result_json: result.to_string(),
        }))
    }
}

/// Serve `WeatherTool` on an already-bound listener. Intended for tests that
/// want a deterministic, OS-assigned ephemeral port (bind with `"…:0"`, read
/// back `local_addr()`, then pass the listener here) instead of racing a
/// freshly spawned server to bind a hardcoded address.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
) -> std::result::Result<(), tonic::transport::Error> {
    let incoming = TcpListenerStream::new(listener);
    Server::builder()
        .add_service(ToolServiceServer::new(WeatherTool))
        .serve_with_incoming(incoming)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_echoes_location_with_canned_data() {
        let v = weather_for("San Francisco");
        assert_eq!(v["location"], "San Francisco");
        assert_eq!(v["temperature_f"], 68);
        assert_eq!(v["conditions"], "Partly cloudy");
    }
}
