#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_tool_client::pb::tool_service_server::ToolServiceServer;
use flux_tool_example::WeatherTool;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _telemetry_guard =
        flux_core::telemetry::init_tracing(false, flux_core::telemetry::log_format_from_env());
    let addr = "127.0.0.1:50051".parse()?;
    tracing::info!(%addr, "get_weather tool server listening");
    Server::builder()
        .add_service(ToolServiceServer::new(WeatherTool))
        .serve(addr)
        .await?;
    Ok(())
}
