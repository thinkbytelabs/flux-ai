#![recursion_limit = "256"]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod launcher;
mod resume;
mod routes;
mod sse;

pub use launcher::{OrchestratorLauncher, RunLauncher};

use flux_session::EventStore;
use flux_workflow::Workflow;
use std::collections::HashMap;
use std::sync::Arc;

/// Shared, cheaply-cloneable server state. `store` is read by the SSE handler
/// and written by the launcher; `launcher` starts runs; `workflows` is the
/// by-id registry loaded at startup.
#[derive(Clone)]
pub struct AppState<S: EventStore, L: RunLauncher<S>> {
    pub store: S,
    pub launcher: L,
    pub workflows: Arc<HashMap<String, Workflow>>,
    pub tail_poll_ms: u64,
}

pub fn router<S, L>(state: AppState<S, L>) -> axum::Router
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/runs", post(routes::create_run::<S, L>))
        .route("/chat", post(routes::chat::<S, L>))
        .route("/workflows", get(routes::list_workflows::<S, L>))
        .route("/runs/{run_id}", get(routes::get_run::<S, L>))
        .route("/runs/{run_id}/approve", post(routes::approve::<S, L>))
        .route("/runs/{run_id}/cancel", post(routes::cancel::<S, L>))
        .route("/runs/{run_id}/events", get(sse::stream_events::<S, L>))
        .with_state(state)
}
