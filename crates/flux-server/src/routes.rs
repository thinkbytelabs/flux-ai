use crate::resume::{awaiting_approval, rebuild_state};
use crate::{AppState, RunLauncher};
use axum::extract::Path;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::response::{IntoResponse, Json};
use flux_core::workflow_event::WorkflowEvent;
use flux_session::EventStore;
use flux_workflow::Workflow;
use serde::Deserialize;
use sha2::Digest;

/// Errors carry the same JSON shape as successes so clients need one parser.
pub(crate) fn error_response(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}

/// Shallow-merge approval `data` into a run's stored `context`. New keys are
/// added; a key already in the stored context may not change (F2: approval data
/// must not rewrite request-scoped context such as a tenant id). Errs when either
/// side is present but not a JSON object, or on a conflicting key.
fn merge_context(
    stored: Option<serde_json::Value>,
    data: Option<serde_json::Value>,
) -> Result<Option<serde_json::Value>, String> {
    let Some(data) = data else { return Ok(stored) };
    let serde_json::Value::Object(incoming) = data else {
        return Err("approve `data` must be a JSON object".into());
    };
    match stored {
        None => Ok(Some(serde_json::Value::Object(incoming))),
        Some(serde_json::Value::Object(mut base)) => {
            for (k, v) in incoming {
                match base.get(&k) {
                    Some(existing) if existing != &v => {
                        return Err(format!("approve data may not change run context key '{k}'"));
                    }
                    _ => {
                        base.insert(k, v);
                    }
                }
            }
            Ok(Some(serde_json::Value::Object(base)))
        }
        Some(_) => Err("cannot merge approve `data` into a non-object run context".into()),
    }
}

/// The run's answer: the last `StageComplete` of the workflow's `output` stage
/// when it completed, else the last `StageComplete` in the log (F1).
fn final_reply(
    events: &[(flux_session::Offset, WorkflowEvent)],
    output: Option<&str>,
) -> Option<(String, Option<serde_json::Value>)> {
    let pick = |want: Option<&str>| {
        events.iter().rev().find_map(|(_, e)| match e {
            WorkflowEvent::StageComplete { stage, output } if want.is_none_or(|w| w == stage) => {
                Some((output.message.clone(), output.structured.clone()))
            }
            _ => None,
        })
    };
    output.and_then(|o| pick(Some(o))).or_else(|| pick(None))
}

#[derive(Deserialize)]
pub(crate) struct CreateRun {
    workflow_id: String,
    input: String,
    #[serde(default)]
    conversation_id: Option<String>,
    #[serde(default)]
    context: Option<serde_json::Value>,
}

/// A client's `Idempotency-Key`, the endpoint it is scoped to, and a hash of
/// the request it came with.
struct Idempotency {
    scope: &'static str,
    key: String,
    body_hash: String,
}

/// Hash of the parsed JSON in canonical form, so a retry that only re-formats
/// or re-orders the body still counts as the same request.
fn body_hash(value: &serde_json::Value) -> String {
    let digest = sha2::Sha256::digest(canonical_json(value).as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// `v` as JSON text with object keys sorted, recursively. Independent of
/// serde_json's `preserve_order` feature, which another crate may enable.
fn canonical_json(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let fields: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical_json(&map[k])))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", items.join(","))
        }
        scalar => scalar.to_string(),
    }
}

/// Parse a `/runs` or `/chat` body and its optional `Idempotency-Key`.
// The error is the handler's own response, returned once per request.
#[allow(clippy::result_large_err)]
fn parse_run_request(
    scope: &'static str,
    headers: &axum::http::HeaderMap,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Result<(CreateRun, Option<Idempotency>), Response> {
    let Json(value) = body.map_err(|rej| {
        error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid request body: {}", rej.body_text()),
        )
    })?;
    let idempotency = match headers.get("idempotency-key") {
        None => None,
        Some(raw) => {
            let key = raw
                .to_str()
                .ok()
                .filter(|k| !k.is_empty() && k.len() <= 255)
                .ok_or_else(|| {
                    error_response(
                        StatusCode::BAD_REQUEST,
                        "Idempotency-Key must be 1-255 printable ASCII characters",
                    )
                })?;
            Some(Idempotency {
                scope,
                key: key.to_string(),
                body_hash: body_hash(&value),
            })
        }
    };
    let req: CreateRun = serde_json::from_value(value).map_err(|e| {
        error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid request body: {e}"),
        )
    })?;
    Ok((req, idempotency))
}

/// What `prepare_run` set up.
enum Prepared {
    /// A new run, ready to launch.
    New { run_id: String, workflow: Workflow },
    /// A retry of an earlier request with the same `Idempotency-Key`.
    Replay { run_id: String, workflow: Workflow },
}

/// Resolve the workflow, mint a run_id, claim the idempotency key, and persist
/// the run's status/manifest/conversation_id/context. Returns what to launch
/// (or replay), or an error `Response` to return as-is. Shared by `create_run`
/// and `chat`.
async fn prepare_run<S, L>(
    state: &AppState<S, L>,
    req: &CreateRun,
    idempotency: Option<&Idempotency>,
) -> Result<Prepared, Response>
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let Some(workflow) = state.workflows.get(&req.workflow_id).cloned() else {
        return Err(error_response(
            StatusCode::NOT_FOUND,
            format!("unknown workflow_id '{}'", req.workflow_id),
        ));
    };
    let run_id = uuid::Uuid::new_v4().to_string();
    if let Some(idem) = idempotency {
        match state
            .store
            .claim_idempotency_key(idem.scope, &idem.key, &run_id, &idem.body_hash)
            .await
        {
            Ok(flux_session::IdempotencyClaim::Claimed) => {}
            Ok(flux_session::IdempotencyClaim::Existing { run_id, body_hash })
                if body_hash == idem.body_hash =>
            {
                return Ok(Prepared::Replay { run_id, workflow });
            }
            Ok(flux_session::IdempotencyClaim::Existing { .. }) => {
                return Err(error_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "Idempotency-Key reused with a different request",
                ))
            }
            Err(_) => {
                return Err(error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to start run",
                ))
            }
        }
    }
    if write_run_state(state, &run_id, req).await.is_err() {
        // The run never started: free the key so a retry starts fresh
        // rather than replaying a run that does not exist.
        if let Some(idem) = idempotency {
            let _ = state
                .store
                .release_idempotency_key(idem.scope, &idem.key)
                .await;
        }
        return Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to start run",
        ));
    }
    Ok(Prepared::New { run_id, workflow })
}

async fn write_run_state<S, L>(
    state: &AppState<S, L>,
    run_id: &str,
    req: &CreateRun,
) -> flux_core::error::Result<()>
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    state
        .store
        .set_status(run_id, flux_session::RunStatus::Running)
        .await?;
    state
        .store
        .set_manifest(run_id, &req.workflow_id, &req.input)
        .await?;
    if let Some(cid) = req.conversation_id.as_deref() {
        state.store.set_conversation_id(run_id, cid).await?;
    }
    if let Some(ctx) = &req.context {
        state.store.set_context(run_id, ctx).await?;
    }
    Ok(())
}

/// POST /runs — validate, mint a run_id, spawn the run in the background, 202.
pub(crate) async fn create_run<S, L>(
    State(state): State<AppState<S, L>>,
    headers: axum::http::HeaderMap,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let (req, idempotency) = match parse_run_request("runs", &headers, body) {
        Ok(x) => x,
        Err(resp) => return resp,
    };
    match start_run(&state, req, idempotency).await {
        Ok(Started::New { run_id, .. }) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({ "run_id": run_id })),
        )
            .into_response(),
        // 200, not 202: nothing new was started.
        Ok(Started::Replay { run_id, .. }) => (
            StatusCode::OK,
            Json(serde_json::json!({ "run_id": run_id })),
        )
            .into_response(),
        Err(resp) => resp,
    }
}

/// POST /chat — run a workflow to completion synchronously and return its reply.
pub(crate) async fn chat<S, L>(
    State(state): State<AppState<S, L>>,
    headers: axum::http::HeaderMap,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let (req, idempotency) = match parse_run_request("chat", &headers, body) {
        Ok(x) => x,
        Err(resp) => return resp,
    };
    match start_run(&state, req, idempotency).await {
        Ok(Started::New {
            run_id,
            output,
            run,
        }) => {
            if run.await.is_err() {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, "run task failed");
            }
            chat_reply(&state, &run_id, output.as_deref()).await
        }
        Ok(Started::Replay { run_id, output }) => {
            replay_chat(&state, &run_id, output.as_deref()).await
        }
        Err(resp) => resp,
    }
}

/// A run `start_run` set up, or the earlier run an idempotent retry maps to.
enum Started {
    New {
        run_id: String,
        output: Option<String>,
        run: tokio::task::JoinHandle<()>,
    },
    Replay {
        run_id: String,
        output: Option<String>,
    },
}

/// Set up and launch a run in tasks of their own, so a client that
/// disconnects drops only its wait for the answer. Abandoning setup midway
/// would leave the idempotency key bound to a run that never starts, or a run
/// stuck `running` with nothing executing it; abandoning the run would cut
/// its tool calls off halfway. Stop a run on purpose with the cancel endpoint.
async fn start_run<S, L>(
    state: &AppState<S, L>,
    req: CreateRun,
    idempotency: Option<Idempotency>,
) -> Result<Started, Response>
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let state = state.clone();
    let setup = tokio::spawn(async move {
        match prepare_run(&state, &req, idempotency.as_ref()).await? {
            Prepared::Replay { run_id, workflow } => Ok(Started::Replay {
                run_id,
                output: workflow.output,
            }),
            Prepared::New { run_id, workflow } => {
                let output = workflow.output.clone();
                let launcher = state.launcher.clone();
                let store = state.store.clone();
                let spawned_id = run_id.clone();
                let run = tokio::spawn(async move {
                    launcher
                        .launch(
                            store,
                            spawned_id,
                            workflow,
                            req.input,
                            req.conversation_id,
                            req.context,
                            flux_orchestrator::ResumeState::default(),
                        )
                        .await;
                });
                Ok(Started::New {
                    run_id,
                    output,
                    run,
                })
            }
        }
    });
    match setup.await {
        Ok(started) => started,
        Err(_) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "run setup failed",
        )),
    }
}

/// How long a `/chat` replay waits for the original run's status to appear:
/// the key is claimed just before that run writes its state.
const REPLAY_START_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Answer a `/chat` retry: wait for the original run to stop, then reply
/// exactly as the first call would have.
async fn replay_chat<S, L>(state: &AppState<S, L>, run_id: &str, output: Option<&str>) -> Response
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    use flux_session::RunStatus;
    let started = tokio::time::Instant::now();
    loop {
        match state.store.get_status(run_id).await {
            Ok(Some(RunStatus::Running | RunStatus::Cancelling)) => {}
            Ok(Some(_)) => return chat_reply(state, run_id, output).await,
            Ok(None) if started.elapsed() < REPLAY_START_GRACE => {}
            Ok(None) => return error_response(StatusCode::GONE, "run state expired"),
            Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
        }
        tokio::time::sleep(std::time::Duration::from_millis(state.tail_poll_ms)).await;
    }
}

/// The `/chat` response for a run that has stopped: its reply, or why not.
async fn chat_reply<S, L>(state: &AppState<S, L>, run_id: &str, output: Option<&str>) -> Response
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    match state.store.get_status(run_id).await {
        Ok(Some(flux_session::RunStatus::Complete)) => {
            // The final StageComplete sits within the last round's fold plus
            // the terminal event — a bounded tail read instead of replaying
            // every token delta of the run. Fall back to the full log only in
            // the (never-expected) case the tail window misses it.
            // With an `output` stage, later stages may have streamed many events
            // after it, so read the whole log; otherwise a bounded tail suffices.
            let read = if output.is_some() {
                state.store.read_after(run_id, 0).await
            } else {
                state.store.read_tail(run_id, 32).await
            };
            let events = match read {
                Ok(e) => e,
                Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
            };
            let found = final_reply(&events, output);
            let (reply, structured) = match found {
                Some(r) => r,
                None => match state.store.read_after(run_id, 0).await {
                    Ok(all) => final_reply(&all, output).unwrap_or_default(),
                    Err(_) => {
                        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error")
                    }
                },
            };
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "run_id": run_id,
                    "status": "complete",
                    "reply": reply,
                    "structured": structured,
                })),
            )
                .into_response()
        }
        Ok(Some(flux_session::RunStatus::WaitingApproval)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "run_id": run_id, "status": "waiting_approval" })),
        )
            .into_response(),
        Ok(Some(flux_session::RunStatus::Cancelled)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "run_id": run_id, "status": "cancelled" })),
        )
            .into_response(),
        Ok(Some(flux_session::RunStatus::Failed)) => {
            // The WorkflowFailed event is the last event of a failed run, so
            // a bounded tail read is enough to find it.
            let events = state.store.read_tail(run_id, 16).await.unwrap_or_default();
            let error = events
                .iter()
                .rev()
                .find_map(|(_, e)| match e {
                    WorkflowEvent::WorkflowFailed { error, .. } => Some(error.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "run failed".to_string());
            (
                StatusCode::OK,
                Json(serde_json::json!({ "run_id": run_id, "status": "failed", "error": error })),
            )
                .into_response()
        }
        _ => error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
    }
}

/// GET /runs/{run_id} — cheap status + outcome poll, no SSE required.
pub(crate) async fn get_run<S, L>(
    State(state): State<AppState<S, L>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let status = match state.store.get_status(&run_id).await {
        Ok(Some(s)) => s,
        Ok(None) => return error_response(StatusCode::NOT_FOUND, "unknown run_id"),
        Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
    };
    let mut body = serde_json::json!({ "run_id": run_id, "status": status.as_str() });
    match status {
        flux_session::RunStatus::Complete => {
            let output = match state.store.get_manifest(&run_id).await {
                Ok(Some((workflow_id, _))) => state
                    .workflows
                    .get(&workflow_id)
                    .and_then(|w| w.output.clone()),
                _ => None,
            };
            let read = if output.is_some() {
                state.store.read_after(&run_id, 0).await
            } else {
                state.store.read_tail(&run_id, 32).await
            };
            if let Ok(events) = read {
                if let Some((reply, structured)) = final_reply(&events, output.as_deref()) {
                    body["reply"] = serde_json::Value::String(reply);
                    body["structured"] = structured.unwrap_or(serde_json::Value::Null);
                }
            }
        }
        flux_session::RunStatus::Failed => {
            if let Ok(events) = state.store.read_tail(&run_id, 16).await {
                if let Some(error) = events.iter().rev().find_map(|(_, e)| match e {
                    WorkflowEvent::WorkflowFailed { error, .. } => Some(error.clone()),
                    _ => None,
                }) {
                    body["error"] = serde_json::Value::String(error);
                }
            }
        }
        _ => {}
    }
    (StatusCode::OK, Json(body)).into_response()
}

/// GET /workflows — the ids a client may pass as workflow_id.
pub(crate) async fn list_workflows<S, L>(State(state): State<AppState<S, L>>) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let mut ids: Vec<&str> = state.workflows.keys().map(String::as_str).collect();
    ids.sort_unstable();
    Json(serde_json::json!({ "workflows": ids }))
}

#[derive(Deserialize)]
pub(crate) struct ApproveBody {
    stage_id: String,
    #[serde(default)]
    data: Option<serde_json::Value>,
}

/// Why the post-CAS resume preparation failed, split by what the client
/// should do about it.
enum ResumeFailure {
    /// The run's Redis state expired — not resumable, run is failed.
    Expired,
    /// Transient store trouble — the gate is restored, the client may retry.
    Internal(&'static str),
    /// The approval was refused (e.g. data conflicts with run context) — gate restored, 400.
    Rejected(String),
    /// The named stage is not a gate the run is paused on — gate restored, 409.
    NotWaiting(String),
}

/// Everything `approve` does between winning the CAS and spawning the resumed
/// launch. Pure reads plus the approval/context writes; no status changes.
async fn prepare_resume<S, L>(
    state: &AppState<S, L>,
    run_id: &str,
    req: ApproveBody,
) -> Result<
    (
        Workflow,
        String,
        Option<String>,
        Option<serde_json::Value>,
        flux_orchestrator::ResumeState,
    ),
    ResumeFailure,
>
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    // Check the approval data against the run context BEFORE recording the approval:
    // a refused approval (400) must not leave the stage approved.
    let context = match state.store.get_context(run_id).await {
        Ok(c) => c,
        Err(_) => return Err(ResumeFailure::Internal("store error")),
    };
    let merged = match merge_context(context, req.data) {
        Ok(c) => c,
        Err(msg) if msg.starts_with("approve data may not change") => {
            return Err(ResumeFailure::Rejected(msg))
        }
        // `data` was validated pre-CAS; this only fires on a non-object
        // *stored* context, which is server-side state damage.
        Err(_) => {
            return Err(ResumeFailure::Internal(
                "cannot merge approve data into run context",
            ))
        }
    };
    let events = match state.store.read_after(run_id, 0).await {
        Ok(e) => e,
        Err(_) => return Err(ResumeFailure::Internal("store error")),
    };
    // A paused run always logged at least its StageWaitingApproval event; an
    // empty log means the events key expired out from under the meta key.
    if events.is_empty() {
        return Err(ResumeFailure::Expired);
    }
    // Only a gate the run is actually paused on may be approved: recording any
    // other stage would pre-approve a later gate so it runs with no human asked.
    if !awaiting_approval(&events).contains(&req.stage_id) {
        return Err(ResumeFailure::NotWaiting(format!(
            "stage '{}' is not awaiting approval",
            req.stage_id
        )));
    }
    if state
        .store
        .record_approval(run_id, &req.stage_id)
        .await
        .is_err()
    {
        return Err(ResumeFailure::Internal("failed to record approval"));
    }
    let (workflow_id, input) = match state.store.get_manifest(run_id).await {
        Ok(Some(m)) => m,
        // A paused run always wrote its manifest; None means TTL took it.
        Ok(None) => return Err(ResumeFailure::Expired),
        Err(_) => return Err(ResumeFailure::Internal("store error")),
    };
    let Some(workflow) = state.workflows.get(&workflow_id).cloned() else {
        return Err(ResumeFailure::Internal(
            "manifest references unknown workflow",
        ));
    };
    let (completed, skipped) = rebuild_state(&events);
    let approved = match state.store.approved_stages(run_id).await {
        Ok(a) => a,
        Err(_) => return Err(ResumeFailure::Internal("store error")),
    };
    let resume = flux_orchestrator::ResumeState {
        completed,
        skipped,
        approved,
    };
    let conversation_id = match state.store.get_conversation_id(run_id).await {
        Ok(c) => c,
        Err(_) => return Err(ResumeFailure::Internal("store error")),
    };
    if let Some(ctx) = &merged {
        if state.store.set_context(run_id, ctx).await.is_err() {
            return Err(ResumeFailure::Internal("store error"));
        }
    }
    Ok((workflow, input, conversation_id, merged, resume))
}

/// Put a paused run's gate back after a failed approve. We own the `running`
/// value approve swapped in, so this CAS cannot clobber a concurrent winner.
/// If a cancel flipped it to `cancelling` meanwhile, no run exists to observe
/// that, so finish the cancel here instead of stranding the run.
async fn restore_gate<S: EventStore>(store: &S, run_id: &str) {
    use flux_session::RunStatus;
    if let Ok(false) = store
        .compare_and_swap_status(run_id, RunStatus::Running, RunStatus::WaitingApproval)
        .await
    {
        if let Ok(true) = store
            .compare_and_swap_status(run_id, RunStatus::Cancelling, RunStatus::Cancelled)
            .await
        {
            let _ = store
                .append(run_id, &WorkflowEvent::WorkflowCancelled)
                .await;
        }
    }
}

/// POST /runs/{run_id}/cancel — stop a run. A paused run is cancelled at once;
/// a running one is marked `cancelling` and its launcher stops it (see
/// `launcher::supervise`), whichever server instance runs it.
pub(crate) async fn cancel<S, L>(
    State(state): State<AppState<S, L>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    use flux_session::RunStatus;
    let accepted = || {
        (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({ "run_id": run_id })),
        )
            .into_response()
    };
    // A run can move between running and waiting_approval while we try, so
    // retry the pair of swaps while it is still live rather than misreport a
    // just-paused run as finished.
    for _ in 0..3 {
        // The same CAS approve uses, so a racing approve and cancel cannot
        // both claim the gate.
        match state
            .store
            .compare_and_swap_status(&run_id, RunStatus::WaitingApproval, RunStatus::Cancelled)
            .await
        {
            Ok(true) => {
                if state
                    .store
                    .append(&run_id, &WorkflowEvent::WorkflowCancelled)
                    .await
                    .is_err()
                {
                    tracing::warn!(run_id = %run_id, "cancelled a paused run but could not log WorkflowCancelled");
                }
                return accepted();
            }
            Ok(false) => {}
            Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
        }
        match state
            .store
            .compare_and_swap_status(&run_id, RunStatus::Running, RunStatus::Cancelling)
            .await
        {
            Ok(true) => return accepted(),
            Ok(false) => {}
            Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
        }
        match state.store.get_status(&run_id).await {
            Ok(None) => return error_response(StatusCode::NOT_FOUND, "unknown run_id"),
            Ok(Some(RunStatus::Cancelling)) => return accepted(),
            Ok(Some(RunStatus::Running | RunStatus::WaitingApproval)) => continue,
            Ok(Some(_)) => return error_response(StatusCode::CONFLICT, "run has already finished"),
            Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
        }
    }
    error_response(
        StatusCode::CONFLICT,
        "run status kept changing; retry the cancel",
    )
}

/// POST /runs/{run_id}/approve — record the approval and resume the run.
pub(crate) async fn approve<S, L>(
    State(state): State<AppState<S, L>>,
    Path(run_id): Path<String>,
    body: Result<Json<ApproveBody>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid request body: {}", rej.body_text()),
            )
        }
    };
    // Validate the body BEFORE the CAS: a 400 must leave the gate intact and
    // consume nothing.
    if let Some(data) = &req.data {
        if !data.is_object() {
            return error_response(
                StatusCode::BAD_REQUEST,
                "approve `data` must be a JSON object",
            );
        }
    }
    // Single-flight gate: atomically claim the run for resume. Of any number of
    // concurrent approves, only one wins the WaitingApproval -> Running swap;
    // the rest fall through to a 404/409 below. This closes the prior TOCTOU
    // between a status read and a separate set_status(Running).
    match state
        .store
        .compare_and_swap_status(
            &run_id,
            flux_session::RunStatus::WaitingApproval,
            flux_session::RunStatus::Running,
        )
        .await
    {
        Ok(true) => {}
        Ok(false) => {
            return match state.store.get_status(&run_id).await {
                Ok(None) => error_response(StatusCode::NOT_FOUND, "unknown run_id"),
                Ok(Some(_)) => error_response(StatusCode::CONFLICT, "run is not awaiting approval"),
                Err(_) => error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
            };
        }
        Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "store error"),
    }
    match prepare_resume(&state, &run_id, req).await {
        Ok((workflow, input, conversation_id, merged, resume)) => {
            let launcher = state.launcher.clone();
            let store = state.store.clone();
            let spawned_id = run_id.clone();
            tokio::spawn(async move {
                launcher
                    .launch(
                        store,
                        spawned_id,
                        workflow,
                        input,
                        conversation_id,
                        merged,
                        resume,
                    )
                    .await;
            });
            (
                StatusCode::ACCEPTED,
                Json(serde_json::json!({ "run_id": run_id })),
            )
                .into_response()
        }
        Err(ResumeFailure::Expired) => {
            // The paused run's Redis state (events and/or manifest) is gone —
            // TTL clocks differ per key, so meta can outlive them. Resuming
            // would re-execute completed stages (duplicate LLM calls and tool
            // side effects) with no record. Terminal-fail the run instead.
            let _ = state
                .store
                .set_status(&run_id, flux_session::RunStatus::Failed)
                .await;
            error_response(StatusCode::GONE, "run state expired; submit a new run")
        }
        Err(ResumeFailure::Rejected(msg)) => {
            restore_gate(&state.store, &run_id).await;
            error_response(StatusCode::BAD_REQUEST, msg)
        }
        Err(ResumeFailure::NotWaiting(msg)) => {
            restore_gate(&state.store, &run_id).await;
            error_response(StatusCode::CONFLICT, msg)
        }
        Err(ResumeFailure::Internal(msg)) => {
            // Put the gate back so a retry can succeed.
            restore_gate(&state.store, &run_id).await;
            error_response(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::body_hash;
    use crate::{router, AppState, RunLauncher};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use flux_session::{EventStore, MemoryStore, RunStatus};
    use flux_workflow::Workflow;
    use http_body_util::BodyExt;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tower::ServiceExt; // oneshot

    #[test]
    fn body_hash_ignores_key_order_at_every_depth() {
        let a = serde_json::json!({"x": 1, "opts": {"p": [1, {"b": 1, "a": 2}], "q": null}});
        let b: serde_json::Value =
            serde_json::from_str(r#"{"opts":{"q":null,"p":[1,{"a":2,"b":1}]},"x":1}"#).unwrap();
        assert_eq!(body_hash(&a), body_hash(&b));
        let c = serde_json::json!({"x": 1, "opts": {"p": [1, {"b": 1, "a": 3}], "q": null}});
        assert_ne!(body_hash(&a), body_hash(&c));
        // Array order is significant.
        assert_ne!(
            body_hash(&serde_json::json!([1, 2])),
            body_hash(&serde_json::json!([2, 1]))
        );
    }

    #[derive(Clone)]
    struct FakeLauncher;

    impl RunLauncher<MemoryStore> for FakeLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            use flux_core::workflow_event::WorkflowEvent;
            let _ = store
                .append(
                    &run_id,
                    &WorkflowEvent::StageStarted {
                        stage: "extract".into(),
                    },
                )
                .await;
            let _ = store
                .append(&run_id, &WorkflowEvent::WorkflowComplete)
                .await;
            let _ = store.set_status(&run_id, RunStatus::Complete).await;
        }
    }

    #[derive(Clone)]
    struct CountingLauncher(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl RunLauncher<MemoryStore> for CountingLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = store.set_status(&run_id, RunStatus::Complete).await;
        }
    }

    fn test_workflow() -> Workflow {
        let yaml = "workflow_id: triage\nstages:\n  - id: extract\n    agent: classify\n    input: \"{{ workflow.input }}\"\n";
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    fn state() -> AppState<MemoryStore, FakeLauncher> {
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        AppState {
            store: MemoryStore::default(),
            launcher: FakeLauncher,
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        }
    }

    #[tokio::test]
    async fn post_runs_unknown_workflow_is_404() {
        let app = router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"ghost","input":"x"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn post_runs_bad_body_is_400() {
        let app = router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"nope":true}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_runs_valid_is_202_with_run_id() {
        let app = router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"triage","input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v.get("run_id").and_then(|r| r.as_str()).is_some());
    }

    #[tokio::test]
    async fn post_runs_with_conversation_id_persists_it() {
        let st = state();
        let store_probe = st.store.clone();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"workflow_id":"triage","input":"hi","conversation_id":"conv-1"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let run_id = v
            .get("run_id")
            .and_then(|r| r.as_str())
            .unwrap()
            .to_string();
        // set_conversation_id runs synchronously in the handler before the
        // response returns, so it is observable immediately.
        assert_eq!(
            store_probe.get_conversation_id(&run_id).await.unwrap(),
            Some("conv-1".to_string())
        );
    }

    #[tokio::test]
    async fn post_runs_with_context_persists_it() {
        let st = state();
        let store_probe = st.store.clone();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"workflow_id":"triage","input":"hi","context":{"projectId":"p1"}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let run_id = v
            .get("run_id")
            .and_then(|r| r.as_str())
            .unwrap()
            .to_string();
        assert_eq!(
            store_probe.get_context(&run_id).await.unwrap(),
            Some(serde_json::json!({ "projectId": "p1" }))
        );
    }

    #[tokio::test]
    async fn approve_unknown_run_is_404() {
        let app = router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/runs/nope/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"act"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn approve_run_not_waiting_is_409() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::Running)
            .await
            .unwrap();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"act"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn approve_bad_body_is_400() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"nope":true}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn approve_waiting_run_is_202_and_records_approval() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        st.store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let store_probe = st.store.clone();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert!(store_probe
            .approved_stages("run1")
            .await
            .unwrap()
            .contains("extract"));
    }

    #[tokio::test]
    async fn approve_of_a_stage_that_is_not_waiting_is_409_and_the_gate_survives() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        st.store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let store_probe = st.store.clone();
        let app = router(st);
        // Pre-approving a later gate would let it run with no human asked.
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"act"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert!(store_probe
            .approved_stages("run1")
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            store_probe.get_status("run1").await.unwrap(),
            Some(RunStatus::WaitingApproval)
        );
    }

    #[tokio::test]
    async fn concurrent_approves_resume_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        store.set_manifest("run1", "triage", "hello").await.unwrap();
        store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let app = router(AppState {
            store: store.clone(),
            launcher: CountingLauncher(counter.clone()),
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        });
        let mk = || {
            Request::builder()
                .method("POST")
                .uri("/runs/run1/approve")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"stage_id":"extract"}"#))
                .unwrap()
        };
        // Two simultaneous approves against the same shared store.
        let (r1, r2) = tokio::join!(app.clone().oneshot(mk()), app.clone().oneshot(mk()));
        let mut codes = [r1.unwrap().status(), r2.unwrap().status()];
        codes.sort_by_key(|c| c.as_u16());
        assert_eq!(codes[0], StatusCode::ACCEPTED);
        assert_eq!(codes[1], StatusCode::CONFLICT);
        // The winner spawns the launcher; let the spawned task run. The loser
        // never spawns, so the counter can only ever reach 1.
        for _ in 0..1000 {
            if counter.load(Ordering::SeqCst) >= 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[derive(Clone)]
    struct ContextRecordingLauncher(std::sync::Arc<tokio::sync::Mutex<Option<serde_json::Value>>>);

    impl RunLauncher<MemoryStore> for ContextRecordingLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            *self.0.lock().await = context;
            let _ = store.set_status(&run_id, RunStatus::Complete).await;
        }
    }

    #[tokio::test]
    async fn approve_threads_stored_context_into_resume() {
        use serde_json::json;
        let recorded = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        store.set_manifest("run1", "triage", "hi").await.unwrap();
        store
            .set_context("run1", &json!({ "projectId": "p1" }))
            .await
            .unwrap();
        store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let app = router(AppState {
            store: store.clone(),
            launcher: ContextRecordingLauncher(recorded.clone()),
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        // The approve handler spawns the resumed launch; let the spawned task
        // run, then read back the context it actually received.
        for _ in 0..1000 {
            if recorded.lock().await.is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(*recorded.lock().await, Some(json!({ "projectId": "p1" })));
    }

    #[test]
    fn final_reply_prefers_the_output_stage() {
        use flux_core::output::AgentOutput;
        use flux_core::workflow_event::WorkflowEvent;
        let done = |stage: &str, msg: &str| WorkflowEvent::StageComplete {
            stage: stage.into(),
            output: AgentOutput {
                message: msg.into(),
                structured: None,
            },
        };
        let events = vec![(1, done("narrate", "answer")), (2, done("audit", "logged"))];
        assert_eq!(
            super::final_reply(&events, Some("narrate")).unwrap().0,
            "answer"
        );
        assert_eq!(super::final_reply(&events, None).unwrap().0, "logged");
        // output stage never completed (skipped) -> last StageComplete
        assert_eq!(
            super::final_reply(&events, Some("missing")).unwrap().0,
            "logged"
        );
    }

    #[test]
    fn merge_context_data_none_passes_stored_through() {
        let stored = Some(serde_json::json!({ "a": 1 }));
        assert_eq!(super::merge_context(stored.clone(), None).unwrap(), stored);
    }

    #[test]
    fn merge_context_stored_none_adopts_data() {
        let data = serde_json::json!({ "a": 1 });
        assert_eq!(
            super::merge_context(None, Some(data.clone())).unwrap(),
            Some(data)
        );
    }

    #[test]
    fn merge_context_adds_new_keys_and_keeps_stored() {
        let stored = serde_json::json!({ "a": 1, "keep": 2 });
        let data = serde_json::json!({ "b": 3 });
        let merged = super::merge_context(Some(stored), Some(data))
            .unwrap()
            .unwrap();
        assert_eq!(merged["a"], 1);
        assert_eq!(merged["keep"], 2);
        assert_eq!(merged["b"], 3);
    }

    #[test]
    fn merge_context_rejects_changing_an_existing_key() {
        // F2: approval data must not rewrite run context (e.g. a tenant id).
        let stored = serde_json::json!({ "account_id": "t1" });
        let data = serde_json::json!({ "account_id": "t2" });
        let err = super::merge_context(Some(stored), Some(data)).unwrap_err();
        assert!(err.contains("account_id"), "{err}");
    }

    #[test]
    fn merge_context_allows_repeating_an_identical_value() {
        let stored = serde_json::json!({ "account_id": "t1" });
        let data = serde_json::json!({ "account_id": "t1", "note": "ok" });
        let merged = super::merge_context(Some(stored), Some(data))
            .unwrap()
            .unwrap();
        assert_eq!(merged["account_id"], "t1");
        assert_eq!(merged["note"], "ok");
    }

    #[test]
    fn merge_context_non_object_data_errors() {
        assert!(super::merge_context(None, Some(serde_json::json!(5))).is_err());
    }

    #[test]
    fn merge_context_non_object_stored_with_data_errors() {
        let stored = Some(serde_json::json!("not-an-object"));
        let data = Some(serde_json::json!({ "a": 1 }));
        assert!(super::merge_context(stored, data).is_err());
    }

    #[tokio::test]
    async fn approve_with_data_merges_into_context_and_persists() {
        use serde_json::json;
        let recorded = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        store.set_manifest("run1", "triage", "hi").await.unwrap();
        store
            .set_context("run1", &json!({ "a": 1, "keep": 2 }))
            .await
            .unwrap();
        store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let store_probe = store.clone();
        let app = router(AppState {
            store: store.clone(),
            launcher: ContextRecordingLauncher(recorded.clone()),
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract","data":{"b":3}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        for _ in 0..1000 {
            if recorded.lock().await.is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let got = recorded.lock().await.clone().unwrap();
        assert_eq!(got["a"], 1); // stored preserved
        assert_eq!(got["keep"], 2); // stored preserved
        assert_eq!(got["b"], 3); // data added
                                 // merge was persisted
        assert_eq!(
            store_probe.get_context("run1").await.unwrap(),
            Some(json!({ "a": 1, "keep": 2, "b": 3 }))
        );
    }

    #[tokio::test]
    async fn approve_with_conflicting_context_key_is_400_and_stays_waiting() {
        use serde_json::json;
        let recorded = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        store.set_manifest("run1", "triage", "hi").await.unwrap();
        store
            .set_context("run1", &json!({ "account_id": "t1" }))
            .await
            .unwrap();
        store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let store_probe = store.clone();
        let app = router(AppState {
            store: store.clone(),
            launcher: ContextRecordingLauncher(recorded.clone()),
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"stage_id":"extract","data":{"account_id":"t2"}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("account_id"));
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(recorded.lock().await.is_none(), "run must not resume");
        assert_eq!(
            store_probe.get_status("run1").await.unwrap(),
            Some(RunStatus::WaitingApproval)
        );
        assert_eq!(
            store_probe.get_context("run1").await.unwrap(),
            Some(json!({ "account_id": "t1" }))
        );
        // A refused approval must not count as an approval of the stage.
        assert!(store_probe
            .approved_stages("run1")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn approve_with_non_object_data_is_400() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract","data":5}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn approve_with_bad_data_is_400_and_the_gate_survives_for_a_retry() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        // The run must have at least one event: a paused run always logged its
        // StageWaitingApproval, and the expired-state guard keys on emptiness.
        st.store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let app = router(st.clone());
        let bad = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract","data":5}"#))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(bad).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        // The 400 must not have consumed the gate: a corrected retry succeeds.
        let good = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract","data":{"a":1}}"#))
            .unwrap();
        assert_eq!(
            app.oneshot(good).await.unwrap().status(),
            StatusCode::ACCEPTED
        );
    }

    #[tokio::test]
    async fn approve_on_expired_run_state_is_410_and_marks_the_run_failed() {
        // Meta key survived TTL but the event log did not: resuming would
        // silently re-execute every stage from scratch. Refuse loudly instead.
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        // no events appended — the expired-log shape
        let store_probe = st.store.clone();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::GONE);
        assert_eq!(
            store_probe.get_status("run1").await.unwrap(),
            Some(RunStatus::Failed)
        );
    }

    #[tokio::test]
    async fn approve_restores_the_gate_when_resume_preparation_fails() {
        // Manifest names a workflow the server does not have: a 500, after which
        // the run must be back in WaitingApproval (not stuck Running forever).
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "ghost", "hi").await.unwrap();
        st.store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let store_probe = st.store.clone();
        let app = router(st);
        let req = Request::builder()
            .method("POST")
            .uri("/runs/run1/approve")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"stage_id":"extract"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            store_probe.get_status("run1").await.unwrap(),
            Some(RunStatus::WaitingApproval)
        );
    }

    #[test]
    fn final_reply_takes_last_stage_complete() {
        use flux_core::output::AgentOutput;
        use flux_core::workflow_event::WorkflowEvent;
        let events = vec![
            (
                1u64,
                WorkflowEvent::StageComplete {
                    stage: "a".into(),
                    output: AgentOutput {
                        message: "first".into(),
                        structured: None,
                    },
                },
            ),
            (
                2u64,
                WorkflowEvent::StageComplete {
                    stage: "b".into(),
                    output: AgentOutput {
                        message: "last".into(),
                        structured: Some(serde_json::json!({"k":1})),
                    },
                },
            ),
            (3u64, WorkflowEvent::WorkflowComplete),
        ];
        let (msg, structured) = super::final_reply(&events, None).unwrap();
        assert_eq!(msg, "last");
        assert_eq!(structured, Some(serde_json::json!({"k":1})));
    }

    #[test]
    fn final_reply_none_without_stage_complete() {
        use flux_core::workflow_event::WorkflowEvent;
        let events = vec![(1u64, WorkflowEvent::StageStarted { stage: "a".into() })];
        assert!(super::final_reply(&events, None).is_none());
    }

    #[derive(Clone)]
    struct ChatCompleteLauncher;
    impl RunLauncher<MemoryStore> for ChatCompleteLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            use flux_core::output::AgentOutput;
            use flux_core::workflow_event::WorkflowEvent;
            let _ = store
                .append(
                    &run_id,
                    &WorkflowEvent::StageComplete {
                        stage: "reply".into(),
                        output: AgentOutput {
                            message: "hi-back".into(),
                            structured: None,
                        },
                    },
                )
                .await;
            let _ = store.set_status(&run_id, RunStatus::Complete).await;
        }
    }

    #[derive(Clone)]
    struct ChatWaitingLauncher;
    impl RunLauncher<MemoryStore> for ChatWaitingLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            use flux_core::workflow_event::WorkflowEvent;
            let _ = store
                .append(
                    &run_id,
                    &WorkflowEvent::StageWaitingApproval {
                        stage: "gate".into(),
                    },
                )
                .await;
            let _ = store.set_status(&run_id, RunStatus::WaitingApproval).await;
        }
    }

    /// Like `ChatCompleteLauncher`, after a delay; records every run_id it launched.
    #[derive(Clone, Default)]
    struct SlowChatLauncher(Arc<std::sync::Mutex<Vec<String>>>);

    impl RunLauncher<MemoryStore> for SlowChatLauncher {
        async fn launch(
            &self,
            store: MemoryStore,
            run_id: String,
            workflow: Workflow,
            input: String,
            conversation_id: Option<String>,
            context: Option<serde_json::Value>,
            resume: flux_orchestrator::ResumeState,
        ) {
            self.0.lock().unwrap().push(run_id.clone());
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            ChatCompleteLauncher
                .launch(
                    store,
                    run_id,
                    workflow,
                    input,
                    conversation_id,
                    context,
                    resume,
                )
                .await;
        }
    }

    #[tokio::test]
    async fn a_chat_run_survives_the_client_disconnecting() {
        let launcher = SlowChatLauncher::default();
        let st = chat_state(launcher.clone());
        let probe = st.store.clone();
        let app = router(st);
        // The client gives up long before the run finishes.
        let gave_up = tokio::time::timeout(
            std::time::Duration::from_millis(20),
            app.oneshot(post("/chat", r#"{"workflow_id":"triage","input":"hi"}"#)),
        )
        .await;
        assert!(gave_up.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        let run_id = launcher.0.lock().unwrap()[0].clone();
        assert_eq!(
            probe.get_status(&run_id).await.unwrap(),
            Some(RunStatus::Complete)
        );
    }

    fn keyed(uri: &str, body: &str, key: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("idempotency-key", key)
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn json_of(resp: axum::response::Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    const RUN_BODY: &str = r#"{"workflow_id":"triage","input":"hi"}"#;

    fn counting_state() -> (
        AppState<MemoryStore, CountingLauncher>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let st = AppState {
            store: MemoryStore::default(),
            launcher: CountingLauncher(counter.clone()),
            workflows: Arc::new(HashMap::from([("triage".to_string(), test_workflow())])),
            tail_poll_ms: 10,
        };
        (st, counter)
    }

    async fn settle() {
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn runs_retry_with_the_same_key_returns_the_original_run() {
        let (st, counter) = counting_state();
        let app = router(st);
        let (s1, b1) = json_of(
            app.clone()
                .oneshot(keyed("/runs", RUN_BODY, "k1"))
                .await
                .unwrap(),
        )
        .await;
        let (s2, b2) = json_of(app.oneshot(keyed("/runs", RUN_BODY, "k1")).await.unwrap()).await;
        settle().await;
        assert_eq!(s1, StatusCode::ACCEPTED);
        assert_eq!(s2, StatusCode::OK);
        assert_eq!(b1["run_id"], b2["run_id"]);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn concurrent_runs_with_one_key_start_one_run() {
        let (st, counter) = counting_state();
        let app = router(st);
        let (r1, r2) = tokio::join!(
            app.clone().oneshot(keyed("/runs", RUN_BODY, "k1")),
            app.clone().oneshot(keyed("/runs", RUN_BODY, "k1")),
        );
        let (_, b1) = json_of(r1.unwrap()).await;
        let (_, b2) = json_of(r2.unwrap()).await;
        settle().await;
        assert_eq!(b1["run_id"], b2["run_id"]);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_key_reused_with_a_different_body_is_422() {
        let (st, counter) = counting_state();
        let app = router(st);
        app.clone()
            .oneshot(keyed("/runs", RUN_BODY, "k1"))
            .await
            .unwrap();
        let other = r#"{"workflow_id":"triage","input":"something else"}"#;
        let (status, body) = json_of(app.oneshot(keyed("/runs", other, "k1")).await.unwrap()).await;
        settle().await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["error"].as_str().unwrap().contains("Idempotency-Key"));
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_invalid_idempotency_key_is_400() {
        let (st, counter) = counting_state();
        let app = router(st);
        let long = "k".repeat(256);
        for key in ["", long.as_str()] {
            let resp = app
                .clone()
                .oneshot(keyed("/runs", RUN_BODY, key))
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::BAD_REQUEST,
                "key len {}",
                key.len()
            );
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// MemoryStore with injectable trouble: the next `set_manifest` fails
    /// once (`fail_next`), every `set_manifest` takes `slow_manifest_ms`, and
    /// the next running→cancelling CAS finds the run just paused
    /// (`pause_before_cancel`).
    #[derive(Clone, Default)]
    struct FlakyManifestStore {
        inner: MemoryStore,
        fail_next: Arc<std::sync::atomic::AtomicBool>,
        slow_manifest_ms: Arc<std::sync::atomic::AtomicU64>,
        pause_before_cancel: Arc<std::sync::atomic::AtomicBool>,
    }

    impl EventStore for FlakyManifestStore {
        async fn append(
            &self,
            run_id: &str,
            event: &flux_core::workflow_event::WorkflowEvent,
        ) -> flux_core::error::Result<flux_session::Offset> {
            self.inner.append(run_id, event).await
        }
        async fn read_after(
            &self,
            run_id: &str,
            after: flux_session::Offset,
        ) -> flux_core::error::Result<
            Vec<(
                flux_session::Offset,
                flux_core::workflow_event::WorkflowEvent,
            )>,
        > {
            self.inner.read_after(run_id, after).await
        }
        async fn read_tail(
            &self,
            run_id: &str,
            count: usize,
        ) -> flux_core::error::Result<
            Vec<(
                flux_session::Offset,
                flux_core::workflow_event::WorkflowEvent,
            )>,
        > {
            self.inner.read_tail(run_id, count).await
        }
        async fn set_status(
            &self,
            run_id: &str,
            status: RunStatus,
        ) -> flux_core::error::Result<()> {
            self.inner.set_status(run_id, status).await
        }
        async fn get_status(&self, run_id: &str) -> flux_core::error::Result<Option<RunStatus>> {
            self.inner.get_status(run_id).await
        }
        async fn compare_and_swap_status(
            &self,
            run_id: &str,
            expected: RunStatus,
            new: RunStatus,
        ) -> flux_core::error::Result<bool> {
            if expected == RunStatus::Running
                && new == RunStatus::Cancelling
                && self
                    .pause_before_cancel
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                self.inner
                    .set_status(run_id, RunStatus::WaitingApproval)
                    .await?;
            }
            self.inner
                .compare_and_swap_status(run_id, expected, new)
                .await
        }
        async fn record_approval(&self, run_id: &str, stage: &str) -> flux_core::error::Result<()> {
            self.inner.record_approval(run_id, stage).await
        }
        async fn approved_stages(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<std::collections::HashSet<String>> {
            self.inner.approved_stages(run_id).await
        }
        async fn set_manifest(
            &self,
            run_id: &str,
            workflow_id: &str,
            input: &str,
        ) -> flux_core::error::Result<()> {
            if self
                .fail_next
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(flux_core::error::FluxError::Config("redis blip".into()));
            }
            let slow = self
                .slow_manifest_ms
                .load(std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(slow)).await;
            self.inner.set_manifest(run_id, workflow_id, input).await
        }
        async fn get_manifest(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<(String, String)>> {
            self.inner.get_manifest(run_id).await
        }
        async fn set_conversation_id(
            &self,
            run_id: &str,
            conversation_id: &str,
        ) -> flux_core::error::Result<()> {
            self.inner
                .set_conversation_id(run_id, conversation_id)
                .await
        }
        async fn get_conversation_id(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<String>> {
            self.inner.get_conversation_id(run_id).await
        }
        async fn set_context(
            &self,
            run_id: &str,
            context: &serde_json::Value,
        ) -> flux_core::error::Result<()> {
            self.inner.set_context(run_id, context).await
        }
        async fn get_context(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<serde_json::Value>> {
            self.inner.get_context(run_id).await
        }
        async fn claim_idempotency_key(
            &self,
            scope: &str,
            key: &str,
            run_id: &str,
            body_hash: &str,
        ) -> flux_core::error::Result<flux_session::IdempotencyClaim> {
            self.inner
                .claim_idempotency_key(scope, key, run_id, body_hash)
                .await
        }
        async fn release_idempotency_key(
            &self,
            scope: &str,
            key: &str,
        ) -> flux_core::error::Result<()> {
            self.inner.release_idempotency_key(scope, key).await
        }
    }

    #[derive(Clone)]
    struct AnyStoreCountingLauncher(Arc<std::sync::atomic::AtomicUsize>);

    impl<S: EventStore + Clone + Send + Sync + 'static> RunLauncher<S> for AnyStoreCountingLauncher {
        async fn launch(
            &self,
            _store: S,
            _run_id: String,
            _workflow: Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn flaky_app(store: FlakyManifestStore) -> (axum::Router, Arc<std::sync::atomic::AtomicUsize>) {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = router(AppState {
            store,
            launcher: AnyStoreCountingLauncher(counter.clone()),
            workflows: Arc::new(HashMap::from([("triage".to_string(), test_workflow())])),
            tail_poll_ms: 10,
        });
        (app, counter)
    }

    #[tokio::test]
    async fn a_disconnect_during_run_setup_still_starts_the_run() {
        // The client gives up while the run's state is being written: the
        // key is already claimed, so abandoning setup would bind it to a run
        // that never starts.
        let store = FlakyManifestStore::default();
        store
            .slow_manifest_ms
            .store(100, std::sync::atomic::Ordering::SeqCst);
        let (app, counter) = flaky_app(store);
        let gave_up = tokio::time::timeout(
            std::time::Duration::from_millis(20),
            app.oneshot(keyed("/runs", RUN_BODY, "k1")),
        )
        .await;
        assert!(gave_up.is_err());
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancel_of_a_run_that_pauses_meanwhile_cancels_it() {
        let store = FlakyManifestStore::default();
        store.set_status("run1", RunStatus::Running).await.unwrap();
        store
            .pause_before_cancel
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let probe = store.clone();
        let (app, _) = flaky_app(store);
        let resp = app.oneshot(post("/runs/run1/cancel", "")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            probe.get_status("run1").await.unwrap(),
            Some(RunStatus::Cancelled)
        );
    }

    #[tokio::test]
    async fn a_key_whose_run_failed_to_start_is_released() {
        let store = FlakyManifestStore::default();
        store
            .fail_next
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = router(AppState {
            store,
            launcher: AnyStoreCountingLauncher(counter.clone()),
            workflows: Arc::new(HashMap::from([("triage".to_string(), test_workflow())])),
            tail_poll_ms: 10,
        });
        let first = app
            .clone()
            .oneshot(keyed("/runs", RUN_BODY, "k1"))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let retry = app.oneshot(keyed("/runs", RUN_BODY, "k1")).await.unwrap();
        assert_eq!(retry.status(), StatusCode::ACCEPTED);
        settle().await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chat_retry_while_running_waits_for_the_original() {
        let launcher = SlowChatLauncher::default();
        let mut st = chat_state(launcher.clone());
        st.tail_poll_ms = 10;
        let app = router(st);
        let (r1, r2) = tokio::join!(
            app.clone().oneshot(keyed("/chat", RUN_BODY, "c1")),
            app.clone().oneshot(keyed("/chat", RUN_BODY, "c1")),
        );
        let (s1, b1) = json_of(r1.unwrap()).await;
        let (s2, b2) = json_of(r2.unwrap()).await;
        assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
        assert_eq!(b1["run_id"], b2["run_id"]);
        assert_eq!(b1["reply"], "hi-back");
        assert_eq!(b1["reply"], b2["reply"]);
        assert_eq!(launcher.0.lock().unwrap().len(), 1);
    }

    fn chat_state<L: RunLauncher<MemoryStore>>(launcher: L) -> AppState<MemoryStore, L> {
        let mut wfs = HashMap::new();
        wfs.insert("triage".to_string(), test_workflow());
        AppState {
            store: MemoryStore::default(),
            launcher,
            workflows: Arc::new(wfs),
            tail_poll_ms: 250,
        }
    }

    #[tokio::test]
    async fn chat_complete_returns_reply() {
        let app = router(chat_state(ChatCompleteLauncher));
        let req = Request::builder()
            .method("POST")
            .uri("/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"triage","input":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "complete");
        assert_eq!(v["reply"], "hi-back");
        assert!(v.get("run_id").and_then(|r| r.as_str()).is_some());
    }

    #[tokio::test]
    async fn chat_waiting_approval_has_no_reply() {
        let app = router(chat_state(ChatWaitingLauncher));
        let req = Request::builder()
            .method("POST")
            .uri("/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"triage","input":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "waiting_approval");
        assert!(v.get("reply").is_none());
    }

    #[tokio::test]
    async fn chat_unknown_workflow_is_404() {
        let app = router(chat_state(ChatCompleteLauncher));
        let req = Request::builder()
            .method("POST")
            .uri("/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"ghost","input":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_bad_body_is_400() {
        let app = router(chat_state(ChatCompleteLauncher));
        let req = Request::builder()
            .method("POST")
            .uri("/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"nope":true}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    fn post(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn paused_state() -> AppState<MemoryStore, FakeLauncher> {
        let st = state();
        st.store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        st.store.set_manifest("run1", "triage", "hi").await.unwrap();
        st.store
            .append(
                "run1",
                &flux_core::workflow_event::WorkflowEvent::StageWaitingApproval {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        st
    }

    #[tokio::test]
    async fn restoring_a_gate_that_was_cancelled_meanwhile_ends_the_run_cancelled() {
        // approve swapped the gate to running, its preparation failed, and a
        // cancel flipped running -> cancelling in between: nothing is running,
        // so leaving it cancelling would strand it.
        let st = state();
        st.store
            .set_status("run1", RunStatus::Cancelling)
            .await
            .unwrap();
        super::restore_gate(&st.store, "run1").await;
        assert_eq!(
            st.store.get_status("run1").await.unwrap(),
            Some(RunStatus::Cancelled)
        );
    }

    #[tokio::test]
    async fn cancel_unknown_run_is_404() {
        let resp = router(state())
            .oneshot(post("/runs/nope/cancel", ""))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn cancel_finished_run_is_409() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::Complete)
            .await
            .unwrap();
        let resp = router(st)
            .oneshot(post("/runs/run1/cancel", ""))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn cancel_paused_run_is_202_and_cancelled() {
        let st = paused_state().await;
        let probe = st.store.clone();
        let app = router(st);
        let resp = app
            .clone()
            .oneshot(post("/runs/run1/cancel", ""))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            probe.get_status("run1").await.unwrap(),
            Some(RunStatus::Cancelled)
        );
        let log = probe.read_after("run1", 0).await.unwrap();
        assert_eq!(
            log.last().map(|(_, e)| e),
            Some(&flux_core::workflow_event::WorkflowEvent::WorkflowCancelled)
        );
        let resp = app
            .oneshot(post("/runs/run1/approve", r#"{"stage_id":"extract"}"#))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn cancel_and_approve_race_has_one_winner() {
        let app = router(paused_state().await);
        let (c, a) = tokio::join!(
            app.clone().oneshot(post("/runs/run1/cancel", "")),
            app.clone()
                .oneshot(post("/runs/run1/approve", r#"{"stage_id":"extract"}"#)),
        );
        let c = c.unwrap().status();
        let a = a.unwrap().status();
        // Cancel may also win after approve by flipping the resumed run to
        // cancelling (202), so the invariant is: approve and a cancel of the
        // paused run never both claim the gate.
        assert!(c.is_success() || a.is_success(), "{c} {a}");
        if a.is_success() {
            assert!(
                c == StatusCode::ACCEPTED || c == StatusCode::CONFLICT,
                "{c}"
            );
        } else {
            assert_eq!(a, StatusCode::CONFLICT);
        }
    }

    #[tokio::test]
    async fn cancel_running_run_sets_cancelling() {
        let st = state();
        st.store
            .set_status("run1", RunStatus::Running)
            .await
            .unwrap();
        let probe = st.store.clone();
        let resp = router(st)
            .oneshot(post("/runs/run1/cancel", ""))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            probe.get_status("run1").await.unwrap(),
            Some(RunStatus::Cancelling)
        );
    }

    async fn get_status_body(status: RunStatus) -> serde_json::Value {
        let st = state();
        st.store.set_status("run1", status).await.unwrap();
        let resp = router(st)
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/runs/run1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn get_run_reports_cancelled() {
        assert_eq!(
            get_status_body(RunStatus::Cancelled).await["status"],
            "cancelled"
        );
    }

    #[tokio::test]
    async fn get_run_reports_cancelling() {
        assert_eq!(
            get_status_body(RunStatus::Cancelling).await["status"],
            "cancelling"
        );
    }

    #[tokio::test]
    async fn get_run_unknown_is_404_with_json_error() {
        let app = router(state());
        let req = Request::builder()
            .method("GET")
            .uri("/runs/nope")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["error"].is_string());
    }

    #[tokio::test]
    async fn get_run_after_completion_returns_status_and_reply() {
        let st = chat_state(ChatCompleteLauncher);
        let app = router(st);
        // drive a run to completion synchronously via /chat
        let req = Request::builder()
            .method("POST")
            .uri("/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"workflow_id":"triage","input":"hi"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let run_id = v["run_id"].as_str().unwrap();

        let req = Request::builder()
            .method("GET")
            .uri(format!("/runs/{run_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"], "complete");
        assert_eq!(v["reply"], "hi-back");
    }

    #[tokio::test]
    async fn list_workflows_returns_ids() {
        let app = router(state());
        let req = Request::builder()
            .method("GET")
            .uri("/workflows")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["workflows"], serde_json::json!(["triage"]));
    }

    #[tokio::test]
    async fn bad_body_400_carries_the_rejection_detail() {
        let app = router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/runs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"nope":true}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let msg = v["error"].as_str().unwrap();
        assert!(
            msg.contains("workflow_id"),
            "serde's field detail must survive: {msg}"
        );
    }
}
