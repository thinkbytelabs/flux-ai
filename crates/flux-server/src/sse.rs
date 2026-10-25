use crate::{AppState, RunLauncher};
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use flux_session::{is_terminal, EventStore, Offset};
use std::convert::Infallible;
use tokio_stream::Stream;

/// Produce an SSE event stream for a run: replay events with offset > `after`,
/// then poll-tail new ones; end the stream after emitting a terminal event.
pub(crate) fn event_stream<S>(
    store: S,
    run_id: String,
    after: Offset,
    tail_poll_ms: u64,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: EventStore + Clone + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let mut offset = after;
        loop {
            match store.read_after(&run_id, offset).await {
                Ok(batch) if !batch.is_empty() => {
                    for (off, event) in batch {
                        offset = off;
                        let terminal = is_terminal(&event);
                        let data = match serde_json::to_string(&event) {
                            Ok(s) => s,
                            Err(_) => continue, // skip un-serializable event
                        };
                        let sse = Event::default().id(off.to_string()).data(data);
                        if tx.send(Ok(sse)).await.is_err() {
                            return; // client disconnected
                        }
                        if terminal {
                            return; // stream ends after the terminal event
                        }
                    }
                }
                Ok(_) => {
                    // No new events. If the run already reached a terminal status
                    // but emitted no terminal event, or its status expired (TTL),
                    // nothing more will come: stop rather than loop forever.
                    // Otherwise wait and poll again.
                    match store.get_status(&run_id).await {
                        Ok(Some(flux_session::RunStatus::Complete))
                        | Ok(Some(flux_session::RunStatus::Failed))
                        | Ok(Some(flux_session::RunStatus::Cancelled))
                        | Ok(None) => match store.read_after(&run_id, offset).await {
                            Ok(tail) if tail.is_empty() => return,
                            _ => {}
                        },
                        _ => {}
                    }
                    // Wait out the poll interval — but leave the moment every
                    // receiver is gone. An idle tail (a run parked at an
                    // approval gate) never sends, so a dropped client was
                    // otherwise never detected: one leaked polling task per
                    // SSE connection ever opened on such a run.
                    tokio::select! {
                        _ = tx.closed() => return,
                        _ = tokio::time::sleep(std::time::Duration::from_millis(tail_poll_ms)) => {}
                    }
                }
                Err(_) => return,
            }
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

/// GET /runs/{run_id}/events — full SSE handler.
pub(crate) async fn stream_events<S, L>(
    State(state): State<AppState<S, L>>,
    Path(run_id): Path<String>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse
where
    S: EventStore + Clone + Send + Sync + 'static,
    L: RunLauncher<S>,
{
    // Unknown run: no status and no events → 404.
    let known = matches!(state.store.get_status(&run_id).await, Ok(Some(_)))
        || state
            .store
            .read_tail(&run_id, 1)
            .await
            .map(|v| !v.is_empty())
            .unwrap_or(false);
    if !known {
        return crate::routes::error_response(axum::http::StatusCode::NOT_FOUND, "unknown run_id");
    }

    let after: Offset = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let tail_poll_ms = state.tail_poll_ms;
    // Keep-alive comments stop proxies from closing a stream as idle while the
    // run waits, e.g. at an approval gate.
    Sse::new(event_stream(
        state.store.clone(),
        run_id,
        after,
        tail_poll_ms,
    ))
    .keep_alive(KeepAlive::default())
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::workflow_event::WorkflowEvent;
    use flux_session::{EventStore, MemoryStore, RunStatus};
    use tokio_stream::StreamExt;

    #[derive(Clone)]
    struct CountingStore {
        inner: MemoryStore,
        reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl EventStore for CountingStore {
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
        async fn append(
            &self,
            run_id: &str,
            event: &WorkflowEvent,
        ) -> flux_core::error::Result<flux_session::Offset> {
            self.inner.append(run_id, event).await
        }
        async fn read_after(
            &self,
            run_id: &str,
            after: flux_session::Offset,
        ) -> flux_core::error::Result<Vec<(flux_session::Offset, WorkflowEvent)>> {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.read_after(run_id, after).await
        }
        async fn read_tail(
            &self,
            run_id: &str,
            count: usize,
        ) -> flux_core::error::Result<Vec<(flux_session::Offset, WorkflowEvent)>> {
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
    }

    async fn seeded_store() -> MemoryStore {
        let s = MemoryStore::default();
        s.append(
            "run1",
            &WorkflowEvent::StageStarted {
                stage: "extract".into(),
            },
        )
        .await
        .unwrap();
        s.append(
            "run1",
            &WorkflowEvent::StageSkipped {
                stage: "act".into(),
            },
        )
        .await
        .unwrap();
        s.append("run1", &WorkflowEvent::WorkflowComplete)
            .await
            .unwrap();
        s.set_status("run1", RunStatus::Complete).await.unwrap();
        s
    }

    // Collect the SSE event `data` payloads until the stream ends.
    async fn collect_data<St>(stream: St) -> Vec<String>
    where
        St: tokio_stream::Stream<
            Item = Result<axum::response::sse::Event, std::convert::Infallible>,
        >,
    {
        tokio::pin!(stream);
        let mut out = Vec::new();
        while let Some(Ok(ev)) = stream.next().await {
            // Event has no public getter; re-serialize via Debug is brittle.
            // Instead the stream is built so each Event's data is our JSON; we
            // assert on the count + the parsed events through the store instead.
            let _ = ev;
            out.push(String::new());
        }
        out
    }

    #[tokio::test]
    async fn streams_all_events_from_start_then_closes() {
        let store = seeded_store().await;
        let stream = event_stream(store, "run1".to_string(), 0, 250);
        let items = collect_data(stream).await;
        // 3 seeded events (incl. terminal WorkflowComplete) then the stream closes.
        assert_eq!(items.len(), 3);
    }

    #[tokio::test]
    async fn replays_only_after_last_event_id() {
        let store = seeded_store().await;
        // Resume after offset 1 → expect events 2 and 3 (skip + complete).
        let stream = event_stream(store, "run1".to_string(), 1, 250);
        let items = collect_data(stream).await;
        assert_eq!(items.len(), 2);
    }

    #[tokio::test]
    async fn tail_task_stops_polling_after_the_client_disconnects() {
        // A run parked in WaitingApproval never emits, so the poller never
        // sends — it must notice the receiver is gone via tx.closed().
        let reads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let store = CountingStore {
            inner: MemoryStore::default(),
            reads: reads.clone(),
        };
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        let stream = event_stream(store, "run1".to_string(), 0, 10);
        drop(stream); // the client is gone before/while the task idles
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        let after_grace = reads.load(std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        let later = reads.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            later <= after_grace + 1,
            "poller kept polling after disconnect: {after_grace} -> {later}"
        );
    }

    #[tokio::test]
    async fn the_stream_ends_on_workflow_cancelled() {
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::Cancelled)
            .await
            .unwrap();
        store
            .append("run1", &WorkflowEvent::WorkflowCancelled)
            .await
            .unwrap();
        let stream = event_stream(store, "run1".to_string(), 0, 10);
        let items = tokio::time::timeout(std::time::Duration::from_secs(2), collect_data(stream))
            .await
            .expect("stream must end on WorkflowCancelled");
        assert_eq!(items.len(), 1);
    }

    #[tokio::test]
    async fn the_stream_ends_once_the_runs_state_has_expired() {
        // Status key gone (TTL) and no terminal event: nothing will ever come.
        let store = MemoryStore::default();
        store
            .append(
                "run1",
                &WorkflowEvent::StageStarted {
                    stage: "extract".into(),
                },
            )
            .await
            .unwrap();
        let stream = event_stream(store, "run1".to_string(), 0, 10);
        let items = tokio::time::timeout(std::time::Duration::from_secs(2), collect_data(stream))
            .await
            .expect("stream must end, not poll forever");
        assert_eq!(items.len(), 1);
    }

    #[derive(Clone)]
    struct NoopLauncher;

    impl crate::RunLauncher<MemoryStore> for NoopLauncher {
        async fn launch(
            &self,
            _store: MemoryStore,
            _run_id: String,
            _workflow: flux_workflow::Workflow,
            _input: String,
            _conversation_id: Option<String>,
            _context: Option<serde_json::Value>,
            _resume: flux_orchestrator::ResumeState,
        ) {
        }
    }

    #[tokio::test(start_paused = true)]
    async fn an_idle_stream_sends_keep_alive_comments() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;
        // A run parked at an approval gate emits nothing; without keep-alives a
        // proxy closes the connection as idle.
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        store
            .append(
                "run1",
                &WorkflowEvent::StageWaitingApproval {
                    stage: "act".into(),
                },
            )
            .await
            .unwrap();
        let app = crate::router(AppState {
            store,
            launcher: NoopLauncher,
            workflows: std::sync::Arc::new(std::collections::HashMap::new()),
            tail_poll_ms: 50,
        });
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/runs/run1/events")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = resp.into_body();
        let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
        assert!(
            first.starts_with(b"id:") || first.starts_with(b"data:"),
            "{first:?}"
        );
        let next = tokio::time::timeout(std::time::Duration::from_secs(60), body.frame())
            .await
            .expect("an idle stream must send a keep-alive within 60s")
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert!(
            next.starts_with(b":"),
            "expected an SSE comment, got {next:?}"
        );
    }
}
