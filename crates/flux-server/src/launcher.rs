use flux_session::EventStore;
use flux_workflow::Workflow;
use futures_util::FutureExt;
use std::future::Future;

use flux_config::AgentConfig;
use flux_core::workflow_event::WorkflowEvent;
use flux_llm::LlmRegistry;
use flux_memory::{PostgresMemory, RedisMemory, TieredMemory};
use flux_orchestrator::run_workflow_cancellable;
use flux_runner::NoopTransport;
use flux_session::RunStatus;
use flux_tool_client::{AnyTransport, MultiTransport};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Launches a workflow run: executes it (in the background, spawned by the
/// caller) and writes its `WorkflowEvent`s + final status into the `store`.
/// The real implementation (`OrchestratorLauncher`) runs `flux_orchestrator::
/// run_workflow`; tests use a scripted fake.
pub trait RunLauncher<S: EventStore>: Clone + Send + Sync + 'static {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        store: S,
        run_id: String,
        workflow: Workflow,
        input: String,
        conversation_id: Option<String>,
        context: Option<serde_json::Value>,
        resume: flux_orchestrator::ResumeState,
    ) -> impl Future<Output = ()> + Send;
}

/// Drain `rx` into the store. Returns true iff every append succeeded. After
/// the first failure the remaining events are still received (so the run's
/// senders never block on a full channel) but not written: the log is already
/// unusable for resume — it is the resume source of truth, and a log with a
/// hole re-executes already-completed stages, duplicating tool side effects.
pub(crate) async fn drain_events<S: EventStore>(
    store: S,
    run_id: String,
    mut rx: tokio::sync::mpsc::Receiver<WorkflowEvent>,
) -> bool {
    let mut ok = true;
    while let Some(event) = rx.recv().await {
        if !ok {
            continue;
        }
        if let Err(e) = store.append(&run_id, &event).await {
            tracing::error!(run_id = %run_id, error = %e, "failed to append workflow event; the run will be marked failed");
            ok = false;
        }
    }
    ok
}

/// Real launcher: runs the M4 DAG executor and relays its events into the store.
#[derive(Clone)]
pub struct OrchestratorLauncher {
    pub llm: Arc<LlmRegistry>,
    pub registry: Arc<HashMap<String, AgentConfig>>,
    pub memory: Option<Arc<TieredMemory<RedisMemory, PostgresMemory>>>,
    pub transport: Option<Arc<MultiTransport<AnyTransport>>>,
    pub tuning: flux_orchestrator::RunTuning,
}

/// Run one workflow with a concrete transport + memory, deduplicating the
/// dispatch arms in `launch`. `T`/`M` are monomorphized per call site.
#[allow(clippy::too_many_arguments)]
async fn drive_run<T, M>(
    workflow: &Workflow,
    registry: &HashMap<String, AgentConfig>,
    llm: &LlmRegistry,
    transport: &T,
    input: &str,
    conversation_id: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    resume: flux_orchestrator::ResumeState,
    tx: &tokio::sync::mpsc::Sender<WorkflowEvent>,
    tuning: flux_orchestrator::RunTuning,
    cancel: &CancellationToken,
) -> flux_core::error::Result<flux_orchestrator::RunOutcome>
where
    T: flux_tool_client::ToolTransport + Sync,
    M: flux_memory::ConversationMemory + Sync,
{
    run_workflow_cancellable(
        workflow,
        registry,
        llm,
        transport,
        input,
        conversation_id,
        context,
        memory,
        resume,
        tx,
        tuning,
        cancel,
    )
    .await
}

/// How often a running run checks the store for a cancel request. The cancel
/// route only writes `cancelling`, so this one path serves a cancel received
/// by any server instance.
const CANCEL_POLL: std::time::Duration = std::time::Duration::from_secs(1);

/// Run `run` with a token that is cancelled once the run's status becomes
/// `cancelling` — immediately if it already is (a cancel that landed between
/// approve's status swap and this relaunch), else within `poll`.
pub(crate) async fn supervise<S, F, Fut, T>(
    store: &S,
    run_id: &str,
    poll: std::time::Duration,
    run: F,
) -> flux_core::error::Result<T>
where
    S: EventStore + Clone + Send + Sync + 'static,
    F: FnOnce(CancellationToken) -> Fut,
    Fut: Future<Output = flux_core::error::Result<T>>,
{
    let cancel = CancellationToken::new();
    if let Ok(Some(RunStatus::Cancelling)) = store.get_status(run_id).await {
        cancel.cancel();
    }
    // Aborts on drop, so a panicking `run` (unwinding) stops the watcher too.
    let _watcher = AbortOnDrop(tokio::spawn({
        let store = store.clone();
        let run_id = run_id.to_string();
        let cancel = cancel.clone();
        async move {
            while !cancel.is_cancelled() {
                tokio::time::sleep(poll).await;
                if let Ok(Some(RunStatus::Cancelling)) = store.get_status(&run_id).await {
                    cancel.cancel();
                }
            }
        }
    }));
    run(cancel).await
}

/// Aborts the wrapped task when dropped.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Record a finished run's status without overwriting a cancel request that
/// arrived meanwhile. A run that completed or failed before it observed the
/// cancel keeps that outcome; one that paused at a gate honours the cancel,
/// since nothing is in flight and a 202'd cancel must not be lost.
pub(crate) async fn write_final_status<S: EventStore>(store: &S, run_id: &str, status: RunStatus) {
    let written = async {
        if status == RunStatus::Cancelled
            || !store
                .compare_and_swap_status(run_id, RunStatus::Running, status)
                .await?
        {
            match store.get_status(run_id).await? {
                Some(RunStatus::Cancelling) if status == RunStatus::WaitingApproval => {
                    store.set_status(run_id, RunStatus::Cancelled).await?;
                    store
                        .append(run_id, &WorkflowEvent::WorkflowCancelled)
                        .await?;
                }
                Some(RunStatus::Cancelling) => {
                    store
                        .compare_and_swap_status(run_id, RunStatus::Cancelling, status)
                        .await?;
                }
                Some(RunStatus::Running) => {
                    store
                        .compare_and_swap_status(run_id, RunStatus::Running, status)
                        .await?;
                }
                // Expired (`None`) or already settled: writing would recreate or
                // clobber it.
                current => tracing::warn!(
                    run_id = %run_id,
                    current = ?current,
                    intended = ?status,
                    "final status not written: run is not running"
                ),
            }
        }
        Ok::<_, flux_core::error::FluxError>(())
    };
    if let Err(e) = written.await {
        tracing::warn!(run_id = %run_id, error = %e, "failed to set terminal run status");
    }
}

impl<S: EventStore + Clone + Send + Sync + 'static> RunLauncher<S> for OrchestratorLauncher {
    async fn launch(
        &self,
        store: S,
        run_id: String,
        workflow: Workflow,
        input: String,
        conversation_id: Option<String>,
        context: Option<serde_json::Value>,
        resume: flux_orchestrator::ResumeState,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel::<WorkflowEvent>(64);

        let writer = tokio::spawn(drain_events(store.clone(), run_id.clone(), rx));

        let cid = conversation_id.as_deref();
        if self.memory.is_none() && cid.is_some() {
            tracing::warn!(
                run_id = %run_id,
                "conversation_id provided but no conversation memory is configured (server started without --database-url); running stateless, no turns will be persisted"
            );
        }
        let events = &tx;
        let run = supervise(&store, &run_id, CANCEL_POLL, |cancel| async move {
            match (&self.transport, &self.memory, cid) {
                (Some(t), Some(mem), Some(_)) => {
                    drive_run(
                        &workflow,
                        &self.registry,
                        &self.llm,
                        t.as_ref(),
                        &input,
                        cid,
                        context.as_ref(),
                        mem.as_ref(),
                        resume,
                        events,
                        self.tuning,
                        &cancel,
                    )
                    .await
                }
                (Some(t), _, _) => {
                    drive_run(
                        &workflow,
                        &self.registry,
                        &self.llm,
                        t.as_ref(),
                        &input,
                        None,
                        context.as_ref(),
                        &flux_orchestrator::NoMemory,
                        resume,
                        events,
                        self.tuning,
                        &cancel,
                    )
                    .await
                }
                (None, Some(mem), Some(_)) => {
                    drive_run(
                        &workflow,
                        &self.registry,
                        &self.llm,
                        &NoopTransport,
                        &input,
                        cid,
                        context.as_ref(),
                        mem.as_ref(),
                        resume,
                        events,
                        self.tuning,
                        &cancel,
                    )
                    .await
                }
                (None, _, _) => {
                    drive_run(
                        &workflow,
                        &self.registry,
                        &self.llm,
                        &NoopTransport,
                        &input,
                        None,
                        context.as_ref(),
                        &flux_orchestrator::NoMemory,
                        resume,
                        events,
                        self.tuning,
                        &cancel,
                    )
                    .await
                }
            }
        });
        let result = guard_panic(run, &tx).await;

        drop(tx);
        let log_ok = writer.await.unwrap_or(false);
        write_final_status(&store, &run_id, final_status(log_ok, result)).await;
    }
}

/// Await `run`; `None` if it panicked. A panic is turned into a terminal
/// `WorkflowFailed` event so SSE clients stop waiting, and the caller records
/// the run as failed — otherwise it would read as running until it expired.
async fn guard_panic<T>(
    run: impl Future<Output = T>,
    events: &tokio::sync::mpsc::Sender<WorkflowEvent>,
) -> Option<T> {
    match std::panic::AssertUnwindSafe(run).catch_unwind().await {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::error!("workflow run panicked");
            let _ = events
                .send(WorkflowEvent::WorkflowFailed {
                    stage: String::new(),
                    error: "internal error: the run panicked".into(),
                })
                .await;
            None
        }
    }
}

/// The status a finished run is recorded with. `result` is `None` when the
/// run panicked.
fn final_status(
    log_ok: bool,
    result: Option<flux_core::error::Result<flux_orchestrator::RunOutcome>>,
) -> RunStatus {
    if !log_ok {
        // The event log is the resume source of truth; a hole in it makes
        // any later resume silently re-execute completed stages.
        return RunStatus::Failed;
    }
    match result {
        Some(Ok(flux_orchestrator::RunOutcome::Completed(_))) => RunStatus::Complete,
        Some(Ok(flux_orchestrator::RunOutcome::Paused { .. })) => RunStatus::WaitingApproval,
        Some(Ok(flux_orchestrator::RunOutcome::Cancelled)) => RunStatus::Cancelled,
        Some(Err(_)) | None => RunStatus::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_session::MemoryStore;

    /// Append always fails; everything else delegates.
    #[derive(Clone, Default)]
    struct FailingAppendStore(MemoryStore);

    impl EventStore for FailingAppendStore {
        async fn claim_idempotency_key(
            &self,
            scope: &str,
            key: &str,
            run_id: &str,
            body_hash: &str,
        ) -> flux_core::error::Result<flux_session::IdempotencyClaim> {
            self.0
                .claim_idempotency_key(scope, key, run_id, body_hash)
                .await
        }
        async fn release_idempotency_key(
            &self,
            scope: &str,
            key: &str,
        ) -> flux_core::error::Result<()> {
            self.0.release_idempotency_key(scope, key).await
        }
        async fn append(
            &self,
            _run_id: &str,
            _event: &WorkflowEvent,
        ) -> flux_core::error::Result<flux_session::Offset> {
            Err(flux_core::error::FluxError::Config("redis blip".into()))
        }
        async fn read_after(
            &self,
            run_id: &str,
            after: flux_session::Offset,
        ) -> flux_core::error::Result<Vec<(flux_session::Offset, WorkflowEvent)>> {
            self.0.read_after(run_id, after).await
        }
        async fn read_tail(
            &self,
            run_id: &str,
            count: usize,
        ) -> flux_core::error::Result<Vec<(flux_session::Offset, WorkflowEvent)>> {
            self.0.read_tail(run_id, count).await
        }
        async fn set_status(
            &self,
            run_id: &str,
            status: flux_session::RunStatus,
        ) -> flux_core::error::Result<()> {
            self.0.set_status(run_id, status).await
        }
        async fn get_status(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<flux_session::RunStatus>> {
            self.0.get_status(run_id).await
        }
        async fn compare_and_swap_status(
            &self,
            run_id: &str,
            expected: flux_session::RunStatus,
            new: flux_session::RunStatus,
        ) -> flux_core::error::Result<bool> {
            self.0.compare_and_swap_status(run_id, expected, new).await
        }
        async fn record_approval(&self, run_id: &str, stage: &str) -> flux_core::error::Result<()> {
            self.0.record_approval(run_id, stage).await
        }
        async fn approved_stages(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<std::collections::HashSet<String>> {
            self.0.approved_stages(run_id).await
        }
        async fn set_manifest(
            &self,
            run_id: &str,
            workflow_id: &str,
            input: &str,
        ) -> flux_core::error::Result<()> {
            self.0.set_manifest(run_id, workflow_id, input).await
        }
        async fn get_manifest(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<(String, String)>> {
            self.0.get_manifest(run_id).await
        }
        async fn set_conversation_id(
            &self,
            run_id: &str,
            conversation_id: &str,
        ) -> flux_core::error::Result<()> {
            self.0.set_conversation_id(run_id, conversation_id).await
        }
        async fn get_conversation_id(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<String>> {
            self.0.get_conversation_id(run_id).await
        }
        async fn set_context(
            &self,
            run_id: &str,
            context: &serde_json::Value,
        ) -> flux_core::error::Result<()> {
            self.0.set_context(run_id, context).await
        }
        async fn get_context(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<serde_json::Value>> {
            self.0.get_context(run_id).await
        }
    }

    /// Counts `get_status` calls; everything else delegates.
    #[derive(Clone, Default)]
    struct CountingStatusStore {
        inner: MemoryStore,
        status_reads: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl EventStore for CountingStatusStore {
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
            status: flux_session::RunStatus,
        ) -> flux_core::error::Result<()> {
            self.inner.set_status(run_id, status).await
        }
        async fn get_status(
            &self,
            run_id: &str,
        ) -> flux_core::error::Result<Option<flux_session::RunStatus>> {
            self.status_reads
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.get_status(run_id).await
        }
        async fn compare_and_swap_status(
            &self,
            run_id: &str,
            expected: flux_session::RunStatus,
            new: flux_session::RunStatus,
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

    #[tokio::test]
    async fn drain_events_reports_success_and_persists() {
        let store = MemoryStore::default();
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.send(WorkflowEvent::StageStarted { stage: "a".into() })
            .await
            .unwrap();
        tx.send(WorkflowEvent::WorkflowComplete).await.unwrap();
        drop(tx);
        assert!(drain_events(store.clone(), "run1".to_string(), rx).await);
        assert_eq!(store.read_after("run1", 0).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn drain_events_reports_failure_but_keeps_draining() {
        // The run must never block on a full channel just because the log
        // broke — the sender side keeps sending until the run finishes.
        let store = FailingAppendStore::default();
        let (tx, rx) = tokio::sync::mpsc::channel(2); // small: forces the drain
        let drainer = tokio::spawn(drain_events(store, "run1".to_string(), rx));
        for _ in 0..16 {
            tx.send(WorkflowEvent::StageStarted { stage: "a".into() })
                .await
                .unwrap();
        }
        drop(tx);
        assert!(!drainer.await.unwrap());
    }

    use flux_orchestrator::RunOutcome;
    use std::time::Duration;

    #[tokio::test]
    async fn a_cancelling_run_ends_cancelled() {
        let store = MemoryStore::default();
        store.set_status("run1", RunStatus::Running).await.unwrap();
        let flipper = store.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            flipper
                .set_status("run1", RunStatus::Cancelling)
                .await
                .unwrap();
        });
        let outcome = tokio::time::timeout(
            Duration::from_secs(2),
            supervise(
                &store,
                "run1",
                Duration::from_millis(10),
                |cancel| async move {
                    cancel.cancelled().await;
                    Ok(RunOutcome::Cancelled)
                },
            ),
        )
        .await
        .expect("the watcher must cancel the run");
        assert!(matches!(outcome, Ok(RunOutcome::Cancelled)));
    }

    #[tokio::test]
    async fn a_panicking_run_stops_the_cancel_watcher() {
        use std::sync::atomic::Ordering;
        let store = CountingStatusStore::default();
        store
            .inner
            .set_status("run1", RunStatus::Running)
            .await
            .unwrap();
        let task = tokio::spawn({
            let store = store.clone();
            async move {
                supervise(
                    &store,
                    "run1",
                    Duration::from_millis(5),
                    |_cancel| async move {
                        if std::hint::black_box(true) {
                            panic!("the run panics");
                        }
                        Ok::<(), flux_core::error::FluxError>(())
                    },
                )
                .await
            }
        });
        assert!(task.await.unwrap_err().is_panic());
        // Let any watcher still alive poll a few more times, then compare.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let settled = store.status_reads.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            store.status_reads.load(Ordering::SeqCst),
            settled,
            "the watcher kept polling after the run panicked"
        );
    }

    #[tokio::test]
    async fn a_run_cancelled_before_relaunch_starts_already_cancelled() {
        // A cancel can land between approve's status swap and the relaunch.
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::Cancelling)
            .await
            .unwrap();
        let seen = supervise(
            &store,
            "run1",
            Duration::from_secs(60),
            |cancel| async move { Ok::<_, flux_core::error::FluxError>(cancel.is_cancelled()) },
        )
        .await
        .unwrap();
        assert!(seen, "token must be cancelled before the run starts");
    }

    #[tokio::test]
    async fn a_run_that_completes_while_cancelling_ends_complete() {
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::Cancelling)
            .await
            .unwrap();
        write_final_status(&store, "run1", RunStatus::Complete).await;
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Complete)
        );
    }

    #[tokio::test]
    async fn a_run_that_pauses_while_cancelling_ends_cancelled() {
        let store = MemoryStore::default();
        store
            .set_status("run1", RunStatus::Cancelling)
            .await
            .unwrap();
        write_final_status(&store, "run1", RunStatus::WaitingApproval).await;
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Cancelled)
        );
        let log = store.read_after("run1", 0).await.unwrap();
        assert_eq!(
            log.last().map(|(_, e)| e),
            Some(&WorkflowEvent::WorkflowCancelled)
        );
    }

    #[tokio::test]
    async fn an_expired_run_is_not_recreated_by_its_final_status() {
        let store = MemoryStore::default();
        write_final_status(&store, "gone", RunStatus::Complete).await;
        assert_eq!(store.get_status("gone").await.unwrap(), None);
    }

    #[tokio::test]
    async fn a_terminal_status_is_not_overwritten() {
        let store = MemoryStore::default();
        store.set_status("run1", RunStatus::Complete).await.unwrap();
        write_final_status(&store, "run1", RunStatus::Failed).await;
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Complete)
        );
    }

    #[tokio::test]
    async fn a_panicking_run_fails_with_a_terminal_event() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<WorkflowEvent>(4);
        let outcome = guard_panic(async { panic!("the run panics") }, &tx).await;
        let outcome: Option<()> = outcome;
        assert!(outcome.is_none());
        let event = rx.try_recv().unwrap();
        assert!(flux_session::is_terminal(&event));
        assert!(matches!(
            event,
            WorkflowEvent::WorkflowFailed { ref error, .. } if error.contains("panicked")
        ));
    }

    #[tokio::test]
    async fn a_run_that_returns_passes_through_the_panic_guard() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<WorkflowEvent>(4);
        assert_eq!(guard_panic(async { 7 }, &tx).await, Some(7));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_panicked_run_is_recorded_as_failed() {
        assert_eq!(final_status(true, None), RunStatus::Failed);
    }

    #[tokio::test]
    async fn an_uncancelled_run_records_its_outcome() {
        let store = MemoryStore::default();
        store.set_status("run1", RunStatus::Running).await.unwrap();
        write_final_status(&store, "run1", RunStatus::WaitingApproval).await;
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::WaitingApproval)
        );
    }
}
