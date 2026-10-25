#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_core::error::{FluxError, Result};
use flux_core::workflow_event::WorkflowEvent;
use redis::AsyncCommands;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A 1-based, monotonic per-run event sequence number. The SSE `Last-Event-ID`
/// a client echoes back is exactly this value.
pub type Offset = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Complete,
    Failed,
    WaitingApproval,
    /// A cancel was requested; the run is still stopping.
    Cancelling,
    /// Terminal: the run stopped because it was cancelled.
    Cancelled,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Running => "running",
            RunStatus::Complete => "complete",
            RunStatus::Failed => "failed",
            RunStatus::WaitingApproval => "waiting_approval",
            RunStatus::Cancelling => "cancelling",
            RunStatus::Cancelled => "cancelled",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<RunStatus> {
        match s {
            "running" => Some(RunStatus::Running),
            "complete" => Some(RunStatus::Complete),
            "failed" => Some(RunStatus::Failed),
            "waiting_approval" => Some(RunStatus::WaitingApproval),
            "cancelling" => Some(RunStatus::Cancelling),
            "cancelled" => Some(RunStatus::Cancelled),
            _ => None,
        }
    }
}

/// True for the events that end a run's stream.
pub fn is_terminal(event: &WorkflowEvent) -> bool {
    matches!(
        event,
        WorkflowEvent::WorkflowComplete
            | WorkflowEvent::WorkflowFailed { .. }
            | WorkflowEvent::WorkflowCancelled
    )
}

/// Append-only, replayable per-run event store with run status. Async methods
/// follow the project's `-> impl Future + Send` trait style (used via generics,
/// never `dyn`).
pub trait EventStore {
    fn append(
        &self,
        run_id: &str,
        event: &WorkflowEvent,
    ) -> impl Future<Output = Result<Offset>> + Send;

    fn read_after(
        &self,
        run_id: &str,
        after: Offset,
    ) -> impl Future<Output = Result<Vec<(Offset, WorkflowEvent)>>> + Send;

    /// The last `count` events, oldest-first (fewer if the run is shorter;
    /// empty for an unknown run or `count == 0`). Exists so "what was the
    /// final reply?" does not replay every token delta of the whole run.
    fn read_tail(
        &self,
        run_id: &str,
        count: usize,
    ) -> impl Future<Output = Result<Vec<(Offset, WorkflowEvent)>>> + Send;

    fn set_status(
        &self,
        run_id: &str,
        status: RunStatus,
    ) -> impl Future<Output = Result<()>> + Send;

    fn get_status(&self, run_id: &str) -> impl Future<Output = Result<Option<RunStatus>>> + Send;

    /// Atomically set status to `new` iff it currently equals `expected`.
    /// Returns true if the swap happened (this caller won the race), false
    /// otherwise (status differed, or the run is unknown).
    fn compare_and_swap_status(
        &self,
        run_id: &str,
        expected: RunStatus,
        new: RunStatus,
    ) -> impl Future<Output = Result<bool>> + Send;

    fn record_approval(&self, run_id: &str, stage: &str)
        -> impl Future<Output = Result<()>> + Send;

    fn approved_stages(
        &self,
        run_id: &str,
    ) -> impl Future<Output = Result<std::collections::HashSet<String>>> + Send;

    fn set_manifest(
        &self,
        run_id: &str,
        workflow_id: &str,
        input: &str,
    ) -> impl Future<Output = Result<()>> + Send;

    fn get_manifest(
        &self,
        run_id: &str,
    ) -> impl Future<Output = Result<Option<(String, String)>>> + Send;

    fn set_conversation_id(
        &self,
        run_id: &str,
        conversation_id: &str,
    ) -> impl Future<Output = Result<()>> + Send;

    fn get_conversation_id(
        &self,
        run_id: &str,
    ) -> impl Future<Output = Result<Option<String>>> + Send;

    fn set_context(
        &self,
        run_id: &str,
        context: &serde_json::Value,
    ) -> impl Future<Output = Result<()>> + Send;

    fn get_context(
        &self,
        run_id: &str,
    ) -> impl Future<Output = Result<Option<serde_json::Value>>> + Send;

    /// Atomically bind an `Idempotency-Key` (within `scope`, e.g. the endpoint)
    /// to `run_id` and the request's `body_hash`, unless it is already bound —
    /// then return that binding instead. Bindings expire with run state.
    fn claim_idempotency_key(
        &self,
        scope: &str,
        key: &str,
        run_id: &str,
        body_hash: &str,
    ) -> impl Future<Output = Result<IdempotencyClaim>> + Send;

    /// Drop a binding whose run never started, so a retry starts fresh.
    fn release_idempotency_key(
        &self,
        scope: &str,
        key: &str,
    ) -> impl Future<Output = Result<()>> + Send;
}

/// Outcome of [`EventStore::claim_idempotency_key`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdempotencyClaim {
    /// The key was free and is now bound to the caller's run.
    Claimed,
    /// The key was already bound by an earlier request.
    Existing { run_id: String, body_hash: String },
}

#[derive(Default)]
struct RunData {
    events: Vec<WorkflowEvent>,
    status: Option<RunStatus>,
    approvals: std::collections::HashSet<String>,
    manifest: Option<(String, String)>,
    conversation_id: Option<String>,
    context: Option<serde_json::Value>,
}

/// In-memory `EventStore` for tests (and single-process use). Clone shares the
/// same backing map via `Arc`.
#[derive(Clone, Default)]
pub struct MemoryStore {
    runs: Arc<Mutex<HashMap<String, RunData>>>,
    /// Never expires.
    idempotency: Arc<Mutex<HashMap<IdempotencySlot, IdempotencyBinding>>>,
}

/// `(scope, key)`.
type IdempotencySlot = (String, String);
/// `(run_id, body_hash)`.
type IdempotencyBinding = (String, String);

impl EventStore for MemoryStore {
    async fn append(&self, run_id: &str, event: &WorkflowEvent) -> Result<Offset> {
        let mut runs = self.runs.lock().await;
        let data = runs.entry(run_id.to_string()).or_default();
        data.events.push(event.clone());
        Ok(data.events.len() as Offset)
    }

    async fn read_after(
        &self,
        run_id: &str,
        after: Offset,
    ) -> Result<Vec<(Offset, WorkflowEvent)>> {
        let runs = self.runs.lock().await;
        let Some(data) = runs.get(run_id) else {
            return Ok(Vec::new());
        };
        let start = after as usize;
        let out = data
            .events
            .iter()
            .enumerate()
            .skip(start)
            .map(|(i, e)| ((i as Offset) + 1, e.clone()))
            .collect();
        Ok(out)
    }

    async fn read_tail(&self, run_id: &str, count: usize) -> Result<Vec<(Offset, WorkflowEvent)>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let runs = self.runs.lock().await;
        let Some(data) = runs.get(run_id) else {
            return Ok(Vec::new());
        };
        let start = data.events.len().saturating_sub(count);
        Ok(data
            .events
            .iter()
            .enumerate()
            .skip(start)
            .map(|(i, e)| ((i as Offset) + 1, e.clone()))
            .collect())
    }

    async fn set_status(&self, run_id: &str, status: RunStatus) -> Result<()> {
        let mut runs = self.runs.lock().await;
        runs.entry(run_id.to_string()).or_default().status = Some(status);
        Ok(())
    }

    async fn compare_and_swap_status(
        &self,
        run_id: &str,
        expected: RunStatus,
        new: RunStatus,
    ) -> Result<bool> {
        let mut runs = self.runs.lock().await;
        let Some(data) = runs.get_mut(run_id) else {
            return Ok(false);
        };
        if data.status == Some(expected) {
            data.status = Some(new);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<RunStatus>> {
        let runs = self.runs.lock().await;
        Ok(runs.get(run_id).and_then(|d| d.status))
    }

    async fn record_approval(&self, run_id: &str, stage: &str) -> Result<()> {
        let mut runs = self.runs.lock().await;
        runs.entry(run_id.to_string())
            .or_default()
            .approvals
            .insert(stage.to_string());
        Ok(())
    }

    async fn approved_stages(&self, run_id: &str) -> Result<std::collections::HashSet<String>> {
        let runs = self.runs.lock().await;
        Ok(runs
            .get(run_id)
            .map(|d| d.approvals.clone())
            .unwrap_or_default())
    }

    async fn set_manifest(&self, run_id: &str, workflow_id: &str, input: &str) -> Result<()> {
        let mut runs = self.runs.lock().await;
        runs.entry(run_id.to_string()).or_default().manifest =
            Some((workflow_id.to_string(), input.to_string()));
        Ok(())
    }

    async fn get_manifest(&self, run_id: &str) -> Result<Option<(String, String)>> {
        let runs = self.runs.lock().await;
        Ok(runs.get(run_id).and_then(|d| d.manifest.clone()))
    }

    async fn set_conversation_id(&self, run_id: &str, conversation_id: &str) -> Result<()> {
        let mut runs = self.runs.lock().await;
        runs.entry(run_id.to_string()).or_default().conversation_id =
            Some(conversation_id.to_string());
        Ok(())
    }

    async fn get_conversation_id(&self, run_id: &str) -> Result<Option<String>> {
        let runs = self.runs.lock().await;
        Ok(runs.get(run_id).and_then(|d| d.conversation_id.clone()))
    }

    async fn set_context(&self, run_id: &str, context: &serde_json::Value) -> Result<()> {
        let mut runs = self.runs.lock().await;
        runs.entry(run_id.to_string()).or_default().context = Some(context.clone());
        Ok(())
    }

    async fn get_context(&self, run_id: &str) -> Result<Option<serde_json::Value>> {
        let runs = self.runs.lock().await;
        Ok(runs.get(run_id).and_then(|d| d.context.clone()))
    }

    async fn claim_idempotency_key(
        &self,
        scope: &str,
        key: &str,
        run_id: &str,
        body_hash: &str,
    ) -> Result<IdempotencyClaim> {
        let mut keys = self.idempotency.lock().await;
        match keys.entry((scope.to_string(), key.to_string())) {
            std::collections::hash_map::Entry::Occupied(e) => {
                let (run_id, body_hash) = e.get().clone();
                Ok(IdempotencyClaim::Existing { run_id, body_hash })
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert((run_id.to_string(), body_hash.to_string()));
                Ok(IdempotencyClaim::Claimed)
            }
        }
    }

    async fn release_idempotency_key(&self, scope: &str, key: &str) -> Result<()> {
        self.idempotency
            .lock()
            .await
            .remove(&(scope.to_string(), key.to_string()));
        Ok(())
    }
}

/// Redis-backed `EventStore`. Per run: a Stream `flux:run:{id}:events` holds the
/// JSON-encoded events (the Nth `XADD` is `Offset` N), and a key
/// `flux:run:{id}:meta` holds the run status. Both keys carry a TTL so state is
/// "hot" (expires after `ttl_secs`). Clone shares the connection manager.
#[derive(Clone)]
pub struct RedisStore {
    conn: redis::aio::ConnectionManager,
    ttl_secs: u64,
}

// Manual impl: `redis::aio::ConnectionManager` isn't `Debug`, so `#[derive]`
// isn't available. Needed for `Result<RedisStore, _>::expect_err` in tests.
impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore")
            .field("ttl_secs", &self.ttl_secs)
            .finish()
    }
}

/// How long a Redis connection attempt may take before failing loudly.
/// `ConnectionManager::new` otherwise retries forever, which turned a
/// missing local Redis into a silent, indefinite hang at server boot.
const REDIS_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl RedisStore {
    pub async fn connect(redis_url: &str, ttl_secs: u64) -> Result<RedisStore> {
        let client =
            redis::Client::open(redis_url).map_err(|e| FluxError::Config(e.to_string()))?;
        let conn = tokio::time::timeout(
            REDIS_CONNECT_TIMEOUT,
            redis::aio::ConnectionManager::new(client),
        )
        .await
        .map_err(|_| {
            FluxError::Config(format!(
                "timed out connecting to Redis at {} (config: redis.url, env: REDIS_URL)",
                flux_core::redact::redact_url(redis_url)
            ))
        })?
        .map_err(|e| {
            FluxError::Config(format!(
                "connecting to Redis at {}: {e}",
                flux_core::redact::redact_url(redis_url)
            ))
        })?;
        Ok(RedisStore { conn, ttl_secs })
    }

    // Every key of a run carries the run id as a `{…}` hash tag, so all of
    // them land in one Redis Cluster slot and one script may touch them all.
    fn events_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:events")
    }

    fn meta_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:meta")
    }

    fn approvals_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:approvals")
    }

    fn manifest_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:manifest")
    }

    fn conversation_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:conversation")
    }

    fn idempotency_key(scope: &str, key: &str) -> String {
        format!("flux:idem:{scope}:{key}")
    }

    fn context_key(run_id: &str) -> String {
        format!("flux:run:{{{run_id}}}:context")
    }
}

impl EventStore for RedisStore {
    async fn append(&self, run_id: &str, event: &WorkflowEvent) -> Result<Offset> {
        use std::sync::OnceLock;
        // One server-side script instead of several round trips. This runs once
        // per streamed token delta, so per-command latency here dominates the
        // event pipeline. Offset stays the 1-based stream position with explicit
        // `{offset}-0` ids, single-writer per run.
        //
        // It also refreshes the TTL of every other key of the run (KEYS[2..]).
        // Those are written once, at creation or pause, so without this a run
        // longer than the TTL lost its status mid-run, and a run paused late in
        // the TTL left little time to approve. EXPIRE on a missing key is a no-op.
        static APPEND_SCRIPT: OnceLock<redis::Script> = OnceLock::new();
        let script = APPEND_SCRIPT.get_or_init(|| {
            redis::Script::new(
                r"local off = redis.call('XLEN', KEYS[1]) + 1
redis.call('XADD', KEYS[1], off .. '-0', 'data', ARGV[1])
for i = 1, #KEYS do
  redis.call('EXPIRE', KEYS[i], ARGV[2])
end
return off",
            )
        });
        let mut conn = self.conn.clone();
        let key = Self::events_key(run_id);
        let payload = serde_json::to_string(event).map_err(|e| FluxError::Decode(e.to_string()))?;
        let offset: u64 = script
            .key(&key)
            .key(Self::meta_key(run_id))
            .key(Self::manifest_key(run_id))
            .key(Self::conversation_key(run_id))
            .key(Self::context_key(run_id))
            .key(Self::approvals_key(run_id))
            .arg(&payload)
            .arg(self.ttl_secs)
            .invoke_async(&mut conn)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(offset)
    }

    async fn read_after(
        &self,
        run_id: &str,
        after: Offset,
    ) -> Result<Vec<(Offset, WorkflowEvent)>> {
        let mut conn = self.conn.clone();
        let key = Self::events_key(run_id);
        // Stream ids are contiguous `{offset}-0`. Seek to entries with offset >
        // `after` via an exclusive range start so only new entries transfer
        // (no full-stream rescan per poll). Requires Redis >= 6.2 for `(` bounds.
        let start = if after == 0 {
            "-".to_string()
        } else {
            format!("({after}-0")
        };
        let reply: redis::streams::StreamRangeReply = conn
            .xrange(&key, &start, "+")
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let mut out = Vec::with_capacity(reply.ids.len());
        for (idx, entry) in reply.ids.iter().enumerate() {
            // Contiguous ids → the i-th returned entry has offset after + i + 1.
            let offset = after + (idx as Offset) + 1;
            let json = match entry.map.get("data") {
                Some(redis::Value::BulkString(bytes)) => String::from_utf8_lossy(bytes).to_string(),
                Some(redis::Value::SimpleString(s)) => s.clone(),
                _ => {
                    return Err(FluxError::Decode(format!(
                        "stream entry {} missing 'data' field",
                        entry.id
                    )))
                }
            };
            let event: WorkflowEvent =
                serde_json::from_str(&json).map_err(|e| FluxError::Decode(e.to_string()))?;
            out.push((offset, event));
        }
        Ok(out)
    }

    async fn read_tail(&self, run_id: &str, count: usize) -> Result<Vec<(Offset, WorkflowEvent)>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut conn = self.conn.clone();
        let key = Self::events_key(run_id);
        let reply: redis::streams::StreamRangeReply = redis::cmd("XREVRANGE")
            .arg(&key)
            .arg("+")
            .arg("-")
            .arg("COUNT")
            .arg(count)
            .query_async(&mut conn)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let mut out = Vec::with_capacity(reply.ids.len());
        for entry in reply.ids.iter() {
            // ids are the explicit `{offset}-0` written by append
            let offset: Offset = entry
                .id
                .split('-')
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    FluxError::Decode(format!("unparseable stream id '{}'", entry.id))
                })?;
            let json = match entry.map.get("data") {
                Some(redis::Value::BulkString(bytes)) => String::from_utf8_lossy(bytes).to_string(),
                Some(redis::Value::SimpleString(s)) => s.clone(),
                _ => {
                    return Err(FluxError::Decode(format!(
                        "stream entry {} missing 'data' field",
                        entry.id
                    )))
                }
            };
            let event: WorkflowEvent =
                serde_json::from_str(&json).map_err(|e| FluxError::Decode(e.to_string()))?;
            out.push((offset, event));
        }
        out.reverse(); // XREVRANGE is newest-first; callers want oldest-first
        Ok(out)
    }

    async fn set_status(&self, run_id: &str, status: RunStatus) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::meta_key(run_id);
        let _: () = conn
            .set_ex(&key, status.as_str(), self.ttl_secs)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn compare_and_swap_status(
        &self,
        run_id: &str,
        expected: RunStatus,
        new: RunStatus,
    ) -> Result<bool> {
        let mut conn = self.conn.clone();
        let key = Self::meta_key(run_id);
        // Atomic compare-and-set on the status key: only swap when the current
        // value equals `expected`. Re-applies the TTL on swap (matching
        // set_status). Runs server-side in one round trip.
        let script = redis::Script::new(
            r"local cur = redis.call('GET', KEYS[1])
if cur == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0",
        );
        let swapped: i64 = script
            .key(&key)
            .arg(expected.as_str())
            .arg(new.as_str())
            .arg(self.ttl_secs)
            .invoke_async(&mut conn)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(swapped == 1)
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<RunStatus>> {
        let mut conn = self.conn.clone();
        let key = Self::meta_key(run_id);
        let raw: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(raw.as_deref().and_then(RunStatus::from_str))
    }

    async fn record_approval(&self, run_id: &str, stage: &str) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::approvals_key(run_id);
        let _: () = conn
            .sadd(&key, stage)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let _: () = conn
            .expire(&key, self.ttl_secs as i64)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn approved_stages(&self, run_id: &str) -> Result<std::collections::HashSet<String>> {
        let mut conn = self.conn.clone();
        let key = Self::approvals_key(run_id);
        let members: std::collections::HashSet<String> = conn
            .smembers(&key)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(members)
    }

    async fn set_manifest(&self, run_id: &str, workflow_id: &str, input: &str) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::manifest_key(run_id);
        let payload = serde_json::to_string(&serde_json::json!({
            "workflow_id": workflow_id,
            "input": input,
        }))
        .map_err(|e| FluxError::Decode(e.to_string()))?;
        let _: () = conn
            .set_ex(&key, payload, self.ttl_secs)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn get_manifest(&self, run_id: &str) -> Result<Option<(String, String)>> {
        let mut conn = self.conn.clone();
        let key = Self::manifest_key(run_id);
        let raw: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let Some(raw) = raw else { return Ok(None) };
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| FluxError::Decode(e.to_string()))?;
        match (
            v.get("workflow_id").and_then(|x| x.as_str()),
            v.get("input").and_then(|x| x.as_str()),
        ) {
            (Some(w), Some(i)) => Ok(Some((w.to_string(), i.to_string()))),
            _ => Err(FluxError::Decode(
                "manifest missing workflow_id/input".into(),
            )),
        }
    }

    async fn set_conversation_id(&self, run_id: &str, conversation_id: &str) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::conversation_key(run_id);
        let _: () = conn
            .set_ex(&key, conversation_id, self.ttl_secs)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn get_conversation_id(&self, run_id: &str) -> Result<Option<String>> {
        let mut conn = self.conn.clone();
        let key = Self::conversation_key(run_id);
        let raw: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(raw)
    }

    async fn set_context(&self, run_id: &str, context: &serde_json::Value) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::context_key(run_id);
        let payload =
            serde_json::to_string(context).map_err(|e| FluxError::Decode(e.to_string()))?;
        let _: () = conn
            .set_ex(&key, payload, self.ttl_secs)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn get_context(&self, run_id: &str) -> Result<Option<serde_json::Value>> {
        let mut conn = self.conn.clone();
        let key = Self::context_key(run_id);
        let raw: Option<String> = conn
            .get(&key)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let Some(raw) = raw else { return Ok(None) };
        let v = serde_json::from_str(&raw).map_err(|e| FluxError::Decode(e.to_string()))?;
        Ok(Some(v))
    }

    async fn claim_idempotency_key(
        &self,
        scope: &str,
        key: &str,
        run_id: &str,
        body_hash: &str,
    ) -> Result<IdempotencyClaim> {
        let mut conn = self.conn.clone();
        let k = Self::idempotency_key(scope, key);
        // A binding can expire between a failed SET NX and the GET; then the
        // key is free again, so claim once more.
        for _ in 0..2 {
            let set: Option<String> = redis::cmd("SET")
                .arg(&k)
                .arg(format!("{run_id} {body_hash}"))
                .arg("NX")
                .arg("EX")
                .arg(self.ttl_secs)
                .query_async(&mut conn)
                .await
                .map_err(|e| FluxError::Config(e.to_string()))?;
            if set.is_some() {
                return Ok(IdempotencyClaim::Claimed);
            }
            let existing: Option<String> = conn
                .get(&k)
                .await
                .map_err(|e| FluxError::Config(e.to_string()))?;
            if let Some(raw) = existing {
                let (run_id, body_hash) = raw.split_once(' ').ok_or_else(|| {
                    FluxError::Decode(format!("malformed idempotency binding at {k}"))
                })?;
                return Ok(IdempotencyClaim::Existing {
                    run_id: run_id.to_string(),
                    body_hash: body_hash.to_string(),
                });
            }
        }
        Err(FluxError::Config(format!(
            "could not claim idempotency key {k}"
        )))
    }

    async fn release_idempotency_key(&self, scope: &str, key: &str) -> Result<()> {
        let mut conn = self.conn.clone();
        let _: () = conn
            .del(Self::idempotency_key(scope, key))
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::output::AgentOutput;
    use flux_core::workflow_event::WorkflowEvent;

    fn ev(stage: &str) -> WorkflowEvent {
        WorkflowEvent::StageStarted {
            stage: stage.into(),
        }
    }

    #[tokio::test]
    async fn append_returns_monotonic_offsets_and_read_after_replays() {
        let store = MemoryStore::default();
        let o1 = store.append("run1", &ev("a")).await.unwrap();
        let o2 = store.append("run1", &ev("b")).await.unwrap();
        let o3 = store.append("run1", &ev("c")).await.unwrap();
        assert_eq!((o1, o2, o3), (1, 2, 3));

        // read from start
        let all = store.read_after("run1", 0).await.unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].0, 1);
        assert_eq!(all[2].0, 3);

        // read only the tail after offset 2
        let tail = store.read_after("run1", 2).await.unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].0, 3);

        // unknown run → empty
        assert!(store.read_after("nope", 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn read_tail_returns_the_last_events_oldest_first() {
        let store = MemoryStore::default();
        for s in ["a", "b", "c", "d"] {
            store.append("run1", &ev(s)).await.unwrap();
        }
        let tail = store.read_tail("run1", 2).await.unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].0, 3);
        assert_eq!(tail[1].0, 4);

        // shorter run than asked → the whole run
        assert_eq!(store.read_tail("run1", 99).await.unwrap().len(), 4);
        // unknown run / zero count → empty
        assert!(store.read_tail("nope", 5).await.unwrap().is_empty());
        assert!(store.read_tail("run1", 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_read_after_preserves_contract() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        // Unique run id so repeated runs never collide; TTL reaps the key.
        let run_id = format!(
            "m5_10_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let o1 = store.append(&run_id, &ev("a")).await.unwrap();
        let o2 = store.append(&run_id, &ev("b")).await.unwrap();
        let o3 = store.append(&run_id, &ev("c")).await.unwrap();
        assert_eq!((o1, o2, o3), (1, 2, 3));

        // read from start
        let all = store.read_after(&run_id, 0).await.unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].0, 1);
        assert_eq!(all[2].0, 3);

        // read only the tail after offset 2
        let tail = store.read_after(&run_id, 2).await.unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].0, 3);

        // reading past the end is empty
        assert!(store.read_after(&run_id, 3).await.unwrap().is_empty());

        // unknown run → empty
        let unknown = format!("{run_id}_unknown");
        assert!(store.read_after(&unknown, 0).await.unwrap().is_empty());

        // read_tail: last 2 events, oldest first
        let tail = store.read_tail(&run_id, 2).await.unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].0, 2);
        assert_eq!(tail[1].0, 3);

        // shorter run than asked → the whole run
        assert_eq!(store.read_tail(&run_id, 99).await.unwrap().len(), 3);
        // unknown run / zero count → empty
        assert!(store.read_tail(&unknown, 5).await.unwrap().is_empty());
        assert!(store.read_tail(&run_id, 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn status_round_trips() {
        let store = MemoryStore::default();
        assert_eq!(store.get_status("run1").await.unwrap(), None);
        store.set_status("run1", RunStatus::Running).await.unwrap();
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Running)
        );
        store.set_status("run1", RunStatus::Complete).await.unwrap();
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Complete)
        );
    }

    #[tokio::test]
    async fn compare_and_swap_status_is_atomic_gate() {
        let store = MemoryStore::default();
        // unknown run → no swap
        assert!(!store
            .compare_and_swap_status("run1", RunStatus::WaitingApproval, RunStatus::Running)
            .await
            .unwrap());
        store
            .set_status("run1", RunStatus::WaitingApproval)
            .await
            .unwrap();
        // expected matches → swaps, returns true
        assert!(store
            .compare_and_swap_status("run1", RunStatus::WaitingApproval, RunStatus::Running)
            .await
            .unwrap());
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Running)
        );
        // second attempt no longer matches → false, status unchanged
        assert!(!store
            .compare_and_swap_status("run1", RunStatus::WaitingApproval, RunStatus::Running)
            .await
            .unwrap());
        assert_eq!(
            store.get_status("run1").await.unwrap(),
            Some(RunStatus::Running)
        );
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_compare_and_swap_status_single_winner() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        let run_id = format!(
            "m6_cas_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        store
            .set_status(&run_id, RunStatus::WaitingApproval)
            .await
            .unwrap();
        let a = store.clone();
        let b = store.clone();
        let id_a = run_id.clone();
        let id_b = run_id.clone();
        let (ra, rb) = tokio::join!(
            async move {
                a.compare_and_swap_status(&id_a, RunStatus::WaitingApproval, RunStatus::Running)
                    .await
                    .unwrap()
            },
            async move {
                b.compare_and_swap_status(&id_b, RunStatus::WaitingApproval, RunStatus::Running)
                    .await
                    .unwrap()
            },
        );
        assert_eq!(
            [ra, rb].iter().filter(|x| **x).count(),
            1,
            "exactly one concurrent CAS must win"
        );
        assert_eq!(
            store.get_status(&run_id).await.unwrap(),
            Some(RunStatus::Running)
        );
    }

    #[test]
    fn waiting_approval_status_round_trips() {
        assert_eq!(RunStatus::WaitingApproval.as_str(), "waiting_approval");
        assert_eq!(
            RunStatus::from_str("waiting_approval"),
            Some(RunStatus::WaitingApproval)
        );
    }

    #[tokio::test]
    async fn approvals_record_and_read() {
        let store = MemoryStore::default();
        assert!(store.approved_stages("run1").await.unwrap().is_empty());
        store.record_approval("run1", "act").await.unwrap();
        store.record_approval("run1", "act").await.unwrap(); // idempotent (set)
        store.record_approval("run1", "notify").await.unwrap();
        let set = store.approved_stages("run1").await.unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains("act"));
        assert!(set.contains("notify"));
    }

    #[tokio::test]
    async fn manifest_round_trips() {
        let store = MemoryStore::default();
        assert_eq!(store.get_manifest("run1").await.unwrap(), None);
        store
            .set_manifest("run1", "triage", "hello world")
            .await
            .unwrap();
        assert_eq!(
            store.get_manifest("run1").await.unwrap(),
            Some(("triage".to_string(), "hello world".to_string()))
        );
    }

    #[tokio::test]
    async fn memory_store_conversation_id_round_trip() {
        let store = MemoryStore::default();
        assert_eq!(store.get_conversation_id("run1").await.unwrap(), None);
        store.set_conversation_id("run1", "conv-abc").await.unwrap();
        assert_eq!(
            store.get_conversation_id("run1").await.unwrap(),
            Some("conv-abc".to_string())
        );
        // a different run is a separate value
        assert_eq!(store.get_conversation_id("run2").await.unwrap(), None);
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_conversation_id_round_trip() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        let run_id = format!(
            "conv_id_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        assert_eq!(store.get_conversation_id(&run_id).await.unwrap(), None);
        store
            .set_conversation_id(&run_id, "conv-xyz")
            .await
            .unwrap();
        assert_eq!(
            store.get_conversation_id(&run_id).await.unwrap(),
            Some("conv-xyz".to_string())
        );
    }

    #[tokio::test]
    async fn memory_store_context_round_trip() {
        let store = MemoryStore::default();
        assert_eq!(store.get_context("run1").await.unwrap(), None);
        let ctx = serde_json::json!({ "projectId": "p1" });
        store.set_context("run1", &ctx).await.unwrap();
        assert_eq!(store.get_context("run1").await.unwrap(), Some(ctx));
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_context_round_trip() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        let run_id = format!(
            "context_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        assert_eq!(store.get_context(&run_id).await.unwrap(), None);
        let ctx = serde_json::json!({ "projectId": "p1", "scope": ["a", "b"] });
        store.set_context(&run_id, &ctx).await.unwrap();
        assert_eq!(store.get_context(&run_id).await.unwrap(), Some(ctx));
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_append_keeps_every_run_key_alive() {
        // A run's keys are written once at creation but must all live as long
        // as the run keeps producing events: a long run otherwise lost its
        // status mid-run, and a pause late in the TTL left minutes to approve.
        use redis::AsyncCommands;
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 600).await.unwrap();
        let run_id = format!(
            "ttl_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        store.set_status(&run_id, RunStatus::Running).await.unwrap();
        store.set_manifest(&run_id, "wf", "hi").await.unwrap();
        store.set_conversation_id(&run_id, "c1").await.unwrap();
        store
            .set_context(&run_id, &serde_json::json!({ "a": 1 }))
            .await
            .unwrap();
        store.record_approval(&run_id, "gate").await.unwrap();
        let keys = [
            RedisStore::meta_key(&run_id),
            RedisStore::manifest_key(&run_id),
            RedisStore::conversation_key(&run_id),
            RedisStore::context_key(&run_id),
            RedisStore::approvals_key(&run_id),
        ];
        let mut conn = store.conn.clone();
        for key in &keys {
            let _: () = conn.expire(key, 5).await.unwrap();
        }
        store
            .append(&run_id, &WorkflowEvent::StageStarted { stage: "s".into() })
            .await
            .unwrap();
        for key in &keys {
            let ttl: i64 = conn.ttl(key).await.unwrap();
            assert!(ttl > 500, "{key} ttl {ttl} was not refreshed by append");
        }
    }

    #[tokio::test]
    async fn connect_to_an_unreachable_redis_fails_within_the_timeout() {
        // 192.0.2.1 is TEST-NET-1 (RFC 5737): unroutable, so the attempt hangs
        // rather than being refused — exactly the case that used to retry forever.
        // This test deliberately takes ~5s (the connect timeout).
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            RedisStore::connect("redis://192.0.2.1:6379", 60),
        )
        .await;
        let inner = result.expect("connect must return within its own 5s cap, not hang");
        let err = inner.expect_err("unreachable redis must be an error");
        let msg = err.to_string();
        assert!(msg.contains("192.0.2.1"), "error must name the URL: {msg}");
        assert!(
            msg.contains("redis.url"),
            "error must name the config key: {msg}"
        );
    }

    async fn claims_a_key_once<S: EventStore>(store: &S, key: &str) {
        assert_eq!(
            store
                .claim_idempotency_key("runs", key, "r1", "h1")
                .await
                .unwrap(),
            IdempotencyClaim::Claimed
        );
        assert_eq!(
            store
                .claim_idempotency_key("runs", key, "r2", "h2")
                .await
                .unwrap(),
            IdempotencyClaim::Existing {
                run_id: "r1".into(),
                body_hash: "h1".into()
            }
        );
    }

    async fn release_frees_the_key<S: EventStore>(store: &S, key: &str) {
        store
            .claim_idempotency_key("runs", key, "r1", "h1")
            .await
            .unwrap();
        store.release_idempotency_key("runs", key).await.unwrap();
        assert_eq!(
            store
                .claim_idempotency_key("runs", key, "r2", "h2")
                .await
                .unwrap(),
            IdempotencyClaim::Claimed
        );
    }

    #[tokio::test]
    async fn memory_store_claims_a_key_once() {
        claims_a_key_once(&MemoryStore::default(), "k").await;
    }

    #[tokio::test]
    async fn memory_store_release_frees_the_key() {
        release_frees_the_key(&MemoryStore::default(), "k").await;
    }

    #[tokio::test]
    async fn claims_are_scoped() {
        let store = MemoryStore::default();
        for scope in ["runs", "chat"] {
            assert_eq!(
                store
                    .claim_idempotency_key(scope, "k", "r", "h")
                    .await
                    .unwrap(),
                IdempotencyClaim::Claimed
            );
        }
    }

    fn unique(prefix: &str) -> String {
        format!(
            "{prefix}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_claims_a_key_once() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        claims_a_key_once(&store, &unique("idem")).await;
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_release_frees_the_key() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let store = RedisStore::connect(&url, 60).await.unwrap();
        release_frees_the_key(&store, &unique("idem")).await;
    }

    #[test]
    fn run_status_round_trips_cancelling_and_cancelled() {
        for st in [RunStatus::Cancelling, RunStatus::Cancelled] {
            assert_eq!(RunStatus::from_str(st.as_str()), Some(st));
        }
        assert_eq!(RunStatus::Cancelling.as_str(), "cancelling");
        assert_eq!(RunStatus::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn is_terminal_detects_terminal_events() {
        assert!(is_terminal(&WorkflowEvent::WorkflowCancelled));
        assert!(is_terminal(&WorkflowEvent::WorkflowComplete));
        assert!(is_terminal(&WorkflowEvent::WorkflowFailed {
            stage: "x".into(),
            error: "boom".into(),
        }));
        assert!(!is_terminal(&WorkflowEvent::StageStarted {
            stage: "x".into()
        }));
        assert!(!is_terminal(&WorkflowEvent::StageComplete {
            stage: "x".into(),
            output: AgentOutput {
                message: "m".into(),
                structured: None
            },
        }));
    }

    #[test]
    fn refine_round_is_not_terminal() {
        assert!(!is_terminal(&WorkflowEvent::RefineRound {
            stage: "x".into(),
            round: 1,
            passed: false,
            feedback: Some("no".into()),
        }));
    }
}
