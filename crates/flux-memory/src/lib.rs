#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! Conversational turn storage: a `ConversationMemory` trait with in-memory,
//! Redis (hot), and Postgres (durable) backends, plus a tiering wrapper.

use flux_core::error::{FluxError, Result};
use redis::AsyncCommands;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Mutex;

pub use flux_core::conversation::{Role, Turn};

/// Append-only per-conversation turn log with a recent-window read. Async
/// methods follow the project's `-> impl Future + Send` trait style (used via
/// generics, never `dyn`).
pub trait ConversationMemory {
    /// Append one turn to the end of a conversation.
    fn append_turn(
        &self,
        conversation_id: &str,
        turn: &Turn,
    ) -> impl Future<Output = Result<()>> + Send;

    /// The last `limit` turns, oldest-first (fewer if the conversation is
    /// shorter; empty for an unknown conversation or `limit == 0`).
    fn recent(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<Turn>>> + Send;

    /// Append several turns in order. The default is the sequential per-turn
    /// loop; backends override it to batch (one statement / one pipeline per
    /// exchange), which also makes the write all-or-nothing per store.
    fn append_turns(
        &self,
        conversation_id: &str,
        turns: &[Turn],
    ) -> impl Future<Output = Result<()>> + Send
    where
        Self: Sync,
    {
        async move {
            for turn in turns {
                self.append_turn(conversation_id, turn).await?;
            }
            Ok(())
        }
    }
}

/// In-memory `ConversationMemory` for tests and single-process use. Clone shares
/// the backing map via `Arc`.
#[derive(Clone, Default)]
pub struct InMemoryMemory {
    convs: Arc<Mutex<HashMap<String, Vec<Turn>>>>,
}

impl ConversationMemory for InMemoryMemory {
    async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
        let mut convs = self.convs.lock().await;
        convs
            .entry(conversation_id.to_string())
            .or_default()
            .push(turn.clone());
        Ok(())
    }

    async fn recent(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let convs = self.convs.lock().await;
        let Some(turns) = convs.get(conversation_id) else {
            return Ok(Vec::new());
        };
        let start = turns.len().saturating_sub(limit);
        Ok(trim_to_exchange_boundary(turns[start..].to_vec()))
    }
}

/// A no-op `ConversationMemory`: reads return empty, writes are dropped. Used as
/// a sentinel so generic callers can pass a real `&M` when no memory is wanted.
#[derive(Clone, Copy, Default)]
pub struct NoMemory;

impl ConversationMemory for NoMemory {
    async fn append_turn(&self, _conversation_id: &str, _turn: &Turn) -> Result<()> {
        Ok(())
    }
    async fn recent(&self, _conversation_id: &str, _limit: usize) -> Result<Vec<Turn>> {
        Ok(Vec::new())
    }
}

/// Redis-backed hot buffer. Each conversation is a LIST at
/// `flux:conv:{id}:turns`, capped to the last `window_cap` turns and expiring
/// after `ttl_secs` of inactivity. Clone shares the connection manager.
#[derive(Clone)]
pub struct RedisMemory {
    conn: redis::aio::ConnectionManager,
    ttl_secs: u64,
    window_cap: isize,
}

/// How long a Redis connection attempt may take before failing loudly.
/// `ConnectionManager::new` otherwise retries forever, which turned a
/// missing local Redis into a silent, indefinite hang at server boot.
const REDIS_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl RedisMemory {
    pub async fn connect(redis_url: &str, ttl_secs: u64, window_cap: usize) -> Result<RedisMemory> {
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
        Ok(RedisMemory {
            conn,
            ttl_secs,
            window_cap: window_cap as isize,
        })
    }

    fn key(conversation_id: &str) -> String {
        format!("flux:conv:{conversation_id}:turns")
    }
}

impl ConversationMemory for RedisMemory {
    async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
        let mut conn = self.conn.clone();
        let key = Self::key(conversation_id);
        let payload = serde_json::to_string(turn).map_err(|e| FluxError::Decode(e.to_string()))?;
        let _: () = conn
            .rpush(&key, payload)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        // Bound hot memory to the last `window_cap` turns.
        let _: () = conn
            .ltrim(&key, -self.window_cap, -1)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let _: () = conn
            .expire(&key, self.ttl_secs as i64)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn recent(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut conn = self.conn.clone();
        let key = Self::key(conversation_id);
        // LRANGE with negative indices returns the last `limit`, oldest-first.
        let raw: Vec<String> = conn
            .lrange(&key, -(limit as isize), -1)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        let mut out = Vec::with_capacity(raw.len());
        for s in raw {
            let turn: Turn =
                serde_json::from_str(&s).map_err(|e| FluxError::Decode(e.to_string()))?;
            out.push(turn);
        }
        Ok(trim_to_exchange_boundary(out))
    }

    async fn append_turns(&self, conversation_id: &str, turns: &[Turn]) -> Result<()> {
        if turns.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.clone();
        let key = Self::key(conversation_id);
        // Serialize first so an encoding error writes nothing at all.
        let mut payloads = Vec::with_capacity(turns.len());
        for turn in turns {
            payloads
                .push(serde_json::to_string(turn).map_err(|e| FluxError::Decode(e.to_string()))?);
        }
        let mut pipe = redis::pipe();
        for payload in &payloads {
            pipe.rpush(&key, payload).ignore();
        }
        pipe.ltrim(&key, -self.window_cap, -1).ignore();
        pipe.expire(&key, self.ttl_secs as i64).ignore();
        let _: () = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }
}

/// Postgres-backed durable system-of-record. Turns live in `conversation_turns`,
/// ordered by a global `BIGSERIAL` (no per-conversation index race). Schema is
/// created on connect. Clone shares the connection pool.
#[derive(Clone)]
pub struct PostgresMemory {
    pool: sqlx::PgPool,
}

impl PostgresMemory {
    pub async fn connect(
        database_url: &str,
        max_connections: u32,
        acquire_timeout_secs: u64,
    ) -> Result<PostgresMemory> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(max_connections)
            .acquire_timeout(std::time::Duration::from_secs(acquire_timeout_secs))
            .connect(database_url)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversation_turns (\
                 id BIGSERIAL PRIMARY KEY, \
                 conversation_id TEXT NOT NULL, \
                 role TEXT NOT NULL, \
                 content TEXT NOT NULL, \
                 tool_calls JSONB, \
                 tool_results JSONB, \
                 created_at TIMESTAMPTZ NOT NULL DEFAULT now())",
        )
        .execute(&pool)
        .await
        .map_err(|e| FluxError::Config(e.to_string()))?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS conversation_turns_cid_id \
                 ON conversation_turns (conversation_id, id)",
        )
        .execute(&pool)
        .await
        .map_err(|e| FluxError::Config(e.to_string()))?;
        // The CREATE TABLE above is IF NOT EXISTS, so it will not alter an
        // existing table. Add the tool columns explicitly.
        for stmt in [
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS tool_calls JSONB",
            "ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS tool_results JSONB",
        ] {
            sqlx::query(stmt)
                .execute(&pool)
                .await
                .map_err(|e| FluxError::Config(e.to_string()))?;
        }
        Ok(PostgresMemory { pool })
    }
}

impl ConversationMemory for PostgresMemory {
    async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
        let calls = if turn.tool_calls.is_empty() {
            None
        } else {
            Some(
                serde_json::to_value(&turn.tool_calls)
                    .map_err(|e| FluxError::Decode(e.to_string()))?,
            )
        };
        let results = if turn.tool_results.is_empty() {
            None
        } else {
            Some(
                serde_json::to_value(&turn.tool_results)
                    .map_err(|e| FluxError::Decode(e.to_string()))?,
            )
        };
        sqlx::query(
            "INSERT INTO conversation_turns \
                 (conversation_id, role, content, tool_calls, tool_results) \
                 VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(conversation_id)
        .bind(turn.role.as_str())
        .bind(&turn.content)
        .bind(calls)
        .bind(results)
        .execute(&self.pool)
        .await
        .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }

    async fn recent(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        use sqlx::Row;
        let rows = sqlx::query(
            "SELECT role, content, tool_calls, tool_results FROM conversation_turns \
                 WHERE conversation_id = $1 ORDER BY id DESC LIMIT $2",
        )
        .bind(conversation_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| FluxError::Config(e.to_string()))?;
        // Query is DESC (newest first); reverse to oldest-first.
        let mut out = Vec::with_capacity(rows.len());
        for row in rows.iter().rev() {
            let role_s: String = row
                .try_get("role")
                .map_err(|e| FluxError::Decode(e.to_string()))?;
            let content: String = row
                .try_get("content")
                .map_err(|e| FluxError::Decode(e.to_string()))?;
            let role = Role::from_str(&role_s)
                .ok_or_else(|| FluxError::Decode(format!("unknown role '{role_s}'")))?;
            let calls: Option<serde_json::Value> = row
                .try_get("tool_calls")
                .map_err(|e| FluxError::Decode(e.to_string()))?;
            let results: Option<serde_json::Value> = row
                .try_get("tool_results")
                .map_err(|e| FluxError::Decode(e.to_string()))?;
            out.push(Turn {
                role,
                content,
                tool_calls: match calls {
                    Some(v) => {
                        serde_json::from_value(v).map_err(|e| FluxError::Decode(e.to_string()))?
                    }
                    None => Vec::new(),
                },
                tool_results: match results {
                    Some(v) => {
                        serde_json::from_value(v).map_err(|e| FluxError::Decode(e.to_string()))?
                    }
                    None => Vec::new(),
                },
            });
        }
        Ok(trim_to_exchange_boundary(out))
    }

    async fn append_turns(&self, conversation_id: &str, turns: &[Turn]) -> Result<()> {
        if turns.is_empty() {
            return Ok(());
        }
        // Serialize first so an encoding error writes nothing at all.
        let mut rows: Vec<(&Turn, Option<serde_json::Value>, Option<serde_json::Value>)> =
            Vec::with_capacity(turns.len());
        for turn in turns {
            let calls = if turn.tool_calls.is_empty() {
                None
            } else {
                Some(
                    serde_json::to_value(&turn.tool_calls)
                        .map_err(|e| FluxError::Decode(e.to_string()))?,
                )
            };
            let results = if turn.tool_results.is_empty() {
                None
            } else {
                Some(
                    serde_json::to_value(&turn.tool_results)
                        .map_err(|e| FluxError::Decode(e.to_string()))?,
                )
            };
            rows.push((turn, calls, results));
        }
        let mut qb = sqlx::QueryBuilder::new(
            "INSERT INTO conversation_turns \
                 (conversation_id, role, content, tool_calls, tool_results) ",
        );
        qb.push_values(rows, |mut b, (turn, calls, results)| {
            b.push_bind(conversation_id)
                .push_bind(turn.role.as_str())
                .push_bind(turn.content.clone())
                .push_bind(calls)
                .push_bind(results);
        });
        qb.build()
            .execute(&self.pool)
            .await
            .map_err(|e| FluxError::Config(e.to_string()))?;
        Ok(())
    }
}

/// Tiers a hot store `H` over a durable store `D`. Writes go through to the
/// durable store first (system-of-record), then best-effort to hot. Reads
/// prefer hot and fall back to durable when hot is empty (e.g. TTL-expired).
pub struct TieredMemory<H, D> {
    hot: H,
    durable: D,
}

impl<H, D> TieredMemory<H, D> {
    pub fn new(hot: H, durable: D) -> TieredMemory<H, D> {
        TieredMemory { hot, durable }
    }
}

impl<H, D> ConversationMemory for TieredMemory<H, D>
where
    H: ConversationMemory + Send + Sync,
    D: ConversationMemory + Send + Sync,
{
    async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
        // Durable system-of-record first: a failure here fails the append.
        self.durable.append_turn(conversation_id, turn).await?;
        // Hot is best-effort; durability is already guaranteed by the durable write.
        if let Err(e) = self.hot.append_turn(conversation_id, turn).await {
            tracing::warn!(
                conversation_id,
                error = %e,
                "hot store append failed; durable write succeeded"
            );
        }
        Ok(())
    }

    async fn append_turns(&self, conversation_id: &str, turns: &[Turn]) -> Result<()> {
        // Durable system-of-record first, as one batch; hot is best-effort.
        self.durable.append_turns(conversation_id, turns).await?;
        if let Err(e) = self.hot.append_turns(conversation_id, turns).await {
            tracing::warn!(
                conversation_id,
                error = %e,
                "hot store batch append failed; durable write succeeded"
            );
        }
        Ok(())
    }

    async fn recent(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
        let hot = self.hot.recent(conversation_id, limit).await?;
        // A hot buffer holding *fewer* turns than requested may be a freshly
        // repopulated buffer after TTL expiry / restart — the full history
        // lives only in the durable store, and "non-empty" used to be enough
        // to skip it, giving the conversation permanent amnesia. Only a hot
        // window that satisfies the request outright can skip the durable read.
        if hot.len() >= limit {
            return Ok(hot);
        }
        match self.durable.recent(conversation_id, limit).await {
            Ok(durable) if durable.len() > hot.len() => Ok(durable),
            Ok(_) => Ok(hot),
            Err(e) if !hot.is_empty() => {
                tracing::warn!(
                    conversation_id,
                    error = %e,
                    "durable recall failed; serving the partial hot window"
                );
                Ok(hot)
            }
            Err(e) => Err(e),
        }
    }
}

/// Reduce a turn-count-sliced window to one a provider will accept: every
/// tool call paired with its result, and a plain user turn first.
///
/// Two independent defects have to be removed.
///
/// **Unmatched tool activity.** A window can contain a user turn carrying
/// `tool_results` whose matching assistant `tool_calls` fell outside the slice,
/// or an assistant `tool_calls` turn whose `tool_results` never got appended (a
/// partial write between the two — see `append_exchange` in
/// flux-orchestrator). Providers reject either half on its own, so any turn
/// whose tool ids have no counterpart *inside the window* is dropped. Position
/// is irrelevant: an orphan left in the middle of the log by a failed append
/// is replayed on every later recall and fails every one of them, which is why
/// matching on ids beats trimming the ends.
///
/// **Leading role.** The window must open on a plain user turn (no
/// `tool_results`), so trim forward to the first one.
///
/// The two passes interact — dropping an orphan can expose a new boundary, and
/// trimming a prefix can orphan a result whose call was in that prefix — so
/// they alternate until neither removes anything.
///
/// A window lying entirely inside one tool-calling exchange has no acceptable
/// prefix at all and reduces to empty. That is a slice too narrow to replay,
/// not a corrupt log; `recall_window` retries such a window wider rather than
/// letting the caller lose the history.
pub fn trim_to_exchange_boundary(turns: Vec<Turn>) -> Vec<Turn> {
    let original = turns.len();
    let mut turns = turns;
    loop {
        let before = turns.len();
        turns = drop_unmatched_tool_activity(turns);
        turns = trim_leading_to_plain_user(turns);
        if turns.len() == before {
            break;
        }
    }
    if turns.is_empty() && original > 0 {
        tracing::warn!(
            total_turns = original,
            "trimmed window to empty; no clean exchange boundary found"
        );
    } else if turns.len() < original {
        tracing::debug!(
            dropped_turns = original - turns.len(),
            "trimmed turns to exchange boundary"
        );
    }
    turns
}

/// Drop every turn carrying a tool call with no matching result, or a tool
/// result with no matching call, considering only ids present in this window.
/// Dropping a turn can orphan another, so iterate to a fixpoint.
fn drop_unmatched_tool_activity(mut turns: Vec<Turn>) -> Vec<Turn> {
    loop {
        // Scoped so the id borrows end before `turns` is rebuilt below.
        let keep: Vec<bool> = {
            let call_ids: std::collections::HashSet<&str> = turns
                .iter()
                .flat_map(|t| t.tool_calls.iter().map(|c| c.id.as_str()))
                .collect();
            let result_ids: std::collections::HashSet<&str> = turns
                .iter()
                .flat_map(|t| t.tool_results.iter().map(|r| r.id.as_str()))
                .collect();
            turns
                .iter()
                .map(|t| {
                    t.tool_calls
                        .iter()
                        .all(|c| result_ids.contains(c.id.as_str()))
                        && t.tool_results
                            .iter()
                            .all(|r| call_ids.contains(r.id.as_str()))
                })
                .collect()
        };
        if keep.iter().all(|k| *k) {
            return turns;
        }
        let mut kept = Vec::with_capacity(turns.len());
        for (turn, keep) in turns.into_iter().zip(keep) {
            if keep {
                kept.push(turn);
            }
        }
        turns = kept;
    }
}

/// Drop leading turns until the window opens on a user turn carrying no
/// `tool_results`; empty if it has none.
fn trim_leading_to_plain_user(turns: Vec<Turn>) -> Vec<Turn> {
    match turns
        .iter()
        .position(|t| t.role == Role::User && t.tool_results.is_empty())
    {
        Some(0) => turns,
        Some(i) => turns[i..].to_vec(),
        None => Vec::new(),
    }
}

/// How much wider than the requested window `recall_window` may read when the
/// requested window trims to nothing. Four covers the default 4-iteration tool
/// loop (one user turn plus two turns per iteration) at any window size.
const WIDEN_FACTOR: usize = 4;

/// Read a recall window, retrying wider when the requested width trims to
/// nothing.
///
/// `recent` returns a `trim_to_exchange_boundary`-reduced slice, and a window
/// that lands wholly inside one tool-calling exchange reduces to empty — the
/// turns that would make it replayable are just outside the slice, so the
/// caller would see total amnesia despite a healthy log. Reading wider
/// recovers it. Overshooting `history_window` on those turns is a better trade
/// than replaying no history at all.
///
/// An empty first read is ambiguous — a trimmed-away window and a conversation
/// with no turns look alike from here — so a brand-new conversation costs one
/// extra empty read. That is an indexed miss, and it buys not having to thread
/// the distinction through all four backends.
pub async fn recall_window<M: ConversationMemory + Sync>(
    memory: &M,
    key: &str,
    window: usize,
) -> Result<Vec<Turn>> {
    let turns = memory.recent(key, window).await?;
    if !turns.is_empty() || window == 0 {
        return Ok(turns);
    }
    memory
        .recent(key, window.saturating_mul(WIDEN_FACTOR))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(role: Role, c: &str) -> Turn {
        Turn::text(role, c)
    }

    fn assistant_with_call(id: &str) -> Turn {
        Turn {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: vec![flux_core::conversation::ToolCallRecord {
                id: id.into(),
                name: "t".into(),
                args: serde_json::json!({}),
            }],
            tool_results: Vec::new(),
        }
    }

    fn user_with_result(id: &str) -> Turn {
        Turn {
            role: Role::User,
            content: String::new(),
            tool_calls: Vec::new(),
            tool_results: vec![flux_core::conversation::ToolResultRecord {
                id: id.into(),
                name: "t".into(),
                result: serde_json::json!({ "ok": true }),
            }],
        }
    }

    #[test]
    fn role_round_trips() {
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::from_str("user"), Some(Role::User));
        assert_eq!(Role::from_str("assistant"), Some(Role::Assistant));
        assert_eq!(Role::from_str("nope"), None);
    }

    #[tokio::test]
    async fn no_memory_is_noop() {
        let m = NoMemory;
        m.append_turn("c", &t(Role::User, "x")).await.unwrap();
        assert!(m.recent("c", 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn in_memory_appends_and_reads_window_oldest_first() {
        let m = InMemoryMemory::default();
        m.append_turn("c1", &t(Role::User, "hi")).await.unwrap();
        m.append_turn("c1", &t(Role::Assistant, "hello"))
            .await
            .unwrap();
        m.append_turn("c1", &t(Role::User, "bye")).await.unwrap();
        m.append_turn("c1", &t(Role::Assistant, "ok"))
            .await
            .unwrap();

        let all = m.recent("c1", 10).await.unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], t(Role::User, "hi"));
        assert_eq!(all[2], t(Role::User, "bye"));

        // Window of last 2 turns is [User, Assistant], which is trim-neutral
        // (starts on clean user turn) and pins both size and order.
        let win = m.recent("c1", 2).await.unwrap();
        assert_eq!(win, vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]);

        // Window of last 3 turns is [Assistant, User, Assistant], trimmed to
        // [User, Assistant] (starts with Assistant, first user turn is "bye").
        let trimmed = m.recent("c1", 3).await.unwrap();
        assert_eq!(
            trimmed,
            vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]
        );

        assert!(m.recent("c1", 0).await.unwrap().is_empty());
        assert!(m.recent("nope", 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn in_memory_round_trips_a_tool_turn() {
        let mem = InMemoryMemory::default();
        // Start with a user turn to establish a valid conversation
        mem.append_turn("k", &t(Role::User, "q")).await.unwrap();
        let turn = Turn {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: vec![flux_core::conversation::ToolCallRecord {
                id: "c1".into(),
                name: "echo".into(),
                args: serde_json::json!({ "text": "hi" }),
            }],
            tool_results: Vec::new(),
        };
        mem.append_turn("k", &turn).await.unwrap();
        // Followed by its matching result, so the tool_calls turn isn't the
        // trailing turn (an unmatched trailing tool_calls turn is now
        // trimmed — see the trailing-trim tests below).
        mem.append_turn("k", &user_with_result("c1")).await.unwrap();
        let got = mem.recent("k", 10).await.unwrap();
        assert_eq!(got, vec![t(Role::User, "q"), turn, user_with_result("c1")]);
    }

    #[tokio::test]
    #[ignore = "requires a live redis-server (cargo test -- --ignored)"]
    async fn redis_memory_window_round_trips() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let m = RedisMemory::connect(&url, 60, 100).await.unwrap();
        let cid = format!(
            "flux_mem_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        m.append_turn(&cid, &t(Role::User, "hi")).await.unwrap();
        m.append_turn(&cid, &t(Role::Assistant, "hello"))
            .await
            .unwrap();
        m.append_turn(&cid, &t(Role::User, "bye")).await.unwrap();
        m.append_turn(&cid, &t(Role::Assistant, "ok"))
            .await
            .unwrap();

        // Window of last 2 turns is [User, Assistant], which is trim-neutral
        // (starts on clean user turn) and pins both size and order.
        let win = m.recent(&cid, 2).await.unwrap();
        assert_eq!(win, vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]);

        // Window of last 3 turns is [Assistant, User, Assistant], trimmed to
        // [User, Assistant] (starts with Assistant, first user turn is "bye").
        let trimmed = m.recent(&cid, 3).await.unwrap();
        assert_eq!(
            trimmed,
            vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]
        );

        assert!(m.recent(&format!("{cid}_x"), 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    #[ignore = "requires a live postgres (cargo test -- --ignored)"]
    async fn postgres_memory_window_round_trips() {
        let url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres@127.0.0.1:5432/flux_memory_test".to_string());
        let m = PostgresMemory::connect(&url, 10, 30).await.unwrap();
        let cid = format!(
            "flux_mem_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        m.append_turn(&cid, &t(Role::User, "hi")).await.unwrap();
        m.append_turn(&cid, &t(Role::Assistant, "hello"))
            .await
            .unwrap();
        m.append_turn(&cid, &t(Role::User, "bye")).await.unwrap();
        m.append_turn(&cid, &t(Role::Assistant, "ok"))
            .await
            .unwrap();

        // Window of last 2 turns is [User, Assistant], which is trim-neutral
        // (starts on clean user turn) and pins both size and order.
        let win = m.recent(&cid, 2).await.unwrap();
        assert_eq!(win, vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]);

        // Window of last 3 turns is [Assistant, User, Assistant], trimmed to
        // [User, Assistant] (starts with Assistant, first user turn is "bye").
        let trimmed = m.recent(&cid, 3).await.unwrap();
        assert_eq!(
            trimmed,
            vec![t(Role::User, "bye"), t(Role::Assistant, "ok")]
        );

        assert!(m.recent(&format!("{cid}_x"), 5).await.unwrap().is_empty());

        // Clean up this run's rows.
        sqlx::query("DELETE FROM conversation_turns WHERE conversation_id = $1")
            .bind(&cid)
            .execute(&m.pool)
            .await
            .unwrap();
    }

    /// A durable store that always errors — used to prove write-through fails
    /// the whole append when the system-of-record is unavailable.
    #[derive(Clone, Default)]
    struct FailingStore;

    impl ConversationMemory for FailingStore {
        async fn append_turn(&self, _conversation_id: &str, _turn: &Turn) -> Result<()> {
            Err(FluxError::Config("boom".into()))
        }
        async fn recent(&self, _conversation_id: &str, _limit: usize) -> Result<Vec<Turn>> {
            Err(FluxError::Config("boom".into()))
        }
    }

    #[tokio::test]
    async fn tiered_write_through_writes_both() {
        let hot = InMemoryMemory::default();
        let durable = InMemoryMemory::default();
        let tier = TieredMemory::new(hot.clone(), durable.clone());
        tier.append_turn("c1", &t(Role::User, "hi")).await.unwrap();
        assert_eq!(
            hot.recent("c1", 10).await.unwrap(),
            vec![t(Role::User, "hi")]
        );
        assert_eq!(
            durable.recent("c1", 10).await.unwrap(),
            vec![t(Role::User, "hi")]
        );
    }

    #[tokio::test]
    async fn tiered_recent_falls_back_to_durable_when_hot_empty() {
        let hot = InMemoryMemory::default(); // stays empty
        let durable = InMemoryMemory::default();
        // Add a user turn first to establish a valid conversation
        durable
            .append_turn("c1", &t(Role::User, "q"))
            .await
            .unwrap();
        durable
            .append_turn("c1", &t(Role::Assistant, "from-durable"))
            .await
            .unwrap();
        let tier = TieredMemory::new(hot, durable);
        assert_eq!(
            tier.recent("c1", 10).await.unwrap(),
            vec![t(Role::User, "q"), t(Role::Assistant, "from-durable")]
        );
    }

    #[tokio::test]
    async fn tiered_recent_prefers_durable_when_hot_is_partial() {
        // Post-TTL shape: hot holds only the latest exchange, durable has the
        // full history. Reads must not get amnesia.
        let hot = InMemoryMemory::default();
        let durable = InMemoryMemory::default();
        for (store, n) in [(&durable, 6usize), (&hot, 2usize)] {
            for i in 0..n / 2 {
                store
                    .append_turn("c1", &t(Role::User, &format!("q{i}")))
                    .await
                    .unwrap();
                store
                    .append_turn("c1", &t(Role::Assistant, &format!("a{i}")))
                    .await
                    .unwrap();
            }
        }
        let tier = TieredMemory::new(hot, durable);
        let got = tier.recent("c1", 10).await.unwrap();
        assert_eq!(
            got.len(),
            6,
            "durable's fuller history must win over a partial hot buffer"
        );
    }

    #[tokio::test]
    async fn tiered_recent_degrades_to_hot_when_durable_errors() {
        let hot = InMemoryMemory::default();
        hot.append_turn("c1", &t(Role::User, "q")).await.unwrap();
        hot.append_turn("c1", &t(Role::Assistant, "a"))
            .await
            .unwrap();
        let tier = TieredMemory::new(hot, FailingStore);
        let got = tier.recent("c1", 10).await.unwrap();
        assert_eq!(
            got.len(),
            2,
            "a durable outage must not erase a usable hot window"
        );
    }

    #[tokio::test]
    async fn tiered_append_fails_when_durable_fails() {
        let tier = TieredMemory::new(InMemoryMemory::default(), FailingStore);
        assert!(tier.append_turn("c1", &t(Role::User, "hi")).await.is_err());
    }

    #[tokio::test]
    async fn append_turns_default_appends_in_order() {
        let m = InMemoryMemory::default();
        let turns = vec![t(Role::User, "q"), t(Role::Assistant, "a")];
        m.append_turns("c1", &turns).await.unwrap();
        assert_eq!(m.recent("c1", 10).await.unwrap(), turns);
    }

    #[tokio::test]
    async fn tiered_append_turns_writes_durable_first_and_hot_best_effort() {
        // Hot failure must not fail the batch — durable already holds it.
        #[derive(Clone, Default)]
        struct FailingHot;
        impl ConversationMemory for FailingHot {
            async fn append_turn(&self, _c: &str, _t: &Turn) -> Result<()> {
                Err(FluxError::Config("hot down".into()))
            }
            async fn recent(&self, _c: &str, _l: usize) -> Result<Vec<Turn>> {
                Ok(Vec::new())
            }
        }
        let durable = InMemoryMemory::default();
        let tier = TieredMemory::new(FailingHot, durable.clone());
        let turns = vec![t(Role::User, "q"), t(Role::Assistant, "a")];
        tier.append_turns("c1", &turns).await.unwrap();
        assert_eq!(durable.recent("c1", 10).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn tiered_append_turns_fails_when_durable_fails() {
        let tier = TieredMemory::new(InMemoryMemory::default(), FailingStore);
        assert!(tier
            .append_turns("c1", &[t(Role::User, "q")])
            .await
            .is_err());
    }

    #[test]
    fn trim_drops_a_leading_orphan_tool_result() {
        let turns = vec![
            user_with_result("c1"), // orphan: its tool_call was cut off
            Turn::text(Role::Assistant, "answer"),
            Turn::text(Role::User, "next question"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "next question");
    }

    #[test]
    fn trim_keeps_a_complete_exchange() {
        let turns = vec![
            Turn::text(Role::User, "question"),
            assistant_with_call("c1"),
            user_with_result("c1"),
            Turn::text(Role::Assistant, "answer"),
        ];
        let out = trim_to_exchange_boundary(turns.clone());
        assert_eq!(out, turns);
    }

    #[test]
    fn trim_of_plain_text_history_is_identity() {
        let turns = vec![
            Turn::text(Role::User, "q"),
            Turn::text(Role::Assistant, "a"),
        ];
        assert_eq!(trim_to_exchange_boundary(turns.clone()), turns);
    }

    #[test]
    fn trim_returns_empty_when_no_clean_boundary_exists() {
        let turns = vec![user_with_result("c1"), Turn::text(Role::Assistant, "a")];
        assert!(trim_to_exchange_boundary(turns).is_empty());
    }

    #[test]
    fn trim_assistant_leading_window_to_clean_user_turn() {
        let turns = vec![
            assistant_with_call("c1"),
            user_with_result("c1"),
            Turn::text(Role::Assistant, "answer"),
            Turn::text(Role::User, "next"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "next");
    }

    #[test]
    fn trim_assistant_leading_with_no_clean_boundary_returns_empty() {
        let turns = vec![assistant_with_call("c1"), user_with_result("c1")];
        assert!(trim_to_exchange_boundary(turns).is_empty());
    }

    #[test]
    fn trim_drops_a_trailing_orphan_tool_call() {
        // The log ends on an assistant tool_calls turn whose results never
        // arrived (the append that would carry them failed). Leading trim
        // can't help here — the defect is at the tail.
        let turns = vec![
            Turn::text(Role::User, "q"),
            Turn::text(Role::Assistant, "a1"),
            Turn::text(Role::User, "q2"),
            assistant_with_call("c2"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 3);
        assert_eq!(out.last().unwrap().content, "q2");
    }

    #[test]
    fn trim_of_a_log_with_only_a_trailing_orphan_tool_call_is_empty() {
        let turns = vec![assistant_with_call("c1")];
        assert!(trim_to_exchange_boundary(turns).is_empty());
    }

    #[test]
    fn trim_drops_an_interior_orphan_tool_call() {
        // `append_exchange` failed between the assistant tool_calls turn and
        // the tool_results turn, then a later exchange appended cleanly on top
        // — so the orphan is now in the *middle* of the log. Trimming only the
        // ends leaves it there, and it is replayed into every later prompt: a
        // tool_use with no matching tool_result, which providers reject. That
        // fails every subsequent turn of the conversation, permanently.
        let turns = vec![
            Turn::text(Role::User, "q1"),
            Turn::text(Role::Assistant, "a1"),
            assistant_with_call("orphan"), // results never arrived
            Turn::text(Role::User, "q2"),
            Turn::text(Role::Assistant, "a2"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|t| t.tool_calls.is_empty()));
        assert_eq!(out[0].content, "q1");
        assert_eq!(out[3].content, "a2");
    }

    #[test]
    fn trim_drops_an_interior_orphan_tool_result() {
        // Mirror case: a tool_results turn whose call is gone.
        let turns = vec![
            Turn::text(Role::User, "q1"),
            user_with_result("orphan"),
            Turn::text(Role::Assistant, "a1"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| t.tool_results.is_empty()));
    }

    #[test]
    fn trim_keeps_a_matched_pair_while_dropping_an_unmatched_one() {
        let turns = vec![
            Turn::text(Role::User, "q"),
            assistant_with_call("kept"),
            user_with_result("kept"),
            assistant_with_call("orphan"),
            Turn::text(Role::User, "q2"),
        ];
        let out = trim_to_exchange_boundary(turns);
        assert_eq!(out.len(), 4);
        assert_eq!(out[1].tool_calls[0].id, "kept");
        assert_eq!(out[3].content, "q2");
    }

    #[tokio::test]
    async fn recall_window_widens_when_the_window_lands_inside_a_tool_loop() {
        // Two tool iterations store six turns. A window of 4 slices to
        // [user+results, assistant+calls, user+results, assistant] — which has
        // no replayable prefix at all, because the turns that would make one
        // are outside the slice. Trimming alone yields nothing, so a healthy
        // log reads back as total amnesia.
        let m = InMemoryMemory::default();
        for turn in [
            Turn::text(Role::User, "q"),
            assistant_with_call("c1"),
            user_with_result("c1"),
            assistant_with_call("c2"),
            user_with_result("c2"),
            Turn::text(Role::Assistant, "final"),
        ] {
            m.append_turn("c", &turn).await.unwrap();
        }

        // The narrow read on its own is empty — this is the defect.
        assert!(m.recent("c", 4).await.unwrap().is_empty());

        // Widening reaches the opening user turn and recovers the exchange.
        let got = recall_window(&m, "c", 4).await.unwrap();
        assert_eq!(got.len(), 6);
        assert_eq!(got[0].content, "q");
        assert_eq!(got[5].content, "final");
    }

    #[tokio::test]
    async fn recall_window_returns_empty_for_an_unknown_conversation() {
        let m = InMemoryMemory::default();
        assert!(recall_window(&m, "nope", 8).await.unwrap().is_empty());
        assert!(recall_window(&m, "nope", 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn recall_window_does_not_widen_a_healthy_window() {
        let m = InMemoryMemory::default();
        for turn in [
            Turn::text(Role::User, "q1"),
            Turn::text(Role::Assistant, "a1"),
            Turn::text(Role::User, "q2"),
            Turn::text(Role::Assistant, "a2"),
        ] {
            m.append_turn("c", &turn).await.unwrap();
        }
        // Window of 2 already trims clean, so the width is respected.
        let got = recall_window(&m, "c", 2).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].content, "q2");
    }
}
