#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod args;
mod refine;

use flux_config::AgentConfig;
use flux_core::conversation::Turn;
use flux_core::error::{FluxError, Result};
use flux_core::event::AgentEvent;
use flux_core::output::AgentOutput;
use flux_core::workflow_event::WorkflowEvent;
use flux_llm::LlmRegistry;
use flux_memory::ConversationMemory;
use flux_runner::run_agent;
use flux_tool_client::ToolTransport;
use flux_workflow::{ready_rounds, Workflow};
use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub use args::render_args;
pub use flux_memory::NoMemory;

/// Which of an agent run's turns get persisted. With `replay_tools` off, keep
/// only the outer user question and final assistant answer, stripped of tool
/// activity — the pre-F3 behavior.
pub(crate) fn turns_to_persist(turns: &[Turn], replay_tools: bool) -> Vec<Turn> {
    if replay_tools {
        return turns.to_vec();
    }
    turns
        .iter()
        .filter(|t| t.tool_calls.is_empty() && t.tool_results.is_empty())
        .map(|t| Turn::text(t.role, t.content.clone()))
        .collect()
}

/// Append one exchange's turns to memory as a single batch per store.
/// Batching stores (Postgres multi-row INSERT, Redis pipeline) make the write
/// all-or-nothing per store; across the two tiers it is still not
/// transactional — the durable write can land while the hot write fails —
/// but recall's `trim_to_exchange_boundary` tolerates that, and the tiered
/// read prefers the fuller store.
pub(crate) async fn append_exchange<M: ConversationMemory + Sync>(
    memory: &M,
    key: &str,
    turns: &[Turn],
) -> Result<()> {
    if let Err(e) = memory.append_turns(key, turns).await {
        tracing::error!(
            conversation_key = %key,
            turns = turns.len(),
            error = %e,
            "conversation exchange append failed"
        );
        return Err(e);
    }
    Ok(())
}

/// Runtime tuning threaded into a workflow run. `Default` preserves the
/// pre-config behavior (20-turn window, default retry).
#[derive(Debug, Clone, Copy)]
pub struct RunTuning {
    pub history_window: usize,
    pub retry: flux_runner::RetryConfig,
}

impl Default for RunTuning {
    fn default() -> Self {
        RunTuning {
            history_window: 20,
            retry: flux_runner::RetryConfig::default(),
        }
    }
}

/// Per-agent conversation memory key: one thread per (conversation, agent).
fn conversation_key(conversation_id: &str, agent_id: &str) -> String {
    format!("{conversation_id}:{agent_id}")
}

/// Whether running `wf` needs a tool transport: a stage calls a tool directly,
/// or a stage's agent declares tools. Scanning the whole registry instead
/// (as this check used to) made an unrelated tool-declaring agent file in the
/// agents dir block every workflow — including the shipped zero-tool samples.
pub fn needs_tool_transport(registry: &HashMap<String, AgentConfig>, wf: &Workflow) -> bool {
    wf.stages.iter().any(|s| s.tool.is_some())
        || stage_agent_ids(wf)
            .any(|agent| registry.get(agent).is_some_and(|cfg| !cfg.tools.is_empty()))
}

/// Every agent id a run of `wf` calls: each stage's `agent` and, for refine
/// stages, its `refine.evaluator`. Preflights that must cover every LLM call
/// iterate this rather than `stage.agent` alone.
pub fn stage_agent_ids(wf: &Workflow) -> impl Iterator<Item = &str> {
    wf.stages.iter().flat_map(|s| {
        s.agent
            .as_deref()
            .into_iter()
            .chain(s.refine.as_ref().map(|r| r.evaluator.as_str()))
    })
}

/// Env vars of provider API keys that are unset for the agents `wf` runs
/// (generators and evaluators), in first-seen order, without duplicates.
/// Unknown providers are an error. Agents missing from `registry` are skipped
/// (`validate_agent_refs` reports those).
pub fn missing_provider_keys(
    wf: &Workflow,
    registry: &HashMap<String, AgentConfig>,
    llm: &LlmRegistry,
) -> Result<Vec<String>> {
    let mut missing: Vec<String> = Vec::new();
    for agent in stage_agent_ids(wf) {
        let Some(cfg) = registry.get(agent) else {
            continue;
        };
        if let Some(var) = llm.check(&cfg.model)? {
            if !missing.contains(&var) {
                missing.push(var);
            }
        }
    }
    Ok(missing)
}

/// Load-time check: every agent stage must reference a registered agent.
/// Catching this at load beats catching it mid-run, after upstream stages
/// already burned real LLM calls.
pub fn validate_agent_refs(wf: &Workflow, registry: &HashMap<String, AgentConfig>) -> Result<()> {
    for stage in &wf.stages {
        if let Some(agent) = stage.agent.as_deref() {
            if !registry.contains_key(agent) {
                return Err(FluxError::Config(format!(
                    "stage '{}' references unknown agent '{}'",
                    stage.id, agent
                )));
            }
        }
        if let Some(refine) = &stage.refine {
            if !registry.contains_key(&refine.evaluator) {
                return Err(FluxError::Config(format!(
                    "stage '{}' references unknown evaluator agent '{}'",
                    stage.id, refine.evaluator
                )));
            }
        }
    }
    Ok(())
}

/// Build the minijinja context: { workflow: { input }, <stage_id>: { message, structured } }.
pub fn build_context(
    workflow_input: &str,
    completed: &HashMap<String, AgentOutput>,
    context: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut root = serde_json::Map::new();
    root.insert(
        "workflow".to_string(),
        serde_json::json!({ "input": workflow_input }),
    );
    if let Some(ctx) = context {
        root.insert("ctx".to_string(), ctx.clone());
    }
    for (id, out) in completed {
        root.insert(
            id.clone(),
            serde_json::json!({
                "message": out.message,
                "structured": out.structured,
            }),
        );
    }
    serde_json::Value::Object(root)
}

pub fn render_input(template: &str, ctx: &serde_json::Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.render_str(template, ctx)
        .map_err(|e| FluxError::Config(format!("input template error: {e}")))
}

/// Evaluate a `when` expression to a bool. None => true. A parse/eval error
/// (e.g. a missing field in an operator/coercion position, or a malformed
/// expression) is a Config error, surfaced rather than silently treated as
/// false — that is what catches a typo'd stage or field name instead of quietly
/// skipping the stage. `is defined` is the deliberate exception, yielding
/// `false` for a missing field rather than erroring.
///
/// The other exception is an expression reading a stage that failed under
/// `on_error: continue`. Such a stage's output is replaced wholesale by
/// `{ "error": … }`, so its normal fields are legitimately absent, and
/// strictness turned the natural spelling of a fallback branch —
/// `when: "execute.structured.status == 'ok'"` — into a fatal error that
/// aborted the whole run. That is precisely the opposite of what `on_error:
/// continue` is configured to achieve, and it made the feature usable only via
/// `is defined`. An undefined reference reached through a failed stage is
/// therefore falsy, while a typo (naming a stage that does not exist, or a
/// missing field on a stage that *succeeded*) still errors.
pub fn evaluate_when(when: Option<&str>, ctx: &serde_json::Value) -> Result<bool> {
    let Some(expr_src) = when else {
        return Ok(true);
    };
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let expr = env
        .compile_expression(expr_src)
        .map_err(|e| FluxError::Config(format!("when expression error: {e}")))?;
    match expr.eval(ctx) {
        Ok(value) if value.is_undefined() => {
            if reads_a_failed_stage(&expr, ctx) {
                return Ok(false);
            }
            Err(FluxError::Config(format!(
                "when undefined reference in expression: {expr_src}"
            )))
        }
        Ok(value) => Ok(value.is_true()),
        // Only an undefined-value error is forgiven, and only via a failed
        // stage. A malformed expression stays fatal even when it happens to
        // mention one.
        Err(e) if e.kind() == minijinja::ErrorKind::UndefinedError => {
            if reads_a_failed_stage(&expr, ctx) {
                return Ok(false);
            }
            Err(FluxError::Config(format!("when evaluation error: {e}")))
        }
        Err(e) => Err(FluxError::Config(format!("when evaluation error: {e}"))),
    }
}

/// Whether the expression reads any stage whose output is an `on_error:
/// continue` error envelope.
///
/// Derived from the context rather than from a set of failed stage ids kept by
/// the run loop, because a failed-but-continued stage is recorded as
/// `StageComplete` carrying this envelope — so the envelope is the only part
/// that survives into a resumed run, and a side set would silently make resume
/// behave differently from the original run.
fn reads_a_failed_stage(expr: &minijinja::Expression<'_, '_>, ctx: &serde_json::Value) -> bool {
    let names = expr.undeclared_variables(false);
    // A name with no entry in the context at all is a typo (or a stage that
    // never ran) — never forgiven, even when the same expression also
    // mentions a failed stage.
    if names.iter().any(|name| ctx.get(name).is_none()) {
        return false;
    }
    names.iter().any(|name| is_error_envelope(ctx.get(name)))
}

/// Exactly `{ "error": "<string>" }` under `structured` — the shape
/// `run_workflow` substitutes for a stage that failed with `on_error:
/// continue`. Matching the whole shape, not merely the presence of an `error`
/// key, keeps a tool that legitimately returns an `error` field alongside other
/// data under full strictness.
fn is_error_envelope(stage: Option<&serde_json::Value>) -> bool {
    stage
        .and_then(|s| s.get("structured"))
        .and_then(|s| s.as_object())
        .is_some_and(|o| o.len() == 1 && o.get("error").is_some_and(|e| e.is_string()))
}

pub(crate) enum StageOutcome {
    Completed(AgentOutput),
    Failed(FluxError),
    /// A static configuration defect detected before dispatch — unknown tool,
    /// missing manifest, or arguments that violate the tool's schema. Always
    /// fatal: `on_error` does not apply, because no downstream stage can
    /// repair malformed YAML.
    Invalid(FluxError),
}

/// Run one agent for a stage, relaying its events as
/// `WorkflowEvent::Agent { stage, event }`. Returns once the relay has
/// drained, so every event of this run precedes whatever the caller emits next.
#[allow(clippy::too_many_arguments)]
async fn run_stage_agent(
    stage_id: &str,
    cfg: &AgentConfig,
    history: &[Turn],
    input: &str,
    llm: &LlmRegistry,
    transport: &(impl ToolTransport + Sync),
    context: Option<&serde_json::Value>,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: RunTuning,
    cancel: &CancellationToken,
) -> Result<flux_runner::AgentRun> {
    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentEvent>(64);
    let stage_id = stage_id.to_string();
    let outer_tx = tx.clone();
    let relay = tokio::spawn(async move {
        while let Some(event) = agent_rx.recv().await {
            if outer_tx
                .send(WorkflowEvent::Agent {
                    stage: stage_id.clone(),
                    event,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    });
    let result = run_agent(
        llm,
        transport,
        cfg,
        history,
        input,
        context,
        &agent_tx,
        tuning.retry,
        cancel,
    )
    .await;
    drop(agent_tx);
    let _ = relay.await;
    result
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, fields(stage_id = %stage.id, agent = stage.agent.as_deref().unwrap_or("-"), tool = stage.tool.as_deref().unwrap_or("-")))]
async fn run_one_stage<M: ConversationMemory + Sync>(
    stage: &flux_workflow::Stage,
    cfg: Option<&AgentConfig>,
    evaluator: Option<&AgentConfig>,
    manifest: Option<&flux_tool_client::Manifest>,
    input: String,
    input_args: serde_json::Value,
    ctx: &serde_json::Value,
    llm: &LlmRegistry,
    transport: &(impl ToolTransport + Sync),
    conversation_id: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: RunTuning,
    cancel: &CancellationToken,
) -> StageOutcome {
    tracing::info!("stage started");
    let _ = tx
        .send(WorkflowEvent::StageStarted {
            stage: stage.id.clone(),
        })
        .await;
    if let Some(tool_name) = stage.tool.as_deref() {
        // An omitted `args:` block parses as JSON null, which fails validation
        // against any `{"type":"object"}` schema. A tool stage with no
        // arguments means an empty argument object.
        let input_args = if input_args.is_null() {
            serde_json::json!({})
        } else {
            input_args
        };
        let Some(manifest) = manifest else {
            return StageOutcome::Invalid(FluxError::Config(format!(
                "stage '{}' calls tool '{}' but no tool manifest is available",
                stage.id, tool_name
            )));
        };
        let Some(spec) = manifest.find(tool_name) else {
            return StageOutcome::Invalid(FluxError::Config(format!(
                "stage '{}' references unknown tool '{}'",
                stage.id, tool_name
            )));
        };
        if let Err(e) = flux_tool_client::validate_args(spec, &input_args) {
            return StageOutcome::Invalid(e);
        }
        let call = flux_tool_client::ToolCall {
            name: tool_name.to_string(),
            args: input_args.clone(),
            context: context.cloned(),
        };
        // A call already received by the tool server may still complete
        // there; cancelling only stops the run waiting for it.
        let called = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                return StageOutcome::Failed(FluxError::Provider("cancelled".into()));
            }
            r = transport.call_tool(&call) => r,
        };
        return match called {
            Ok(r) => {
                flux_core::metrics::record_tool_call("ok");
                let flux_tool_client::ToolResult { result, for_model } = r;
                StageOutcome::Completed(AgentOutput {
                    // `message` is the model- and human-facing rendering of the
                    // stage, so it takes the transport's compact projection when
                    // there is one. Templating `{{ stage.message }}` into a
                    // downstream agent's prompt otherwise bypassed that
                    // compaction entirely and shipped the lossless envelope —
                    // base64 image payloads and all — which is the cost
                    // `for_model` exists to avoid on the agent path.
                    //
                    // `structured` stays lossless on purpose: workflow templates
                    // are rendered once and may need the content blocks that the
                    // model-facing projection drops.
                    //
                    // A transport that sets no projection (gRPC) is unaffected —
                    // `message` is byte-identical to before for those.
                    message: for_model.as_ref().unwrap_or(&result).to_string(),
                    structured: Some(result),
                })
            }
            Err(e) => {
                flux_core::metrics::record_tool_call("error");
                StageOutcome::Failed(e)
            }
        };
    }
    let Some(cfg) = cfg else {
        return StageOutcome::Failed(FluxError::Config(format!(
            "stage '{}' has neither a tool nor an agent config",
            stage.id
        )));
    };
    if let Some(refine) = &stage.refine {
        let Some(evaluator) = evaluator else {
            return StageOutcome::Failed(FluxError::Config(format!(
                "stage '{}' refines with no evaluator agent config",
                stage.id
            )));
        };
        return refine::run_refine(
            stage,
            refine,
            cfg,
            evaluator,
            input,
            ctx,
            llm,
            transport,
            conversation_id,
            context,
            memory,
            tx,
            tuning,
            cancel,
        )
        .await;
    }
    // Load this agent's prior turns when the stage opts in and the run is in a
    // conversation.
    let remembering = stage.remember && conversation_id.is_some();
    let history: Vec<Turn> = if remembering {
        let cid = conversation_id.unwrap_or_default();
        let key = conversation_key(cid, &cfg.agent_id);
        match flux_memory::recall_window(memory, &key, tuning.history_window).await {
            Ok(h) => h,
            Err(e) => return StageOutcome::Failed(e),
        }
    } else {
        Vec::new()
    };

    let result = run_stage_agent(
        &stage.id, cfg, &history, &input, llm, transport, context, tx, tuning, cancel,
    )
    .await;
    match result {
        Ok(run) => {
            if remembering {
                let cid = conversation_id.unwrap_or_default();
                let key = conversation_key(cid, &cfg.agent_id);
                let turns = turns_to_persist(&run.turns, stage.replay_tools);
                if let Err(e) = append_exchange(memory, &key, &turns).await {
                    return StageOutcome::Failed(e);
                }
            }
            // The runner returns Ok for an aborted run so the turns above can
            // be persisted — but the stage itself did not succeed. Mapping it
            // to Completed here is what made `/chat` report an empty reply as
            // success and let `on_error: abort` not abort.
            if let Some(reason) = run.aborted {
                return StageOutcome::Failed(FluxError::Provider(format!(
                    "agent run aborted: {reason}"
                )));
            }
            StageOutcome::Completed(run.output)
        }
        Err(e) => StageOutcome::Failed(e),
    }
}

/// Resume state seeded into a (re-)launched workflow run. Empty (`default()`)
/// for a fresh run; rebuilt from the event log + approvals on resume.
#[derive(Debug, Default)]
pub struct ResumeState {
    pub completed: HashMap<String, AgentOutput>,
    pub skipped: HashSet<String>,
    pub approved: HashSet<String>,
}

/// Outcome of a (possibly partial) workflow execution.
#[derive(Debug)]
pub enum RunOutcome {
    Completed(HashMap<String, AgentOutput>),
    Paused {
        waiting: Vec<String>,
    },
    /// Stopped by its cancellation token; `WorkflowCancelled` was emitted.
    Cancelled,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Decision {
    Skip,
    Gate,
    Run,
}

/// Decide a stage that has already passed the resume/dep guards: its deps are
/// all done and none were skipped. `when_true` is the evaluated `when` result.
pub(crate) fn decide_runnable(
    when_true: bool,
    requires_approval: bool,
    approved: bool,
) -> Decision {
    if !when_true {
        Decision::Skip
    } else if requires_approval && !approved {
        Decision::Gate
    } else {
        Decision::Run
    }
}

/// A stage cleared to run this round: stage, agent config, evaluator config
/// (refine stages), rendered input, rendered tool args.
type Runnable<'a> = (
    &'a flux_workflow::Stage,
    Option<&'a AgentConfig>,
    Option<&'a AgentConfig>,
    String,
    serde_json::Value,
);

/// [`run_workflow_cancellable`] with a token nobody cancels.
#[allow(clippy::too_many_arguments)]
pub async fn run_workflow<M: ConversationMemory + Sync>(
    wf: &Workflow,
    registry: &HashMap<String, AgentConfig>,
    llm: &LlmRegistry,
    transport: &(impl ToolTransport + Sync),
    workflow_input: &str,
    conversation_id: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    resume: ResumeState,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: RunTuning,
) -> Result<RunOutcome> {
    run_workflow_cancellable(
        wf,
        registry,
        llm,
        transport,
        workflow_input,
        conversation_id,
        context,
        memory,
        resume,
        tx,
        tuning,
        &CancellationToken::new(),
    )
    .await
}

/// Run a workflow until it completes, pauses at an approval gate, fails, or
/// `cancel` fires. On cancel, in-flight LLM and tool calls are abandoned,
/// remembering stages keep the turns they have, `WorkflowCancelled` is
/// emitted, and the outcome is [`RunOutcome::Cancelled`].
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, fields(workflow_id = %wf.workflow_id))]
pub async fn run_workflow_cancellable<M: ConversationMemory + Sync>(
    wf: &Workflow,
    registry: &HashMap<String, AgentConfig>,
    llm: &LlmRegistry,
    transport: &(impl ToolTransport + Sync),
    workflow_input: &str,
    conversation_id: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    resume: ResumeState,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: RunTuning,
    cancel: &CancellationToken,
) -> Result<RunOutcome> {
    flux_core::metrics::record_run();
    let ResumeState {
        mut completed,
        mut skipped,
        approved,
    } = resume;
    let rounds = ready_rounds(wf)?;
    // A tool stage needs the manifest even when no agent declares tools, so
    // resolve it once up front rather than inside the agent path.
    let manifest: Option<flux_tool_client::Manifest> = if wf.stages.iter().any(|s| s.tool.is_some())
    {
        match transport.list_tools().await {
            Ok(m) => Some(m),
            Err(e) => {
                // Not a single stage's fault: this is a workflow-level
                // failure before any stage ran. "<workflow>" says so rather
                // than misattributing it to whichever stage happens to
                // declare a tool.
                let _ = tx
                    .send(WorkflowEvent::WorkflowFailed {
                        stage: "<workflow>".to_string(),
                        error: e.to_string(),
                    })
                    .await;
                return Err(e);
            }
        }
    } else {
        None
    };
    let stage_by_id: HashMap<&str, &flux_workflow::Stage> =
        wf.stages.iter().map(|s| (s.id.as_str(), s)).collect();

    let mut waiting: Vec<String> = Vec::new();

    for round in rounds {
        if cancel.is_cancelled() {
            let _ = tx.send(WorkflowEvent::WorkflowCancelled).await;
            return Ok(RunOutcome::Cancelled);
        }
        // Classify the round against the fixed pre-round context.
        let ctx = build_context(workflow_input, &completed, context);
        let mut runnable: Vec<Runnable<'_>> = Vec::new();
        for id in &round {
            let Some(stage) = stage_by_id.get(id.as_str()) else {
                continue;
            };
            // Resume: already settled — completed OR skipped — → don't re-run,
            // don't re-emit. Re-classifying a skipped stage re-emitted the
            // whole cascade on every resume, and an approve that merges new
            // context could flip its `when` and run it while its dependents
            // stayed skipped — a branch neither fully run nor fully skipped.
            if completed.contains_key(id) || skipped.contains(id.as_str()) {
                continue;
            }
            // Skip-cascade. `join: all` skips on any skipped dep as soon as it
            // is known; `join: any` must first wait for every dep to settle,
            // then skips only if none of them ran.
            let cascade = match stage.join {
                flux_workflow::Join::All => stage.depends_on.iter().any(|d| skipped.contains(d)),
                flux_workflow::Join::Any => {
                    !stage.depends_on.is_empty()
                        && stage.depends_on.iter().all(|d| skipped.contains(d))
                }
            };
            if cascade {
                let _ = tx
                    .send(WorkflowEvent::StageSkipped { stage: id.clone() })
                    .await;
                tracing::info!(stage = %id, "stage skipped (dependency skipped)");
                flux_core::metrics::record_stage("skipped");
                skipped.insert(id.clone());
                continue;
            }
            // Blocked by an upstream pending gate (a dep is neither completed nor
            // skipped) → leave undone for a future resume.
            if !stage
                .depends_on
                .iter()
                .all(|d| completed.contains_key(d) || skipped.contains(d))
            {
                continue;
            }
            // Deps are all settled (and, under `join: all`, none skipped) →
            // safe to evaluate `when`.
            let when_true = match evaluate_when(stage.when.as_deref(), &ctx) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx
                        .send(WorkflowEvent::WorkflowFailed {
                            stage: id.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    return Err(e);
                }
            };
            match decide_runnable(when_true, stage.requires_approval, approved.contains(id)) {
                Decision::Skip => {
                    let _ = tx
                        .send(WorkflowEvent::StageSkipped { stage: id.clone() })
                        .await;
                    tracing::info!(stage = %id, "stage skipped (when=false)");
                    flux_core::metrics::record_stage("skipped");
                    skipped.insert(id.clone());
                    continue;
                }
                Decision::Gate => {
                    let _ = tx
                        .send(WorkflowEvent::StageWaitingApproval { stage: id.clone() })
                        .await;
                    tracing::info!(stage = %id, "stage waiting for approval");
                    flux_core::metrics::record_stage("waiting_approval");
                    waiting.push(id.clone());
                    continue;
                }
                Decision::Run => {}
            }
            let (input, input_args) = if stage.tool.is_some() {
                match render_args(&stage.args, &ctx) {
                    Ok(a) => (String::new(), a),
                    Err(e) => {
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        return Err(e);
                    }
                }
            } else {
                match render_input(&stage.input, &ctx) {
                    Ok(s) => (s, serde_json::Value::Null),
                    Err(e) => {
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        return Err(e);
                    }
                }
            };
            let cfg: Option<&AgentConfig> = if stage.tool.is_some() {
                None
            } else {
                let agent_id = match stage.agent.as_deref() {
                    Some(a) => a,
                    None => {
                        let e = FluxError::Config(format!(
                            "stage '{}' sets neither agent nor tool",
                            stage.id
                        ));
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        return Err(e);
                    }
                };
                match registry.get(agent_id) {
                    Some(c) => Some(c),
                    None => {
                        let e = FluxError::Config(format!(
                            "stage '{}' references unknown agent '{}'",
                            stage.id, agent_id
                        ));
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        return Err(e);
                    }
                }
            };
            let evaluator: Option<&AgentConfig> = match &stage.refine {
                None => None,
                Some(refine) => match registry.get(&refine.evaluator) {
                    Some(c) => Some(c),
                    None => {
                        let e = FluxError::Config(format!(
                            "stage '{}' references unknown evaluator agent '{}'",
                            stage.id, refine.evaluator
                        ));
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        return Err(e);
                    }
                },
            };
            runnable.push((stage, cfg, evaluator, input, input_args));
        }
        // Run all runnable stages in this round concurrently.
        let futures = runnable
            .iter()
            .map(|(stage, cfg, evaluator, input, input_args)| {
                run_one_stage(
                    stage,
                    *cfg,
                    *evaluator,
                    manifest.as_ref(),
                    input.clone(),
                    input_args.clone(),
                    &ctx,
                    llm,
                    transport,
                    conversation_id,
                    context,
                    memory,
                    tx,
                    tuning,
                    cancel,
                )
            });
        let outcomes = futures_util::future::join_all(futures).await;
        // Before the fold: it would record the interrupted stages as failures.
        if cancel.is_cancelled() {
            let _ = tx.send(WorkflowEvent::WorkflowCancelled).await;
            return Ok(RunOutcome::Cancelled);
        }
        // Fold deterministically by stage id order within the round.
        for ((stage, _, _, _, _), outcome) in runnable.iter().zip(outcomes) {
            match outcome {
                StageOutcome::Completed(output) => {
                    let _ = tx
                        .send(WorkflowEvent::StageComplete {
                            stage: stage.id.clone(),
                            output: output.clone(),
                        })
                        .await;
                    flux_core::metrics::record_stage("completed");
                    completed.insert(stage.id.clone(), output);
                }
                StageOutcome::Failed(e) => match stage.on_error {
                    flux_workflow::OnError::Continue => {
                        let output = AgentOutput {
                            message: e.to_string(),
                            structured: Some(serde_json::json!({ "error": e.to_string() })),
                        };
                        let _ = tx
                            .send(WorkflowEvent::StageComplete {
                                stage: stage.id.clone(),
                                output: output.clone(),
                            })
                            .await;
                        tracing::warn!(stage = %stage.id, error = %e, "stage failed; continuing");
                        flux_core::metrics::record_stage("failed_continue");
                        completed.insert(stage.id.clone(), output);
                    }
                    flux_workflow::OnError::Abort => {
                        let _ = tx
                            .send(WorkflowEvent::WorkflowFailed {
                                stage: stage.id.clone(),
                                error: e.to_string(),
                            })
                            .await;
                        tracing::warn!(stage = %stage.id, error = %e, "stage failed");
                        flux_core::metrics::record_stage("failed");
                        return Err(e);
                    }
                },
                StageOutcome::Invalid(e) => {
                    let _ = tx
                        .send(WorkflowEvent::WorkflowFailed {
                            stage: stage.id.clone(),
                            error: e.to_string(),
                        })
                        .await;
                    tracing::warn!(stage = %stage.id, error = %e, "stage config invalid");
                    flux_core::metrics::record_stage("invalid");
                    return Err(e);
                }
            }
        }
    }

    if !waiting.is_empty() {
        return Ok(RunOutcome::Paused { waiting });
    }
    let _ = tx.send(WorkflowEvent::WorkflowComplete).await;
    Ok(RunOutcome::Completed(completed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::conversation::{Role, Turn};
    use flux_core::output::AgentOutput;
    use std::collections::HashMap;

    fn completed_with_sentiment(s: &str) -> HashMap<String, AgentOutput> {
        let mut m = HashMap::new();
        m.insert(
            "extract".to_string(),
            AgentOutput {
                message: "{...}".into(),
                structured: Some(serde_json::json!({ "sentiment": s, "summary": "late delivery" })),
            },
        );
        m
    }

    #[test]
    fn validate_agent_refs_accepts_known_agents_and_tool_stages() {
        let mut registry = HashMap::new();
        let cfg: AgentConfig =
            serde_yaml_ng::from_str("agent_id: a\ndisplay_name: A\nsystem_prompt: hi\n").unwrap();
        registry.insert("a".to_string(), cfg);
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s1\n    agent: a\n  - id: s2\n    tool: t\n    depends_on: [s1]\n",
        )
        .unwrap();
        validate_agent_refs(&wf, &registry).unwrap();
    }

    #[test]
    fn validate_agent_refs_names_the_stage_and_the_missing_agent() {
        let wf: flux_workflow::Workflow =
            serde_yaml_ng::from_str("workflow_id: w\nstages:\n  - id: s1\n    agent: mok\n")
                .unwrap();
        match validate_agent_refs(&wf, &HashMap::new()) {
            Err(FluxError::Config(m)) => {
                assert!(m.contains("s1") && m.contains("mok"), "got: {m}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    fn refine_workflow() -> flux_workflow::Workflow {
        serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s1\n    agent: gen\n    refine:\n      evaluator: critic\n      evaluate: \"{{ candidate.message }}\"\n",
        )
        .unwrap()
    }

    fn agent_cfg(id: &str, tools: bool) -> AgentConfig {
        let tools_yaml = if tools { "tools:\n  - search\n" } else { "" };
        serde_yaml_ng::from_str(&format!(
            "agent_id: {id}\ndisplay_name: A\nsystem_prompt: hi\n{tools_yaml}"
        ))
        .unwrap()
    }

    #[test]
    fn validate_agent_refs_rejects_an_unknown_evaluator() {
        let mut registry = HashMap::new();
        registry.insert("gen".to_string(), agent_cfg("gen", false));
        match validate_agent_refs(&refine_workflow(), &registry) {
            Err(FluxError::Config(m)) => {
                assert!(
                    m.contains("s1") && m.contains("unknown evaluator agent 'critic'"),
                    "got: {m}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
        registry.insert("critic".to_string(), agent_cfg("critic", false));
        validate_agent_refs(&refine_workflow(), &registry).unwrap();
    }

    #[test]
    fn stage_agent_ids_yields_generators_and_evaluators() {
        let wf = refine_workflow();
        let ids: Vec<&str> = stage_agent_ids(&wf).collect();
        assert_eq!(ids, ["gen", "critic"]);
    }

    #[test]
    fn missing_provider_keys_covers_the_evaluator_agent() {
        let var = "FLUX_TEST_UNSET_EVAL_KEY_5d2a";
        let llm_cfg: flux_settings::LlmCfg = serde_yaml_ng::from_str(&format!(
            "providers:\n  evalco:\n    kind: anthropic\n    api_key_env: {var}\n"
        ))
        .unwrap();
        let llm = LlmRegistry::from_settings(&llm_cfg).unwrap();
        let mut registry = HashMap::new();
        let mut gen = agent_cfg("gen", false);
        gen.model.provider = "mock".into();
        registry.insert("gen".to_string(), gen);
        let mut critic = agent_cfg("critic", false);
        critic.model.provider = "evalco".into();
        registry.insert("critic".to_string(), critic);
        assert_eq!(
            missing_provider_keys(&refine_workflow(), &registry, &llm).unwrap(),
            [var.to_string()]
        );
    }

    #[test]
    fn an_evaluator_with_tools_needs_a_tool_transport() {
        let mut registry = HashMap::new();
        registry.insert("gen".to_string(), agent_cfg("gen", false));
        registry.insert("critic".to_string(), agent_cfg("critic", false));
        assert!(!needs_tool_transport(&registry, &refine_workflow()));
        registry.insert("critic".to_string(), agent_cfg("critic", true));
        assert!(needs_tool_transport(&registry, &refine_workflow()));
    }

    #[test]
    fn build_context_inserts_ctx_when_present() {
        let completed = HashMap::new();
        let context = serde_json::json!({ "projectId": "p1" });
        let ctx = build_context("hi", &completed, Some(&context));
        assert_eq!(ctx["ctx"]["projectId"], "p1");
        let rendered = render_input("proj={{ ctx.projectId }}", &ctx).unwrap();
        assert_eq!(rendered, "proj=p1");
    }

    #[test]
    fn build_context_omits_ctx_when_absent_and_ref_errors() {
        let completed = HashMap::new();
        let ctx = build_context("hi", &completed, None);
        assert!(ctx.get("ctx").is_none());
        // strict-undefined: referencing ctx with no context fails
        assert!(render_input("{{ ctx.projectId }}", &ctx).is_err());
    }

    #[test]
    fn context_exposes_workflow_input_and_stage_fields() {
        let ctx = build_context("hello", &completed_with_sentiment("negative"), None);
        assert_eq!(ctx["workflow"]["input"], "hello");
        assert_eq!(ctx["extract"]["structured"]["sentiment"], "negative");
        assert_eq!(ctx["extract"]["message"], "{...}");
    }

    #[test]
    fn render_input_interpolates_fields() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        let out = render_input("S: {{ extract.structured.summary }}", &ctx).unwrap();
        assert_eq!(out, "S: late delivery");
    }

    #[test]
    fn render_input_errors_on_missing_field() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        // references a field that does not exist on extract.structured
        assert!(render_input("S: {{ extract.structured.summry }}", &ctx).is_err());
    }

    #[test]
    fn evaluate_when_true_false_and_absent() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        assert!(evaluate_when(Some("extract.structured.sentiment == 'negative'"), &ctx).unwrap());
        assert!(!evaluate_when(Some("extract.structured.sentiment == 'positive'"), &ctx).unwrap());
        assert!(evaluate_when(None, &ctx).unwrap());
    }

    #[test]
    fn evaluate_when_error_on_bad_expression() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        assert!(evaluate_when(Some("this is ((( not valid"), &ctx).is_err());
    }

    #[test]
    fn evaluate_when_error_on_missing_field() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        // references a stage/field that does not exist in the context
        assert!(evaluate_when(Some("ghost.structured.sentiment == 'negative'"), &ctx).is_err());
    }

    #[test]
    fn evaluate_when_errors_on_nested_missing_field_in_comparison() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        assert!(evaluate_when(Some("extract.structured.ghost == 'negative'"), &ctx).is_err());
    }

    #[test]
    fn evaluate_when_errors_on_bare_undefined_path() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        match evaluate_when(Some("extract.structured.ghost"), &ctx) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_when_is_defined_on_missing_field_is_false_not_an_error() {
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        assert!(!evaluate_when(Some("extract.structured.error is defined"), &ctx).unwrap());
    }

    /// The `{ "error": … }` output `run_workflow` substitutes for a stage that
    /// failed under `on_error: continue`.
    fn completed_with_failed_stage() -> HashMap<String, AgentOutput> {
        let mut completed = HashMap::new();
        completed.insert(
            "execute".to_string(),
            AgentOutput {
                message: "tool `boom` failed".into(),
                structured: Some(serde_json::json!({ "error": "tool `boom` failed" })),
            },
        );
        completed
    }

    #[test]
    fn a_comparison_against_a_failed_stage_is_false_not_fatal() {
        // The natural spelling of a fallback branch. Before, this aborted the
        // whole run — the opposite of what `on_error: continue` configures.
        let ctx = build_context("hi", &completed_with_failed_stage(), None);
        assert!(!evaluate_when(Some("execute.structured.status == 'ok'"), &ctx).unwrap());
        // Inverted spellings must not accidentally become *true* either; the
        // field is absent, so no branch keyed on its value should fire.
        assert!(!evaluate_when(Some("execute.structured.status"), &ctx).unwrap());
        assert!(!evaluate_when(Some("execute.structured.count > 3"), &ctx).unwrap());
    }

    #[test]
    fn a_failed_stages_error_field_is_still_readable() {
        // Leniency must not hide the envelope itself — gating a recovery stage
        // on the failure having happened is the main reason to branch here.
        let ctx = build_context("hi", &completed_with_failed_stage(), None);
        assert!(evaluate_when(Some("execute.structured.error is defined"), &ctx).unwrap());
        assert!(evaluate_when(Some("'boom' in execute.structured.error"), &ctx).unwrap());
    }

    #[test]
    fn a_typo_is_still_fatal_when_another_stage_failed() {
        // The whole point of strictness: a misspelled stage id must not be
        // excused just because some *other* stage failed.
        let mut completed = completed_with_failed_stage();
        completed.insert(
            "extract".to_string(),
            AgentOutput {
                message: "{}".into(),
                structured: Some(serde_json::json!({ "sentiment": "negative" })),
            },
        );
        let ctx = build_context("hi", &completed, None);
        assert!(evaluate_when(Some("ghost.structured.status == 'ok'"), &ctx).is_err());
        // A missing field on a stage that SUCCEEDED is still a typo, too.
        assert!(evaluate_when(Some("extract.structured.ghost == 'ok'"), &ctx).is_err());
    }

    #[test]
    fn a_typo_beside_a_failed_stage_in_the_same_expression_is_still_fatal() {
        // `reslt` is a typo (absent from the context entirely). Mentioning the
        // failed stage `execute` in the same expression must not forgive it.
        let ctx = build_context("hi", &completed_with_failed_stage(), None);
        assert!(evaluate_when(
            Some("execute.structured.status == 'ok' or reslt.structured.y"),
            &ctx
        )
        .is_err());
    }

    #[test]
    fn a_malformed_expression_stays_fatal_even_naming_a_failed_stage() {
        let ctx = build_context("hi", &completed_with_failed_stage(), None);
        assert!(evaluate_when(Some("execute.structured.x == ((("), &ctx).is_err());
    }

    #[test]
    fn an_error_field_beside_other_data_keeps_full_strictness() {
        // Not the envelope shape, so this is a successful stage that happens to
        // carry an `error` key — a typo against it must still be caught.
        let mut completed = HashMap::new();
        completed.insert(
            "probe".to_string(),
            AgentOutput {
                message: "{}".into(),
                structured: Some(serde_json::json!({ "error": null, "data": 1 })),
            },
        );
        let ctx = build_context("hi", &completed, None);
        assert!(evaluate_when(Some("probe.structured.ghost == 'ok'"), &ctx).is_err());
    }

    #[test]
    fn evaluate_when_present_but_falsy_field_is_false() {
        let mut completed = HashMap::new();
        completed.insert(
            "extract".to_string(),
            AgentOutput {
                message: "{}".into(),
                structured: Some(serde_json::json!({ "flag": false })),
            },
        );
        let ctx = build_context("hi", &completed, None);
        assert!(!evaluate_when(Some("extract.structured.flag"), &ctx).unwrap());
    }

    #[test]
    fn context_excludes_uncompleted_siblings() {
        // Only 'extract' completed; a sibling 'other' in the same round is absent.
        let ctx = build_context("hi", &completed_with_sentiment("negative"), None);
        assert!(ctx.get("other").is_none());
        assert!(ctx.get("extract").is_some());
    }

    #[test]
    fn conversation_key_is_per_agent() {
        assert_eq!(super::conversation_key("conv1", "advisor"), "conv1:advisor");
    }

    #[tokio::test]
    async fn memory_round_trip_contract_for_a_stage_key() {
        // The orchestrator stores a stage's turns at {conversation_id}:{agent_id}
        // and reads them back oldest-first on the next run. Exercise that contract
        // directly (no LLM) with the in-memory store.
        use flux_memory::{ConversationMemory, InMemoryMemory};
        let mem = InMemoryMemory::default();
        let key = super::conversation_key("conv1", "advisor");
        mem.append_turn(&key, &Turn::text(Role::User, "q1"))
            .await
            .unwrap();
        mem.append_turn(&key, &Turn::text(Role::Assistant, "a1"))
            .await
            .unwrap();
        let got = mem.recent(&key, 20).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], Turn::text(Role::User, "q1"));
        // a different agent under the same conversation is a separate thread
        let other = super::conversation_key("conv1", "classify");
        assert!(mem.recent(&other, 20).await.unwrap().is_empty());
    }

    struct EchoTransport;

    impl flux_tool_client::ToolTransport for EchoTransport {
        async fn list_tools(&self) -> Result<flux_tool_client::Manifest> {
            Ok(flux_tool_client::Manifest {
                tools: vec![flux_tool_client::ToolSpec {
                    name: "echo".into(),
                    description: "echoes its args".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                        "required": ["text"]
                    }),
                }],
            })
        }
        async fn call_tool(
            &self,
            call: &flux_tool_client::ToolCall,
        ) -> Result<flux_tool_client::ToolResult> {
            Ok(flux_tool_client::ToolResult::new(
                serde_json::json!({ "echoed": call.args["text"] }),
            ))
        }
    }

    #[tokio::test]
    async fn tool_stage_completes_with_its_result_as_structured() {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: echo\n    args:\n      text: \"{{ workflow.input }}\"\n",
        )
        .unwrap();
        let registry = HashMap::new();
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();

        let outcome = run_workflow(
            &wf,
            &registry,
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        .unwrap();

        let RunOutcome::Completed(done) = outcome else {
            panic!("expected Completed");
        };
        let out = done.get("e").expect("stage e completed");
        assert_eq!(
            out.structured,
            Some(serde_json::json!({ "echoed": "hello" }))
        );
        assert_eq!(out.message, "{\"echoed\":\"hello\"}");

        drop(tx);
        let mut saw_started = false;
        let mut saw_complete = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                WorkflowEvent::StageStarted { ref stage } if stage == "e" => saw_started = true,
                WorkflowEvent::StageComplete { ref stage, .. } if stage == "e" => {
                    saw_complete = true
                }
                _ => {}
            }
        }
        assert!(saw_started && saw_complete);
    }

    #[tokio::test]
    async fn tool_stage_with_bad_args_fails_validation() {
        // `text` is required by the spec; omit it.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: echo\n    args:\n      other: \"x\"\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_stage_honours_requires_approval() {
        // The gate is decided before dispatch, so it must work on a tool stage
        // exactly as on an agent stage — this is what replaces an in-tool
        // "if (!confirmed) abort" check.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: echo\n    requires_approval: true\n    args:\n      text: \"hi\"\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        .unwrap();
        match outcome {
            RunOutcome::Paused { waiting } => assert_eq!(waiting, vec!["e".to_string()]),
            other => panic!("expected Paused, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_stage_with_unknown_tool_is_config_error() {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: ghost\n    args: {}\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        match run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        {
            Err(FluxError::Config(m)) => assert!(m.contains("ghost")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_stage_with_no_args_block_sends_an_empty_object() {
        // `echo` requires `text`, so a null-coerced-to-{} call must fail
        // validation rather than fail a type check against `null`.
        let wf: flux_workflow::Workflow =
            serde_yaml_ng::from_str("workflow_id: w\nstages:\n  - id: e\n    tool: echo\n")
                .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        match run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        {
            Err(FluxError::Config(m)) => assert!(m.contains("text")),
            other => panic!("expected a required-property validation error, got {other:?}"),
        }
    }

    struct FailingTransport;

    impl flux_tool_client::ToolTransport for FailingTransport {
        async fn list_tools(&self) -> Result<flux_tool_client::Manifest> {
            Ok(flux_tool_client::Manifest {
                tools: vec![flux_tool_client::ToolSpec {
                    name: "boom".into(),
                    description: "always fails".into(),
                    parameters: serde_json::json!({ "type": "object", "properties": {} }),
                }],
            })
        }
        async fn call_tool(
            &self,
            _call: &flux_tool_client::ToolCall,
        ) -> Result<flux_tool_client::ToolResult> {
            Err(FluxError::Provider("upstream exploded".into()))
        }
    }

    /// Shaped like the MCP transport: a lossless `result` carrying a bulky
    /// payload, plus the compact `for_model` projection that omits it.
    struct ProjectingTransport;

    impl flux_tool_client::ToolTransport for ProjectingTransport {
        async fn list_tools(&self) -> Result<flux_tool_client::Manifest> {
            Ok(flux_tool_client::Manifest {
                tools: vec![flux_tool_client::ToolSpec {
                    name: "snap".into(),
                    description: "returns a bulky payload".into(),
                    parameters: serde_json::json!({ "type": "object", "properties": {} }),
                }],
            })
        }
        async fn call_tool(
            &self,
            _call: &flux_tool_client::ToolCall,
        ) -> Result<flux_tool_client::ToolResult> {
            Ok(flux_tool_client::ToolResult {
                result: serde_json::json!({
                    "content": [{ "type": "image", "data": "BULKYPAYLOAD" }],
                    "text": "a chart",
                    "structured_content": null,
                }),
                for_model: Some(serde_json::json!({
                    "text": "a chart",
                    "structured_content": null,
                    "non_text_blocks": ["image"],
                })),
            })
        }
    }

    async fn tool_stage_output(
        transport: &(impl flux_tool_client::ToolTransport + Sync),
        tool: &str,
        args: &str,
    ) -> AgentOutput {
        let yaml =
            format!("workflow_id: w\nstages:\n  - id: s\n    tool: {tool}\n    args: {args}\n");
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(&yaml).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(256);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            transport,
            "hi",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        .expect("run completes");
        match outcome {
            RunOutcome::Completed(mut done) => done.remove("s").expect("stage s completed"),
            other => panic!("expected a completed run, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_tool_stages_message_uses_the_compact_projection() {
        let out = tool_stage_output(&ProjectingTransport, "snap", "{}").await;
        // `message` is what gets templated into a downstream agent's prompt, so
        // the bulky payload must not be in it.
        assert!(
            !out.message.contains("BULKYPAYLOAD"),
            "message must not carry the lossless payload: {}",
            out.message
        );
        assert!(out.message.contains("a chart"), "got: {}", out.message);

        // `structured` stays lossless — workflow templates may need the blocks.
        let structured = out.structured.expect("tool stages set structured");
        assert_eq!(structured["content"][0]["data"], "BULKYPAYLOAD");
    }

    #[tokio::test]
    async fn a_transport_with_no_projection_keeps_the_full_result_as_message() {
        // gRPC sets no `for_model`, so its stages must be byte-identical to
        // before this change.
        let out = tool_stage_output(&EchoTransport, "echo", "{ text: \"hi\" }").await;
        let structured = out.structured.expect("tool stages set structured");
        assert_eq!(out.message, structured.to_string());
    }

    async fn run_boom(on_error: &str) -> Result<RunOutcome> {
        let yaml = format!(
            "workflow_id: w\nstages:\n  - id: b\n    tool: boom\n    args: {{}}\n    on_error: {on_error}\n"
        );
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(&yaml).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &FailingTransport,
            "hi",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
    }

    #[tokio::test]
    async fn on_error_abort_fails_the_run() {
        assert!(run_boom("abort").await.is_err());
    }

    /// Run arbitrary workflow YAML against `FailingTransport` (its only tool,
    /// `boom`, always fails).
    async fn run_yaml(yaml: &str) -> Result<RunOutcome> {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(256);
        let llm = flux_llm::LlmRegistry::new();
        run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &FailingTransport,
            "hi",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
    }

    #[tokio::test]
    async fn a_fallback_gated_on_a_failed_stages_field_skips_rather_than_aborting_the_run() {
        // End to end: `on_error: continue` must actually let the run continue
        // when a downstream `when` reads a field the failure erased. Unit tests
        // on evaluate_when cannot show this — the abort happened in the run
        // loop, which turns any `when` error into a whole-run failure.
        let outcome = run_yaml(concat!(
            "workflow_id: w\n",
            "stages:\n",
            "  - id: b\n    tool: boom\n    args: {}\n    on_error: continue\n",
            "  - id: fallback\n    tool: boom\n    depends_on: [b]\n",
            "    when: \"b.structured.status == 'ok'\"\n    args: {}\n",
        ))
        .await
        .expect("the run must survive the failed stage, not abort on the `when`");

        match outcome {
            RunOutcome::Completed(done) => {
                assert!(done.contains_key("b"), "the failed stage is recorded");
                assert!(
                    !done.contains_key("fallback"),
                    "when was false, so the stage is skipped rather than run"
                );
            }
            other => panic!("expected a completed run, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_recovery_stage_gated_on_the_error_envelope_still_runs() {
        // The positive branch: leniency must not make every `when` on a failed
        // stage false. Gating recovery on the failure is the main use.
        let outcome = run_yaml(concat!(
            "workflow_id: w\n",
            "stages:\n",
            "  - id: b\n    tool: boom\n    args: {}\n    on_error: continue\n",
            "  - id: recover\n    tool: boom\n    depends_on: [b]\n",
            "    when: \"b.structured.error is defined\"\n    args: {}\n",
            "    on_error: continue\n",
        ))
        .await
        .expect("run completes");

        match outcome {
            RunOutcome::Completed(done) => assert!(
                done.contains_key("recover"),
                "the recovery stage must have run"
            ),
            other => panic!("expected a completed run, got {other:?}"),
        }
    }

    struct FailingManifestTransport;

    impl flux_tool_client::ToolTransport for FailingManifestTransport {
        async fn list_tools(&self) -> Result<flux_tool_client::Manifest> {
            Err(FluxError::Provider("manifest fetch exploded".into()))
        }
        async fn call_tool(
            &self,
            _call: &flux_tool_client::ToolCall,
        ) -> Result<flux_tool_client::ToolResult> {
            Err(FluxError::Provider(
                "call_tool should not be reached".into(),
            ))
        }
    }

    #[tokio::test]
    async fn manifest_fetch_failure_emits_workflow_failed_before_returning() {
        // Every other early return in run_workflow sends WorkflowFailed
        // before returning; the eager manifest fetch must too, or the HTTP
        // layer falls back to "run failed" naming no stage or tool, and SSE
        // clients see a stream that closes with no reason.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: echo\n    args: {}\n",
        )
        .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &FailingManifestTransport,
            "hi",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        assert!(result.is_err());

        drop(tx);
        let mut saw_failed = false;
        while let Some(ev) = rx.recv().await {
            if let WorkflowEvent::WorkflowFailed { error, .. } = ev {
                assert!(error.contains("manifest fetch exploded"));
                saw_failed = true;
            }
        }
        assert!(saw_failed, "expected a WorkflowFailed event to be emitted");
    }

    #[tokio::test]
    async fn on_error_continue_surfaces_the_error_as_structured() {
        let RunOutcome::Completed(done) = run_boom("continue").await.unwrap() else {
            panic!("expected Completed");
        };
        let out = done.get("b").expect("stage b completed");
        let err = out.structured.as_ref().unwrap()["error"].as_str().unwrap();
        assert!(err.contains("upstream exploded"));
    }

    #[tokio::test]
    async fn on_error_continue_does_not_absorb_schema_validation_failure() {
        // `text` is required by EchoTransport's schema; omit it. This is a
        // static YAML defect, not a runtime failure, so `on_error: continue`
        // must not swallow it.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: echo\n    args:\n      other: \"x\"\n    on_error: continue\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        match result {
            Err(FluxError::Config(m)) => assert!(m.contains("text") || m.contains("valid")),
            other => panic!("expected a validation Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_error_continue_does_not_absorb_unknown_tool_name() {
        // An unknown tool name is a static YAML defect too — always fatal.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: e\n    tool: ghost\n    args: {}\n    on_error: continue\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        match run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        {
            Err(FluxError::Config(m)) => assert!(m.contains("ghost")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn on_error_continue_absorbs_agent_output_schema_failure() {
        // The converse of the two tests above: an LLM response failing its
        // own output_schema is a runtime failure of the agent call itself,
        // not a static workflow-YAML defect, so on_error: continue MUST
        // absorb it — unlike unknown-tool/bad-args, which never do. This is
        // the whole reason StageOutcome::Invalid exists as a variant
        // separate from Failed: finalize_output also returns
        // FluxError::Config here, so a fold keyed on that error variant
        // alone (instead of on which StageOutcome arm produced it) would
        // wrongly treat this the same as the pre-dispatch tool defects.
        let agent_yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: '{\"other\":\"x\"}'\nguardrails:\n  output_schema:\n    type: object\n    properties:\n      sentiment: { type: string }\n    required: [sentiment]\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(agent_yaml).unwrap();
        let mut registry = HashMap::new();
        registry.insert("a".to_string(), cfg);

        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s\n    agent: a\n    input: \"hi\"\n    on_error: continue\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow(
            &wf,
            &registry,
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        .unwrap();
        let RunOutcome::Completed(done) = outcome else {
            panic!("expected Completed");
        };
        let out = done.get("s").expect("stage s completed");
        let err = out.structured.as_ref().unwrap()["error"].as_str().unwrap();
        assert!(err.contains("schema"), "unexpected error text: {err}");
    }

    fn zero_iteration_agent_registry() -> HashMap<String, AgentConfig> {
        let agent_yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: 'hi'\nguardrails:\n  max_iterations: 0\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(agent_yaml).unwrap();
        let mut registry = HashMap::new();
        registry.insert("a".to_string(), cfg);
        registry
    }

    #[tokio::test]
    async fn an_agent_that_exhausts_max_iterations_fails_the_stage() {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s\n    agent: a\n    input: \"hi\"\n",
        )
        .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &zero_iteration_agent_registry(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        // on_error defaults to abort: the run must fail, not complete with "".
        assert!(result.is_err());
        drop(tx);
        let mut saw_stage_complete = false;
        let mut saw_workflow_failed = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                WorkflowEvent::StageComplete { .. } => saw_stage_complete = true,
                WorkflowEvent::WorkflowFailed { .. } => saw_workflow_failed = true,
                _ => {}
            }
        }
        assert!(
            !saw_stage_complete,
            "an aborted agent run must not complete its stage"
        );
        assert!(saw_workflow_failed);
    }

    #[tokio::test]
    async fn an_exhausted_agent_under_on_error_continue_yields_the_error_envelope() {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s\n    agent: a\n    input: \"hi\"\n    on_error: continue\n",
        )
        .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow(
            &wf,
            &zero_iteration_agent_registry(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await
        .unwrap();
        let RunOutcome::Completed(done) = outcome else {
            panic!("expected Completed")
        };
        let err = done["s"].structured.as_ref().unwrap()["error"]
            .as_str()
            .unwrap();
        assert!(err.contains("max_iterations"), "got: {err}");
    }

    #[tokio::test]
    async fn an_exhausted_remembering_agent_still_persists_its_turns() {
        use flux_memory::{ConversationMemory, InMemoryMemory};
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s\n    agent: a\n    input: \"hi\"\n    remember: true\n",
        )
        .unwrap();
        let mem = InMemoryMemory::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &zero_iteration_agent_registry(),
            &llm,
            &EchoTransport,
            "hello",
            Some("conv1"),
            None,
            &mem,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        assert!(result.is_err(), "stage must still fail");
        // But the user's question and the closing note were persisted first.
        let turns = mem.recent("conv1:a", 20).await.unwrap();
        assert_eq!(turns.len(), 2);
    }

    #[tokio::test]
    async fn a_remembering_agent_whose_reply_fails_its_schema_still_persists_its_turns() {
        use flux_memory::{ConversationMemory, InMemoryMemory};
        let agent_yaml = "agent_id: a\ndisplay_name: A\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: '{\"other\":\"x\"}'\nguardrails:\n  output_schema:\n    type: object\n    required: [sentiment]\n";
        let cfg: AgentConfig = serde_yaml_ng::from_str(agent_yaml).unwrap();
        let mut registry = HashMap::new();
        registry.insert("a".to_string(), cfg);
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: s\n    agent: a\n    input: \"hi\"\n    remember: true\n",
        )
        .unwrap();
        let mem = InMemoryMemory::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let result = run_workflow(
            &wf,
            &registry,
            &llm,
            &EchoTransport,
            "hello",
            Some("conv1"),
            None,
            &mem,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        assert!(result.is_err(), "stage must still fail");
        // Without the user's question in memory, the next turn would answer
        // as if it had never been asked.
        let turns = mem.recent("conv1:a", 20).await.unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].content, "hi");
    }

    /// Serves one tool, `slow`, whose call never returns.
    struct PendingTransport;

    impl flux_tool_client::ToolTransport for PendingTransport {
        async fn list_tools(&self) -> Result<flux_tool_client::Manifest> {
            Ok(flux_tool_client::Manifest {
                tools: vec![flux_tool_client::ToolSpec {
                    name: "slow".into(),
                    description: "never returns".into(),
                    parameters: serde_json::json!({ "type": "object" }),
                }],
            })
        }
        async fn call_tool(
            &self,
            _call: &flux_tool_client::ToolCall,
        ) -> Result<flux_tool_client::ToolResult> {
            std::future::pending().await
        }
    }

    async fn run_cancellable(
        yaml: &str,
        transport: &PendingTransport,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> (Result<RunOutcome>, Vec<WorkflowEvent>) {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow_cancellable(
            &wf,
            &join_registry(),
            &llm,
            transport,
            "in",
            None,
            None,
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
            cancel,
        )
        .await;
        drop(tx);
        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        (outcome, events)
    }

    #[tokio::test]
    async fn a_cancelled_run_stops_before_the_next_stage() {
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let (outcome, events) = run_cancellable(
            "workflow_id: w\nstages:\n  - id: a\n    agent: m\n    input: hi\n",
            &PendingTransport,
            &cancel,
        )
        .await;
        assert!(matches!(outcome, Ok(RunOutcome::Cancelled)), "{outcome:?}");
        assert!(events.contains(&WorkflowEvent::WorkflowCancelled));
        assert!(!events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::StageStarted { .. })));
    }

    #[tokio::test]
    async fn cancel_interrupts_a_stage_waiting_on_a_tool() {
        let cancel = tokio_util::sync::CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let (outcome, events) = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_cancellable(
                "workflow_id: w\nstages:\n  - id: t\n    tool: slow\n    args: {}\n  - id: after\n    agent: m\n    depends_on: [t]\n    input: hi\n",
                &PendingTransport,
                &cancel,
            ),
        )
        .await
        .expect("cancel must interrupt the pending tool call");
        assert!(matches!(outcome, Ok(RunOutcome::Cancelled)), "{outcome:?}");
        assert!(events.contains(&WorkflowEvent::WorkflowCancelled));
        assert!(!events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::WorkflowFailed { .. })));
        assert!(!events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::StageStarted { stage } if stage == "after")));
    }

    /// Records every appended turn as written, without the recall-time trim
    /// `InMemoryMemory::recent` applies (which drops an unanswered tool call).
    #[derive(Default)]
    struct RecordingMemory {
        turns: std::sync::Mutex<Vec<(String, Turn)>>,
    }

    impl ConversationMemory for RecordingMemory {
        async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
            self.turns
                .lock()
                .unwrap()
                .push((conversation_id.to_string(), turn.clone()));
            Ok(())
        }

        async fn recent(&self, _conversation_id: &str, _limit: usize) -> Result<Vec<Turn>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn a_remembering_agent_cancelled_mid_tool_call_keeps_its_turns() {
        let backend = std::sync::Arc::new(flux_llm::MockBackend::scripted(vec![
            flux_llm::MockTurn::ToolCalls(vec![(
                "c1".into(),
                "slow".into(),
                serde_json::json!({}),
            )]),
        ]));
        let mut llm = flux_llm::LlmRegistry::new();
        llm.register("scripted", backend);
        let agent: AgentConfig = serde_yaml_ng::from_str(
            "agent_id: s\ndisplay_name: S\nsystem_prompt: hi\ntools: [slow]\nmodel:\n  provider: scripted\n",
        )
        .unwrap();
        let registry = HashMap::from([("s".to_string(), agent)]);
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: s\n    input: hi\n    remember: true\n    replay_tools: true\n",
        )
        .unwrap();
        let mem = RecordingMemory::default();
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);

        let watcher = {
            let cancel = cancel.clone();
            async move {
                let mut events = Vec::new();
                while let Some(e) = rx.recv().await {
                    if matches!(
                        e,
                        WorkflowEvent::Agent {
                            event: AgentEvent::ToolCall { .. },
                            ..
                        }
                    ) {
                        cancel.cancel();
                    }
                    events.push(e);
                }
                events
            }
        };
        let run = async {
            let outcome = run_workflow_cancellable(
                &wf,
                &registry,
                &llm,
                &PendingTransport,
                "in",
                Some("conv1"),
                None,
                &mem,
                ResumeState::default(),
                &tx,
                RunTuning::default(),
                &cancel,
            )
            .await;
            drop(tx);
            outcome
        };
        let (outcome, events) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            futures_util::future::join(run, watcher),
        )
        .await
        .expect("cancel must interrupt the pending tool call");

        assert!(matches!(outcome, Ok(RunOutcome::Cancelled)), "{outcome:?}");
        assert!(events.contains(&WorkflowEvent::WorkflowCancelled));
        let turns: Vec<Turn> = mem
            .turns
            .lock()
            .unwrap()
            .iter()
            .map(|(key, t)| {
                assert_eq!(key, "conv1:s");
                t.clone()
            })
            .collect();
        assert_eq!(turns.len(), 3, "{turns:?}");
        assert_eq!(turns[0].role, Role::User);
        assert_eq!(turns[0].content, "hi");
        assert_eq!(turns[1].role, Role::Assistant);
        assert_eq!(turns[1].tool_calls.len(), 1);
        assert_eq!(turns[1].tool_calls[0].id, "c1");
        assert_eq!(turns[2].role, Role::Assistant);
        assert_eq!(turns[2].content, "[run stopped: cancelled]");
    }

    fn join_registry() -> HashMap<String, AgentConfig> {
        let ok: AgentConfig = serde_yaml_ng::from_str(
            "agent_id: m\ndisplay_name: M\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: 'done'\n",
        )
        .unwrap();
        let bad: AgentConfig = serde_yaml_ng::from_str(
            "agent_id: bad\ndisplay_name: B\nsystem_prompt: hi\nmodel:\n  provider: mock\n  mock_response: '{\"other\":\"x\"}'\nguardrails:\n  output_schema:\n    type: object\n    required: [sentiment]\n",
        )
        .unwrap();
        HashMap::from([("m".to_string(), ok), ("bad".to_string(), bad)])
    }

    /// Run `yaml` with context `{"x": x}`; return the outcome and every event.
    async fn run_join(yaml: &str, x: bool) -> (Result<RunOutcome>, Vec<WorkflowEvent>) {
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        flux_workflow::validate(&wf).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(256);
        let llm = flux_llm::LlmRegistry::new();
        let ctx = serde_json::json!({ "x": x });
        let outcome = run_workflow(
            &wf,
            &join_registry(),
            &llm,
            &EchoTransport,
            "in",
            None,
            Some(&ctx),
            &NoMemory,
            ResumeState::default(),
            &tx,
            RunTuning::default(),
        )
        .await;
        drop(tx);
        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        (outcome, events)
    }

    const JOIN_WF: &str = "workflow_id: w\nstages:\n  - id: a\n    agent: m\n    when: \"ctx.x\"\n    input: a\n  - id: b\n    agent: m\n    when: \"not ctx.x\"\n    input: b\n  - id: merge\n    agent: m\n    depends_on: [a, b]\n    join: any\n    input: \"{{ a.message if a is defined else b.message }}\"\n  - id: after\n    agent: m\n    depends_on: [merge]\n    input: z\n";

    fn completed(events: &[WorkflowEvent], id: &str) -> bool {
        events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::StageComplete { stage, .. } if stage == id))
    }

    #[tokio::test]
    async fn join_any_runs_after_whichever_branch_ran() {
        for x in [true, false] {
            let (outcome, events) = run_join(JOIN_WF, x).await;
            assert!(matches!(outcome, Ok(RunOutcome::Completed(_))), "x={x}");
            assert!(completed(&events, "merge"), "x={x}: merge did not run");
            assert!(completed(&events, "after"), "x={x}");
        }
    }

    #[tokio::test]
    async fn join_any_is_skipped_when_every_dependency_was_skipped() {
        let yaml = JOIN_WF.replace("when: \"not ctx.x\"", "when: \"ctx.x\"");
        let (outcome, events) = run_join(&yaml, false).await;
        assert!(matches!(outcome, Ok(RunOutcome::Completed(_))));
        for id in ["merge", "after"] {
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, WorkflowEvent::StageSkipped { stage } if stage == id)),
                "{id} not skipped"
            );
        }
    }

    #[tokio::test]
    async fn join_any_waits_for_a_dependency_paused_at_a_gate() {
        let yaml = JOIN_WF.replace(
            "when: \"not ctx.x\"",
            "when: \"ctx.x\"\n    requires_approval: true",
        );
        let (outcome, events) = run_join(&yaml, true).await;
        assert!(matches!(outcome, Ok(RunOutcome::Paused { .. })));
        assert!(completed(&events, "a"));
        assert!(!events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::StageStarted { stage } if stage == "merge")));
    }

    #[tokio::test]
    async fn a_dependency_failed_under_continue_counts_as_completed_for_join_any() {
        let yaml = JOIN_WF.replace(
            "  - id: b\n    agent: m\n    when: \"not ctx.x\"\n",
            "  - id: b\n    agent: bad\n    on_error: continue\n",
        );
        let (outcome, events) = run_join(&yaml, false).await;
        assert!(
            matches!(outcome, Ok(RunOutcome::Completed(_))),
            "{outcome:?}"
        );
        assert!(completed(&events, "merge"));
    }

    #[test]
    fn decide_runnable_covers_gate_cases() {
        // when false → always Skip, regardless of gate
        assert!(matches!(
            decide_runnable(false, false, false),
            Decision::Skip
        ));
        assert!(matches!(
            decide_runnable(false, true, false),
            Decision::Skip
        ));
        // when true, no gate → Run
        assert!(matches!(decide_runnable(true, false, false), Decision::Run));
        // when true, gate not approved → Gate
        assert!(matches!(decide_runnable(true, true, false), Decision::Gate));
        // when true, gate approved → Run
        assert!(matches!(decide_runnable(true, true, true), Decision::Run));
    }

    #[test]
    fn turns_to_persist_without_replay_keeps_only_the_outer_pair() {
        let turns = vec![
            Turn::text(Role::User, "q"),
            Turn {
                role: Role::Assistant,
                content: String::new(),
                tool_calls: vec![flux_core::conversation::ToolCallRecord {
                    id: "c1".into(),
                    name: "t".into(),
                    args: serde_json::json!({}),
                }],
                tool_results: Vec::new(),
            },
            Turn {
                role: Role::User,
                content: String::new(),
                tool_calls: Vec::new(),
                tool_results: vec![flux_core::conversation::ToolResultRecord {
                    id: "c1".into(),
                    name: "t".into(),
                    result: serde_json::json!({}),
                }],
            },
            Turn::text(Role::Assistant, "final"),
        ];
        let kept = turns_to_persist(&turns, false);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].content, "q");
        assert_eq!(kept[1].content, "final");
        assert!(kept
            .iter()
            .all(|t| t.tool_calls.is_empty() && t.tool_results.is_empty()));

        assert_eq!(turns_to_persist(&turns, true).len(), 4);
    }

    /// Test double: succeeds for the first `fail_after` appends, then fails
    /// every append after that. Lets us prove `append_exchange` stops
    /// mid-sequence and leaves the earlier turn(s) persisted.
    struct FlakyMemory {
        inner: flux_memory::InMemoryMemory,
        fail_after: usize,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ConversationMemory for FlakyMemory {
        async fn append_turn(&self, conversation_id: &str, turn: &Turn) -> Result<()> {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n >= self.fail_after {
                return Err(FluxError::Config("simulated append failure".into()));
            }
            self.inner.append_turn(conversation_id, turn).await
        }

        async fn recent(&self, conversation_id: &str, limit: usize) -> Result<Vec<Turn>> {
            self.inner.recent(conversation_id, limit).await
        }
    }

    #[tokio::test]
    async fn append_exchange_stops_on_mid_sequence_failure_leaving_partial_state() {
        let mem = FlakyMemory {
            inner: flux_memory::InMemoryMemory::default(),
            fail_after: 1,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let turns = vec![
            Turn::text(Role::User, "q1"),
            Turn::text(Role::Assistant, "a1"),
        ];
        let key = "conv1:advisor";
        assert!(append_exchange(&mem, key, &turns).await.is_err());
        // The user turn made it in before the assistant append failed: this
        // is exactly the odd-length exposure the fix makes loud rather than
        // silent, not one it eliminates.
        let persisted = mem.inner.recent(key, 20).await.unwrap();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].content, "q1");
    }

    #[tokio::test]
    async fn append_exchange_succeeds_when_every_turn_appends() {
        let mem = flux_memory::InMemoryMemory::default();
        let turns = vec![
            Turn::text(Role::User, "q1"),
            Turn::text(Role::Assistant, "a1"),
        ];
        let key = "conv1:advisor";
        append_exchange(&mem, key, &turns).await.unwrap();
        let persisted = mem.recent(key, 20).await.unwrap();
        assert_eq!(persisted.len(), 2);
    }

    #[tokio::test]
    async fn resume_does_not_reprocess_previously_skipped_stages() {
        // b was skipped in the original run; the approve merged ctx.flag = "x",
        // so b's `when` is now true. A skipped stage is settled: it must not run
        // on resume, and no duplicate StageSkipped may be emitted for b or its
        // cascaded dependent c.
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(concat!(
            "workflow_id: w\n",
            "stages:\n",
            "  - id: a\n    tool: echo\n    args: { text: \"hi\" }\n",
            "  - id: b\n    tool: echo\n    depends_on: [a]\n",
            "    when: \"ctx.flag == 'x'\"\n    args: { text: \"hi\" }\n",
            "  - id: c\n    tool: echo\n    depends_on: [b]\n    args: { text: \"hi\" }\n",
        ))
        .unwrap();
        let mut completed = HashMap::new();
        completed.insert(
            "a".to_string(),
            AgentOutput {
                message: "done".into(),
                structured: Some(serde_json::json!({"echoed":"hi"})),
            },
        );
        let resume = ResumeState {
            completed,
            skipped: ["b".to_string(), "c".to_string()].into_iter().collect(),
            approved: Default::default(),
        };
        let context = serde_json::json!({ "flag": "x" });
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let llm = flux_llm::LlmRegistry::new();
        let outcome = run_workflow(
            &wf,
            &HashMap::new(),
            &llm,
            &EchoTransport,
            "hello",
            None,
            Some(&context),
            &NoMemory,
            resume,
            &tx,
            RunTuning::default(),
        )
        .await
        .unwrap();
        let RunOutcome::Completed(done) = outcome else {
            panic!("expected Completed")
        };
        assert!(
            !done.contains_key("b"),
            "a previously-skipped stage must not run on resume"
        );
        assert!(!done.contains_key("c"));
        drop(tx);
        while let Some(ev) = rx.recv().await {
            assert!(
                !matches!(ev, WorkflowEvent::StageSkipped { .. }),
                "resume must not re-emit skip events for settled stages"
            );
        }
    }

    // ---- refine (evaluator-optimizer) stages ----

    #[test]
    fn evaluate_pass_accepts_only_booleans() {
        let expr = "verdict.structured.pass";
        let vars = |pass: serde_json::Value| serde_json::json!({ "verdict": { "structured": { "pass": pass } } });
        assert!(refine::evaluate_pass(expr, &vars(serde_json::json!(true))).unwrap());
        assert!(!refine::evaluate_pass(expr, &vars(serde_json::json!(false))).unwrap());
        for (value, kind) in [
            (serde_json::json!("false"), "string"),
            (serde_json::json!(1), "number"),
            (serde_json::Value::Null, "null"),
        ] {
            let err = refine::evaluate_pass(expr, &vars(value.clone()))
                .expect_err("a non-boolean pass must be an error")
                .to_string();
            assert!(err.contains(expr), "{value}: {err}");
            assert!(err.contains(kind), "{value}: {err}");
        }
        let missing = serde_json::json!({ "verdict": { "structured": {} } });
        let err = refine::evaluate_pass(expr, &missing)
            .unwrap_err()
            .to_string();
        assert!(err.contains(expr) && err.contains("undefined"), "{err}");
    }

    const JUDGE_STRICT: &str = "guardrails:\n  output_schema:\n    type: object\n    properties:\n      pass: { type: boolean }\n      feedback: { type: string }\n    required: [pass]\n";

    struct RefineCase<'a> {
        stage_extra: &'a str,
        refine_extra: &'a str,
        generator: Vec<flux_llm::MockTurn>,
        judge: Vec<flux_llm::MockTurn>,
        judge_schema: &'a str,
        conversation_id: Option<&'a str>,
        cancel_on_judge: bool,
    }

    impl Default for RefineCase<'_> {
        fn default() -> Self {
            RefineCase {
                stage_extra: "",
                refine_extra: "",
                generator: Vec::new(),
                judge: Vec::new(),
                judge_schema: JUDGE_STRICT,
                conversation_id: None,
                cancel_on_judge: false,
            }
        }
    }

    struct RefineRun {
        outcome: Result<RunOutcome>,
        events: Vec<WorkflowEvent>,
        generator: std::sync::Arc<flux_llm::MockBackend>,
        judge: std::sync::Arc<flux_llm::MockBackend>,
    }

    fn text(s: &str) -> flux_llm::MockTurn {
        flux_llm::MockTurn::Text(s.to_string())
    }

    /// Run one refine stage `s` (generator `gen`, evaluator `judge`, both
    /// scripted) on workflow input "the question".
    async fn run_refine<M: ConversationMemory + Sync>(
        case: RefineCase<'_>,
        memory: &M,
    ) -> RefineRun {
        let generator = std::sync::Arc::new(flux_llm::MockBackend::scripted(case.generator));
        let judge = std::sync::Arc::new(flux_llm::MockBackend::scripted(case.judge));
        let mut llm = flux_llm::LlmRegistry::new();
        llm.register("gen", generator.clone());
        llm.register("judge", judge.clone());
        let gen_cfg: AgentConfig = serde_yaml_ng::from_str(
            "agent_id: gen\ndisplay_name: G\nsystem_prompt: write\nmodel:\n  provider: gen\n",
        )
        .unwrap();
        let judge_cfg: AgentConfig = serde_yaml_ng::from_str(&format!(
            "agent_id: judge\ndisplay_name: J\nsystem_prompt: judge\nmodel:\n  provider: judge\n{}",
            case.judge_schema
        ))
        .unwrap();
        let registry = HashMap::from([
            ("gen".to_string(), gen_cfg),
            ("judge".to_string(), judge_cfg),
        ]);
        let yaml = format!(
            "workflow_id: w\nstages:\n  - id: s\n    agent: gen\n    input: \"{{{{ workflow.input }}}}\"\n{}    refine:\n      evaluator: judge\n      evaluate: \"Q: {{{{ input }}}} C: {{{{ candidate.message }}}}\"\n{}",
            case.stage_extra, case.refine_extra
        );
        let wf: flux_workflow::Workflow = serde_yaml_ng::from_str(&yaml).unwrap();
        let cancel = CancellationToken::new();
        let (tx, mut rx) = mpsc::channel(256);
        let watcher = {
            let cancel = cancel.clone();
            let cancel_on_judge = case.cancel_on_judge;
            async move {
                let mut events = Vec::new();
                while let Some(e) = rx.recv().await {
                    if cancel_on_judge
                        && matches!(&e, WorkflowEvent::Agent { event: AgentEvent::Started { agent_id }, .. } if agent_id == "judge")
                    {
                        cancel.cancel();
                    }
                    events.push(e);
                }
                events
            }
        };
        let run = async {
            let outcome = run_workflow_cancellable(
                &wf,
                &registry,
                &llm,
                &EchoTransport,
                "the question",
                case.conversation_id,
                None,
                memory,
                ResumeState::default(),
                &tx,
                RunTuning::default(),
                &cancel,
            )
            .await;
            drop(tx);
            outcome
        };
        let (outcome, events) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            futures_util::future::join(run, watcher),
        )
        .await
        .expect("refine run must finish");
        RefineRun {
            outcome,
            events,
            generator,
            judge,
        }
    }

    fn rounds(events: &[WorkflowEvent]) -> Vec<(u32, bool, Option<String>)> {
        events
            .iter()
            .filter_map(|e| match e {
                WorkflowEvent::RefineRound {
                    round,
                    passed,
                    feedback,
                    ..
                } => Some((*round, *passed, feedback.clone())),
                _ => None,
            })
            .collect()
    }

    fn last_user_text(req: &flux_llm::ChatRequest) -> String {
        req.messages
            .iter()
            .rev()
            .find(|m| m.role == flux_llm::Role::User)
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    flux_llm::Content::Text(t) => Some(t.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default()
    }

    fn workflow_failed_error(events: &[WorkflowEvent]) -> Option<String> {
        events.iter().find_map(|e| match e {
            WorkflowEvent::WorkflowFailed { error, .. } => Some(error.clone()),
            _ => None,
        })
    }

    fn completed_output(outcome: &Result<RunOutcome>) -> AgentOutput {
        match outcome {
            Ok(RunOutcome::Completed(done)) => done["s"].clone(),
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    const FAIL_JOIN: &str = r#"{"pass":false,"feedback":"use a JOIN"}"#;
    const PASS: &str = r#"{"pass":true,"feedback":"ok"}"#;

    #[tokio::test]
    async fn refine_passes_on_the_first_round() {
        let r = run_refine(
            RefineCase {
                generator: vec![text("SELECT 1")],
                judge: vec![text(PASS)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert_eq!(completed_output(&r.outcome).message, "SELECT 1");
        assert_eq!(r.generator.requests().len(), 1);
        assert_eq!(r.judge.requests().len(), 1);
        assert_eq!(rounds(&r.events), vec![(1, true, None)]);
        assert_eq!(
            last_user_text(&r.judge.requests()[0]),
            "Q: the question C: SELECT 1"
        );
    }

    #[tokio::test]
    async fn refine_feeds_feedback_into_the_next_round() {
        let r = run_refine(
            RefineCase {
                generator: vec![text("SELECT a, b"), text("SELECT a JOIN b")],
                judge: vec![text(FAIL_JOIN), text(PASS)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert_eq!(completed_output(&r.outcome).message, "SELECT a JOIN b");
        let reqs = r.generator.requests();
        assert_eq!(reqs.len(), 2);
        let revise = last_user_text(&reqs[1]);
        assert!(revise.contains("use a JOIN"), "{revise}");
        assert!(revise.contains("SELECT a, b"), "{revise}");
        assert!(revise.contains("the question"), "{revise}");
        let rs = rounds(&r.events);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].0, 1);
        assert!(!rs[0].1);
        assert!(rs[0].2.as_deref().is_some_and(|f| f.contains("use a JOIN")));
        assert_eq!(rs[1], (2, true, None));
    }

    #[tokio::test]
    async fn refine_exhausted_fails_the_stage() {
        let r = run_refine(
            RefineCase {
                refine_extra: "      max_rounds: 2\n",
                generator: vec![text("a"), text("b")],
                judge: vec![text(FAIL_JOIN), text(FAIL_JOIN)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert!(r.outcome.is_err());
        let err = workflow_failed_error(&r.events).expect("WorkflowFailed");
        assert!(err.contains("no candidate passed after 2 rounds"), "{err}");
        assert!(err.contains("use a JOIN"), "{err}");
        assert!(!r
            .events
            .iter()
            .any(|e| matches!(e, WorkflowEvent::StageComplete { .. })));
        assert_eq!(rounds(&r.events).len(), 2);
    }

    #[tokio::test]
    async fn refine_exhausted_with_continue_completes_with_an_error_envelope() {
        let r = run_refine(
            RefineCase {
                stage_extra: "    on_error: continue\n",
                refine_extra: "      max_rounds: 1\n",
                generator: vec![text("a")],
                judge: vec![text(FAIL_JOIN)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        let out = completed_output(&r.outcome);
        let structured = out.structured.expect("error envelope");
        let obj = structured.as_object().unwrap();
        assert_eq!(obj.len(), 1);
        let err = obj["error"].as_str().unwrap();
        assert!(err.contains("no candidate passed after 1 rounds"), "{err}");
    }

    #[tokio::test]
    async fn refine_accept_last_returns_the_last_candidate() {
        let r = run_refine(
            RefineCase {
                refine_extra: "      max_rounds: 2\n      on_exhausted: accept_last\n",
                generator: vec![text("first"), text("second")],
                judge: vec![text(FAIL_JOIN), text(FAIL_JOIN)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert_eq!(completed_output(&r.outcome).message, "second");
        assert_eq!(rounds(&r.events).len(), 2);
    }

    #[tokio::test]
    async fn a_string_false_verdict_fails_the_stage() {
        // The judge's schema does not type `pass`, so the string "false"
        // reaches the predicate: it must fail the stage, never pass by
        // truthiness.
        let r = run_refine(
            RefineCase {
                generator: vec![text("a")],
                judge: vec![text(r#"{"pass":"false"}"#)],
                judge_schema: "guardrails:\n  output_schema:\n    type: object\n",
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert!(r.outcome.is_err());
        let err = workflow_failed_error(&r.events).expect("WorkflowFailed");
        assert!(err.contains("verdict.structured.pass"), "{err}");
        assert!(err.contains("string"), "{err}");
        assert!(rounds(&r.events).is_empty());
    }

    #[tokio::test]
    async fn a_verdict_with_no_structured_output_fails_the_stage() {
        let r = run_refine(
            RefineCase {
                generator: vec![text("a")],
                judge: vec![text(r#"{"pass":false}"#)],
                judge_schema: "",
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert!(r.outcome.is_err());
        let err = workflow_failed_error(&r.events).expect("WorkflowFailed");
        assert!(err.contains("verdict.structured.pass"), "{err}");
    }

    #[tokio::test]
    async fn custom_feedback_and_revise_templates_are_used() {
        let r = run_refine(
            RefineCase {
                refine_extra: "      feedback: \"FB={{ verdict.structured.feedback }}\"\n      revise: \"REVISE {{ input }} | {{ candidate.message }} | {{ feedback }} | {{ round }}\"\n",
                generator: vec![text("v1"), text("v2")],
                judge: vec![text(FAIL_JOIN), text(PASS)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert_eq!(completed_output(&r.outcome).message, "v2");
        assert_eq!(
            last_user_text(&r.generator.requests()[1]),
            "REVISE the question | v1 | FB=use a JOIN | 1"
        );
        assert_eq!(rounds(&r.events)[0].2.as_deref(), Some("FB=use a JOIN"));
    }

    #[tokio::test]
    async fn cancel_during_the_evaluator_ends_the_run() {
        let r = run_refine(
            RefineCase {
                generator: vec![text("a")],
                judge: vec![flux_llm::MockTurn::Stall],
                cancel_on_judge: true,
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        assert!(
            matches!(r.outcome, Ok(RunOutcome::Cancelled)),
            "{:?}",
            r.outcome
        );
        assert!(r.events.contains(&WorkflowEvent::WorkflowCancelled));
        assert!(!r.events.iter().any(|e| matches!(
            e,
            WorkflowEvent::StageComplete { .. }
                | WorkflowEvent::RefineRound { .. }
                | WorkflowEvent::WorkflowFailed { .. }
        )));
    }

    #[tokio::test]
    async fn a_remembering_refine_stage_persists_only_the_accepted_exchange() {
        let mem = RecordingMemory::default();
        let r = run_refine(
            RefineCase {
                stage_extra: "    remember: true\n",
                generator: vec![text("draft"), text("final")],
                judge: vec![text(FAIL_JOIN), text(PASS)],
                conversation_id: Some("conv1"),
                ..Default::default()
            },
            &mem,
        )
        .await;
        assert_eq!(completed_output(&r.outcome).message, "final");
        let turns = mem.turns.lock().unwrap().clone();
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert!(turns.iter().all(|(key, _)| key == "conv1:gen"));
        assert_eq!(turns[0].1, Turn::text(Role::User, "the question"));
        assert_eq!(turns[1].1, Turn::text(Role::Assistant, "final"));
    }

    #[tokio::test]
    async fn a_failed_remembering_refine_stage_persists_a_stop_note() {
        let mem = RecordingMemory::default();
        let r = run_refine(
            RefineCase {
                stage_extra: "    remember: true\n",
                refine_extra: "      max_rounds: 1\n",
                generator: vec![text("draft")],
                judge: vec![text(FAIL_JOIN)],
                conversation_id: Some("conv1"),
                ..Default::default()
            },
            &mem,
        )
        .await;
        assert!(r.outcome.is_err());
        let turns = mem.turns.lock().unwrap().clone();
        assert_eq!(turns.len(), 2, "{turns:?}");
        assert_eq!(turns[0].1, Turn::text(Role::User, "the question"));
        assert_eq!(turns[1].1.role, Role::Assistant);
        assert_eq!(
            turns[1].1.content,
            "[run stopped: refine: no candidate passed after 1 rounds]"
        );
        assert!(
            !turns[1].1.content.contains("use a JOIN"),
            "critique leaked"
        );
    }

    #[tokio::test]
    async fn refine_events_are_ordered_per_round() {
        let r = run_refine(
            RefineCase {
                generator: vec![text("a"), text("b")],
                judge: vec![text(FAIL_JOIN), text(PASS)],
                ..Default::default()
            },
            &NoMemory,
        )
        .await;
        // (kind, round-ordinal) positions: per round, the generator's
        // ResponseComplete precedes the judge's Started, which precedes the
        // round's RefineRound.
        let mut current_agent = String::new();
        let mut gen_done = Vec::new();
        let mut judge_started = Vec::new();
        let mut round_events = Vec::new();
        for (i, e) in r.events.iter().enumerate() {
            match e {
                WorkflowEvent::Agent {
                    event: AgentEvent::Started { agent_id },
                    ..
                } => {
                    current_agent = agent_id.clone();
                    if agent_id == "judge" {
                        judge_started.push(i);
                    }
                }
                WorkflowEvent::Agent {
                    event: AgentEvent::ResponseComplete { .. },
                    ..
                } if current_agent == "gen" => gen_done.push(i),
                WorkflowEvent::RefineRound { .. } => round_events.push(i),
                _ => {}
            }
        }
        assert_eq!(gen_done.len(), 2);
        assert_eq!(judge_started.len(), 2);
        assert_eq!(round_events.len(), 2);
        for round in 0..2 {
            assert!(gen_done[round] < judge_started[round]);
            assert!(judge_started[round] < round_events[round]);
        }
        assert!(round_events[0] < gen_done[1]);
    }
}
