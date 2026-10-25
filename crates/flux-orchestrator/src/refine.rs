//! Evaluator-optimizer ("refine") stages: generate a candidate, have an
//! evaluator agent judge it, and revise with its feedback until it passes or
//! the round budget runs out.

use crate::{append_exchange, conversation_key, run_stage_agent, RunTuning, StageOutcome};
use flux_config::AgentConfig;
use flux_core::conversation::{Role, Turn};
use flux_core::error::{FluxError, Result};
use flux_core::output::AgentOutput;
use flux_core::workflow_event::WorkflowEvent;
use flux_llm::LlmRegistry;
use flux_memory::ConversationMemory;
use flux_tool_client::ToolTransport;
use flux_workflow::{OnExhausted, Refine, Stage};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// The generator's input for rounds after the first when `revise` is unset.
pub(crate) const DEFAULT_REVISE: &str = "{{ input }}

Your previous answer:
{{ candidate.message }}

A reviewer rejected it: {{ feedback }}
Write a corrected answer.";

/// Evaluate the `pass` expression. Only a real boolean is an answer: an
/// undefined, string (`"false"` is truthy), number or null result is an
/// error naming the expression and what it produced — never coerced.
pub(crate) fn evaluate_pass(expr_src: &str, vars: &serde_json::Value) -> Result<bool> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let expr_error = |e: minijinja::Error| {
        FluxError::Config(format!("refine pass expression '{expr_src}' error: {e}"))
    };
    let expr = env.compile_expression(expr_src).map_err(expr_error)?;
    let kind = match expr.eval(vars) {
        Ok(v) if v.kind() == minijinja::value::ValueKind::Bool => return Ok(v.is_true()),
        Ok(v) => match v.kind() {
            minijinja::value::ValueKind::None => "null".to_string(),
            other => other.to_string(),
        },
        Err(e) if e.kind() == minijinja::ErrorKind::UndefinedError => "undefined".to_string(),
        Err(e) => return Err(expr_error(e)),
    };
    Err(FluxError::Config(format!(
        "refine pass expression '{expr_src}' must be a boolean, got {kind}"
    )))
}

fn render(name: &str, template: &str, vars: &serde_json::Value) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.render_str(template, vars)
        .map_err(|e| FluxError::Config(format!("refine {name} template error: {e}")))
}

fn set(vars: &mut serde_json::Value, name: &str, value: serde_json::Value) {
    if let Some(m) = vars.as_object_mut() {
        m.insert(name.to_string(), value);
    }
}

fn output_value(out: &AgentOutput) -> serde_json::Value {
    serde_json::json!({ "message": out.message, "structured": out.structured })
}

/// Why the loop stopped without a candidate. `note` is the reason recorded in
/// memory as `[run stopped: <note>]`; `None` skips persisting (memory itself
/// failed).
struct Stop {
    error: FluxError,
    note: Option<String>,
}

impl From<FluxError> for Stop {
    fn from(error: FluxError) -> Self {
        let note = Some(error.to_string());
        Stop { error, note }
    }
}

/// What a cancelled agent stage returns today; the run loop sees the token
/// and reports `WorkflowCancelled` instead of folding it.
fn cancelled() -> Stop {
    Stop {
        error: FluxError::Provider("agent run aborted: cancelled".into()),
        note: Some("cancelled".into()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_refine<M: ConversationMemory + Sync>(
    stage: &Stage,
    refine: &Refine,
    generator: &AgentConfig,
    evaluator: &AgentConfig,
    input: String,
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
    // Only the generator remembers; the evaluator never touches memory.
    let key = conversation_id
        .filter(|_| stage.remember)
        .map(|cid| conversation_key(cid, &generator.agent_id));
    let result = refine_loop(
        stage,
        refine,
        generator,
        evaluator,
        &input,
        ctx,
        llm,
        transport,
        key.as_deref(),
        context,
        memory,
        tx,
        tuning,
        cancel,
    )
    .await;
    match result {
        Ok(candidate) => {
            if let Some(key) = &key {
                let turns = [
                    Turn::text(Role::User, input),
                    Turn::text(Role::Assistant, candidate.message.clone()),
                ];
                if let Err(e) = append_exchange(memory, key, &turns).await {
                    return StageOutcome::Failed(e);
                }
            }
            StageOutcome::Completed(candidate)
        }
        Err(Stop { error, note }) => {
            if let (Some(key), Some(note)) = (&key, note) {
                let turns = [
                    Turn::text(Role::User, input),
                    Turn::text(Role::Assistant, format!("[run stopped: {note}]")),
                ];
                // Already logged by append_exchange; the loop's own error is
                // the one worth reporting.
                let _ = append_exchange(memory, key, &turns).await;
            }
            StageOutcome::Failed(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn refine_loop<M: ConversationMemory + Sync>(
    stage: &Stage,
    refine: &Refine,
    generator: &AgentConfig,
    evaluator: &AgentConfig,
    input: &str,
    ctx: &serde_json::Value,
    llm: &LlmRegistry,
    transport: &(impl ToolTransport + Sync),
    memory_key: Option<&str>,
    context: Option<&serde_json::Value>,
    memory: &M,
    tx: &mpsc::Sender<WorkflowEvent>,
    tuning: RunTuning,
    cancel: &CancellationToken,
) -> std::result::Result<AgentOutput, Stop> {
    let mut vars = if ctx.is_object() {
        ctx.clone()
    } else {
        serde_json::json!({})
    };
    set(&mut vars, "input", input.into());
    let revise = refine.revise.as_deref().unwrap_or(DEFAULT_REVISE);
    let mut generator_input = input.to_string();
    let mut last: Option<(AgentOutput, String)> = None;

    for round in 1..=refine.max_rounds {
        if cancel.is_cancelled() {
            return Err(cancelled());
        }
        let history = match memory_key {
            Some(key) => flux_memory::recall_window(memory, key, tuning.history_window)
                .await
                .map_err(|error| Stop { error, note: None })?,
            None => Vec::new(),
        };
        let run = run_stage_agent(
            &stage.id,
            generator,
            &history,
            &generator_input,
            llm,
            transport,
            context,
            tx,
            tuning,
            cancel,
        )
        .await;
        if cancel.is_cancelled() {
            return Err(cancelled());
        }
        let run = run?;
        if let Some(reason) = run.aborted {
            return Err(Stop {
                error: FluxError::Provider(format!("agent run aborted: {reason}")),
                note: Some(reason),
            });
        }
        let candidate = run.output;
        set(&mut vars, "round", round.into());
        set(&mut vars, "candidate", output_value(&candidate));
        if let Some(m) = vars.as_object_mut() {
            m.remove("verdict");
            m.remove("feedback");
        }

        let evaluate_input = render("evaluate", &refine.evaluate, &vars)?;
        let verdict = run_stage_agent(
            &stage.id,
            evaluator,
            &[],
            &evaluate_input,
            llm,
            transport,
            context,
            tx,
            tuning,
            cancel,
        )
        .await;
        // An interrupted round reports no RefineRound.
        if cancel.is_cancelled() {
            return Err(cancelled());
        }
        let verdict = verdict?;
        if let Some(reason) = verdict.aborted {
            return Err(Stop {
                error: FluxError::Provider(format!("refine evaluator aborted: {reason}")),
                note: Some(reason),
            });
        }
        set(&mut vars, "verdict", output_value(&verdict.output));

        let passed = evaluate_pass(&refine.pass, &vars)?;
        let feedback = if passed {
            None
        } else {
            Some(render("feedback", &refine.feedback, &vars)?)
        };
        let _ = tx
            .send(WorkflowEvent::RefineRound {
                stage: stage.id.clone(),
                round,
                passed,
                feedback: feedback.clone(),
            })
            .await;
        tracing::info!(round, passed, "refine round evaluated");
        let Some(feedback) = feedback else {
            return Ok(candidate);
        };
        if round < refine.max_rounds {
            set(&mut vars, "feedback", feedback.clone().into());
            generator_input = render("revise", revise, &vars)?;
        }
        last = Some((candidate, feedback));
    }

    match (refine.on_exhausted, last) {
        (OnExhausted::AcceptLast, Some((candidate, _))) => Ok(candidate),
        (_, last) => Err(Stop {
            error: FluxError::Provider(format!(
                "refine: no candidate passed after {} rounds; last feedback: {}",
                refine.max_rounds,
                last.map(|(_, f)| f).unwrap_or_default()
            )),
            // The critique stays out of the generator's conversation history.
            note: Some(format!(
                "refine: no candidate passed after {} rounds",
                refine.max_rounds
            )),
        }),
    }
}
