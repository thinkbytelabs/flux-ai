#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use flux_core::error::{FluxError, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub workflow_id: String,
    pub stages: Vec<Stage>,
    /// The stage whose output is the run's answer for `/chat` and `GET /runs/{id}`.
    /// Unset: the last stage to complete (the previous behavior).
    #[serde(default)]
    pub output: Option<String>,
}

/// What a stage's failure does to the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnError {
    /// Fail the whole run. The default, and the pre-existing behavior.
    #[default]
    Abort,
    /// Complete the stage with `{"error": …}` so downstream stages can branch.
    Continue,
}

/// When a stage with skipped dependencies still runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Join {
    /// Skipped if any dependency was skipped. The default.
    #[default]
    All,
    /// Skipped only if every dependency was skipped, so a stage can rejoin
    /// branches that `when` made mutually exclusive.
    Any,
}

/// What an exhausted refine loop does when no round passed.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnExhausted {
    /// Fail the stage. The default.
    #[default]
    Fail,
    /// Complete the stage with the last candidate.
    AcceptLast,
}

fn default_pass() -> String {
    "verdict.structured.pass".into()
}

fn default_feedback() -> String {
    "{{ verdict.message }}".into()
}

fn default_max_rounds() -> u32 {
    3
}

/// Evaluator-optimizer loop on an agent stage: generate, evaluate, revise.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refine {
    /// Agent id of the evaluator.
    pub evaluator: String,
    /// Template for the evaluator's input.
    pub evaluate: String,
    /// Expression over the verdict; must evaluate to a bool.
    #[serde(default = "default_pass")]
    pub pass: String,
    /// Template for the feedback handed to the revision.
    #[serde(default = "default_feedback")]
    pub feedback: String,
    /// Template for the revision input. Unset: a built-in prompt.
    #[serde(default)]
    pub revise: Option<String>,
    #[serde(default = "default_max_rounds")]
    pub max_rounds: u32,
    #[serde(default)]
    pub on_exhausted: OnExhausted,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub id: String,
    #[serde(default)]
    pub agent: Option<String>,
    /// Tool name for a tool stage. Exactly one of `agent` / `tool` is set.
    #[serde(default)]
    pub tool: Option<String>,
    /// Argument tree for a tool stage. String leaves are minijinja templates.
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(default)]
    pub input: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub requires_approval: bool,
    #[serde(default)]
    pub remember: bool,
    /// Persist and replay this agent's tool calls and results across turns.
    /// Off by default: replaying tool output re-sends earlier result sets on
    /// every later turn.
    #[serde(default)]
    pub replay_tools: bool,
    #[serde(default)]
    pub on_error: OnError,
    #[serde(default)]
    pub join: Join,
    #[serde(default)]
    pub refine: Option<Refine>,
}

/// Template variables Flux defines alongside the stage outputs.
const RESERVED_NAMES: [&str; 7] = [
    "workflow",
    "ctx",
    "candidate",
    "verdict",
    "feedback",
    "input",
    "round",
];

/// Template variables available inside a `refine` block, besides stage outputs.
pub const REFINE_VARIABLES: &[&str] = &["input", "candidate", "verdict", "feedback", "round"];

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn validate(wf: &Workflow) -> Result<()> {
    if wf.workflow_id.trim().is_empty() {
        return Err(FluxError::Config("workflow_id must not be empty".into()));
    }
    let mut ids: HashSet<&str> = HashSet::new();
    for stage in &wf.stages {
        if stage.id.trim().is_empty() {
            return Err(FluxError::Config("each stage needs a non-empty id".into()));
        }
        // Stage outputs share the template namespace with these, and templates
        // reference a stage as `<id>.message`, so the id must be an identifier.
        if RESERVED_NAMES.contains(&stage.id.as_str()) {
            return Err(FluxError::Config(format!(
                "stage id '{}' is reserved for the template variable of the same name",
                stage.id
            )));
        }
        if !is_identifier(&stage.id) {
            return Err(FluxError::Config(format!(
                "stage id '{}' must be an identifier (letters, digits, `_`; not starting with a digit)",
                stage.id
            )));
        }
        let has_agent = stage.agent.as_deref().is_some_and(|a| !a.trim().is_empty());
        let has_tool = stage.tool.as_deref().is_some_and(|t| !t.trim().is_empty());
        if has_agent == has_tool {
            return Err(FluxError::Config(format!(
                "stage '{}' must set exactly one of `agent` or `tool`",
                stage.id
            )));
        }
        if has_agent && !stage.args.is_null() {
            return Err(FluxError::Config(format!(
                "stage '{}' sets `args`, which is only valid on a tool stage",
                stage.id
            )));
        }
        if has_tool && (stage.remember || stage.replay_tools) {
            return Err(FluxError::Config(format!(
                "stage '{}' is a tool stage; `remember` and `replay_tools` only apply to agent stages",
                stage.id
            )));
        }
        if stage.replay_tools && !stage.remember {
            return Err(FluxError::Config(format!(
                "stage '{}' sets `replay_tools` without `remember`, which has no effect",
                stage.id
            )));
        }
        if let Some(refine) = &stage.refine {
            if has_tool {
                return Err(FluxError::Config(format!(
                    "stage '{}' is a tool stage; `refine` only applies to agent stages",
                    stage.id
                )));
            }
            if stage.replay_tools {
                return Err(FluxError::Config(format!(
                    "stage '{}' sets `refine` with `replay_tools`, which cannot be combined",
                    stage.id
                )));
            }
            if !(1..=10).contains(&refine.max_rounds) {
                return Err(FluxError::Config(format!(
                    "stage '{}' refine max_rounds must be between 1 and 10, got {}",
                    stage.id, refine.max_rounds
                )));
            }
        }
        if !ids.insert(stage.id.as_str()) {
            return Err(FluxError::Config(format!(
                "duplicate stage id '{}'",
                stage.id
            )));
        }
    }
    for stage in &wf.stages {
        for dep in &stage.depends_on {
            if !ids.contains(dep.as_str()) {
                return Err(FluxError::Config(format!(
                    "stage '{}' depends on unknown stage '{}'",
                    stage.id, dep
                )));
            }
        }
    }
    if let Some(out) = &wf.output {
        if !ids.contains(out.as_str()) {
            return Err(FluxError::Config(format!(
                "workflow output '{out}' is not a stage id"
            )));
        }
    }
    // Acyclicity is proven by a successful ready_rounds computation.
    ready_rounds(wf)?;
    check_template_references(wf, &ids)?;
    Ok(())
}

/// Every stage a template reads must be a transitive dependency. Otherwise the
/// read either fails mid-run, after earlier stages have spent LLM calls, or
/// works only because the stage happened to land in an earlier round. Names
/// that are not stage ids are left to the strict runtime check: they may be
/// minijinja globals such as `range`.
fn check_template_references(wf: &Workflow, ids: &HashSet<&str>) -> Result<()> {
    for stage in &wf.stages {
        let upstream = ancestors(wf, &stage.id);
        let mut names = template_names(&stage.id, "input", &stage.input)?;
        if let Some(when) = &stage.when {
            let env = minijinja::Environment::new();
            let expr = env.compile_expression(when).map_err(|e| {
                FluxError::Config(format!("stage '{}' when expression error: {e}", stage.id))
            })?;
            names.extend(expr.undeclared_variables(false));
        }
        collect_arg_names(&stage.id, &stage.args, &mut names)?;
        if let Some(refine) = &stage.refine {
            // `verdict` and `feedback` only exist later in the round: `verdict`
            // from the evaluator's reply on, `feedback` only in `revise`.
            let evaluate = template_names(&stage.id, "refine.evaluate", &refine.evaluate)?;
            reject_round_vars(
                &stage.id,
                "refine.evaluate",
                &evaluate,
                &["verdict", "feedback"],
            )?;
            names.extend(evaluate);
            let feedback = template_names(&stage.id, "refine.feedback", &refine.feedback)?;
            reject_round_vars(&stage.id, "refine.feedback", &feedback, &["feedback"])?;
            names.extend(feedback);
            if let Some(revise) = &refine.revise {
                names.extend(template_names(&stage.id, "refine.revise", revise)?);
            }
            let env = minijinja::Environment::new();
            let expr = env.compile_expression(&refine.pass).map_err(|e| {
                FluxError::Config(format!(
                    "stage '{}' refine.pass expression error: {e}",
                    stage.id
                ))
            })?;
            names.extend(expr.undeclared_variables(false));
        }
        for name in &names {
            if ids.contains(name.as_str()) && !upstream.contains(name.as_str()) {
                return Err(FluxError::Config(format!(
                    "stage '{}' reads stage '{name}', which is not in its depends_on (directly or transitively)",
                    stage.id
                )));
            }
        }
    }
    Ok(())
}

fn reject_round_vars(
    stage: &str,
    field: &str,
    used: &HashSet<String>,
    forbidden: &[&str],
) -> Result<()> {
    for var in forbidden {
        if used.contains(*var) {
            return Err(FluxError::Config(format!(
                "stage '{stage}' {field} uses '{var}', which is not defined at that point in a refine round"
            )));
        }
    }
    Ok(())
}

fn template_names(stage: &str, field: &str, src: &str) -> Result<HashSet<String>> {
    let env = minijinja::Environment::new();
    let tmpl = env
        .template_from_str(src)
        .map_err(|e| FluxError::Config(format!("stage '{stage}' {field} template error: {e}")))?;
    Ok(tmpl.undeclared_variables(false))
}

fn collect_arg_names(
    stage: &str,
    node: &serde_json::Value,
    out: &mut HashSet<String>,
) -> Result<()> {
    match node {
        serde_json::Value::String(s) => out.extend(template_names(stage, "args", s)?),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_arg_names(stage, item, out)?;
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                collect_arg_names(stage, value, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The stages `id` depends on, directly or transitively. Assumes an acyclic graph.
fn ancestors<'a>(wf: &'a Workflow, id: &str) -> HashSet<&'a str> {
    let mut seen = HashSet::new();
    let mut stack: Vec<&str> = vec![id];
    while let Some(current) = stack.pop() {
        if let Some(stage) = wf.stages.iter().find(|s| s.id == current) {
            for dep in &stage.depends_on {
                if seen.insert(dep.as_str()) {
                    stack.push(dep.as_str());
                }
            }
        }
    }
    seen
}

/// Group stage ids into dependency rounds: each round's stages depend only on
/// stages in earlier rounds. Stages in the same round are independent and may
/// run in parallel. Errors (Config) if the graph has a cycle.
pub fn ready_rounds(wf: &Workflow) -> Result<Vec<Vec<String>>> {
    let mut done: HashSet<String> = HashSet::new();
    let mut rounds: Vec<Vec<String>> = Vec::new();
    while done.len() < wf.stages.len() {
        let mut round: Vec<String> = Vec::new();
        for stage in &wf.stages {
            if done.contains(&stage.id) {
                continue;
            }
            if stage.depends_on.iter().all(|d| done.contains(d)) {
                round.push(stage.id.clone());
            }
        }
        if round.is_empty() {
            return Err(FluxError::Config("workflow has a dependency cycle".into()));
        }
        for id in &round {
            done.insert(id.clone());
        }
        rounds.push(round);
    }
    Ok(rounds)
}

pub fn load_workflow(path: &Path) -> Result<Workflow> {
    let text = std::fs::read_to_string(path)?;
    let wf: Workflow =
        serde_yaml_ng::from_str(&text).map_err(|e| FluxError::Config(e.to_string()))?;
    validate(&wf)?;
    Ok(wf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
workflow_id: triage
stages:
  - id: extract
    agent: classify
    input: "{{ workflow.input }}"
  - id: act
    agent: advisor
    depends_on: [extract]
    when: "extract.structured.sentiment == 'negative'"
    input: "Advise: {{ extract.structured.summary }}"
"#;

    #[test]
    fn join_defaults_to_all_and_parses_any() {
        let wf = one_stage("a");
        assert_eq!(wf.stages[0].join, Join::All);
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n    join: any\n",
        )
        .unwrap();
        assert_eq!(wf.stages[0].join, Join::Any);
    }

    #[test]
    fn rejects_an_unknown_join_value() {
        let r: std::result::Result<Workflow, _> = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n    join: some\n",
        );
        assert!(r.is_err());
    }

    fn one_stage(id: &str) -> Workflow {
        serde_yaml_ng::from_str(&format!(
            "workflow_id: w\nstages:\n  - id: {id}\n    agent: x\n    input: hi\n"
        ))
        .unwrap()
    }

    #[test]
    fn a_stage_id_may_not_shadow_a_template_variable() {
        // Stage outputs share the template namespace with `workflow` and `ctx`.
        for id in ["workflow", "ctx"] {
            let err = validate(&one_stage(id)).unwrap_err().to_string();
            assert!(err.contains("reserved"), "{id}: {err}");
        }
    }

    #[test]
    fn a_stage_id_must_be_referenceable_in_a_template() {
        // `my-stage.message` would parse as a subtraction.
        for id in ["my-stage", "1st", "a.b"] {
            let err = validate(&one_stage(id)).unwrap_err().to_string();
            assert!(err.contains("identifier"), "{id}: {err}");
        }
        assert!(validate(&one_stage("file_bug_2")).is_ok());
    }

    #[test]
    fn a_template_may_not_read_a_stage_it_does_not_depend_on() {
        // b and c share a round: whether c sees b's output would be timing luck.
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n  - id: b\n    agent: x\n    depends_on: [a]\n    input: hi\n  - id: c\n    agent: x\n    depends_on: [a]\n    input: \"{{ b.message }}\"\n",
        )
        .unwrap();
        let err = validate(&wf).unwrap_err().to_string();
        assert!(
            err.contains("'c'") && err.contains("'b'") && err.contains("depends_on"),
            "{err}"
        );
    }

    #[test]
    fn a_when_or_args_reference_needs_the_dependency_too() {
        let when: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n  - id: b\n    agent: x\n    when: \"a.message == 'x'\"\n    input: hi\n",
        )
        .unwrap();
        assert!(validate(&when)
            .unwrap_err()
            .to_string()
            .contains("depends_on"));
        let args: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n  - id: b\n    tool: t\n    args: { q: \"{{ a.structured }}\" }\n",
        )
        .unwrap();
        assert!(validate(&args)
            .unwrap_err()
            .to_string()
            .contains("depends_on"));
    }

    #[test]
    fn a_template_may_read_a_transitive_dependency_and_builtins() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"{{ workflow.input }} {{ ctx.user }}\"\n  - id: b\n    agent: x\n    depends_on: [a]\n    input: hi\n  - id: c\n    agent: x\n    depends_on: [b]\n    input: \"{{ a.message }}{% for i in range(2) %}{{ i }}{% endfor %}\"\n",
        )
        .unwrap();
        validate(&wf).unwrap();
    }

    #[test]
    fn output_must_name_a_stage() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\noutput: nope\nstages:\n  - id: a\n    agent: x\n    input: hi\n",
        )
        .unwrap();
        let err = validate(&wf).unwrap_err().to_string();
        assert!(err.contains("output") && err.contains("nope"), "{err}");
    }

    #[test]
    fn output_naming_a_stage_is_valid() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\noutput: a\nstages:\n  - id: a\n    agent: x\n    input: hi\n",
        )
        .unwrap();
        assert!(validate(&wf).is_ok());
        assert_eq!(wf.output.as_deref(), Some("a"));
    }

    #[test]
    fn parses_and_validates_a_workflow() {
        let wf: Workflow = serde_yaml_ng::from_str(GOOD).unwrap();
        assert_eq!(wf.workflow_id, "triage");
        assert_eq!(wf.stages.len(), 2);
        assert_eq!(wf.stages[1].depends_on, vec!["extract".to_string()]);
        assert_eq!(
            wf.stages[1].when.as_deref(),
            Some("extract.structured.sentiment == 'negative'")
        );
        validate(&wf).unwrap();
    }

    #[test]
    fn rejects_missing_dependency() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"\"\n    depends_on: [ghost]\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_stage_id() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"\"\n  - id: a\n    agent: y\n    input: \"\"\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(validate(&wf).is_err());
    }

    #[test]
    fn rejects_cycle() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"\"\n    depends_on: [b]\n  - id: b\n    agent: y\n    input: \"\"\n    depends_on: [a]\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(ready_rounds(&wf).is_err());
    }

    #[test]
    fn ready_rounds_groups_a_diamond() {
        // a -> b, a -> c, (b,c) -> d
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"\"\n  - id: b\n    agent: x\n    input: \"\"\n    depends_on: [a]\n  - id: c\n    agent: x\n    input: \"\"\n    depends_on: [a]\n  - id: d\n    agent: x\n    input: \"\"\n    depends_on: [b, c]\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        let rounds = ready_rounds(&wf).unwrap();
        assert_eq!(rounds[0], vec!["a".to_string()]);
        assert_eq!(rounds.len(), 3);
        let mut mid = rounds[1].clone();
        mid.sort();
        assert_eq!(mid, vec!["b".to_string(), "c".to_string()]);
        assert_eq!(rounds[2], vec!["d".to_string()]);
    }

    #[test]
    fn stage_parses_requires_approval_default_false() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"\"\n  - id: b\n    agent: y\n    input: \"\"\n    requires_approval: true\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(!wf.stages[0].requires_approval);
        assert!(wf.stages[1].requires_approval);
    }

    #[test]
    fn remember_defaults_false_and_parses_true() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"{{ workflow.input }}\"\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(!wf.stages[0].remember);

        let yaml2 = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: \"hi\"\n    remember: true\n";
        let wf2: Workflow = serde_yaml_ng::from_str(yaml2).unwrap();
        assert!(wf2.stages[0].remember);
    }

    #[test]
    fn parses_a_tool_stage() {
        let yaml = r#"
workflow_id: w
stages:
  - id: exec
    tool: safe_exec
    args:
      sql: "{{ compile.structured.sql }}"
      limit: 100
"#;
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        validate(&wf).unwrap();
        assert_eq!(wf.stages[0].tool.as_deref(), Some("safe_exec"));
        assert!(wf.stages[0].agent.is_none());
        assert_eq!(wf.stages[0].args["limit"], 100);
    }

    #[test]
    fn rejects_stage_with_neither_agent_nor_tool() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    input: \"\"\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(m)) => assert!(m.contains("exactly one")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_stage_with_both_agent_and_tool() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    tool: t\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(validate(&wf).is_err());
    }

    #[test]
    fn rejects_args_on_an_agent_stage() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    args:\n      k: v\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(m)) => assert!(m.contains("args")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn existing_agent_stages_still_validate() {
        let wf: Workflow = serde_yaml_ng::from_str(GOOD).unwrap();
        validate(&wf).unwrap();
        assert_eq!(wf.stages[0].agent.as_deref(), Some("classify"));
    }

    #[test]
    fn on_error_defaults_to_abort_and_parses_continue() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n  - id: b\n    agent: y\n    on_error: continue\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(wf.stages[0].on_error, OnError::Abort);
        assert_eq!(wf.stages[1].on_error, OnError::Continue);
    }

    #[test]
    fn replay_tools_defaults_false_and_parses_true() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n  - id: b\n    agent: y\n    replay_tools: true\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(!wf.stages[0].replay_tools);
        assert!(wf.stages[1].replay_tools);
    }

    #[test]
    fn rejects_remember_on_a_tool_stage() {
        let yaml = "workflow_id: w\nstages:\n  - id: exec\n    tool: t\n    remember: true\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(m)) => assert!(m.contains("exec")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_replay_tools_on_a_tool_stage() {
        let yaml = "workflow_id: w\nstages:\n  - id: exec\n    tool: t\n    replay_tools: true\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(m)) => assert!(m.contains("exec")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_replay_tools_without_remember() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    replay_tools: true\n";
        let wf: Workflow = serde_yaml_ng::from_str(yaml).unwrap();
        match validate(&wf) {
            Err(FluxError::Config(m)) => assert!(m.contains("'a'")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_an_unknown_stage_field() {
        // `whn:` is a typo of `when`. Before deny_unknown_fields it silently
        // no-opped, running a gated stage unconditionally.
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    whn: \"1 == 1\"\n";
        assert!(serde_yaml_ng::from_str::<Workflow>(yaml).is_err());
    }

    #[test]
    fn rejects_an_unknown_workflow_field() {
        let yaml = "workflow_id: w\nstagez: []\nstages:\n  - id: a\n    agent: x\n";
        assert!(serde_yaml_ng::from_str::<Workflow>(yaml).is_err());
    }

    fn refine_wf(refine: &str) -> Workflow {
        serde_yaml_ng::from_str(&format!(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    input: hi\n  - id: b\n    agent: x\n    depends_on: [a]\n    input: hi\n    refine:\n{refine}"
        ))
        .unwrap()
    }

    #[test]
    fn refine_block_parses_with_defaults() {
        let wf = refine_wf("      evaluator: judge\n      evaluate: \"{{ candidate.message }}\"\n");
        let r = wf.stages[1].refine.as_ref().unwrap();
        assert_eq!(r.evaluator, "judge");
        assert_eq!(r.pass, "verdict.structured.pass");
        assert_eq!(r.feedback, "{{ verdict.message }}");
        assert!(r.revise.is_none());
        assert_eq!(r.max_rounds, 3);
        assert_eq!(r.on_exhausted, OnExhausted::Fail);
        validate(&wf).unwrap();
    }

    #[test]
    fn refine_on_a_tool_stage_is_rejected() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: t\n    tool: q\n    refine:\n      evaluator: judge\n      evaluate: e\n",
        )
        .unwrap();
        let err = validate(&wf).unwrap_err().to_string();
        assert!(err.contains("'t'") && err.contains("refine"), "{err}");
    }

    #[test]
    fn refine_max_rounds_must_be_between_1_and_10() {
        for n in [0, 11] {
            let wf = refine_wf(&format!(
                "      evaluator: judge\n      evaluate: e\n      max_rounds: {n}\n"
            ));
            let err = validate(&wf).unwrap_err().to_string();
            assert!(
                err.contains("'b'") && err.contains("max_rounds"),
                "{n}: {err}"
            );
        }
        for n in [1, 10] {
            let wf = refine_wf(&format!(
                "      evaluator: judge\n      evaluate: e\n      max_rounds: {n}\n"
            ));
            validate(&wf).unwrap();
        }
    }

    #[test]
    fn refine_with_replay_tools_is_rejected() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    remember: true\n    replay_tools: true\n    refine:\n      evaluator: judge\n      evaluate: e\n",
        )
        .unwrap();
        let err = validate(&wf).unwrap_err().to_string();
        assert!(err.contains("'a'") && err.contains("replay_tools"), "{err}");
    }

    #[test]
    fn refine_variables_are_reserved_stage_ids() {
        for id in ["candidate", "round", "verdict", "feedback", "input"] {
            let err = validate(&one_stage(id)).unwrap_err().to_string();
            assert!(err.contains("reserved"), "{id}: {err}");
        }
    }

    #[test]
    fn refine_templates_may_read_refine_variables_and_ancestors() {
        let wf = refine_wf(
            "      evaluator: judge\n      evaluate: \"{{ input }} {{ candidate.message }} {{ round }} {{ a.message }}\"\n      pass: \"verdict.structured.score > round and a.message\"\n      feedback: \"{{ verdict.message }} {{ a.message }}\"\n      revise: \"{{ input }} {{ candidate.message }} {{ feedback }} {{ round }}\"\n",
        );
        validate(&wf).unwrap();
    }

    #[test]
    fn refine_template_reading_a_non_ancestor_is_rejected() {
        let wf: Workflow = serde_yaml_ng::from_str(
            "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    refine:\n      evaluator: judge\n      evaluate: \"{{ later.message }}\"\n  - id: later\n    agent: x\n",
        )
        .unwrap();
        let err = validate(&wf).unwrap_err().to_string();
        assert!(err.contains("'a'") && err.contains("'later'"), "{err}");
    }

    #[test]
    fn refine_unparsable_template_or_expression_is_rejected() {
        for field in [
            "evaluate: \"{{ \"",
            "pass: \"1 +\"",
            "feedback: \"{{ \"",
            "revise: \"{% if %}\"",
        ] {
            let wf = refine_wf(&format!(
                "      evaluator: judge\n      {field}\n      {}\n",
                if field.starts_with("evaluate") {
                    ""
                } else {
                    "evaluate: e"
                }
            ));
            assert!(validate(&wf).is_err(), "{field}");
        }
    }

    #[test]
    fn refine_variables_are_checked_against_their_round_scope() {
        for (field, var) in [
            ("evaluate: \"{{ verdict.message }}\"", "verdict"),
            ("evaluate: \"{{ feedback }}\"", "feedback"),
            ("feedback: \"{{ feedback }}\"", "feedback"),
        ] {
            let extra = if field.starts_with("evaluate") {
                String::new()
            } else {
                "      evaluate: e\n".to_string()
            };
            let wf = refine_wf(&format!("      evaluator: judge\n{extra}      {field}\n"));
            let err = validate(&wf).unwrap_err().to_string();
            assert!(
                err.contains("'b'") && err.contains(var) && err.contains("refine."),
                "{field}: {err}"
            );
        }
        let wf = refine_wf(
            "      evaluator: judge\n      evaluate: e\n      pass: \"verdict.structured.pass\"\n      revise: \"{{ feedback }} {{ verdict.message }}\"\n",
        );
        validate(&wf).unwrap();
    }

    #[test]
    fn refine_unknown_field_is_rejected() {
        let yaml = "workflow_id: w\nstages:\n  - id: a\n    agent: x\n    refine:\n      evaluator: judge\n      evaluate: e\n      max_round: 3\n";
        assert!(serde_yaml_ng::from_str::<Workflow>(yaml).is_err());
    }

    #[test]
    fn repo_workflows_dir_still_parses() {
        // The shipped samples must survive the stricter parse.
        let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../workflows"));
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                load_workflow(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            }
        }
    }
}
