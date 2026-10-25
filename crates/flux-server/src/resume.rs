use flux_core::output::AgentOutput;
use flux_core::workflow_event::WorkflowEvent;
use flux_session::Offset;
use std::collections::{HashMap, HashSet};

/// Fold a run's event log into the executor's resume state: completed outputs
/// (from StageComplete) and the skipped set (from StageSkipped). Other events
/// (Started, Agent, WaitingApproval, terminal) do not contribute.
pub(crate) fn rebuild_state(
    events: &[(Offset, WorkflowEvent)],
) -> (HashMap<String, AgentOutput>, HashSet<String>) {
    let mut completed = HashMap::new();
    let mut skipped = HashSet::new();
    for (_, event) in events {
        match event {
            WorkflowEvent::StageComplete { stage, output } => {
                completed.insert(stage.clone(), output.clone());
            }
            WorkflowEvent::StageSkipped { stage } => {
                skipped.insert(stage.clone());
            }
            _ => {}
        }
    }
    (completed, skipped)
}

/// The gates a paused run is waiting on: every stage that logged
/// StageWaitingApproval and has not since completed or been skipped.
pub(crate) fn awaiting_approval(events: &[(Offset, WorkflowEvent)]) -> HashSet<String> {
    let (completed, skipped) = rebuild_state(events);
    events
        .iter()
        .filter_map(|(_, event)| match event {
            WorkflowEvent::StageWaitingApproval { stage }
                if !completed.contains_key(stage) && !skipped.contains(stage) =>
            {
                Some(stage.clone())
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::output::AgentOutput;
    use flux_core::workflow_event::WorkflowEvent;

    #[test]
    fn rebuild_state_folds_completed_and_skipped() {
        let out = AgentOutput {
            message: "m".into(),
            structured: None,
        };
        let events = vec![
            (1u64, WorkflowEvent::StageStarted { stage: "a".into() }),
            (
                2u64,
                WorkflowEvent::StageComplete {
                    stage: "a".into(),
                    output: out.clone(),
                },
            ),
            (3u64, WorkflowEvent::StageSkipped { stage: "b".into() }),
            (
                4u64,
                WorkflowEvent::StageWaitingApproval { stage: "c".into() },
            ),
        ];
        let (completed, skipped) = rebuild_state(&events);
        assert_eq!(completed.len(), 1);
        assert!(completed.contains_key("a"));
        assert_eq!(completed["a"].message, "m");
        assert_eq!(skipped.len(), 1);
        assert!(skipped.contains("b"));
        // a waiting gate is neither completed nor skipped
        assert!(!completed.contains_key("c"));
        assert!(!skipped.contains("c"));
    }

    #[test]
    fn awaiting_approval_excludes_a_gate_that_later_completed() {
        let out = AgentOutput {
            message: "m".into(),
            structured: None,
        };
        let events = vec![
            (
                1u64,
                WorkflowEvent::StageWaitingApproval { stage: "a".into() },
            ),
            (
                2u64,
                WorkflowEvent::StageComplete {
                    stage: "a".into(),
                    output: out,
                },
            ),
            (
                3u64,
                WorkflowEvent::StageWaitingApproval { stage: "b".into() },
            ),
        ];
        let waiting = awaiting_approval(&events);
        assert_eq!(waiting.len(), 1);
        assert!(waiting.contains("b"));
    }
}
