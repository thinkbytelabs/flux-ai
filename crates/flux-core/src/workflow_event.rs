use crate::event::AgentEvent;
use crate::output::AgentOutput;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowEvent {
    StageStarted {
        stage: String,
    },
    StageSkipped {
        stage: String,
    },
    StageWaitingApproval {
        stage: String,
    },
    Agent {
        stage: String,
        event: AgentEvent,
    },
    /// One generate-then-evaluate round of a refine stage finished.
    /// `feedback` is `None` when the candidate passed.
    RefineRound {
        stage: String,
        round: u32,
        passed: bool,
        feedback: Option<String>,
    },
    StageComplete {
        stage: String,
        output: AgentOutput,
    },
    WorkflowComplete,
    WorkflowFailed {
        stage: String,
        error: String,
    },
    /// The run was stopped by a cancel request. Terminal, like the two above.
    WorkflowCancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_events_construct_and_compare() {
        let a = WorkflowEvent::StageStarted { stage: "x".into() };
        let b = WorkflowEvent::StageStarted { stage: "x".into() };
        assert_eq!(a, b);
        let wrap = WorkflowEvent::Agent {
            stage: "x".into(),
            event: AgentEvent::Thought { text: "hi".into() },
        };
        assert_ne!(a, wrap);
    }

    #[test]
    fn workflow_event_round_trips_through_json() {
        let original = WorkflowEvent::StageComplete {
            stage: "extract".into(),
            output: AgentOutput {
                message: "hi".into(),
                structured: Some(serde_json::json!({ "sentiment": "negative" })),
            },
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(original, back);

        let agent = WorkflowEvent::Agent {
            stage: "extract".into(),
            event: AgentEvent::ToolCall {
                tool: "search".into(),
                args: serde_json::json!({ "q": "x" }),
            },
        };
        let back2: WorkflowEvent =
            serde_json::from_str(&serde_json::to_string(&agent).unwrap()).unwrap();
        assert_eq!(agent, back2);
    }

    #[test]
    fn stage_waiting_approval_round_trips_and_is_not_terminal_shaped() {
        let ev = WorkflowEvent::StageWaitingApproval {
            stage: "act".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: WorkflowEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, back);
        // distinct from the terminal events
        assert_ne!(ev, WorkflowEvent::WorkflowComplete);
    }

    #[test]
    fn refine_round_round_trips_through_json() {
        for feedback in [None, Some("too long".to_string())] {
            let ev = WorkflowEvent::RefineRound {
                stage: "draft".into(),
                round: 2,
                passed: feedback.is_none(),
                feedback,
            };
            let back: WorkflowEvent =
                serde_json::from_str(&serde_json::to_string(&ev).unwrap()).unwrap();
            assert_eq!(ev, back);
        }
    }
}
