use crate::output::AgentOutput;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AgentEvent {
    Started {
        agent_id: String,
    },
    Thought {
        text: String,
    },
    Text {
        delta: String,
    },
    ToolCall {
        tool: String,
        args: serde_json::Value,
    },
    ToolResult {
        tool: String,
        result: serde_json::Value,
    },
    /// Token usage of one LLM call; `None` when the provider did not report
    /// it. `model` is the provider-reported model, else the agent's profile.
    Usage {
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        model: String,
    },
    ResponseComplete {
        output: AgentOutput,
    },
    Failed {
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_event_is_constructable_and_comparable() {
        let a = AgentEvent::Thought { text: "hi".into() };
        let b = AgentEvent::Thought { text: "hi".into() };
        assert_eq!(a, b);
    }

    #[test]
    fn tool_events_are_constructable() {
        let call = AgentEvent::ToolCall {
            tool: "get_weather".into(),
            args: serde_json::json!({ "location": "SF" }),
        };
        let result = AgentEvent::ToolResult {
            tool: "get_weather".into(),
            result: serde_json::json!({ "temperature_f": 68 }),
        };
        assert_ne!(call, result);
    }
}
