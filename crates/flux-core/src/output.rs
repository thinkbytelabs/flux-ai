#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AgentOutput {
    pub message: String,
    pub structured: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_output_carries_optional_structured() {
        let text = AgentOutput {
            message: "hello".into(),
            structured: None,
        };
        let structured = AgentOutput {
            message: "{\"a\":1}".into(),
            structured: Some(serde_json::json!({ "a": 1 })),
        };
        assert!(text.structured.is_none());
        assert_eq!(structured.structured, Some(serde_json::json!({ "a": 1 })));
    }
}
