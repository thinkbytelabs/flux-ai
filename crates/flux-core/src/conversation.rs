//! Shared conversational domain types: a chat turn and its role.

/// Who produced a turn. The agent's system prompt is never a stored turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Role> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// One tool call the assistant made within a turn.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// One tool result fed back to the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolResultRecord {
    pub id: String,
    pub name: String,
    pub result: serde_json::Value,
}

/// One conversation turn. `tool_calls` is only ever non-empty on an assistant
/// turn, `tool_results` only on a user turn.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<ToolResultRecord>,
}

impl Turn {
    /// A plain text turn with no tool activity.
    pub fn text(role: Role, content: impl Into<String>) -> Turn {
        Turn {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_round_trips() {
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::from_str("user"), Some(Role::User));
        assert_eq!(Role::from_str("assistant"), Some(Role::Assistant));
        assert_eq!(Role::from_str("nope"), None);
    }

    #[test]
    fn turn_deserializes_pre_existing_json_without_tool_fields() {
        let stored = r#"{"role":"assistant","content":"hi"}"#;
        let turn: Turn = serde_json::from_str(stored).unwrap();
        assert_eq!(turn.content, "hi");
        assert!(turn.tool_calls.is_empty());
        assert!(turn.tool_results.is_empty());
    }

    #[test]
    fn turn_omits_empty_tool_fields_when_serializing() {
        let turn = Turn::text(Role::User, "hi");
        let s = serde_json::to_string(&turn).unwrap();
        assert!(!s.contains("tool_calls"));
        assert!(!s.contains("tool_results"));
    }

    #[test]
    fn turn_round_trips_tool_calls_and_results() {
        let turn = Turn {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: vec![ToolCallRecord {
                id: "c1".into(),
                name: "echo".into(),
                args: serde_json::json!({ "text": "hi" }),
            }],
            tool_results: Vec::new(),
        };
        let s = serde_json::to_string(&turn).unwrap();
        let back: Turn = serde_json::from_str(&s).unwrap();
        assert_eq!(back, turn);
    }
}
