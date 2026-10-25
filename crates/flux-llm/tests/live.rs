//! Live smoke tests against the real provider APIs. Ignored by default; run
//! with `cargo test -p flux-llm --test live -- --ignored`. Each test returns
//! early when its API key variable is unset.

use flux_config::ModelConfig;
use flux_llm::{
    stream_chat, ChatRequest, Content, LlmEvent, LlmRegistry, Message, OutputFormat, Reasoning,
    RetryConfig, Role, ToolDef,
};
use flux_settings::LlmCfg;

async fn text_and_usage(provider: &str, key_var: &str, model_var: &str, default_model: &str) {
    if std::env::var(key_var).map_or(true, |v| v.is_empty()) {
        eprintln!("skipping {provider} live test: {key_var} is not set");
        return;
    }
    let model_name = std::env::var(model_var).unwrap_or_else(|_| default_model.to_string());
    let model = ModelConfig {
        provider: provider.into(),
        ..ModelConfig::default()
    };
    let registry = LlmRegistry::from_settings(&LlmCfg::default()).expect("registry");
    let backend = registry.backend_for(&model).expect("backend");
    let req = ChatRequest {
        model: model_name,
        system: "Reply in one short sentence.".into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text("Say hello.".into())],
        }],
        tools: vec![],
        output: OutputFormat::Text,
        max_tokens: Some(64),
        temperature: None,
        reasoning: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    stream_chat(backend.as_ref(), req, &tx, RetryConfig::default())
        .await
        .expect("stream_chat");
    drop(tx);
    let (mut text, mut usage) = (String::new(), None);
    while let Some(ev) = rx.recv().await {
        match ev {
            LlmEvent::Text(t) => text.push_str(&t),
            LlmEvent::Usage(u) => usage = Some(u),
            _ => {}
        }
    }
    assert!(!text.trim().is_empty(), "no text from {provider}");
    let usage = usage.expect("usage event");
    assert!(usage.output_tokens.is_some(), "no output_tokens: {usage:?}");
}

#[tokio::test]
#[ignore = "needs ANTHROPIC_API_KEY"]
async fn anthropic_live_text_and_usage() {
    text_and_usage(
        "anthropic",
        "ANTHROPIC_API_KEY",
        "FLUX_LIVE_ANTHROPIC_MODEL",
        "claude-haiku-4-5",
    )
    .await;
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn openai_live_text_and_usage() {
    text_and_usage(
        "openai",
        "OPENAI_API_KEY",
        "FLUX_LIVE_OPENAI_MODEL",
        "gpt-4o-mini",
    )
    .await;
}

/// Runs one `stream_chat` call and returns every event it emitted.
async fn collect_events(backend: &dyn flux_llm::LlmBackend, req: ChatRequest) -> Vec<LlmEvent> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    stream_chat(backend, req, &tx, RetryConfig::default())
        .await
        .expect("stream_chat");
    drop(tx);
    let mut events = Vec::new();
    while let Some(ev) = rx.recv().await {
        events.push(ev);
    }
    events
}

/// Proves Anthropic accepts the thinking block replayed in a tool loop.
#[tokio::test]
#[ignore = "needs ANTHROPIC_API_KEY"]
async fn anthropic_live_tool_loop_with_thinking() {
    if std::env::var("ANTHROPIC_API_KEY").map_or(true, |v| v.is_empty()) {
        eprintln!("skipping anthropic live test: ANTHROPIC_API_KEY is not set");
        return;
    }
    let model_name = std::env::var("FLUX_LIVE_ANTHROPIC_THINKING_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-5".to_string());
    let model = ModelConfig {
        provider: "anthropic".into(),
        ..ModelConfig::default()
    };
    let registry = LlmRegistry::from_settings(&LlmCfg::default()).expect("registry");
    let backend = registry.backend_for(&model).expect("backend");

    let mut req = ChatRequest {
        model: model_name,
        system: "Use the tools provided to answer.".into(),
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text("What is the weather in Paris?".into())],
        }],
        tools: vec![ToolDef {
            name: "get_weather".into(),
            description: "Get the current weather for a city.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        }],
        output: OutputFormat::Text,
        max_tokens: Some(2048),
        temperature: None,
        reasoning: Some(Reasoning {
            effort: None,
            budget_tokens: Some(1024),
        }),
    };

    // Turn 1: the model thinks, then calls the tool.
    let events = collect_events(backend.as_ref(), req.clone()).await;
    let mut opaque = Vec::new();
    let mut text = String::new();
    let mut call = None;
    for ev in events {
        match ev {
            LlmEvent::Opaque(block) => opaque.push(Content::Opaque(block)),
            LlmEvent::Text(t) => text.push_str(&t),
            LlmEvent::ToolCall { id, name, args } if name == "get_weather" => {
                call = Some(Content::ToolCall { id, name, args });
            }
            _ => {}
        }
    }
    let call = call.expect("a get_weather tool call");
    assert!(!opaque.is_empty(), "no opaque thinking block in turn 1");
    let Content::ToolCall { id, name, .. } = &call else {
        unreachable!()
    };
    let result = Content::ToolResult {
        id: id.clone(),
        name: name.clone(),
        result: serde_json::json!({ "temp_c": 18 }),
    };

    // Turn 2: replay thinking blocks first, then text, then the tool call.
    let mut assistant = opaque;
    if !text.is_empty() {
        assistant.push(Content::Text(text));
    }
    assistant.push(call);
    req.messages.push(Message {
        role: Role::Assistant,
        content: assistant,
    });
    req.messages.push(Message {
        role: Role::User,
        content: vec![result],
    });
    let (mut answer, mut usage) = (String::new(), None);
    for ev in collect_events(backend.as_ref(), req).await {
        match ev {
            LlmEvent::Text(t) => answer.push_str(&t),
            LlmEvent::Usage(u) => usage = Some(u),
            _ => {}
        }
    }
    assert!(!answer.trim().is_empty(), "no text in turn 2");
    let usage = usage.expect("usage event");
    assert!(usage.output_tokens.is_some(), "no output_tokens: {usage:?}");
}
