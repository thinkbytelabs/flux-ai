//! Rendering of a tool stage's `args` tree against the workflow context.

use flux_core::error::{FluxError, Result};

/// If `s` is exactly one `{{ … }}` with nothing but whitespace around it,
/// return the inner expression source.
fn bare_expression(s: &str) -> Option<&str> {
    let t = s.trim();
    let inner = t.strip_prefix("{{")?.strip_suffix("}}")?;
    // Reject "{{ a }}{{ b }}" and "{{ a }} tail": a single pair only.
    if inner.contains("}}") || inner.contains("{{") {
        return None;
    }
    Some(inner.trim())
}

/// Render one `args` tree. String leaves are minijinja templates; a leaf that
/// is exactly one `{{ expr }}` yields `expr`'s native JSON value so that
/// objects and numbers survive as objects and numbers.
pub fn render_args(args: &serde_json::Value, ctx: &serde_json::Value) -> Result<serde_json::Value> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    render_node(&env, args, ctx)
}

fn render_node(
    env: &minijinja::Environment<'_>,
    node: &serde_json::Value,
    ctx: &serde_json::Value,
) -> Result<serde_json::Value> {
    match node {
        serde_json::Value::String(s) => match bare_expression(s) {
            Some(src) => {
                let expr = env
                    .compile_expression(src)
                    .map_err(|e| FluxError::Config(format!("args expression error: {e}")))?;
                let value = expr
                    .eval(ctx)
                    .map_err(|e| FluxError::Config(format!("args evaluation error: {e}")))?;
                if value.is_undefined() {
                    return Err(FluxError::Config(format!(
                        "args undefined reference in expression: {src}"
                    )));
                }
                serde_json::to_value(value)
                    .map_err(|e| FluxError::Config(format!("args value error: {e}")))
            }
            None => {
                let rendered = env
                    .render_str(s, ctx)
                    .map_err(|e| FluxError::Config(format!("args template error: {e}")))?;
                Ok(serde_json::Value::String(rendered))
            }
        },
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), render_node(env, v, ctx)?);
            }
            Ok(serde_json::Value::Object(out))
        }
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(render_node(env, v, ctx)?);
            }
            Ok(serde_json::Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> serde_json::Value {
        json!({
            "workflow": { "input": "hi" },
            "ctx": { "project_id": "p1", "limit": 50 },
            "transform": {
                "message": "{...}",
                "structured": { "select": ["a", "b"], "status": "ok", "n": 3 }
            }
        })
    }

    #[test]
    fn bare_expression_yields_native_object() {
        let out = render_args(&json!({ "plan": "{{ transform.structured }}" }), &ctx()).unwrap();
        assert!(out["plan"].is_object());
        assert_eq!(out["plan"]["select"], json!(["a", "b"]));
    }

    #[test]
    fn bare_expression_yields_native_number_not_string() {
        let out = render_args(&json!({ "n": "{{ transform.structured.n }}" }), &ctx()).unwrap();
        assert_eq!(out["n"], json!(3));
        assert!(out["n"].is_number());
    }

    #[test]
    fn bare_expression_tolerates_surrounding_whitespace() {
        let out = render_args(&json!({ "n": "  {{ transform.structured.n }}  " }), &ctx()).unwrap();
        assert_eq!(out["n"], json!(3));
    }

    #[test]
    fn interpolated_string_stays_a_string() {
        let out = render_args(
            &json!({ "note": "status is {{ transform.structured.status }}" }),
            &ctx(),
        )
        .unwrap();
        assert_eq!(out["note"], json!("status is ok"));
    }

    #[test]
    fn two_expressions_in_one_leaf_stay_a_string() {
        let out = render_args(
            &json!({ "k": "{{ ctx.project_id }}{{ transform.structured.status }}" }),
            &ctx(),
        )
        .unwrap();
        assert_eq!(out["k"], json!("p1ok"));
    }

    #[test]
    fn non_string_leaves_pass_through() {
        let out = render_args(&json!({ "a": 100, "b": true, "c": null }), &ctx()).unwrap();
        assert_eq!(out, json!({ "a": 100, "b": true, "c": null }));
    }

    #[test]
    fn nested_objects_and_arrays_are_walked() {
        let out = render_args(
            &json!({
                "outer": { "inner": "{{ ctx.project_id }}" },
                "list": ["{{ ctx.limit }}", "literal"]
            }),
            &ctx(),
        )
        .unwrap();
        assert_eq!(out["outer"]["inner"], json!("p1"));
        assert_eq!(out["list"][0], json!(50));
        assert_eq!(out["list"][1], json!("literal"));
    }

    #[test]
    fn undefined_reference_is_a_config_error() {
        match render_args(&json!({ "k": "{{ transform.structured.ghost }}" }), &ctx()) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn undefined_top_level_reference_is_a_config_error() {
        match render_args(&json!({ "k": "{{ ghost.structured }}" }), &ctx()) {
            Err(FluxError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn null_args_render_to_null() {
        assert_eq!(
            render_args(&serde_json::Value::Null, &ctx()).unwrap(),
            serde_json::Value::Null
        );
    }
}
