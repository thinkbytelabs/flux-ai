//! Metrics facade. No-ops unless the `otel` feature is active AND init_tracing
//! set up a MeterProvider. The orchestration libs call these unconditionally,
//! so opentelemetry never becomes their dependency. Only counts/outcomes are
//! recorded — never message content.

#[cfg(not(feature = "otel"))]
mod imp {
    #[inline]
    pub fn record_run() {}
    #[inline]
    pub fn record_stage(_outcome: &str) {}
    #[inline]
    pub fn record_tool_call(_outcome: &str) {}
    #[inline]
    pub fn record_llm_call(_input_tokens: Option<u32>, _output_tokens: Option<u32>) {}
}

#[cfg(feature = "otel")]
mod imp {
    use crate::telemetry::meter_provider;
    use opentelemetry::metrics::{Counter, MeterProvider as _};
    use opentelemetry::KeyValue;
    use std::sync::OnceLock;

    /// Lazily-built `u64` counter instruments, keyed by name. Each is built once
    /// from the global meter; if the meter provider is unset they are never
    /// initialized and the record fns no-op.
    struct Counters {
        runs: Counter<u64>,
        stages: Counter<u64>,
        tool_calls: Counter<u64>,
        llm_calls: Counter<u64>,
        tokens_input: Counter<u64>,
        tokens_output: Counter<u64>,
    }

    static COUNTERS: OnceLock<Counters> = OnceLock::new();

    /// Fetch (or lazily build) the counter set. Returns `None` when no global
    /// `MeterProvider` was installed (otel built but endpoint absent / init not
    /// called), so callers no-op safely without panicking.
    fn counters() -> Option<&'static Counters> {
        if let Some(c) = COUNTERS.get() {
            return Some(c);
        }
        let provider = meter_provider()?;
        let meter = provider.meter("flux");
        let built = Counters {
            runs: meter.u64_counter("flux.runs").build(),
            stages: meter.u64_counter("flux.stages").build(),
            tool_calls: meter.u64_counter("flux.tool_calls").build(),
            llm_calls: meter.u64_counter("flux.llm_calls").build(),
            tokens_input: meter.u64_counter("flux.tokens.input").build(),
            tokens_output: meter.u64_counter("flux.tokens.output").build(),
        };
        // First writer wins; if another thread set it, use the stored one.
        let _ = COUNTERS.set(built);
        COUNTERS.get()
    }

    pub fn record_run() {
        if let Some(c) = counters() {
            c.runs.add(1, &[]);
        }
    }

    pub fn record_stage(outcome: &str) {
        if let Some(c) = counters() {
            c.stages
                .add(1, &[KeyValue::new("outcome", outcome.to_string())]);
        }
    }

    pub fn record_tool_call(outcome: &str) {
        if let Some(c) = counters() {
            c.tool_calls
                .add(1, &[KeyValue::new("outcome", outcome.to_string())]);
        }
    }

    pub fn record_llm_call(input_tokens: Option<u32>, output_tokens: Option<u32>) {
        if let Some(c) = counters() {
            c.llm_calls.add(1, &[]);
            if let Some(n) = input_tokens {
                c.tokens_input.add(u64::from(n), &[]);
            }
            if let Some(n) = output_tokens {
                c.tokens_output.add(u64::from(n), &[]);
            }
        }
    }
}

pub use imp::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_fns_are_callable_and_noop_safe() {
        // Without the otel feature these are no-ops; must never panic.
        record_run();
        record_stage("completed");
        record_tool_call("ok");
        record_llm_call(Some(10), Some(20));
        record_llm_call(None, None);
    }
}
