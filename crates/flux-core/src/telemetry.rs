//! Process-wide logging/tracing initialization.
//!
//! `init_tracing` builds an env-filtered `tracing_subscriber` whose format is
//! supplied by the caller as a [`LogFormat`] (pretty or JSON). Config-aware
//! binaries derive it from `settings.telemetry.log_format` (which already folds
//! in the `FLUX_LOG_FORMAT` env override) via [`log_format_from_str`]; binaries
//! without config use [`log_format_from_env`]. Callers invoke it once at startup
//! and hold the returned [`TracingGuard`] for the process lifetime; library
//! crates only emit spans/events.
//!
//! When the `otel` feature is enabled AND `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
//! an OpenTelemetry/OTLP export layer is added alongside the fmt layer and the
//! guard flushes the exporters on drop. With the feature off (the default), the
//! behavior is identical to the fmt-only subscriber.

use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter, Layer, Registry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Pretty,
    Json,
}

/// Pretty unless `s` equals "json" (case-insensitive). Shared by the config
/// path (`settings.telemetry.log_format`) and the env path.
pub fn log_format_from_str(s: &str) -> LogFormat {
    if s.eq_ignore_ascii_case("json") {
        LogFormat::Json
    } else {
        LogFormat::Pretty
    }
}

/// Pretty unless `FLUX_LOG_FORMAT` equals "json" (case-insensitive). For
/// binaries without a `Settings` (e.g. the example tool server).
pub fn log_format_from_env() -> LogFormat {
    log_format_from_str(&std::env::var("FLUX_LOG_FORMAT").unwrap_or_default())
}

/// Boxed fmt layer reproducing the M5h output: pretty/JSON switch crossed with
/// stderr/stdout writer. `.boxed()` unifies the four arms into one type.
fn build_fmt_layer<S>(to_stderr: bool, format: LogFormat) -> Box<dyn Layer<S> + Send + Sync>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    match (format, to_stderr) {
        (LogFormat::Json, true) => fmt::layer().json().with_writer(std::io::stderr).boxed(),
        (LogFormat::Json, false) => fmt::layer().json().boxed(),
        (LogFormat::Pretty, true) => fmt::layer().with_writer(std::io::stderr).boxed(),
        (LogFormat::Pretty, false) => fmt::layer().boxed(),
    }
}

/// Held for process lifetime; on drop, flushes/shuts down OTel exporters
/// (no-op unless the `otel` feature is active and an endpoint was configured).
#[derive(Default)]
pub struct TracingGuard {
    // Held only for its `Drop` (flushes OTLP exporters); never read directly.
    #[cfg(feature = "otel")]
    #[allow(dead_code)]
    otel: Option<otel::OtelProviders>,
}

/// Initialize the global tracing subscriber. `to_stderr` writes logs to stderr
/// (used by the CLI so its product output on stdout stays clean); otherwise
/// stdout. `format` selects pretty/JSON output (derive it from config via
/// [`log_format_from_str`] or from env via [`log_format_from_env`]). Idempotent:
/// a second call is a no-op (`try_init`). Honors `RUST_LOG` via `EnvFilter`,
/// defaulting to a filter scoping third-party crates to warn.
///
/// Returns a [`TracingGuard`] that must be held for the process lifetime; when
/// the `otel` feature is active, dropping it flushes the OTLP exporters.
pub fn init_tracing(to_stderr: bool, format: LogFormat) -> TracingGuard {
    // Default filter: our crates at info, third-party dependency chatter at
    // warn. RUST_LOG, when set, replaces this default wholesale.
    const DEFAULT_FILTER: &str = "info,h2=warn,hyper=warn,tower=warn,sqlx=warn";
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let fmt_layer = build_fmt_layer(to_stderr, format);

    #[cfg(feature = "otel")]
    {
        match otel::build_otel(otel::endpoint_from_env()) {
            Some((otel_layer, providers)) => {
                let _ = Registry::default()
                    .with(filter)
                    .with(fmt_layer)
                    .with(otel_layer)
                    .try_init();
                TracingGuard {
                    otel: Some(providers),
                }
            }
            None => {
                let _ = Registry::default().with(filter).with(fmt_layer).try_init();
                TracingGuard { otel: None }
            }
        }
    }
    #[cfg(not(feature = "otel"))]
    {
        let _ = Registry::default().with(filter).with(fmt_layer).try_init();
        TracingGuard::default()
    }
}

/// Crate-internal accessor for the globally installed OTLP meter provider.
/// Consumed by the [`crate::metrics`] facade (Task 2) so it can fetch a `Meter`
/// without threading state through call sites. `None` until OTLP export is
/// active.
#[cfg(feature = "otel")]
pub(crate) use otel::meter_provider;

#[cfg(feature = "otel")]
mod otel {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
    use opentelemetry_sdk::metrics::periodic_reader_with_async_runtime::PeriodicReader;
    use opentelemetry_sdk::metrics::SdkMeterProvider;
    use opentelemetry_sdk::runtime;
    use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use std::sync::OnceLock;
    use tracing_subscriber::Layer;

    /// Global meter provider, set when OTLP export is active so Task 2's metrics
    /// facade can fetch a `Meter` without threading state through call sites.
    static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();

    /// Returns the globally installed OTLP meter provider, if any.
    /// Consumed by the metrics facade (Task 2).
    pub(crate) fn meter_provider() -> Option<&'static SdkMeterProvider> {
        METER_PROVIDER.get()
    }

    /// Owns the SDK providers; dropping the guard flushes/shuts them down.
    pub struct OtelProviders {
        tracer: SdkTracerProvider,
        meter: SdkMeterProvider,
    }

    /// OTLP endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT`; `None` disables export.
    pub fn endpoint_from_env() -> Option<String> {
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .ok()
            .filter(|e| !e.is_empty())
    }

    /// Build the OTLP trace layer plus tracer/meter providers via the gRPC
    /// (tonic) transport. Span batching and metric export run fully async on the
    /// Tokio runtime (`runtime::Tokio`). Returns `None` (fmt-only fallback) when
    /// no endpoint is configured or when building an exporter fails — never panics.
    pub fn build_otel<S>(
        endpoint: Option<String>,
    ) -> Option<(Box<dyn Layer<S> + Send + Sync>, OtelProviders)>
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
    {
        let endpoint = endpoint?;

        let span_exporter = match SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
        {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "failed to build OTLP span exporter; tracing export disabled");
                return None;
            }
        };
        let metric_exporter = match MetricExporter::builder()
            .with_tonic()
            .with_endpoint(&endpoint)
            .build()
        {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "failed to build OTLP metric exporter; metrics export disabled");
                return None;
            }
        };

        // Async, non-blocking export driven on the Tokio runtime: the batch span
        // processor and periodic metric reader spawn their work as Tokio tasks.
        let span_processor = BatchSpanProcessor::builder(span_exporter, runtime::Tokio).build();
        let metric_reader = PeriodicReader::builder(metric_exporter, runtime::Tokio).build();

        let tracer_provider = SdkTracerProvider::builder()
            .with_span_processor(span_processor)
            .build();
        let meter_provider = SdkMeterProvider::builder()
            .with_reader(metric_reader)
            .build();

        let tracer = tracer_provider.tracer("flux");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer).boxed();

        // Publish globally for the metrics facade (best-effort; first wins).
        let _ = METER_PROVIDER.set(meter_provider.clone());

        Some((
            otel_layer,
            OtelProviders {
                tracer: tracer_provider,
                meter: meter_provider,
            },
        ))
    }

    impl Drop for OtelProviders {
        fn drop(&mut self) {
            // Best-effort flush/shutdown; ignore errors during teardown.
            let _ = self.tracer.shutdown();
            let _ = self.meter.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_defaults_to_pretty_and_switches_on_json() {
        assert_eq!(log_format_from_str(""), LogFormat::Pretty);
        assert_eq!(log_format_from_str("pretty"), LogFormat::Pretty);
        assert_eq!(log_format_from_str("anything"), LogFormat::Pretty);
        assert_eq!(log_format_from_str("json"), LogFormat::Json);
        assert_eq!(log_format_from_str("JSON"), LogFormat::Json); // case-insensitive
    }

    #[test]
    fn format_defaults_to_pretty_and_switches_on_env() {
        std::env::remove_var("FLUX_LOG_FORMAT");
        assert_eq!(log_format_from_env(), LogFormat::Pretty);

        std::env::set_var("FLUX_LOG_FORMAT", "json");
        assert_eq!(log_format_from_env(), LogFormat::Json);

        std::env::set_var("FLUX_LOG_FORMAT", "JSON");
        assert_eq!(log_format_from_env(), LogFormat::Json); // case-insensitive

        std::env::set_var("FLUX_LOG_FORMAT", "pretty");
        assert_eq!(log_format_from_env(), LogFormat::Pretty);
        std::env::remove_var("FLUX_LOG_FORMAT");
    }

    #[test]
    fn init_returns_guard() {
        // init_tracing must return a guard (type-level check) and be idempotent.
        let _g = init_tracing(true, LogFormat::Pretty);
    }
}
