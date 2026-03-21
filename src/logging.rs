//! Unified logging configuration for Rust LLM.
//!
//! Provides consistent log initialization across server and tests.

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug, Clone)]
struct LoggingConfig {
    /// Log level filter (e.g., "info", "debug", "info,pegainfer=debug").
    /// Falls back to RUST_LOG environment variable if set.
    level: String,
    /// Enable colored output (info=green, warn=yellow, error=red).
    colored: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            colored: true,
        }
    }
}

/// Default noisy modules to reduce log spam.
const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 5] = [
    ("h2", "warn"),
    ("hyper", "warn"),
    ("hyper_util", "warn"),
    ("axum", "warn"),
    ("tower", "warn"),
];

fn apply_default_module_levels(mut filter: String) -> String {
    for (module, level) in DEFAULT_NOISY_MODULE_LEVELS {
        let module_pattern = format!("{module}=");
        if !filter.contains(&module_pattern) {
            if !filter.is_empty() {
                filter.push(',');
            }
            filter.push_str(module);
            filter.push('=');
            filter.push_str(level);
        }
    }
    filter
}

/// Initialize logging with the given configuration.
///
/// This function is idempotent - subsequent calls after the first are no-ops.
/// The RUST_LOG environment variable takes precedence over the configured level.
/// When RUST_LOG is not set, noisy dependency modules default to warn to
/// keep debug output focused on application components.
fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig { level, colored } = config;

        let filter_str =
            std::env::var("RUST_LOG").unwrap_or_else(|_| apply_default_module_levels(level));

        let mut builder = logforth::starter_log::builder();

        // Parse filter from string using EnvFilterBuilder
        let filter =
            logforth::filter::env_filter::EnvFilterBuilder::from_env_or("RUST_LOG", filter_str)
                .build();

        if colored {
            let layout = TextLayout::default()
                .info_color(Green)
                .warn_color(Yellow)
                .error_color(Red);
            builder = builder.dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default().with_layout(layout))
            });
        } else {
            builder = builder.dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default())
            });
        }

        builder.apply();
    });
}

/// Initialize logging to stderr without colors.
///
/// Convenience function for tests use case.
pub fn init_stderr(level: &str) {
    init(LoggingConfig {
        level: level.to_string(),
        colored: false,
    });
}

/// Initialize logging with default settings (stderr, colored, "info" level).
pub fn init_default() {
    init(LoggingConfig::default());
}
