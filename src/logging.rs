//! Unified logging configuration for Rust LLM.
//!
//! Provides consistent log initialization across server and tests.

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use std::sync::Once;

static INIT: Once = Once::new();

/// Output destination for logs.
#[derive(Debug, Clone, Copy, Default)]
pub enum LogOutput {
    #[default]
    Stderr,
    Stdout,
}

/// Configuration for logging initialization.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level filter (e.g., "info", "debug", "info,rust_llm=debug").
    /// Falls back to RUST_LOG environment variable if set.
    pub level: String,
    /// Output destination.
    pub output: LogOutput,
    /// Enable colored output (info=green, warn=yellow, error=red).
    pub colored: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            output: LogOutput::Stderr,
            colored: true,
        }
    }
}

impl LoggingConfig {
    /// Create a new config with the given log level.
    pub fn new(level: impl Into<String>) -> Self {
        Self {
            level: level.into(),
            ..Default::default()
        }
    }

    /// Set output to stdout.
    pub fn stdout(mut self) -> Self {
        self.output = LogOutput::Stdout;
        self
    }

    /// Set output to stderr.
    pub fn stderr(mut self) -> Self {
        self.output = LogOutput::Stderr;
        self
    }

    /// Enable colored output.
    pub fn colored(mut self) -> Self {
        self.colored = true;
        self
    }

    /// Disable colored output.
    pub fn no_color(mut self) -> Self {
        self.colored = false;
        self
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
pub fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig {
            level,
            output,
            colored,
        } = config;

        let filter_str =
            std::env::var("RUST_LOG").unwrap_or_else(|_| apply_default_module_levels(level));

        let mut builder = logforth::starter_log::builder();

        // Parse filter from string using EnvFilterBuilder
        let filter = logforth::filter::env_filter::EnvFilterBuilder::from_env_or("RUST_LOG", filter_str).build();

        match (output, colored) {
            (LogOutput::Stdout, true) => {
                let layout = TextLayout::default()
                    .info_color(Green)
                    .warn_color(Yellow)
                    .error_color(Red);
                builder = builder.dispatch(|d| {
                    d.filter(filter)
                        .diagnostic(ThreadLocalDiagnostic::default())
                        .append(logforth::append::Stdout::default().with_layout(layout))
                });
            }
            (LogOutput::Stdout, false) => {
                builder = builder.dispatch(|d| {
                    d.filter(filter)
                        .diagnostic(ThreadLocalDiagnostic::default())
                        .append(logforth::append::Stdout::default())
                });
            }
            (LogOutput::Stderr, true) => {
                let layout = TextLayout::default()
                    .info_color(Green)
                    .warn_color(Yellow)
                    .error_color(Red);
                builder = builder.dispatch(|d| {
                    d.filter(filter)
                        .diagnostic(ThreadLocalDiagnostic::default())
                        .append(logforth::append::Stderr::default().with_layout(layout))
                });
            }
            (LogOutput::Stderr, false) => {
                builder = builder.dispatch(|d| {
                    d.filter(filter)
                        .diagnostic(ThreadLocalDiagnostic::default())
                        .append(logforth::append::Stderr::default())
                });
            }
        }

        builder.apply();
    });
}

/// Initialize logging to stderr without colors.
///
/// Convenience function for tests use case.
pub fn init_stderr(level: &str) {
    init(LoggingConfig::new(level).stderr().no_color());
}

/// Initialize logging with default settings (stderr, colored, "info" level).
pub fn init_default() {
    init(LoggingConfig::default());
}
