use std::fmt;
use std::path::Path;

use anyhow::Result;

// ── Stop-sequence helpers ───────────────────────────────────────────────

/// Truncate at the first occurrence of any stop string (OpenAI-compatible).
/// Returns the prefix of `text` up to (but not including) the earliest stop.
pub fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for s in stops {
        let s = s.as_str();
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s) {
            earliest = Some(match earliest {
                None => pos,
                Some(e) => std::cmp::min(e, pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

/// If `new_full` (accumulated text) ends with any of `stops`, return the delta to send
/// (from `sent_len` up to but not including the stop) and the matching stop.
/// Prefers the longest matching stop when several match at the end.
pub fn truncate_at_stop<'a>(
    new_full: &str,
    sent_len: usize,
    stops: &[&'a str],
) -> Option<(String, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if new_full.ends_with(s) {
            let len = s.len();
            if best.is_none_or(|(l, _)| len > l) {
                best = Some((len, *s));
            }
        }
    }
    best.map(|(stop_len, stop)| {
        let end = new_full.len().saturating_sub(stop_len);
        let to_send = if end >= sent_len {
            new_full[sent_len..end].to_string()
        } else {
            String::new()
        };
        (to_send, stop)
    })
}

// ── Shared types ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct StreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

// ── Model type detection ────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

/// Detect model type from config.json.
pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    let config_path = format!("{}/config.json", model_path);
    let content = std::fs::read_to_string(&config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    if json.get("text_config").is_some() {
        return Ok(ModelType::Qwen35);
    }

    Ok(ModelType::Qwen3)
}

// ── Utility ─────────────────────────────────────────────────────────────

pub fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{truncate_at_first_stop, truncate_at_stop};

    #[test]
    fn test_truncate_at_first_stop() {
        let stops: Vec<String> = vec!["\n\n".into(), "END".into()];
        assert_eq!(
            truncate_at_first_stop("4\n\nand more", &stops),
            Some("4".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("helloEND", &stops),
            Some("hello".to_string())
        );
        assert_eq!(truncate_at_first_stop("hello", &stops), None);
        assert_eq!(truncate_at_first_stop("", &stops), None);
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
        );
        let stops_nl: Vec<String> = vec!["\n".into()];
        assert_eq!(
            truncate_at_first_stop("hello\nworld", &stops_nl),
            Some("hello".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("ab", &["ab".to_string()]),
            Some(String::new())
        );
    }

    #[test]
    fn test_truncate_at_stop() {
        let stops = ["\n"];
        assert_eq!(
            truncate_at_stop("hello\n", 0, &stops),
            Some(("hello".to_string(), "\n"))
        );
        assert_eq!(truncate_at_stop("hello", 0, &stops), None);
        assert_eq!(
            truncate_at_stop("ab\n", 2, &stops),
            Some((String::new(), "\n"))
        );
    }
}
