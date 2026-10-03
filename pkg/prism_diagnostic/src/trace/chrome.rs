//! Chrome Trace Event Format (JSON) export.
//!
//! Serializes completed spans from every registered thread into the
//! `chrome://tracing` / Perfetto `traceEvents` array. Each span becomes a
//! complete event (`"ph":"X"`) carrying `ts`/`dur` in microseconds, a `pid`,
//! a `tid`, the span `name`, an optional `cat`, and optional `args`. The JSON
//! is hand-rolled with proper string escaping; there is no `serde` dependency.

extern crate alloc;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Write as _;

use super::ring::{registered_threads, SpanRecord, ThreadTrace};

/// Process id reported in exported events. The Prism model is single-process;
/// threads are distinguished by their `tid`.
const PROCESS_ID: u64 = 1;

/// Serialize every registered thread's retained spans to a Chrome Trace JSON
/// string.
pub fn export_string() -> String {
    export_threads(&registered_threads())
}

fn export_threads(threads: &[Arc<ThreadTrace>]) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("{\"traceEvents\":[");
    let mut first = true;

    // Thread-name metadata events so UIs label each track.
    for trace in threads {
        if !first {
            out.push(',');
        }
        first = false;
        write_thread_name_event(&mut out, trace.thread_id, &trace.thread_name);
    }

    // Complete ("X") events, one per retained span.
    for trace in threads {
        let spans = match trace.buffer.lock() {
            Ok(buf) => buf.snapshot(),
            Err(_) => Vec::new(),
        };
        for span in &spans {
            if !first {
                out.push(',');
            }
            first = false;
            write_complete_event(&mut out, trace.thread_id, span);
        }
    }

    out.push_str("],\"displayTimeUnit\":\"ms\"}");
    out
}

fn write_complete_event(out: &mut String, tid: u64, span: &SpanRecord) {
    out.push_str("{\"name\":");
    write_json_string(out, &span.name);
    if let Some(cat) = &span.category {
        out.push_str(",\"cat\":");
        write_json_string(out, cat);
    }
    out.push_str(",\"ph\":\"X\",\"pid\":");
    let _ = write!(out, "{PROCESS_ID}");
    out.push_str(",\"tid\":");
    let _ = write!(out, "{tid}");
    out.push_str(",\"ts\":");
    write_micros(out, span.start_nanos);
    out.push_str(",\"dur\":");
    write_micros(out, span.duration_nanos);
    if !span.args.is_empty() {
        out.push_str(",\"args\":{");
        for (i, (key, value)) in span.args.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json_string(out, key);
            out.push(':');
            write_json_string(out, value);
        }
        out.push('}');
    }
    out.push('}');
}

fn write_thread_name_event(out: &mut String, tid: u64, name: &str) {
    out.push_str("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":");
    let _ = write!(out, "{PROCESS_ID}");
    out.push_str(",\"tid\":");
    let _ = write!(out, "{tid}");
    out.push_str(",\"args\":{\"name\":");
    write_json_string(out, name);
    out.push_str("}}");
}

/// Write `nanos` as a microseconds JSON number with 3 fractional digits (exact
/// integer division, so no floating-point rounding error creeps in).
fn write_micros(out: &mut String, nanos: u64) {
    let micros = nanos / 1000;
    let frac = nanos % 1000;
    let _ = write!(out, "{micros}.{frac:03}");
}

/// Append `s` as a JSON string literal with the mandatory escapes.
fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Write the Chrome Trace JSON for all registered threads to `path`.
pub fn export_to_file(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    use std::io::Write as _;
    let json = export_string();
    let mut file = std::fs::File::create(path)?;
    file.write_all(json.as_bytes())
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use crate::span::Scope;
    use crate::trace::ring;

    /// Minimal structural check: brackets/braces balance outside strings and
    /// every string literal is closed (enough to catch a malformed export).
    fn is_balanced_json(s: &str) -> bool {
        let mut depth: i64 = 0;
        let mut in_str = false;
        let mut escaped = false;
        for c in s.chars() {
            if in_str {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return false;
            }
        }
        depth == 0 && !in_str
    }

    #[test]
    fn balance_helper_rejects_broken_json() {
        assert!(is_balanced_json("{\"a\":[1,2]}"));
        assert!(!is_balanced_json("{\"a\":[1,2]"));
        assert!(is_balanced_json("{\"a\":\"]}\"}"));
    }

    #[test]
    fn chrome_export_is_wellformed_and_contains_spans() {
        ring::clear_current_thread();
        {
            let _s = Scope::new("alpha_scope")
                .with_category("render")
                .with_arg("frame", "7");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let json = super::export_string();
        assert!(json.contains("\"traceEvents\""));
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("alpha_scope"));
        assert!(json.contains("\"cat\":\"render\""));
        assert!(json.contains("\"ts\":"));
        assert!(json.contains("\"dur\":"));
        assert!(json.contains("\"args\":"));
        assert!(json.contains("\"frame\""));
        assert!(is_balanced_json(&json), "unbalanced JSON: {json}");
    }

    #[test]
    fn json_strings_are_escaped() {
        ring::clear_current_thread();
        {
            let _s = Scope::new("weird\"name\twith\\escapes");
        }
        let json = super::export_string();
        assert!(json.contains("weird\\\"name\\twith\\\\escapes"));
        assert!(is_balanced_json(&json), "unbalanced JSON: {json}");
    }

    #[test]
    fn multi_thread_spans_all_appear() {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let name = format!("mt_scope_{i}");
                    let _s = Scope::new(name);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let json = super::export_string();
        for i in 0..4 {
            let needle = format!("mt_scope_{i}");
            assert!(json.contains(&needle), "missing {needle}");
        }
        assert!(is_balanced_json(&json), "unbalanced JSON");
    }

    #[test]
    fn export_to_file_writes_json() {
        ring::clear_current_thread();
        {
            let _s = Scope::new("file_scope");
        }
        let mut path = std::env::temp_dir();
        path.push(format!("prism_diag_trace_{}.json", std::process::id()));
        super::export_to_file(&path).unwrap();
        let contents = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
        assert!(contents.contains("file_scope"));
        assert!(is_balanced_json(&contents));
        let _ = std::fs::remove_file(&path);
    }
}
