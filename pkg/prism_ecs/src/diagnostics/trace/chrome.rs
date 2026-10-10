//! Chrome Trace Event Format export (design §16.6).
//!
//! [`to_chrome_json`] serialises a [`TraceBuffer`] into the JSON object that
//! `chrome://tracing`, Perfetto, and the `catapult` tooling consume:
//!
//! ```json
//! { "displayTimeUnit": "ns", "traceEvents": [
//!   { "name": "physics", "cat": "system", "ph": "X",
//!     "ts": 12.345, "dur": 8.000, "pid": 1, "tid": 0,
//!     "args": { "dirty_chunks": 4 } }
//! ] }
//! ```
//!
//! The exporter is self-contained: no `serde`, no float formatting machinery.
//! Timestamps are stored in nanoseconds but the format expects **microseconds**,
//! so each `ts` / `dur` is emitted as fixed-point `micros.nnn` (three fractional
//! digits = the original nanosecond precision), produced by integer arithmetic
//! so output is byte-for-byte deterministic across platforms.

use alloc::string::String;

use super::buffer::TraceBuffer;
use super::event::{EventPhase, TraceArgValue, TraceEvent};

/// The fixed process id reported for every event. The `no_std` core has no OS
/// process concept; a single synthetic process keeps all tracks in one view.
const SYNTHETIC_PID: u32 = 1;

/// Serialise `buffer` into a Chrome Trace Event Format JSON string.
///
/// Events are emitted in retained record order. The result is always a valid
/// JSON object, including `{"displayTimeUnit":"ns","traceEvents":[]}` for an
/// empty buffer.
#[must_use]
pub fn to_chrome_json(buffer: &TraceBuffer) -> String {
    let mut out = String::with_capacity(64 + buffer.len() * 96);
    out.push_str("{\"displayTimeUnit\":\"ns\",\"traceEvents\":[");
    for (i, ev) in buffer.events().iter().enumerate() {
        if i != 0 {
            out.push(',');
        }
        write_event(&mut out, ev);
    }
    out.push_str("]}");
    out
}

fn write_event(out: &mut String, ev: &TraceEvent) {
    out.push_str("{\"name\":");
    write_json_string(out, ev.name());
    out.push_str(",\"cat\":");
    write_json_string(out, ev.category());
    out.push_str(",\"ph\":");
    write_json_string(out, ev.phase().chrome_ph());
    out.push_str(",\"ts\":");
    write_micros(out, ev.timestamp_ns());
    if ev.phase().has_duration() {
        out.push_str(",\"dur\":");
        write_micros(out, ev.duration_ns());
    }
    if matches!(ev.phase(), EventPhase::Instant) {
        // Thread-scoped instant marker, the usual devtools default.
        out.push_str(",\"s\":\"t\"");
    }
    out.push_str(",\"pid\":");
    push_u64(out, u64::from(SYNTHETIC_PID));
    out.push_str(",\"tid\":");
    push_u64(out, u64::from(ev.track().get()));
    if !ev.args().is_empty() {
        out.push_str(",\"args\":{");
        for (i, arg) in ev.args().iter().enumerate() {
            if i != 0 {
                out.push(',');
            }
            write_json_string(out, arg.key());
            out.push(':');
            write_arg_value(out, arg.value());
        }
        out.push('}');
    }
    out.push('}');
}

fn write_arg_value(out: &mut String, value: &TraceArgValue) {
    match value {
        TraceArgValue::Int(v) => push_i64(out, *v),
        TraceArgValue::Uint(v) => push_u64(out, *v),
        TraceArgValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        TraceArgValue::Str(s) => write_json_string(out, s),
    }
}

/// Emit `ns` nanoseconds as microseconds with three fractional digits, e.g.
/// `12345` ns → `12.345`. Pure integer arithmetic for deterministic output.
fn write_micros(out: &mut String, ns: u64) {
    let micros = ns / 1000;
    let frac = ns % 1000;
    push_u64(out, micros);
    out.push('.');
    // Zero-pad the fractional nanosecond remainder to three digits.
    out.push((b'0' + (frac / 100) as u8) as char);
    out.push((b'0' + ((frac / 10) % 10) as u8) as char);
    out.push((b'0' + (frac % 10) as u8) as char);
}

/// Append `value` as a JSON string literal with full escaping of the characters
/// the JSON grammar requires (`"`, `\`, and control codes `< 0x20`).
fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => push_unicode_escape(out, c as u32),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append a `\u00XX` escape for a control code point.
fn push_unicode_escape(out: &mut String, code: u32) {
    out.push_str("\\u");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(HEX[((code >> 12) & 0xF) as usize] as char);
    out.push(HEX[((code >> 8) & 0xF) as usize] as char);
    out.push(HEX[((code >> 4) & 0xF) as usize] as char);
    out.push(HEX[(code & 0xF) as usize] as char);
}

/// Append the decimal digits of `value` without `std` formatting.
fn push_u64(out: &mut String, value: u64) {
    if value == 0 {
        out.push('0');
        return;
    }
    // u64 max is 20 decimal digits.
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut v = value;
    while v != 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    for &byte in &buf[i..] {
        out.push(byte as char);
    }
}

/// Append the decimal digits of a signed `value`, with a leading `-` when
/// negative. `i64::MIN` is handled without overflow by negating as `u64`.
fn push_i64(out: &mut String, value: i64) {
    if value < 0 {
        out.push('-');
        push_u64(out, (value as i128).unsigned_abs() as u64);
    } else {
        push_u64(out, value as u64);
    }
}
