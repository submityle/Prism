//! Human-readable event formatting.

extern crate alloc;

use alloc::format;
use alloc::string::String;

use crate::model::{Event, FieldValue};

/// Render an event to a single log line, e.g.
/// `[INFO ] target: message key=value`.
pub fn format_line(event: &Event) -> String {
    let mut line = format!(
        "[{:<5}] {}: {}",
        event.level.label(),
        event.target,
        event.message
    );
    for i in 0..event.fields.len() {
        if let Some(field) = event.fields.get(i) {
            line.push(' ');
            line.push_str(field.key);
            line.push('=');
            match &field.value {
                FieldValue::Str(s) => line.push_str(s),
                FieldValue::I64(v) => line.push_str(&format!("{v}")),
                FieldValue::U64(v) => line.push_str(&format!("{v}")),
                FieldValue::F64(v) => line.push_str(&format!("{v}")),
                FieldValue::Bool(v) => line.push_str(&format!("{v}")),
            }
        }
    }
    line
}
