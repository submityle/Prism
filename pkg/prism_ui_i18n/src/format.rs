//! Argument interpolation engine.
//!
//! This module parses message templates containing `{name}` placeholders and
//! substitutes values supplied through an [`Args`] map. Braces are escaped by
//! doubling them (`{{` renders a literal `{`, `}}` a literal `}`), matching the
//! common `ICU` `MessageFormat` convention.
//!
//! All integer rendering is performed with pure integer arithmetic — no
//! floating-point operations are used anywhere in this crate.

use alloc::string::String;
use alloc::vec::Vec;

/// A value that can be interpolated into a message template.
///
/// Kept intentionally small: a borrowed/owned string or a signed integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// A text value, rendered verbatim.
    Str(String),
    /// A signed integer value, rendered in base-10 without floating point.
    Num(i64),
}

impl Value {
    /// Append the rendered form of this value to `buf`.
    pub(crate) fn render_into(&self, buf: &mut String) {
        match self {
            Value::Str(s) => buf.push_str(s),
            Value::Num(n) => push_i64(buf, *n),
        }
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Str(String::from(value))
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::Str(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Num(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Num(value as i64)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Value::Num(value as i64)
    }
}

/// An ordered collection of named arguments for interpolation.
///
/// Lookups are by exact key match. The builder-style [`Args::with`] enables
/// ergonomic inline construction, while [`Args::set`] mutates in place.
#[derive(Clone, Debug, Default)]
pub struct Args {
    entries: Vec<(String, Value)>,
}

impl Args {
    /// Create an empty argument map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace `key` with `value`, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.set(key, value);
        self
    }

    /// Insert or replace `key` with `value` in place.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<Value>) -> &mut Self {
        let key = key.into();
        let value = value.into();
        for entry in &mut self.entries {
            if entry.0 == key {
                entry.1 = value;
                return self;
            }
        }
        self.entries.push((key, value));
        self
    }

    /// Look up the value bound to `key`, if any.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// Render `template`, substituting `{name}` placeholders from `args`.
///
/// Escaping rules:
/// * `{{` renders a literal `{`.
/// * `}}` renders a literal `}`.
///
/// An unknown placeholder (no matching key in `args`) is left verbatim,
/// including its surrounding braces, so authors can spot typos. A `{` with no
/// closing `}` is emitted literally together with the characters that followed.
pub fn interpolate(template: &str, args: &Args) -> String {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    out.push('{');
                } else {
                    let mut name = String::new();
                    let mut closed = false;
                    while let Some(&nc) = chars.peek() {
                        if nc == '}' {
                            chars.next();
                            closed = true;
                            break;
                        }
                        name.push(nc);
                        chars.next();
                    }
                    if closed {
                        match args.get(&name) {
                            Some(value) => value.render_into(&mut out),
                            None => {
                                out.push('{');
                                out.push_str(&name);
                                out.push('}');
                            }
                        }
                    } else {
                        out.push('{');
                        out.push_str(&name);
                    }
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                }
                out.push('}');
            }
            other => out.push(other),
        }
    }
    out
}

/// Append the base-10 representation of `n` to `buf` using only integer math.
pub(crate) fn push_i64(buf: &mut String, n: i64) {
    if n == 0 {
        buf.push('0');
        return;
    }
    let negative = n < 0;
    let mut magnitude = n.unsigned_abs();
    let mut digits = [0u8; 20];
    let mut len = 0;
    while magnitude > 0 {
        digits[len] = b'0' + (magnitude % 10) as u8;
        magnitude /= 10;
        len += 1;
    }
    if negative {
        buf.push('-');
    }
    while len > 0 {
        len -= 1;
        buf.push(digits[len] as char);
    }
}
