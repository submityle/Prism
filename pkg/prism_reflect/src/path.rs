//! Parsed access paths into reflected values ([`ParsedPath`]).
//!
//! A path is a sequence of accessors that navigate from a root `&dyn Reflect`
//! into a nested field, element, or entry. The textual grammar mirrors Rust's
//! own access syntax:
//!
//! - `.name` or a leading bare `name` — a named struct field (or named enum
//!   variant field).
//! - `#n` — a tuple-struct field or tuple-enum-variant field by position.
//! - `[n]` — a list or array element by index.
//! - `["key"]` / `['key']` — a map entry by string key.
//!
//! For example `.transform.translation[0]`, `#0.name`, and `["player"].health`
//! are all valid. Resolve a parsed path with [`reflect_path`] (shared) or
//! [`reflect_path_mut`] (mutable).

use crate::reflect::Reflect;
use crate::{ReflectMut, ReflectRef};
use core::fmt;
use std::string::String;
use std::vec::Vec;

/// A single navigation step into a reflected value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// A named field, matched against [`Struct`](crate::Struct) fields or named [`Enum`](crate::Enum)
    /// variant fields.
    Field(String),
    /// A positional field of a [`TupleStruct`](crate::TupleStruct) or tuple [`Enum`](crate::Enum) variant.
    TupleIndex(usize),
    /// An element index into a [`List`](crate::List) or [`Array`](crate::Array).
    ListIndex(usize),
    /// A string key into a [`Map`](crate::Map).
    Key(String),
}

/// A parsed sequence of [`Access`] steps navigating into a reflected value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedPath {
    segments: Vec<Access>,
}

/// An error produced while parsing a textual access path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePathError {
    message: String,
    offset: usize,
}

impl ParsePathError {
    /// The human-readable reason the path failed to parse.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The byte offset within the input where parsing failed.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl fmt::Display for ParsePathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid access path at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParsePathError {}

/// Whether `c` may start a bare identifier segment.
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Whether `c` may continue a bare identifier segment.
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

impl ParsedPath {
    /// Parse `input` into a [`ParsedPath`].
    ///
    /// # Errors
    /// Returns a [`ParsePathError`] when the input contains an unexpected
    /// character, an empty or malformed segment, or an unterminated bracket.
    pub fn parse(input: &str) -> Result<Self, ParsePathError> {
        let bytes = input.as_bytes();
        let mut segments = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as char;
            match c {
                '.' => {
                    i += 1;
                    let start = i;
                    while i < bytes.len() && is_ident_continue(bytes[i] as char) {
                        i += 1;
                    }
                    if i == start {
                        return Err(ParsePathError {
                            message: String::from("expected a field name after `.`"),
                            offset: start,
                        });
                    }
                    segments.push(Access::Field(input[start..i].to_string()));
                }
                '#' => {
                    i += 1;
                    let start = i;
                    while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                    if i == start {
                        return Err(ParsePathError {
                            message: String::from("expected a tuple index after `#`"),
                            offset: start,
                        });
                    }
                    let index = parse_index(&input[start..i], start)?;
                    segments.push(Access::TupleIndex(index));
                }
                '[' => {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                        let quote = bytes[i];
                        i += 1;
                        let start = i;
                        while i < bytes.len() && bytes[i] != quote {
                            i += 1;
                        }
                        if i >= bytes.len() {
                            return Err(ParsePathError {
                                message: String::from("unterminated quoted map key"),
                                offset: start,
                            });
                        }
                        let key = input[start..i].to_string();
                        i += 1; // consume the closing quote
                        if i >= bytes.len() || bytes[i] != b']' {
                            return Err(ParsePathError {
                                message: String::from("expected `]` after quoted map key"),
                                offset: i,
                            });
                        }
                        i += 1; // consume `]`
                        segments.push(Access::Key(key));
                    } else {
                        let start = i;
                        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                            i += 1;
                        }
                        if i == start {
                            return Err(ParsePathError {
                                message: String::from("expected an index or quoted key after `[`"),
                                offset: start,
                            });
                        }
                        let index = parse_index(&input[start..i], start)?;
                        if i >= bytes.len() || bytes[i] != b']' {
                            return Err(ParsePathError {
                                message: String::from("expected `]` after list index"),
                                offset: i,
                            });
                        }
                        i += 1; // consume `]`
                        segments.push(Access::ListIndex(index));
                    }
                }
                c if is_ident_start(c) => {
                    let start = i;
                    while i < bytes.len() && is_ident_continue(bytes[i] as char) {
                        i += 1;
                    }
                    segments.push(Access::Field(input[start..i].to_string()));
                }
                _ => {
                    return Err(ParsePathError {
                        message: String::from("unexpected character in access path"),
                        offset: i,
                    });
                }
            }
        }
        Ok(Self { segments })
    }

    /// The ordered accessor steps this path will follow.
    #[must_use]
    pub fn segments(&self) -> &[Access] {
        &self.segments
    }

    /// Whether this path has no segments (resolves to the root value).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

impl core::str::FromStr for ParsedPath {
    type Err = ParsePathError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Parse a decimal index, mapping overflow to a [`ParsePathError`].
fn parse_index(digits: &str, offset: usize) -> Result<usize, ParsePathError> {
    digits.parse::<usize>().map_err(|_| ParsePathError {
        message: String::from("index does not fit in `usize`"),
        offset,
    })
}

/// Follow a single [`Access`] into a shared reflected value.
fn access<'a>(current: &'a dyn Reflect, step: &Access) -> Option<&'a dyn Reflect> {
    match (current.reflect_ref(), step) {
        (ReflectRef::Struct(value), Access::Field(name)) => value.field(name),
        (ReflectRef::Enum(value), Access::Field(name)) => value.field(name),
        (ReflectRef::TupleStruct(value), Access::TupleIndex(index)) => value.field(*index),
        (ReflectRef::Enum(value), Access::TupleIndex(index)) => value.field_at(*index),
        (ReflectRef::List(value), Access::ListIndex(index)) => value.get(*index),
        (ReflectRef::Array(value), Access::ListIndex(index)) => value.get(*index),
        (ReflectRef::Map(value), Access::Key(key)) => {
            let key = key.clone();
            value.get(&key as &dyn Reflect)
        }
        _ => None,
    }
}

/// Follow a single [`Access`] into a mutable reflected value.
fn access_mut<'a>(current: &'a mut dyn Reflect, step: &Access) -> Option<&'a mut dyn Reflect> {
    match (current.reflect_mut(), step) {
        (ReflectMut::Struct(value), Access::Field(name)) => value.field_mut(name),
        (ReflectMut::Enum(value), Access::Field(name)) => value.field_mut(name),
        (ReflectMut::TupleStruct(value), Access::TupleIndex(index)) => value.field_mut(*index),
        (ReflectMut::Enum(value), Access::TupleIndex(index)) => value.field_at_mut(*index),
        (ReflectMut::List(value), Access::ListIndex(index)) => value.get_mut(*index),
        (ReflectMut::Array(value), Access::ListIndex(index)) => value.get_mut(*index),
        (ReflectMut::Map(value), Access::Key(key)) => {
            let key = key.clone();
            value.get_mut(&key as &dyn Reflect)
        }
        _ => None,
    }
}

/// Resolve `path` against `root`, returning a shared reference to the targeted
/// value or `None` if any step does not match the value's shape.
#[must_use]
pub fn reflect_path<'a>(root: &'a dyn Reflect, path: &ParsedPath) -> Option<&'a dyn Reflect> {
    let mut current = root;
    for step in &path.segments {
        current = access(current, step)?;
    }
    Some(current)
}

/// Resolve `path` against `root`, returning a mutable reference to the targeted
/// value or `None` if any step does not match the value's shape.
#[must_use]
pub fn reflect_path_mut<'a>(
    root: &'a mut dyn Reflect,
    path: &ParsedPath,
) -> Option<&'a mut dyn Reflect> {
    let mut current = root;
    for step in &path.segments {
        current = access_mut(current, step)?;
    }
    Some(current)
}
