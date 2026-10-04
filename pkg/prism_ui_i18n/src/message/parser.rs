//! Recursive-descent parser for the `ICU`-style message format.
//!
//! Grammar (whitespace between tokens is insignificant):
//!
//! ```text
//! message       := (text | '#' | '{{' | '}}' | argument)*
//! argument      := '{' name (',' arg_type)? '}'
//! arg_type      := 'select' ',' select_arm+
//!                | ('plural' | 'selectordinal') ',' ('offset' ':' int)? plural_arm+
//! select_arm    := key '{' message '}'
//! plural_arm    := ('=' int | keyword) '{' message '}'
//! keyword       := 'zero' | 'one' | 'two' | 'few' | 'many' | 'other'
//! ```
//!
//! Brace escaping follows this crate's convention (`{{` → `{`, `}}` → `}`)
//! rather than `ICU`'s apostrophe quoting, matching
//! [`crate::format::interpolate`]. The `}}` escape applies only to top-level
//! text: inside a switch arm a single `}` closes the arm, so consecutive
//! structural closes (e.g. `other {x}}`) parse correctly. A
//! `select`/`plural`/`selectordinal` must declare an `other` arm; omitting it
//! is a parse error.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::plural::PluralCategory;

use super::ast::{Node, PluralArm, PluralSelector, SelectArm};

/// A message-pattern parse failure, with a human-readable reason and the
/// character offset at which it was detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageParseError {
    /// Character index (not byte index) where parsing failed.
    pub position: usize,
    /// A short description of what went wrong.
    pub reason: ParseErrorKind,
}

/// The category of a [`MessageParseError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// An opening `{` was never closed.
    UnclosedBrace,
    /// An argument had an empty name.
    EmptyName,
    /// An unknown argument type (expected `select`/`plural`/`selectordinal`).
    UnknownType(String),
    /// A `select`/`plural`/`selectordinal` lacked a required `other` arm.
    MissingOther,
    /// A plural keyword selector was not a valid `CLDR` category.
    BadKeyword(String),
    /// A malformed `offset:` clause.
    BadOffset,
    /// A malformed `=N` exact selector.
    BadExactSelector,
    /// An expected character was missing.
    Expected(char),
    /// Trailing input remained after the message (an unmatched `}`).
    UnexpectedClose,
}

impl fmt::Display for MessageParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "message parse error at {}: ", self.position)?;
        match &self.reason {
            ParseErrorKind::UnclosedBrace => f.write_str("unclosed '{'"),
            ParseErrorKind::EmptyName => f.write_str("empty argument name"),
            ParseErrorKind::UnknownType(t) => write!(f, "unknown argument type '{t}'"),
            ParseErrorKind::MissingOther => f.write_str("missing required 'other' arm"),
            ParseErrorKind::BadKeyword(k) => write!(f, "invalid plural keyword '{k}'"),
            ParseErrorKind::BadOffset => f.write_str("malformed 'offset:' value"),
            ParseErrorKind::BadExactSelector => f.write_str("malformed '=N' selector"),
            ParseErrorKind::Expected(c) => write!(f, "expected '{c}'"),
            ParseErrorKind::UnexpectedClose => f.write_str("unexpected '}'"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MessageParseError {}

/// Parse `src` into a flat node list.
pub(crate) fn parse(src: &str) -> Result<Vec<Node>, MessageParseError> {
    let chars: Vec<char> = src.chars().collect();
    let mut p = Parser { chars, pos: 0 };
    let nodes = p.parse_message(false)?;
    if p.pos < p.chars.len() {
        // Only a stray '}' can stop a top-level message early.
        return Err(p.err(ParseErrorKind::UnexpectedClose));
    }
    Ok(nodes)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn err(&self, reason: ParseErrorKind) -> MessageParseError {
        MessageParseError {
            position: self.pos,
            reason,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: char) -> Result<(), MessageParseError> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(ParseErrorKind::Expected(c)))
        }
    }

    /// Parse a (sub-)message. When `nested`, an unescaped `}` ends the message
    /// and is left unconsumed for the caller; at top level a lone `}` is a
    /// literal character.
    fn parse_message(&mut self, nested: bool) -> Result<Vec<Node>, MessageParseError> {
        let mut nodes: Vec<Node> = Vec::new();
        let mut text = String::new();

        macro_rules! flush {
            () => {
                if !text.is_empty() {
                    nodes.push(Node::Text(core::mem::take(&mut text)));
                }
            };
        }

        while let Some(c) = self.peek() {
            match c {
                '{' => {
                    if self.peek2() == Some('{') {
                        self.pos += 2;
                        text.push('{');
                    } else {
                        flush!();
                        let node = self.parse_argument()?;
                        nodes.push(node);
                    }
                }
                '}' => {
                    if nested {
                        // Inside a switch arm a single `}` closes the arm. We
                        // deliberately do not treat `}}` as an escape here: an
                        // arm that ends right before its enclosing switch's
                        // close produces `}}` (arm close + switch close), and
                        // that structural reading must win. Literal `}}`
                        // escaping therefore applies only to top-level text.
                        break;
                    } else if self.peek2() == Some('}') {
                        self.pos += 2;
                        text.push('}');
                    } else {
                        self.pos += 1;
                        text.push('}');
                    }
                }
                '#' => {
                    self.pos += 1;
                    flush!();
                    nodes.push(Node::Pound);
                }
                _ => {
                    self.pos += 1;
                    text.push(c);
                }
            }
        }

        flush!();
        Ok(nodes)
    }

    /// Parse an argument, assuming the opening `{` is the current character.
    fn parse_argument(&mut self) -> Result<Node, MessageParseError> {
        self.expect('{')?;
        self.skip_ws();
        let name = self.read_identifier();
        if name.is_empty() {
            return Err(self.err(ParseErrorKind::EmptyName));
        }
        self.skip_ws();
        match self.peek() {
            Some('}') => {
                self.pos += 1;
                Ok(Node::Arg(name))
            }
            Some(',') => {
                self.pos += 1;
                self.skip_ws();
                let ty = self.read_identifier();
                self.skip_ws();
                self.expect(',')?;
                match ty.as_str() {
                    "select" => self.parse_select(name),
                    "plural" => self.parse_plural(name, false),
                    "selectordinal" => self.parse_plural(name, true),
                    _ => Err(self.err(ParseErrorKind::UnknownType(ty))),
                }
            }
            Some(_) => Err(self.err(ParseErrorKind::Expected('}'))),
            None => Err(self.err(ParseErrorKind::UnclosedBrace)),
        }
    }

    fn parse_select(&mut self, name: String) -> Result<Node, MessageParseError> {
        let mut arms: Vec<SelectArm> = Vec::new();
        let mut has_other = false;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(self.err(ParseErrorKind::UnclosedBrace)),
                _ => {}
            }
            let key = self.read_identifier();
            if key.is_empty() {
                return Err(self.err(ParseErrorKind::Expected('}')));
            }
            if key == "other" {
                has_other = true;
            }
            self.skip_ws();
            self.expect('{')?;
            let body = self.parse_message(true)?;
            self.expect('}')?;
            arms.push(SelectArm { key, body });
        }
        if !has_other {
            return Err(self.err(ParseErrorKind::MissingOther));
        }
        Ok(Node::Select { name, arms })
    }

    fn parse_plural(&mut self, name: String, ordinal: bool) -> Result<Node, MessageParseError> {
        self.skip_ws();
        let offset = self.parse_offset()?;
        let mut arms: Vec<PluralArm> = Vec::new();
        let mut has_other = false;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                None => return Err(self.err(ParseErrorKind::UnclosedBrace)),
                _ => {}
            }
            let selector = self.parse_plural_selector()?;
            if selector == PluralSelector::Category(PluralCategory::Other) {
                has_other = true;
            }
            self.skip_ws();
            self.expect('{')?;
            let body = self.parse_message(true)?;
            self.expect('}')?;
            arms.push(PluralArm { selector, body });
        }
        if !has_other {
            return Err(self.err(ParseErrorKind::MissingOther));
        }
        Ok(Node::Plural {
            name,
            ordinal,
            offset,
            arms,
        })
    }

    /// Parse an optional `offset:<int>` clause (returns 0 when absent).
    fn parse_offset(&mut self) -> Result<i64, MessageParseError> {
        let save = self.pos;
        let word = self.read_identifier();
        if word != "offset" {
            self.pos = save;
            return Ok(0);
        }
        self.skip_ws();
        if self.expect(':').is_err() {
            return Err(self.err(ParseErrorKind::BadOffset));
        }
        self.skip_ws();
        self.read_integer().ok_or_else(|| self.err(ParseErrorKind::BadOffset))
    }

    fn parse_plural_selector(&mut self) -> Result<PluralSelector, MessageParseError> {
        if self.peek() == Some('=') {
            self.pos += 1;
            let n = self
                .read_integer()
                .ok_or_else(|| self.err(ParseErrorKind::BadExactSelector))?;
            return Ok(PluralSelector::Exact(n));
        }
        let kw = self.read_identifier();
        let cat = match kw.as_str() {
            "zero" => PluralCategory::Zero,
            "one" => PluralCategory::One,
            "two" => PluralCategory::Two,
            "few" => PluralCategory::Few,
            "many" => PluralCategory::Many,
            "other" => PluralCategory::Other,
            _ => return Err(self.err(ParseErrorKind::BadKeyword(kw))),
        };
        Ok(PluralSelector::Category(cat))
    }

    /// Read an identifier: ASCII alphanumerics plus `_`, `.`, and `-`.
    fn read_identifier(&mut self) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
                s.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        s
    }

    /// Read an optionally-signed base-10 integer.
    fn read_integer(&mut self) -> Option<i64> {
        let start = self.pos;
        let mut negative = false;
        if self.peek() == Some('-') {
            negative = true;
            self.pos += 1;
        }
        let mut value: i64 = 0;
        let mut digits = 0;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                value = value.saturating_mul(10).saturating_add(i64::from(d));
                self.pos += 1;
                digits += 1;
            } else {
                break;
            }
        }
        if digits == 0 {
            self.pos = start;
            return None;
        }
        Some(if negative { -value } else { value })
    }
}
