//! An `ICU`-style `MessageFormat` subset for inline `select` / `plural` /
//! `selectordinal` arguments.
//!
//! This module compiles a message pattern once into a [`MessagePattern`] and
//! renders it repeatedly against a set of [`Args`]. It extends the simple
//! `{name}` substitution of [`crate::format::interpolate`] with the three
//! grammatical switches `ICU` `MessageFormat` is best known for:
//!
//! ```text
//! {gender, select, male {he} female {she} other {they}}
//! {count, plural, one {# file} other {# files}}
//! {place, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}
//! ```
//!
//! # Supported syntax
//!
//! * `{name}` — a bare substitution (identical to [`crate::format::interpolate`]).
//! * `{name, select, key {..} other {..}}` — a string switch.
//! * `{name, plural, offset:N? (=N | keyword) {..} ... other {..}}` — a cardinal
//!   numeric switch.
//! * `{name, selectordinal, ...}` — an ordinal numeric switch.
//! * `#` — inside a `plural`/`selectordinal` arm renders the value minus the
//!   `offset`; elsewhere it is a literal `#`.
//! * `{{` / `}}` — literal `{` / `}` in top-level text (this crate's escaping
//!   convention, matching [`crate::format::interpolate`]; `ICU`'s apostrophe
//!   quoting is *not* used). Inside a `select`/`plural` arm a single `}` closes
//!   the arm, so `}}` there reads as two structural closes rather than an
//!   escaped brace; a literal `}` inside an arm body is therefore not
//!   expressible (a literal `{` via `{{` still is).
//!
//! # Deliberate omissions
//!
//! Number/date/time *skeletons* (e.g. `{n, number, percent}`) are **not**
//! supported: faithful number formatting needs a per-locale symbol table this
//! crate does not yet carry, and a half-implementation would silently diverge
//! from `ICU`. Numbers render as plain base-10 integers.
//!
//! # Selection semantics
//!
//! * Exact `=N` selectors match the argument's *original* value; keyword
//!   (`one`/`few`/...) selection uses the value *after* subtracting `offset`.
//! * `plural` uses the locale's registered cardinal [`PluralRules`];
//!   `selectordinal` resolves the locale's ordinal rules.
//! * A missing argument renders a `select`/`plural` `other` arm, and a bare
//!   `{name}` is left verbatim as `{name}` (matching the simple interpolator).

mod ast;
mod eval;
mod parser;

#[cfg(all(test, feature = "std"))]
mod tests;

use alloc::string::String;
use alloc::vec::Vec;

use crate::format::Args;
use crate::plural::PluralRules;

use ast::Node;
use eval::{eval, EvalCtx};

pub use parser::{MessageParseError, ParseErrorKind};

/// A compiled `ICU`-style message pattern.
///
/// Parse a pattern once with [`MessagePattern::parse`], then render it many
/// times with [`MessagePattern::format`]. Parsing validates structure up front
/// (balanced braces, a required `other` arm, well-formed selectors), so
/// rendering never fails.
#[derive(Clone, Debug)]
pub struct MessagePattern {
    nodes: Vec<Node>,
}

impl MessagePattern {
    /// Compile `src` into a reusable pattern.
    ///
    /// # Errors
    ///
    /// Returns a [`MessageParseError`] describing the first structural problem
    /// (unclosed brace, unknown argument type, missing `other` arm, malformed
    /// selector, or stray `}`).
    pub fn parse(src: &str) -> Result<Self, MessageParseError> {
        Ok(MessagePattern {
            nodes: parser::parse(src)?,
        })
    }

    /// Render the pattern against `args`.
    ///
    /// `cardinal` is the active locale's registered cardinal rules (used by
    /// `plural`), and `locale` is its identifier (used to resolve
    /// `selectordinal` rules). `#` outside any plural renders as a literal `#`.
    pub fn format(&self, args: &Args, cardinal: PluralRules, locale: &str) -> String {
        let ctx = EvalCtx {
            args,
            cardinal,
            locale,
        };
        let mut out = String::new();
        eval(&self.nodes, &ctx, None, &mut out);
        out
    }
}
