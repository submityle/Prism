//! A classic, line-based C-style preprocessor driven by shader-defs.
//!
//! Given a source string and a [`ShaderDefs`] set, [`preprocess`] evaluates the
//! conditional-compilation family of directives and returns the surviving text:
//!
//! ```text
//! #ifdef NAME        emit the block when NAME is defined (any value)
//! #ifndef NAME       emit the block when NAME is not defined
//! #if EXPR           emit the block when EXPR evaluates to non-zero
//! #elif EXPR         else-if branch (only when no earlier branch was taken)
//! #else              fallback branch
//! #endif             close the innermost conditional
//! ```
//!
//! A directive is any line whose first non-whitespace character is `#`. Only the
//! conditional family above is interpreted; every other `#`-line (for example a
//! downstream `#import` consumed by [`crate::compose`], or a backend-native
//! directive) is passed through verbatim so this layer composes cleanly. The
//! `#if` / `#elif` expressions are evaluated by [`crate::expr`], giving the full
//! integer grammar including `defined(NAME)`.
//!
//! Evaluation is total: malformed nesting (a stray `#elif` / `#else` / `#endif`,
//! an unterminated block, a missing identifier, or a bad expression) is reported
//! as a [`PreprocessError`] rather than panicking. Consumed directive lines are
//! dropped from the output.
//!
//! [`ShaderDefs`]: crate::def::ShaderDefs

use alloc::string::String;
use alloc::vec::Vec;

use crate::def::ShaderDefs;
use crate::expr::{self, ExprError};

/// Something that went wrong while preprocessing a source string.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PreprocessError {
    /// An `#ifdef` or `#ifndef` directive was missing its identifier.
    MissingDefineName {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// An `#if` or `#elif` expression could not be evaluated.
    BadExpression {
        /// 1-based line number of the offending directive.
        line: usize,
        /// The underlying expression error.
        error: ExprError,
    },
    /// An `#elif` appeared with no open conditional block.
    ElifWithoutIf {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// An `#elif` appeared after the block's `#else`.
    ElifAfterElse {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// An `#else` appeared with no open conditional block.
    ElseWithoutIf {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// A second `#else` appeared in the same conditional block.
    DuplicateElse {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// An `#endif` appeared with no open conditional block.
    EndifWithoutIf {
        /// 1-based line number of the offending directive.
        line: usize,
    },
    /// The source ended while a conditional block was still open.
    UnterminatedConditional {
        /// 1-based line number of the directive that opened the block.
        line: usize,
    },
}

/// One open conditional block on the directive stack.
struct Frame {
    /// Whether the enclosing scope is emitting, so this block can be live.
    parent_emitting: bool,
    /// Whether any branch of this block has already been taken.
    branch_taken: bool,
    /// Whether the current branch is emitting its lines.
    active: bool,
    /// Whether the block's `#else` branch has already been seen.
    else_seen: bool,
    /// 1-based line number of the directive that opened this block.
    open_line: usize,
}

/// A parsed `#`-directive: its keyword and the trimmed remainder.
struct Directive<'a> {
    /// The directive keyword without the leading `#` (for example `ifdef`).
    keyword: &'a str,
    /// The remainder of the line after the keyword, trimmed.
    argument: &'a str,
}

/// Splits a source line into a [`Directive`] when it is one.
///
/// A directive is a line whose first non-whitespace byte is `#`. Returns `None`
/// for ordinary content lines.
fn parse_directive(line: &str) -> Option<Directive<'_>> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('#')?;
    let rest = rest.trim_start();
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let (keyword, argument) = rest.split_at(end);
    Some(Directive { keyword, argument: argument.trim() })
}

/// Returns the first whitespace-delimited identifier of `argument`, if any.
fn first_token(argument: &str) -> Option<&str> {
    argument.split_whitespace().next()
}

/// Preprocesses `source` against `defs`, returning the surviving text.
///
/// # Errors
///
/// Returns a [`PreprocessError`] when the conditional directives are unbalanced,
/// an identifier is missing, or an `#if` / `#elif` expression fails to evaluate.
pub fn preprocess(source: &str, defs: &ShaderDefs) -> Result<String, PreprocessError> {
    let mut stack: Vec<Frame> = Vec::new();
    let mut output: Vec<&str> = Vec::new();

    for (index, line) in source.lines().enumerate() {
        let line_no = index + 1;
        let Some(directive) = parse_directive(line) else {
            if stack.last().is_none_or(|frame| frame.active) {
                output.push(line);
            }
            continue;
        };

        match directive.keyword {
            "ifdef" | "ifndef" => {
                let name = first_token(directive.argument)
                    .ok_or(PreprocessError::MissingDefineName { line: line_no })?;
                let defined = defs.contains(name);
                let cond = if directive.keyword == "ifdef" { defined } else { !defined };
                let parent_emitting = stack.last().is_none_or(|frame| frame.active);
                let active = parent_emitting && cond;
                stack.push(Frame {
                    parent_emitting,
                    branch_taken: active,
                    active,
                    else_seen: false,
                    open_line: line_no,
                });
            }
            "if" => {
                let cond = evaluate_condition(directive.argument, defs, line_no)?;
                let parent_emitting = stack.last().is_none_or(|frame| frame.active);
                let active = parent_emitting && cond;
                stack.push(Frame {
                    parent_emitting,
                    branch_taken: active,
                    active,
                    else_seen: false,
                    open_line: line_no,
                });
            }
            "elif" => {
                let cond = evaluate_condition(directive.argument, defs, line_no)?;
                let frame = stack
                    .last_mut()
                    .ok_or(PreprocessError::ElifWithoutIf { line: line_no })?;
                if frame.else_seen {
                    return Err(PreprocessError::ElifAfterElse { line: line_no });
                }
                if frame.branch_taken {
                    frame.active = false;
                } else if frame.parent_emitting && cond {
                    frame.active = true;
                    frame.branch_taken = true;
                } else {
                    frame.active = false;
                }
            }
            "else" => {
                let frame = stack
                    .last_mut()
                    .ok_or(PreprocessError::ElseWithoutIf { line: line_no })?;
                if frame.else_seen {
                    return Err(PreprocessError::DuplicateElse { line: line_no });
                }
                frame.else_seen = true;
                frame.active = frame.parent_emitting && !frame.branch_taken;
                frame.branch_taken = true;
            }
            "endif" => {
                stack
                    .pop()
                    .ok_or(PreprocessError::EndifWithoutIf { line: line_no })?;
            }
            _ => {
                // Not a conditional directive: pass the line through unchanged
                // when the enclosing scope is emitting (e.g. `#import`).
                if stack.last().is_none_or(|frame| frame.active) {
                    output.push(line);
                }
            }
        }
    }

    if let Some(frame) = stack.last() {
        return Err(PreprocessError::UnterminatedConditional { line: frame.open_line });
    }

    Ok(output.join("\n"))
}

/// Evaluates an `#if` / `#elif` expression into a boolean, mapping errors.
fn evaluate_condition(
    argument: &str,
    defs: &ShaderDefs,
    line: usize,
) -> Result<bool, PreprocessError> {
    match expr::evaluate(argument, defs) {
        Ok(value) => Ok(value != 0),
        Err(error) => Err(PreprocessError::BadExpression { line, error }),
    }
}

impl PreprocessError {
    /// The 1-based source line the error refers to.
    #[must_use]
    pub fn line(&self) -> usize {
        match *self {
            Self::MissingDefineName { line }
            | Self::BadExpression { line, .. }
            | Self::ElifWithoutIf { line }
            | Self::ElifAfterElse { line }
            | Self::ElseWithoutIf { line }
            | Self::DuplicateElse { line }
            | Self::EndifWithoutIf { line }
            | Self::UnterminatedConditional { line } => line,
        }
    }
}

