//! Error types for the constraint DSL pipeline.
//!
//! A single [`DslError`] enum spans the whole toolchain (lexing, parsing,
//! compilation, and evaluation) so that callers of the public API only ever
//! match one error type. Each variant carries a human-readable message and,
//! where meaningful, a byte position into the source text.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! layered lexer/parser/compiler/interpreter error model is standard,
//! publicly documented compiler-construction knowledge.

/// An error produced anywhere in the constraint-DSL pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DslError {
    /// The lexer encountered an invalid character or malformed token.
    Lex {
        /// Byte offset into the source where the problem was detected.
        position: usize,
        /// Human-readable description of the lexical error.
        message: String,
    },
    /// The parser encountered an unexpected token or unfinished construct.
    Parse {
        /// Byte offset into the source where the problem was detected.
        position: usize,
        /// Human-readable description of the syntax error.
        message: String,
    },
    /// The compiler could not lower the AST (unknown function, arity mismatch,
    /// unknown variable, or an illegal compliance expression).
    Compile {
        /// Human-readable description of the compilation error.
        message: String,
    },
    /// The virtual machine hit a runtime fault (type mismatch, empty stack,
    /// division by zero, or a domain error such as `sqrt` of a negative).
    Eval {
        /// Human-readable description of the evaluation error.
        message: String,
    },
}

impl DslError {
    /// Builds a [`DslError::Lex`] at `position` with `message`.
    #[must_use]
    pub fn lex(position: usize, message: impl Into<String>) -> Self {
        DslError::Lex {
            position,
            message: message.into(),
        }
    }

    /// Builds a [`DslError::Parse`] at `position` with `message`.
    #[must_use]
    pub fn parse(position: usize, message: impl Into<String>) -> Self {
        DslError::Parse {
            position,
            message: message.into(),
        }
    }

    /// Builds a [`DslError::Compile`] with `message`.
    #[must_use]
    pub fn compile(message: impl Into<String>) -> Self {
        DslError::Compile {
            message: message.into(),
        }
    }

    /// Builds a [`DslError::Eval`] with `message`.
    #[must_use]
    pub fn eval(message: impl Into<String>) -> Self {
        DslError::Eval {
            message: message.into(),
        }
    }
}

impl core::fmt::Display for DslError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DslError::Lex { position, message } => {
                write!(f, "lexical error at byte {position}: {message}")
            }
            DslError::Parse { position, message } => {
                write!(f, "syntax error at byte {position}: {message}")
            }
            DslError::Compile { message } => write!(f, "compile error: {message}"),
            DslError::Eval { message } => write!(f, "evaluation error: {message}"),
        }
    }
}

impl std::error::Error for DslError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_position_and_message() {
        let e = DslError::lex(7, "bad char");
        let s = format!("{e}");
        assert!(s.contains("byte 7"));
        assert!(s.contains("bad char"));
    }

    #[test]
    fn variants_compare_by_value() {
        assert_eq!(DslError::compile("x"), DslError::compile("x"));
        assert_ne!(DslError::compile("x"), DslError::eval("x"));
    }

    #[test]
    fn implements_std_error() {
        fn assert_error<T: std::error::Error>(_: &T) {}
        assert_error(&DslError::eval("boom"));
    }
}
