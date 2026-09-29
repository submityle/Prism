//! Lexical analysis for the constraint DSL.
//!
//! [`tokenize`] converts a source string into a flat list of
//! [`SpannedToken`]s, each tagged with the byte offset where it began. The
//! lexer skips whitespace and `//` line comments, recognises the `constraint`
//! keyword, and emits numbers, identifiers, operators, and punctuation. Unary
//! minus is left for the parser to disambiguate.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! single-pass, character-classifying scanner is standard, publicly
//! documented compiler-construction knowledge.

use crate::dsl::error::DslError;
use crate::math::scalar::Real;

/// A lexical token in the constraint DSL.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// A numeric literal, already parsed into a [`Real`].
    Number(Real),
    /// An identifier (variable or function name).
    Ident(String),
    /// The `constraint` keyword.
    Constraint,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `=`
    Equals,
}

/// A [`Token`] paired with its starting byte offset in the source.
#[derive(Clone, Debug, PartialEq)]
pub struct SpannedToken {
    /// The token itself.
    pub token: Token,
    /// Byte offset of the first character of the token.
    pub position: usize,
}

impl SpannedToken {
    /// Bundles `token` with its source `position`.
    #[must_use]
    pub fn new(token: Token, position: usize) -> Self {
        SpannedToken { token, position }
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Tokenizes `src`, returning the token stream or the first lexical error.
///
/// # Errors
///
/// Returns [`DslError::Lex`] on an illegal character or a malformed numeric
/// literal.
pub fn tokenize(src: &str) -> Result<Vec<SpannedToken>, DslError> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        let c = bytes[i] as char;
        // Whitespace.
        if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
            i += 1;
            continue;
        }
        // Line comment `// ... \n`.
        if c == '/' && i + 1 < n && bytes[i + 1] as char == '/' {
            i += 2;
            while i < n && bytes[i] as char != '\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        match c {
            '+' => {
                tokens.push(SpannedToken::new(Token::Plus, start));
                i += 1;
            }
            '-' => {
                tokens.push(SpannedToken::new(Token::Minus, start));
                i += 1;
            }
            '*' => {
                tokens.push(SpannedToken::new(Token::Star, start));
                i += 1;
            }
            '/' => {
                tokens.push(SpannedToken::new(Token::Slash, start));
                i += 1;
            }
            '(' => {
                tokens.push(SpannedToken::new(Token::LParen, start));
                i += 1;
            }
            ')' => {
                tokens.push(SpannedToken::new(Token::RParen, start));
                i += 1;
            }
            '{' => {
                tokens.push(SpannedToken::new(Token::LBrace, start));
                i += 1;
            }
            '}' => {
                tokens.push(SpannedToken::new(Token::RBrace, start));
                i += 1;
            }
            ',' => {
                tokens.push(SpannedToken::new(Token::Comma, start));
                i += 1;
            }
            ';' => {
                tokens.push(SpannedToken::new(Token::Semicolon, start));
                i += 1;
            }
            '=' => {
                tokens.push(SpannedToken::new(Token::Equals, start));
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                // Numeric literal: digits with an optional single fractional part.
                let mut j = i;
                let mut seen_dot = false;
                let mut seen_digit = false;
                while j < n {
                    let d = bytes[j] as char;
                    if d.is_ascii_digit() {
                        seen_digit = true;
                        j += 1;
                    } else if d == '.' && !seen_dot {
                        seen_dot = true;
                        j += 1;
                    } else {
                        break;
                    }
                }
                if !seen_digit {
                    return Err(DslError::lex(start, "number literal has no digits"));
                }
                let text = &src[i..j];
                let value: Real = text
                    .parse::<Real>()
                    .map_err(|_| DslError::lex(start, "malformed number literal"))?;
                tokens.push(SpannedToken::new(Token::Number(value), start));
                i = j;
            }
            _ if is_ident_start(c) => {
                let mut j = i + 1;
                while j < n && is_ident_continue(bytes[j] as char) {
                    j += 1;
                }
                let text = &src[i..j];
                let token = if text == "constraint" {
                    Token::Constraint
                } else {
                    Token::Ident(text.to_string())
                };
                tokens.push(SpannedToken::new(token, start));
                i = j;
            }
            _ => {
                return Err(DslError::lex(start, format!("unexpected character '{c}'")));
            }
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Token> {
        tokenize(src)
            .unwrap()
            .into_iter()
            .map(|t| t.token)
            .collect()
    }

    #[test]
    fn scans_operators_and_punctuation() {
        let toks = kinds("+-*/(){},;=");
        assert_eq!(
            toks,
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::RBrace,
                Token::Comma,
                Token::Semicolon,
                Token::Equals,
            ]
        );
    }

    #[test]
    fn scans_numbers_and_identifiers() {
        let toks = kinds("rest 3.5 alpha_1");
        assert_eq!(toks[0], Token::Ident("rest".into()));
        match toks[1] {
            Token::Number(v) => assert!((v - 3.5).abs() < 1e-6),
            _ => unreachable!(),
        }
        assert_eq!(toks[2], Token::Ident("alpha_1".into()));
    }

    #[test]
    fn recognises_keyword() {
        assert_eq!(kinds("constraint"), vec![Token::Constraint]);
    }

    #[test]
    fn skips_line_comments_and_whitespace() {
        let toks = kinds("a // comment\n + b");
        assert_eq!(
            toks,
            vec![
                Token::Ident("a".into()),
                Token::Plus,
                Token::Ident("b".into())
            ]
        );
    }

    #[test]
    fn tracks_positions() {
        let toks = tokenize("  a").unwrap();
        assert_eq!(toks[0].position, 2);
    }

    #[test]
    fn rejects_illegal_character() {
        let err = tokenize("a $ b").unwrap_err();
        assert!(matches!(err, DslError::Lex { .. }));
    }
}
