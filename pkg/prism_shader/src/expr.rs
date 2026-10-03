//! A small, total expression evaluator for `#if` / `#elif` directives.
//!
//! The grammar is a conventional C-preprocessor-style integer expression over
//! the current [`ShaderDefs`]:
//!
//! ```text
//! expr   := or
//! or     := and ('||' and)*
//! and    := cmp ('&&' cmp)*
//! cmp    := add (('=='|'!='|'<'|'>'|'<='|'>=') add)*
//! add    := mul (('+'|'-') mul)*
//! mul    := unary (('*'|'/'|'%') unary)*
//! unary  := ('!'|'-') unary | primary
//! primary:= number | 'defined' '(' ident ')' | 'defined' ident
//!         | ident | '(' expr ')'
//! ```
//!
//! All arithmetic is `i64`. A bare identifier evaluates to its def value (via
//! [`ShaderDefValue::as_i64`]) or `0` when undefined, matching C preprocessor
//! semantics. Boolean and comparison operators yield `1` or `0`. Division or
//! remainder by zero and arithmetic overflow are reported as errors rather than
//! panicking, so evaluation is total.
//!
//! [`ShaderDefValue::as_i64`]: crate::def::ShaderDefValue::as_i64

use alloc::vec::Vec;

use crate::def::ShaderDefs;

/// Something that went wrong while tokenising or evaluating a `#if` expression.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExprError {
    /// The expression had no tokens at all.
    Empty,
    /// A character that cannot begin any token was found.
    UnexpectedChar(char),
    /// A lone `&` or `|` (the language only has `&&` and `||`).
    IncompleteOperator(char),
    /// A numeric literal did not fit in `i64`.
    IntegerOverflow,
    /// A token appeared where a value was expected.
    ExpectedValue,
    /// A `(` was opened but never closed.
    ExpectedRParen,
    /// `defined` was not followed by an identifier (optionally parenthesised).
    ExpectedIdentAfterDefined,
    /// Tokens remained after a complete expression was parsed.
    TrailingTokens,
    /// Arithmetic overflowed `i64`.
    ArithmeticOverflow,
    /// Division or remainder by zero.
    DivideByZero,
}

/// A lexical token of a `#if` expression.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Token {
    /// An integer literal.
    Num(i64),
    /// An identifier (def name or the `defined` keyword).
    Ident(Vec<u8>),
    /// `!`
    Not,
    /// `&&`
    AndAnd,
    /// `||`
    OrOr,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `(`
    LParen,
    /// `)`
    RParen,
}

/// Tokenises an expression into a flat token list.
fn tokenize(input: &str) -> Result<Vec<Token>, ExprError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b' ' | b'\t' | b'\r' | b'\n' => index += 1,
            b'(' => {
                tokens.push(Token::LParen);
                index += 1;
            }
            b')' => {
                tokens.push(Token::RParen);
                index += 1;
            }
            b'+' => {
                tokens.push(Token::Plus);
                index += 1;
            }
            b'-' => {
                tokens.push(Token::Minus);
                index += 1;
            }
            b'*' => {
                tokens.push(Token::Star);
                index += 1;
            }
            b'/' => {
                tokens.push(Token::Slash);
                index += 1;
            }
            b'%' => {
                tokens.push(Token::Percent);
                index += 1;
            }
            b'!' => {
                if bytes.get(index + 1) == Some(&b'=') {
                    tokens.push(Token::NotEq);
                    index += 2;
                } else {
                    tokens.push(Token::Not);
                    index += 1;
                }
            }
            b'=' => {
                if bytes.get(index + 1) == Some(&b'=') {
                    tokens.push(Token::EqEq);
                    index += 2;
                } else {
                    return Err(ExprError::IncompleteOperator('='));
                }
            }
            b'<' => {
                if bytes.get(index + 1) == Some(&b'=') {
                    tokens.push(Token::Le);
                    index += 2;
                } else {
                    tokens.push(Token::Lt);
                    index += 1;
                }
            }
            b'>' => {
                if bytes.get(index + 1) == Some(&b'=') {
                    tokens.push(Token::Ge);
                    index += 2;
                } else {
                    tokens.push(Token::Gt);
                    index += 1;
                }
            }
            b'&' => {
                if bytes.get(index + 1) == Some(&b'&') {
                    tokens.push(Token::AndAnd);
                    index += 2;
                } else {
                    return Err(ExprError::IncompleteOperator('&'));
                }
            }
            b'|' => {
                if bytes.get(index + 1) == Some(&b'|') {
                    tokens.push(Token::OrOr);
                    index += 2;
                } else {
                    return Err(ExprError::IncompleteOperator('|'));
                }
            }
            b'0'..=b'9' => {
                let (token, next) = lex_number(bytes, index)?;
                tokens.push(token);
                index = next;
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                let start = index;
                while index < bytes.len() && is_ident_byte(bytes[index]) {
                    index += 1;
                }
                tokens.push(Token::Ident(bytes[start..index].to_vec()));
            }
            other => return Err(ExprError::UnexpectedChar(other as char)),
        }
    }
    Ok(tokens)
}

/// Whether a byte can appear within an identifier.
const fn is_ident_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// Lexes a decimal or `0x`-prefixed hexadecimal literal starting at `start`.
fn lex_number(bytes: &[u8], start: usize) -> Result<(Token, usize), ExprError> {
    let mut index = start;
    let (radix, digits_start) = if bytes[index] == b'0'
        && matches!(bytes.get(index + 1), Some(b'x' | b'X'))
    {
        (16u32, index + 2)
    } else {
        (10u32, index)
    };
    index = digits_start;
    let mut value: i64 = 0;
    let mut saw_digit = false;
    while index < bytes.len() {
        let digit = match bytes[index] {
            byte @ b'0'..=b'9' => u32::from(byte - b'0'),
            byte @ b'a'..=b'f' if radix == 16 => u32::from(byte - b'a') + 10,
            byte @ b'A'..=b'F' if radix == 16 => u32::from(byte - b'A') + 10,
            _ => break,
        };
        value = value
            .checked_mul(i64::from(radix))
            .and_then(|scaled| scaled.checked_add(i64::from(digit)))
            .ok_or(ExprError::IntegerOverflow)?;
        saw_digit = true;
        index += 1;
    }
    if !saw_digit {
        return Err(ExprError::IntegerOverflow);
    }
    Ok((Token::Num(value), index))
}

/// Recursive-descent parser-evaluator over a token slice.
struct Evaluator<'a> {
    /// The token stream being consumed.
    tokens: &'a [Token],
    /// The current read cursor into [`Self::tokens`].
    cursor: usize,
    /// The defs an identifier resolves against.
    defs: &'a ShaderDefs,
}

impl<'a> Evaluator<'a> {
    /// Peeks at the current token without consuming it.
    ///
    /// The returned reference carries the token-slice lifetime `'a`, so a
    /// borrowed identifier can outlive the `&mut self` of a consuming call.
    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.cursor)
    }

    /// Consumes and returns the current token (with the `'a` lifetime).
    fn next(&mut self) -> Option<&'a Token> {
        let token = self.tokens.get(self.cursor);
        if token.is_some() {
            self.cursor += 1;
        }
        token
    }

    /// `or := and ('||' and)*`
    fn parse_or(&mut self) -> Result<i64, ExprError> {
        let mut value = self.parse_and()?;
        while matches!(self.peek(), Some(Token::OrOr)) {
            self.cursor += 1;
            let rhs = self.parse_and()?;
            value = i64::from(value != 0 || rhs != 0);
        }
        Ok(value)
    }

    /// `and := cmp ('&&' cmp)*`
    fn parse_and(&mut self) -> Result<i64, ExprError> {
        let mut value = self.parse_cmp()?;
        while matches!(self.peek(), Some(Token::AndAnd)) {
            self.cursor += 1;
            let rhs = self.parse_cmp()?;
            value = i64::from(value != 0 && rhs != 0);
        }
        Ok(value)
    }

    /// `cmp := add (('=='|'!='|'<'|'>'|'<='|'>=') add)*`
    fn parse_cmp(&mut self) -> Result<i64, ExprError> {
        let mut value = self.parse_add()?;
        while let Some(
            token @ (Token::EqEq
            | Token::NotEq
            | Token::Lt
            | Token::Gt
            | Token::Le
            | Token::Ge),
        ) = self.peek()
        {
            let op = token.clone();
            self.cursor += 1;
            let rhs = self.parse_add()?;
            value = i64::from(match op {
                Token::EqEq => value == rhs,
                Token::NotEq => value != rhs,
                Token::Lt => value < rhs,
                Token::Gt => value > rhs,
                Token::Le => value <= rhs,
                Token::Ge => value >= rhs,
                _ => unreachable!("peek restricted the op set"),
            });
        }
        Ok(value)
    }

    /// `add := mul (('+'|'-') mul)*`
    fn parse_add(&mut self) -> Result<i64, ExprError> {
        let mut value = self.parse_mul()?;
        loop {
            let subtract = match self.peek() {
                Some(Token::Plus) => false,
                Some(Token::Minus) => true,
                _ => break,
            };
            self.cursor += 1;
            let rhs = self.parse_mul()?;
            value = if subtract {
                value.checked_sub(rhs)
            } else {
                value.checked_add(rhs)
            }
            .ok_or(ExprError::ArithmeticOverflow)?;
        }
        Ok(value)
    }

    /// `mul := unary (('*'|'/'|'%') unary)*`
    fn parse_mul(&mut self) -> Result<i64, ExprError> {
        let mut value = self.parse_unary()?;
        while let Some(token @ (Token::Star | Token::Slash | Token::Percent)) = self.peek() {
            let op = token.clone();
            self.cursor += 1;
            let rhs = self.parse_unary()?;
            value = match op {
                Token::Star => value.checked_mul(rhs).ok_or(ExprError::ArithmeticOverflow)?,
                Token::Slash => {
                    if rhs == 0 {
                        return Err(ExprError::DivideByZero);
                    }
                    value.checked_div(rhs).ok_or(ExprError::ArithmeticOverflow)?
                }
                Token::Percent => {
                    if rhs == 0 {
                        return Err(ExprError::DivideByZero);
                    }
                    value.checked_rem(rhs).ok_or(ExprError::ArithmeticOverflow)?
                }
                _ => unreachable!("peek restricted the op set"),
            };
        }
        Ok(value)
    }

    /// `unary := ('!'|'-') unary | primary`
    fn parse_unary(&mut self) -> Result<i64, ExprError> {
        match self.peek() {
            Some(Token::Not) => {
                self.cursor += 1;
                Ok(i64::from(self.parse_unary()? == 0))
            }
            Some(Token::Minus) => {
                self.cursor += 1;
                self.parse_unary()?
                    .checked_neg()
                    .ok_or(ExprError::ArithmeticOverflow)
            }
            _ => self.parse_primary(),
        }
    }

    /// `primary := number | defined(ident) | defined ident | ident | ( expr )`
    fn parse_primary(&mut self) -> Result<i64, ExprError> {
        match self.next() {
            Some(Token::Num(value)) => Ok(*value),
            Some(Token::LParen) => {
                let value = self.parse_or()?;
                match self.next() {
                    Some(Token::RParen) => Ok(value),
                    _ => Err(ExprError::ExpectedRParen),
                }
            }
            Some(Token::Ident(name)) => {
                if name.as_slice() == b"defined" {
                    self.parse_defined()
                } else {
                    Ok(self.lookup(name.clone()))
                }
            }
            _ => Err(ExprError::ExpectedValue),
        }
    }

    /// Parses the operand of a `defined` operator (with or without parens).
    fn parse_defined(&mut self) -> Result<i64, ExprError> {
        let parenthesised = matches!(self.peek(), Some(Token::LParen));
        if parenthesised {
            self.cursor += 1;
        }
        let present = match self.next() {
            Some(Token::Ident(name)) => self.defs_contains(name),
            _ => return Err(ExprError::ExpectedIdentAfterDefined),
        };
        if parenthesised {
            match self.next() {
                Some(Token::RParen) => {}
                _ => return Err(ExprError::ExpectedRParen),
            }
        }
        Ok(i64::from(present))
    }

    /// Whether a def with this raw identifier name exists.
    fn defs_contains(&self, name: &[u8]) -> bool {
        core::str::from_utf8(name).is_ok_and(|name| self.defs.contains(name))
    }

    /// Resolves a bare identifier to its def value, or `0` when undefined.
    fn lookup(&self, name: Vec<u8>) -> i64 {
        core::str::from_utf8(&name)
            .ok()
            .and_then(|name| self.defs.get(name))
            .map_or(0, crate::def::ShaderDefValue::as_i64)
    }
}

/// Evaluates a `#if` / `#elif` expression against `defs`.
///
/// Returns the integer result; callers treat a non-zero result as the branch
/// being taken. Errors are returned (never panics) for malformed input,
/// division by zero, and overflow.
///
/// # Errors
///
/// Returns an [`ExprError`] when the expression is empty, contains an invalid
/// token, is syntactically malformed, has trailing tokens, divides by zero, or
/// overflows `i64`.
pub fn evaluate(input: &str, defs: &ShaderDefs) -> Result<i64, ExprError> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(ExprError::Empty);
    }
    let mut evaluator = Evaluator {
        tokens: &tokens,
        cursor: 0,
        defs,
    };
    let value = evaluator.parse_or()?;
    if evaluator.cursor != tokens.len() {
        return Err(ExprError::TrailingTokens);
    }
    Ok(value)
}
