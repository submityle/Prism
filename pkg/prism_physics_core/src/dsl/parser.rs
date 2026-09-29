//! Recursive-descent parser for the constraint DSL.
//!
//! The parser turns a token stream into an [`Expr`] or a [`ConstraintDecl`].
//! Operator precedence is encoded structurally: `expr` handles `+`/`-`,
//! `term` handles `*`/`/`, `unary` handles a leading `-`, and `primary`
//! handles literals, parenthesised groups, variable references, and function
//! calls.
//!
//! ```text
//! expr    := term (('+' | '-') term)*
//! term    := unary (('*' | '/') unary)*
//! unary   := '-' unary | primary
//! primary := number | ident ('(' args? ')')? | '(' expr ')'
//! args    := expr (',' expr)*
//! decl    := 'constraint' ident '(' params? ')' '{' assignment* '}'
//! ```
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Recursive
//! descent with precedence-climbing layers is standard, publicly documented
//! compiler-construction knowledge.

use crate::dsl::ast::{BinOp, ConstraintDecl, Expr};
use crate::dsl::error::DslError;
use crate::dsl::lexer::{tokenize, SpannedToken, Token};

/// A cursor over a token stream with precedence-aware descent methods.
struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    /// Byte length of the source, used for end-of-input error positions.
    src_len: usize,
}

impl Parser {
    fn new(tokens: Vec<SpannedToken>, src_len: usize) -> Self {
        Parser {
            tokens,
            pos: 0,
            src_len,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    fn position_here(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map_or(self.src_len, |t| t.position)
    }

    fn advance(&mut self) -> Option<SpannedToken> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Token, what: &str) -> Result<(), DslError> {
        match self.peek() {
            Some(tok) if tok == want => {
                self.pos += 1;
                Ok(())
            }
            _ => Err(DslError::parse(
                self.position_here(),
                format!("expected {what}"),
            )),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.parse_term()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_term()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.parse_unary()?;
        while let Some(tok) = self.peek() {
            let op = match tok {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, DslError> {
        if let Some(Token::Minus) = self.peek() {
            self.pos += 1;
            let inner = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, DslError> {
        let position = self.position_here();
        let spanned = self
            .advance()
            .ok_or_else(|| DslError::parse(position, "unexpected end of input"))?;
        match spanned.token {
            Token::Number(v) => Ok(Expr::Const(v)),
            Token::LParen => {
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen, "')' to close grouping")?;
                Ok(inner)
            }
            Token::Ident(name) => {
                if let Some(Token::LParen) = self.peek() {
                    self.pos += 1;
                    let args = self.parse_args()?;
                    self.expect(&Token::RParen, "')' to close call")?;
                    Ok(Expr::Call { func: name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(DslError::parse(
                spanned.position,
                format!("unexpected token {other:?}"),
            )),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, DslError> {
        let mut args = Vec::new();
        if let Some(Token::RParen) = self.peek() {
            return Ok(args);
        }
        loop {
            args.push(self.parse_expr()?);
            match self.peek() {
                Some(Token::Comma) => {
                    self.pos += 1;
                }
                _ => break,
            }
        }
        Ok(args)
    }

    fn parse_constraint(&mut self) -> Result<ConstraintDecl, DslError> {
        self.expect(&Token::Constraint, "'constraint' keyword")?;
        let name = self.expect_ident("constraint name")?;
        self.expect(&Token::LParen, "'(' after constraint name")?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen, "')' after parameter list")?;
        self.expect(&Token::LBrace, "'{' to open constraint body")?;

        let mut residual: Option<Expr> = None;
        let mut compliance: Option<Expr> = None;
        while !matches!(self.peek(), Some(Token::RBrace) | None) {
            let field = self.expect_ident("assignment target ('residual' or 'compliance')")?;
            self.expect(&Token::Equals, "'=' in assignment")?;
            let value = self.parse_expr()?;
            self.expect(&Token::Semicolon, "';' after assignment")?;
            match field.as_str() {
                "residual" => residual = Some(value),
                "compliance" => compliance = Some(value),
                other => {
                    return Err(DslError::parse(
                        self.position_here(),
                        format!("unknown assignment target '{other}'"),
                    ));
                }
            }
        }
        self.expect(&Token::RBrace, "'}' to close constraint body")?;

        let residual = residual.ok_or_else(|| {
            DslError::parse(self.src_len, "constraint body must define 'residual'")
        })?;
        let compliance = compliance.unwrap_or(Expr::Const(0.0));
        Ok(ConstraintDecl {
            name,
            params,
            residual,
            compliance,
        })
    }

    fn parse_params(&mut self) -> Result<Vec<String>, DslError> {
        let mut params = Vec::new();
        if let Some(Token::RParen) = self.peek() {
            return Ok(params);
        }
        loop {
            params.push(self.expect_ident("parameter name")?);
            match self.peek() {
                Some(Token::Comma) => {
                    self.pos += 1;
                }
                _ => break,
            }
        }
        Ok(params)
    }

    fn expect_ident(&mut self, what: &str) -> Result<String, DslError> {
        let position = self.position_here();
        match self.advance() {
            Some(SpannedToken {
                token: Token::Ident(name),
                ..
            }) => Ok(name),
            _ => Err(DslError::parse(position, format!("expected {what}"))),
        }
    }

    fn ensure_consumed(&self) -> Result<(), DslError> {
        if self.pos == self.tokens.len() {
            Ok(())
        } else {
            Err(DslError::parse(
                self.position_here(),
                "trailing tokens after expression",
            ))
        }
    }
}

/// Parses a standalone arithmetic expression from `src`.
///
/// # Errors
///
/// Returns [`DslError::Lex`] or [`DslError::Parse`] on malformed input, or if
/// there are trailing tokens after the expression.
pub fn parse_expression(src: &str) -> Result<Expr, DslError> {
    let tokens = tokenize(src)?;
    let mut parser = Parser::new(tokens, src.len());
    let expr = parser.parse_expr()?;
    parser.ensure_consumed()?;
    Ok(expr)
}

/// Parses a full `constraint` declaration from `src`.
///
/// # Errors
///
/// Returns [`DslError::Lex`] or [`DslError::Parse`] on malformed input, or if
/// there are trailing tokens after the declaration.
pub fn parse_constraint(src: &str) -> Result<ConstraintDecl, DslError> {
    let tokens = tokenize(src)?;
    let mut parser = Parser::new(tokens, src.len());
    let decl = parser.parse_constraint()?;
    parser.ensure_consumed()?;
    Ok(decl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_binds_multiplication_tighter() {
        let e = parse_expression("1 + 2 * 3").unwrap();
        match e {
            Expr::Binary {
                op: BinOp::Add,
                rhs,
                ..
            } => assert!(matches!(*rhs, Expr::Binary { op: BinOp::Mul, .. })),
            _ => unreachable!(),
        }
    }

    #[test]
    fn unary_minus_and_parens() {
        let e = parse_expression("-(a + 1)").unwrap();
        assert!(matches!(e, Expr::Neg(_)));
    }

    #[test]
    fn function_call_with_args() {
        let e = parse_expression("distance(a, b)").unwrap();
        match e {
            Expr::Call { func, args } => {
                assert_eq!(func, "distance");
                assert_eq!(args.len(), 2);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn constraint_declaration_defaults_compliance() {
        let decl =
            parse_constraint("constraint d(rest) { residual = length(b - a) - rest; }").unwrap();
        assert_eq!(decl.name, "d");
        assert_eq!(decl.params, vec!["rest".to_string()]);
        assert_eq!(decl.compliance, Expr::Const(0.0));
    }

    #[test]
    fn constraint_reads_compliance() {
        let decl =
            parse_constraint("constraint d(rest, k) { residual = rest; compliance = k; }").unwrap();
        assert_eq!(decl.compliance, Expr::Var("k".into()));
    }

    #[test]
    fn missing_residual_is_error() {
        let err = parse_constraint("constraint d() { compliance = 1; }").unwrap_err();
        assert!(matches!(err, DslError::Parse { .. }));
    }

    #[test]
    fn unbalanced_parens_is_error() {
        let err = parse_expression("(a + 1").unwrap_err();
        assert!(matches!(err, DslError::Parse { .. }));
    }

    #[test]
    fn trailing_tokens_rejected() {
        let err = parse_expression("a b").unwrap_err();
        assert!(matches!(err, DslError::Parse { .. }));
    }
}
