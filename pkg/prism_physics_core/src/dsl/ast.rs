//! Abstract syntax tree for the constraint DSL.
//!
//! The grammar is deliberately small: arithmetic expressions over scalar and
//! vector variables, a handful of built-in functions, and a `constraint`
//! declaration that binds a `residual` (and optional `compliance`) expression
//! to a named parameter list. These types are the boundary between the parser
//! and the compiler.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! expression/declaration tree is a textbook abstract-syntax representation
//! from standard compiler-construction knowledge.

use crate::math::scalar::Real;

/// A binary arithmetic operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BinOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
}

/// An arithmetic expression node.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// A literal scalar constant.
    Const(Real),
    /// A reference to a named variable (parameter or point), resolved to a
    /// slot at compile time.
    Var(String),
    /// Unary negation of a sub-expression.
    Neg(Box<Expr>),
    /// A binary operation between two sub-expressions.
    Binary {
        /// The operator applied to `lhs` and `rhs`.
        op: BinOp,
        /// Left-hand operand.
        lhs: Box<Expr>,
        /// Right-hand operand.
        rhs: Box<Expr>,
    },
    /// A call to a built-in function with positional arguments.
    Call {
        /// Function name as written in the source.
        func: String,
        /// Positional argument expressions.
        args: Vec<Expr>,
    },
}

/// A parsed `constraint` declaration.
///
/// The `residual` expression is the constraint function `C`; the solver drives
/// it toward zero. The `compliance` expression evaluates to the XPBD
/// compliance (inverse stiffness) and may only reference parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintDecl {
    /// The declared constraint name.
    pub name: String,
    /// Ordered scalar parameter names from the declaration header.
    pub params: Vec<String>,
    /// The residual (constraint-function) expression.
    pub residual: Expr,
    /// The compliance expression; defaults to `Const(0.0)` when omitted.
    pub compliance: Expr,
}

impl Expr {
    /// Returns `true` when this node is a compile-time-known constant.
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(self, Expr::Const(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_detection() {
        assert!(Expr::Const(1.0).is_const());
        assert!(!Expr::Var("x".into()).is_const());
    }

    #[test]
    fn tree_equality() {
        let a = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(Expr::Const(1.0)),
            rhs: Box::new(Expr::Var("x".into())),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
