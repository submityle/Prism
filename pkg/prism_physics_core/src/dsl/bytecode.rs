//! Stack-machine bytecode for compiled constraint expressions.
//!
//! A compiled [`Expr`](crate::dsl::ast::Expr) becomes a [`Program`]: a flat
//! list of [`OpCode`]s executed against an operand stack. Variables are
//! resolved to slot indices at compile time — scalars occupy one slot and
//! points occupy three consecutive slots (x, y, z). Built-in functions are
//! resolved to the [`BuiltinFn`] enum so the VM never dispatches on strings.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. A
//! postfix, stack-based instruction set for arithmetic is standard, publicly
//! documented virtual-machine knowledge.

use crate::math::scalar::Real;

/// A built-in function callable from the DSL.
///
/// The set is deliberately limited to operations expressible with the allowed
/// scalar primitives (no transcendental functions), keeping evaluation
/// deterministic and free of disallowed `f32` methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BuiltinFn {
    /// `abs(x)`: absolute value of a scalar.
    Abs,
    /// `min(a, b)`: smaller of two scalars.
    Min,
    /// `max(a, b)`: larger of two scalars.
    Max,
    /// `clamp(x, lo, hi)`: `x` restricted to `[lo, hi]`.
    Clamp,
    /// `sqrt(x)`: square root of a non-negative scalar.
    Sqrt,
    /// `length(v)`: Euclidean length of a vector.
    Length,
    /// `distance(a, b)`: Euclidean distance between two points.
    Distance,
    /// `dot(a, b)`: dot product of two vectors.
    Dot,
}

impl BuiltinFn {
    /// Returns the number of arguments the function expects.
    #[must_use]
    pub fn arity(self) -> usize {
        match self {
            BuiltinFn::Abs | BuiltinFn::Sqrt | BuiltinFn::Length => 1,
            BuiltinFn::Min | BuiltinFn::Max | BuiltinFn::Distance | BuiltinFn::Dot => 2,
            BuiltinFn::Clamp => 3,
        }
    }

    /// Resolves a source identifier to a built-in function, if it names one.
    #[must_use]
    pub fn from_name(name: &str) -> Option<BuiltinFn> {
        match name {
            "abs" => Some(BuiltinFn::Abs),
            "min" => Some(BuiltinFn::Min),
            "max" => Some(BuiltinFn::Max),
            "clamp" => Some(BuiltinFn::Clamp),
            "sqrt" => Some(BuiltinFn::Sqrt),
            "length" => Some(BuiltinFn::Length),
            "distance" => Some(BuiltinFn::Distance),
            "dot" => Some(BuiltinFn::Dot),
            _ => None,
        }
    }

    /// Returns the canonical source spelling of the function.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFn::Abs => "abs",
            BuiltinFn::Min => "min",
            BuiltinFn::Max => "max",
            BuiltinFn::Clamp => "clamp",
            BuiltinFn::Sqrt => "sqrt",
            BuiltinFn::Length => "length",
            BuiltinFn::Distance => "distance",
            BuiltinFn::Dot => "dot",
        }
    }
}

/// A single stack-machine instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OpCode {
    /// Push a scalar constant onto the stack.
    PushConst(Real),
    /// Push the scalar variable stored at the given slot.
    PushScalar(usize),
    /// Push the point (vector) whose x/y/z occupy `base`, `base+1`, `base+2`.
    PushPoint(usize),
    /// Pop `b`, pop `a`, push `a + b`.
    Add,
    /// Pop `b`, pop `a`, push `a - b`.
    Sub,
    /// Pop `b`, pop `a`, push `a * b`.
    Mul,
    /// Pop `b`, pop `a`, push `a / b`.
    Div,
    /// Pop `a`, push `-a`.
    Neg,
    /// Pop the function's arguments and push its result.
    Call(BuiltinFn),
}

/// A compiled program: an ordered list of stack-machine instructions.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Program {
    /// The instruction sequence, executed front to back.
    pub ops: Vec<OpCode>,
}

impl Program {
    /// Creates an empty program.
    #[must_use]
    pub fn new() -> Self {
        Program { ops: Vec::new() }
    }

    /// Appends an instruction to the program.
    pub fn push(&mut self, op: OpCode) {
        self.ops.push(op);
    }

    /// Returns the number of instructions in the program.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` when the program has no instructions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_matches_definition() {
        assert_eq!(BuiltinFn::Abs.arity(), 1);
        assert_eq!(BuiltinFn::Clamp.arity(), 3);
        assert_eq!(BuiltinFn::Dot.arity(), 2);
    }

    #[test]
    fn name_round_trips() {
        for f in [
            BuiltinFn::Abs,
            BuiltinFn::Min,
            BuiltinFn::Max,
            BuiltinFn::Clamp,
            BuiltinFn::Sqrt,
            BuiltinFn::Length,
            BuiltinFn::Distance,
            BuiltinFn::Dot,
        ] {
            assert_eq!(BuiltinFn::from_name(f.name()), Some(f));
        }
    }

    #[test]
    fn unknown_name_is_none() {
        assert_eq!(BuiltinFn::from_name("sin"), None);
    }

    #[test]
    fn program_len_and_empty() {
        let mut p = Program::new();
        assert!(p.is_empty());
        p.push(OpCode::PushConst(1.0));
        assert_eq!(p.len(), 1);
        assert!(!p.is_empty());
    }
}
