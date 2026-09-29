//! Stack virtual machine that evaluates a compiled [`Program`].
//!
//! [`eval`] runs the bytecode against a flat slice of scalar values (laid out
//! by an [`Environment`](crate::dsl::env::Environment)) and returns the single
//! scalar left on the stack. Values on the stack are either scalars or
//! vectors; arithmetic follows the usual scalar/vector rules and type
//! mismatches surface as [`DslError::Eval`] rather than panics. The machine is
//! fully deterministic and uses only allowed scalar primitives.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Postfix
//! evaluation over an operand stack is standard, publicly documented
//! virtual-machine knowledge.

use glam::Vec3;

use crate::dsl::bytecode::{BuiltinFn, OpCode, Program};
use crate::dsl::error::DslError;
use crate::math::scalar::{Real, EPSILON};

/// A value on the VM operand stack: a scalar or a 3-vector.
#[derive(Clone, Copy, Debug)]
enum Value {
    Scalar(Real),
    Vector(Vec3),
}

impl Value {
    fn as_scalar(self) -> Result<Real, DslError> {
        match self {
            Value::Scalar(s) => Ok(s),
            Value::Vector(_) => Err(DslError::eval("expected scalar, found vector")),
        }
    }

    fn as_vector(self) -> Result<Vec3, DslError> {
        match self {
            Value::Vector(v) => Ok(v),
            Value::Scalar(_) => Err(DslError::eval("expected vector, found scalar")),
        }
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, DslError> {
    stack
        .pop()
        .ok_or_else(|| DslError::eval("operand stack underflow"))
}

fn add(a: Value, b: Value) -> Result<Value, DslError> {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x + y)),
        (Value::Vector(x), Value::Vector(y)) => Ok(Value::Vector(x + y)),
        _ => Err(DslError::eval(
            "'+' requires matching scalar or vector operands",
        )),
    }
}

fn sub(a: Value, b: Value) -> Result<Value, DslError> {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x - y)),
        (Value::Vector(x), Value::Vector(y)) => Ok(Value::Vector(x - y)),
        _ => Err(DslError::eval(
            "'-' requires matching scalar or vector operands",
        )),
    }
}

fn mul(a: Value, b: Value) -> Result<Value, DslError> {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x * y)),
        (Value::Vector(v), Value::Scalar(s)) | (Value::Scalar(s), Value::Vector(v)) => {
            Ok(Value::Vector(v * s))
        }
        (Value::Vector(_), Value::Vector(_)) => Err(DslError::eval(
            "'*' between two vectors is undefined; use dot()",
        )),
    }
}

fn div(a: Value, b: Value) -> Result<Value, DslError> {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => {
            if y.abs() < EPSILON {
                return Err(DslError::eval("division by zero"));
            }
            Ok(Value::Scalar(x / y))
        }
        (Value::Vector(v), Value::Scalar(s)) => {
            if s.abs() < EPSILON {
                return Err(DslError::eval("division by zero"));
            }
            Ok(Value::Vector(v / s))
        }
        _ => Err(DslError::eval("'/' requires a scalar divisor")),
    }
}

fn call(func: BuiltinFn, stack: &mut Vec<Value>) -> Result<Value, DslError> {
    match func {
        BuiltinFn::Abs => {
            let x = pop(stack)?.as_scalar()?;
            Ok(Value::Scalar(x.abs()))
        }
        BuiltinFn::Sqrt => {
            let x = pop(stack)?.as_scalar()?;
            if x < 0.0 {
                return Err(DslError::eval("sqrt of a negative value"));
            }
            Ok(Value::Scalar(x.sqrt()))
        }
        BuiltinFn::Min => {
            let b = pop(stack)?.as_scalar()?;
            let a = pop(stack)?.as_scalar()?;
            Ok(Value::Scalar(a.min(b)))
        }
        BuiltinFn::Max => {
            let b = pop(stack)?.as_scalar()?;
            let a = pop(stack)?.as_scalar()?;
            Ok(Value::Scalar(a.max(b)))
        }
        BuiltinFn::Clamp => {
            let hi = pop(stack)?.as_scalar()?;
            let lo = pop(stack)?.as_scalar()?;
            let x = pop(stack)?.as_scalar()?;
            if lo > hi {
                return Err(DslError::eval("clamp bounds are inverted (lo > hi)"));
            }
            Ok(Value::Scalar(x.clamp(lo, hi)))
        }
        BuiltinFn::Length => {
            let v = pop(stack)?.as_vector()?;
            Ok(Value::Scalar(v.length()))
        }
        BuiltinFn::Distance => {
            let b = pop(stack)?.as_vector()?;
            let a = pop(stack)?.as_vector()?;
            Ok(Value::Scalar((a - b).length()))
        }
        BuiltinFn::Dot => {
            let b = pop(stack)?.as_vector()?;
            let a = pop(stack)?.as_vector()?;
            Ok(Value::Scalar(a.dot(b)))
        }
    }
}

/// Evaluates `program` against the scalar `values` slice.
///
/// The `values` slice must be at least as long as the largest slot the program
/// references; a point at slot `base` reads `values[base..base + 3]`.
///
/// # Errors
///
/// Returns [`DslError::Eval`] on a stack underflow, a type mismatch, an
/// out-of-range slot, a division by zero, a domain error, or if the program
/// does not leave exactly one scalar on the stack.
pub fn eval(program: &Program, values: &[Real]) -> Result<Real, DslError> {
    let mut stack: Vec<Value> = Vec::with_capacity(8);
    for op in &program.ops {
        match *op {
            OpCode::PushConst(c) => stack.push(Value::Scalar(c)),
            OpCode::PushScalar(slot) => {
                let v = values
                    .get(slot)
                    .copied()
                    .ok_or_else(|| DslError::eval("scalar slot out of range"))?;
                stack.push(Value::Scalar(v));
            }
            OpCode::PushPoint(base) => {
                let x = values.get(base).copied();
                let y = values.get(base + 1).copied();
                let z = values.get(base + 2).copied();
                let (Some(x), Some(y), Some(z)) = (x, y, z) else {
                    return Err(DslError::eval("point slot out of range"));
                };
                stack.push(Value::Vector(Vec3::new(x, y, z)));
            }
            OpCode::Add => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(add(a, b)?);
            }
            OpCode::Sub => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(sub(a, b)?);
            }
            OpCode::Mul => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(mul(a, b)?);
            }
            OpCode::Div => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(div(a, b)?);
            }
            OpCode::Neg => {
                let a = pop(&mut stack)?;
                let negated = match a {
                    Value::Scalar(s) => Value::Scalar(-s),
                    Value::Vector(v) => Value::Vector(-v),
                };
                stack.push(negated);
            }
            OpCode::Call(func) => {
                let result = call(func, &mut stack)?;
                stack.push(result);
            }
        }
    }
    if stack.len() != 1 {
        return Err(DslError::eval("program did not produce a single result"));
    }
    pop(&mut stack)?.as_scalar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::bytecode::OpCode;

    fn prog(ops: Vec<OpCode>) -> Program {
        Program { ops }
    }

    #[test]
    fn evaluates_scalar_arithmetic() {
        // (1 + 2) * 3 = 9
        let p = prog(vec![
            OpCode::PushConst(1.0),
            OpCode::PushConst(2.0),
            OpCode::Add,
            OpCode::PushConst(3.0),
            OpCode::Mul,
        ]);
        assert!((eval(&p, &[]).unwrap() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn length_of_point_difference() {
        // length(b - a) with a=(0,0,0), b=(3,4,0) => 5
        let values = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0];
        let p = prog(vec![
            OpCode::PushPoint(3),
            OpCode::PushPoint(0),
            OpCode::Sub,
            OpCode::Call(BuiltinFn::Length),
        ]);
        assert!((eval(&p, &values).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn division_by_zero_errors() {
        let p = prog(vec![
            OpCode::PushConst(1.0),
            OpCode::PushConst(0.0),
            OpCode::Div,
        ]);
        assert!(matches!(eval(&p, &[]), Err(DslError::Eval { .. })));
    }

    #[test]
    fn sqrt_negative_errors() {
        let p = prog(vec![OpCode::PushConst(-1.0), OpCode::Call(BuiltinFn::Sqrt)]);
        assert!(matches!(eval(&p, &[]), Err(DslError::Eval { .. })));
    }

    #[test]
    fn type_mismatch_errors() {
        // length(scalar) is invalid.
        let p = prog(vec![
            OpCode::PushConst(1.0),
            OpCode::Call(BuiltinFn::Length),
        ]);
        assert!(matches!(eval(&p, &[]), Err(DslError::Eval { .. })));
    }

    #[test]
    fn out_of_range_slot_errors() {
        let p = prog(vec![OpCode::PushScalar(5)]);
        assert!(matches!(eval(&p, &[1.0]), Err(DslError::Eval { .. })));
    }

    #[test]
    fn clamp_and_minmax() {
        let p = prog(vec![
            OpCode::PushConst(5.0),
            OpCode::PushConst(0.0),
            OpCode::PushConst(3.0),
            OpCode::Call(BuiltinFn::Clamp),
        ]);
        assert!((eval(&p, &[]).unwrap() - 3.0).abs() < 1e-6);
    }
}
