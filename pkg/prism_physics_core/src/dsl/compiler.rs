//! Lowering of the constraint AST into stack-machine [`Program`]s.
//!
//! [`compile_constraint`] takes a parsed [`ConstraintDecl`], builds the
//! variable-slot [`Environment`], and lowers both the residual and the
//! compliance expressions into [`Program`]s. Free identifiers that are not
//! declared parameters are treated as *points* (the particles the constraint
//! couples); parameters are scalars. The compliance expression may reference
//! only parameters.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Tree-walk
//! lowering to a postfix instruction stream with a symbol table is standard,
//! publicly documented compiler-construction knowledge.

use crate::dsl::ast::{BinOp, ConstraintDecl, Expr};
use crate::dsl::bytecode::{BuiltinFn, OpCode, Program};
use crate::dsl::env::{Environment, VarKind};
use crate::dsl::error::DslError;

/// A fully compiled constraint: its programs plus the slot layout needed to
/// evaluate them.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompiledConstraint {
    /// The constraint name from the declaration.
    pub name: String,
    /// Scalar parameter names, occupying slots `0..params.len()`.
    pub params: Vec<String>,
    /// Point (particle) names, in first-use order, following the parameters.
    pub point_names: Vec<String>,
    /// The slot layout shared by both programs.
    pub env: Environment,
    /// Compiled residual (constraint function) program.
    pub residual: Program,
    /// Compiled compliance program (references parameters only).
    pub compliance: Program,
}

/// Collects free value identifiers (variables, not function names) in
/// first-use order.
fn collect_free_vars(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Const(_) => {}
        Expr::Var(name) => {
            if !out.iter().any(|n| n == name) {
                out.push(name.clone());
            }
        }
        Expr::Neg(inner) => collect_free_vars(inner, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_free_vars(lhs, out);
            collect_free_vars(rhs, out);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                collect_free_vars(arg, out);
            }
        }
    }
}

/// Lowers a single expression into a [`Program`] using an existing
/// [`Environment`] for variable resolution.
///
/// # Errors
///
/// Returns [`DslError::Compile`] for unknown variables, unknown functions, or
/// argument-count mismatches.
pub fn compile_expr(expr: &Expr, env: &Environment) -> Result<Program, DslError> {
    let mut program = Program::new();
    lower(expr, env, &mut program)?;
    Ok(program)
}

fn lower(expr: &Expr, env: &Environment, program: &mut Program) -> Result<(), DslError> {
    match expr {
        Expr::Const(c) => program.push(OpCode::PushConst(*c)),
        Expr::Var(name) => {
            let binding = env
                .lookup(name)
                .ok_or_else(|| DslError::compile(format!("unknown variable '{name}'")))?;
            match binding.kind {
                VarKind::Scalar => program.push(OpCode::PushScalar(binding.base)),
                VarKind::Point => program.push(OpCode::PushPoint(binding.base)),
            }
        }
        Expr::Neg(inner) => {
            lower(inner, env, program)?;
            program.push(OpCode::Neg);
        }
        Expr::Binary { op, lhs, rhs } => {
            lower(lhs, env, program)?;
            lower(rhs, env, program)?;
            program.push(match op {
                BinOp::Add => OpCode::Add,
                BinOp::Sub => OpCode::Sub,
                BinOp::Mul => OpCode::Mul,
                BinOp::Div => OpCode::Div,
            });
        }
        Expr::Call { func, args } => {
            let builtin = BuiltinFn::from_name(func)
                .ok_or_else(|| DslError::compile(format!("unknown function '{func}'")))?;
            if args.len() != builtin.arity() {
                return Err(DslError::compile(format!(
                    "function '{}' expects {} argument(s), got {}",
                    builtin.name(),
                    builtin.arity(),
                    args.len()
                )));
            }
            for arg in args {
                lower(arg, env, program)?;
            }
            program.push(OpCode::Call(builtin));
        }
    }
    Ok(())
}

/// Compiles a full [`ConstraintDecl`] into a [`CompiledConstraint`].
///
/// Parameters occupy the first slots (as scalars); points follow, in the order
/// they first appear in the residual, each taking three slots. The compliance
/// expression is compiled against the same environment but is rejected if it
/// references any point (non-parameter) variable.
///
/// # Errors
///
/// Returns [`DslError::Compile`] on unknown functions, arity mismatches, or a
/// compliance expression that references a non-parameter variable.
pub fn compile_constraint(decl: &ConstraintDecl) -> Result<CompiledConstraint, DslError> {
    // Determine the point variables: residual free vars that are not params.
    let mut residual_vars = Vec::new();
    collect_free_vars(&decl.residual, &mut residual_vars);
    let point_names: Vec<String> = residual_vars
        .into_iter()
        .filter(|name| !decl.params.iter().any(|p| p == name))
        .collect();

    // Build the environment: parameters first (scalars), then points.
    let mut env = Environment::new();
    for param in &decl.params {
        env.define(param, VarKind::Scalar);
    }
    for point in &point_names {
        env.define(point, VarKind::Point);
    }

    let residual = compile_expr(&decl.residual, &env)?;

    // The compliance expression may reference only parameters.
    let mut compliance_vars = Vec::new();
    collect_free_vars(&decl.compliance, &mut compliance_vars);
    for name in &compliance_vars {
        if !decl.params.iter().any(|p| p == name) {
            return Err(DslError::compile(format!(
                "compliance expression may only reference parameters, found '{name}'"
            )));
        }
    }
    let compliance = compile_expr(&decl.compliance, &env)?;

    Ok(CompiledConstraint {
        name: decl.name.clone(),
        params: decl.params.clone(),
        point_names,
        env,
        residual,
        compliance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parser::parse_constraint;
    use crate::dsl::vm::eval;

    #[test]
    fn compiles_distance_residual() {
        let decl =
            parse_constraint("constraint d(rest) { residual = length(b - a) - rest; }").unwrap();
        let compiled = compile_constraint(&decl).unwrap();
        assert_eq!(compiled.params, vec!["rest".to_string()]);
        // Points appear in first-use order: b before a.
        assert_eq!(compiled.point_names, vec!["b".to_string(), "a".to_string()]);
        // Slots: rest=0, b=1..4, a=4..7 => total 7.
        assert_eq!(compiled.env.total_slots(), 7);
    }

    #[test]
    fn residual_evaluates_correctly() {
        let decl =
            parse_constraint("constraint d(rest) { residual = length(b - a) - rest; }").unwrap();
        let compiled = compile_constraint(&decl).unwrap();
        // rest=2, b=(3,4,0), a=(0,0,0) => length=5, residual=3.
        let mut values = vec![0.0; compiled.env.total_slots()];
        let rest = compiled.env.lookup("rest").unwrap().base;
        values[rest] = 2.0;
        let b = compiled.env.lookup("b").unwrap().base;
        values[b] = 3.0;
        values[b + 1] = 4.0;
        let r = eval(&compiled.residual, &values).unwrap();
        assert!((r - 3.0).abs() < 1e-5);
    }

    #[test]
    fn unknown_function_is_compile_error() {
        let decl = parse_constraint("constraint d() { residual = sin(a); }").unwrap();
        assert!(matches!(
            compile_constraint(&decl),
            Err(DslError::Compile { .. })
        ));
    }

    #[test]
    fn arity_mismatch_is_compile_error() {
        let decl = parse_constraint("constraint d() { residual = length(a, b); }").unwrap();
        assert!(matches!(
            compile_constraint(&decl),
            Err(DslError::Compile { .. })
        ));
    }

    #[test]
    fn compliance_referencing_point_is_error() {
        let decl =
            parse_constraint("constraint d() { residual = length(a); compliance = length(a); }")
                .unwrap();
        assert!(matches!(
            compile_constraint(&decl),
            Err(DslError::Compile { .. })
        ));
    }
}
