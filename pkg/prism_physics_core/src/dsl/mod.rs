//! Declarative constraint DSL with parameter hot-reload.
//!
//! This module implements a small, safe domain-specific language for authoring
//! custom physics constraints as text. The pipeline is a classic compiler
//! front end followed by a stack virtual machine:
//!
//! 1. [`lexer::tokenize`] scans source into tokens.
//! 2. [`parser::parse_constraint`] / [`parser::parse_expression`] build an
//!    [`ast`] tree via recursive descent.
//! 3. [`compiler::compile_constraint`] lowers the tree into stack
//!    [`bytecode`], resolving variables to slots and functions to
//!    [`bytecode::BuiltinFn`].
//! 4. [`vm::eval`] evaluates a compiled [`bytecode::Program`] deterministically.
//! 5. [`constraint::DslConstraint`] plugs a compiled residual into an XPBD
//!    projection using finite-difference gradients.
//! 6. [`param::ParameterStore`] and [`reload::HotReloadRegistry`] provide
//!    range-bounded, versioned live editing for editor sliders.
//!
//! The built-in function set is deliberately restricted to operations that can
//! be evaluated with the engine's allowed scalar primitives (`abs`, `min`,
//! `max`, `clamp`, `sqrt`, `length`, `distance`, `dot`), avoiding disallowed
//! transcendental `f32` methods so evaluation stays deterministic.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! lexer/parser/compiler/VM design is standard compiler-construction
//! knowledge; the projection uses the canonical XPBD update (Macklin et al.
//! 2016).

pub mod ast;
pub mod bytecode;
pub mod compiler;
pub mod constraint;
pub mod env;
pub mod error;
pub mod lexer;
pub mod param;
pub mod parser;
pub mod reload;
pub mod vm;

pub use ast::{BinOp, ConstraintDecl, Expr};
pub use bytecode::{BuiltinFn, OpCode, Program};
pub use compiler::{compile_constraint, compile_expr, CompiledConstraint};
pub use constraint::DslConstraint;
pub use env::{Environment, VarBinding, VarKind};
pub use error::DslError;
pub use lexer::{tokenize, SpannedToken, Token};
pub use param::{ParamSpec, ParamValue, ParameterStore};
pub use parser::{parse_constraint, parse_expression};
pub use reload::HotReloadRegistry;
pub use vm::eval;
