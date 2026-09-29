//! Integration tests for milestone M8: the constraint DSL and hot reload.
//!
//! These exercises drive the public API only: parse and compile a residual,
//! evaluate it in the VM, project it as an XPBD constraint, verify lexer and
//! parser error paths, and drive the hot-reload registry and parameter store.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It tests
//! standard compiler-construction and XPBD behaviour.

use glam::Vec3;

use prism_physics_core::dsl::{
    compile_constraint, eval, parse_constraint, parse_expression, DslConstraint, DslError,
    HotReloadRegistry, ParamValue, ParameterStore,
};

const EPS: f32 = 1e-4;

#[test]
fn parse_compile_and_evaluate_distance_residual() {
    let decl =
        parse_constraint("constraint distance(rest) { residual = length(b - a) - rest; }").unwrap();
    let compiled = compile_constraint(&decl).unwrap();

    let mut values = vec![0.0_f32; compiled.env.total_slots()];
    let rest = compiled.env.lookup("rest").unwrap().base;
    let a = compiled.env.lookup("a").unwrap().base;
    let b = compiled.env.lookup("b").unwrap().base;
    values[rest] = 2.0;
    // a = (0,0,0), b = (3,4,0) => length = 5, residual = 3.
    values[a] = 0.0;
    values[a + 1] = 0.0;
    values[a + 2] = 0.0;
    values[b] = 3.0;
    values[b + 1] = 4.0;
    values[b + 2] = 0.0;

    let result = eval(&compiled.residual, &values).unwrap();
    assert!((result - 3.0).abs() < EPS, "residual was {result}");
}

#[test]
fn dsl_constraint_projection_reduces_residual() {
    let decl =
        parse_constraint("constraint distance(rest) { residual = length(b - a) - rest; }").unwrap();
    let compiled = compile_constraint(&decl).unwrap();
    // Compiled point order is [b, a]; bind b -> particle 1, a -> particle 0.
    let mut constraint = DslConstraint::from_compiled(&compiled, &[1.0], &[1, 0]).unwrap();

    let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
    let inverse_masses = [1.0_f32, 1.0_f32];

    let before = constraint.residual_value(&positions).unwrap().abs();
    constraint.project(&mut positions, &inverse_masses, 1.0 / 60.0);
    let after = constraint.residual_value(&positions).unwrap().abs();

    assert!(after < before, "expected {after} < {before}");
}

#[test]
fn lexer_and_parser_errors_do_not_panic() {
    // Illegal character (lexical error).
    let lex_err = parse_expression("a $ b");
    assert!(matches!(lex_err, Err(DslError::Lex { .. })));

    // Unbalanced parenthesis (syntax error).
    let parse_err = parse_expression("(a + 1");
    assert!(matches!(parse_err, Err(DslError::Parse { .. })));

    // Missing residual (syntax error at declaration level).
    let decl_err = parse_constraint("constraint d() { compliance = 1; }");
    assert!(matches!(decl_err, Err(DslError::Parse { .. })));
}

#[test]
fn hot_reload_registry_versions_and_preserves_on_failure() {
    let mut store = ParameterStore::new();
    store.define_real("rest", 1.0, 0.0, 10.0);
    let mut registry = HotReloadRegistry::new(
        "constraint d(rest) { residual = length(b - a) - rest; }",
        store,
    )
    .unwrap();

    // set_param bumps the version and updates the value.
    let v0 = registry.version();
    registry.set_param("rest", ParamValue::Real(4.0)).unwrap();
    assert_eq!(registry.version(), v0 + 1);
    assert!((registry.params().get_real("rest").unwrap() - 4.0).abs() < EPS);

    // reload_source with a different expression changes behaviour.
    let v1 = registry.version();
    let before = registry.compiled().clone();
    registry
        .reload_source("constraint d(rest) { residual = distance(a, b) - rest; }")
        .unwrap();
    assert_eq!(registry.version(), v1 + 1);
    assert_ne!(registry.compiled(), &before);

    // A broken source is rejected and leaves the previous program untouched.
    let v2 = registry.version();
    let good = registry.compiled().clone();
    let err = registry.reload_source("constraint d( { residual =");
    assert!(err.is_err());
    assert_eq!(registry.version(), v2);
    assert_eq!(registry.compiled(), &good);
}

#[test]
fn parameter_store_clamps_out_of_range() {
    let mut store = ParameterStore::new();
    store.define_real("stiffness", 0.5, 0.0, 1.0);

    assert!(store.set_real("stiffness", 10.0));
    assert!((store.get_real("stiffness").unwrap() - 1.0).abs() < EPS);

    assert!(store.set_real("stiffness", -5.0));
    assert!((store.get_real("stiffness").unwrap() - 0.0).abs() < EPS);
}
