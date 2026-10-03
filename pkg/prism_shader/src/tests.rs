//! Unit tests for the shader composition kernel.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::compose::{ComposeError, ShaderComposer};
use crate::def::{ShaderDefValue, ShaderDefs};
use crate::expr::{ExprError, evaluate};
use crate::module::ShaderModule;
use crate::permutation::PermutationId;
use crate::preprocess::{PreprocessError, preprocess};

fn defs_of(pairs: &[(&str, ShaderDefValue)]) -> ShaderDefs {
    let mut defs = ShaderDefs::new();
    for &(name, value) in pairs {
        defs.insert(name, value);
    }
    defs
}

// ----- def -----------------------------------------------------------------

#[test]
fn defs_insert_get_and_truthiness() {
    let mut defs = ShaderDefs::new();
    assert!(defs.is_empty());
    assert_eq!(defs.insert("A", ShaderDefValue::Int(3)), None);
    assert_eq!(defs.insert("A", ShaderDefValue::Int(4)), Some(ShaderDefValue::Int(3)));
    assert!(defs.define("B").is_none());
    assert_eq!(defs.len(), 2);
    assert!(defs.contains("B"));
    assert_eq!(defs.get("A"), Some(ShaderDefValue::Int(4)));
    assert_eq!(defs.remove("A"), Some(ShaderDefValue::Int(4)));
    assert_eq!(defs.get("A"), None);
    assert!(ShaderDefValue::Bool(true).is_truthy());
    assert!(!ShaderDefValue::Bool(false).is_truthy());
    assert!(!ShaderDefValue::UInt(0).is_truthy());
    assert!(ShaderDefValue::UInt(7).is_truthy());
    assert_eq!(ShaderDefValue::Int(-5).as_i64(), -5);
}

#[test]
fn defs_iterate_in_name_sorted_order() {
    let defs = defs_of(&[
        ("Zebra", ShaderDefValue::Bool(true)),
        ("alpha", ShaderDefValue::Int(1)),
        ("Beta", ShaderDefValue::UInt(2)),
    ]);
    let names: Vec<&str> = defs.iter().map(|(name, _)| name).collect();
    assert_eq!(names, ["Beta", "Zebra", "alpha"]);
}

// ----- permutation ---------------------------------------------------------

#[test]
fn permutation_id_is_order_independent_and_stable() {
    let a = defs_of(&[("A", ShaderDefValue::Int(1)), ("B", ShaderDefValue::Bool(true))]);
    let b = defs_of(&[("B", ShaderDefValue::Bool(true)), ("A", ShaderDefValue::Int(1))]);
    assert_eq!(PermutationId::of(&a), PermutationId::of(&b));
    // Stable across repeated computation.
    assert_eq!(PermutationId::of(&a).get(), PermutationId::of(&a).get());
}

#[test]
fn permutation_id_empty_and_distinctness() {
    assert_eq!(PermutationId::empty(), PermutationId::of(&ShaderDefs::new()));
    let one = defs_of(&[("A", ShaderDefValue::Int(1))]);
    let two = defs_of(&[("A", ShaderDefValue::Int(2))]);
    assert_ne!(PermutationId::of(&one), PermutationId::of(&two));
    // A value's type participates: Int(1) and UInt(1) must not alias.
    let int_one = defs_of(&[("A", ShaderDefValue::Int(1))]);
    let uint_one = defs_of(&[("A", ShaderDefValue::UInt(1))]);
    assert_ne!(PermutationId::of(&int_one), PermutationId::of(&uint_one));
    // Name boundaries are unambiguous: {"ab":1} differs from {"a":?,"b":1}.
    let joined = defs_of(&[("ab", ShaderDefValue::Int(1))]);
    let split = defs_of(&[("a", ShaderDefValue::Int(0)), ("b", ShaderDefValue::Int(1))]);
    assert_ne!(PermutationId::of(&joined), PermutationId::of(&split));
    assert_ne!(PermutationId::empty(), PermutationId::of(&one));
}

// ----- expr -----------------------------------------------------------------

#[test]
fn expr_arithmetic_and_precedence() {
    let defs = ShaderDefs::new();
    assert_eq!(evaluate("1 + 2 * 3", &defs), Ok(7));
    assert_eq!(evaluate("(1 + 2) * 3", &defs), Ok(9));
    assert_eq!(evaluate("10 % 3", &defs), Ok(1));
    assert_eq!(evaluate("-4 + 1", &defs), Ok(-3));
    assert_eq!(evaluate("0x10 + 1", &defs), Ok(17));
    assert_eq!(evaluate("!0", &defs), Ok(1));
    assert_eq!(evaluate("!5", &defs), Ok(0));
}

#[test]
fn expr_comparisons_and_logic() {
    let defs = ShaderDefs::new();
    assert_eq!(evaluate("2 < 3 && 3 <= 3", &defs), Ok(1));
    assert_eq!(evaluate("2 > 3 || 1 == 1", &defs), Ok(1));
    assert_eq!(evaluate("4 != 4", &defs), Ok(0));
    assert_eq!(evaluate("5 >= 6", &defs), Ok(0));
}

#[test]
fn expr_defined_and_bare_idents() {
    let defs = defs_of(&[("QUALITY", ShaderDefValue::Int(2)), ("FEATURE", ShaderDefValue::Bool(true))]);
    assert_eq!(evaluate("defined(QUALITY)", &defs), Ok(1));
    assert_eq!(evaluate("defined MISSING", &defs), Ok(0));
    assert_eq!(evaluate("QUALITY == 2", &defs), Ok(1));
    assert_eq!(evaluate("FEATURE && QUALITY > 1", &defs), Ok(1));
    // Undefined bare identifier is zero.
    assert_eq!(evaluate("UNDEFINED", &defs), Ok(0));
}

#[test]
fn expr_error_cases() {
    let defs = ShaderDefs::new();
    assert_eq!(evaluate("", &defs), Err(ExprError::Empty));
    assert_eq!(evaluate("1 / 0", &defs), Err(ExprError::DivideByZero));
    assert_eq!(evaluate("1 2", &defs), Err(ExprError::TrailingTokens));
    assert_eq!(evaluate("(1 + 2", &defs), Err(ExprError::ExpectedRParen));
    assert!(matches!(evaluate("1 &", &defs), Err(ExprError::IncompleteOperator('&'))));
}

// ----- preprocess -----------------------------------------------------------

#[test]
fn preprocess_ifdef_and_ifndef() {
    let defs = defs_of(&[("FEATURE", ShaderDefValue::Bool(true))]);
    let src = "a\n#ifdef FEATURE\nb\n#endif\n#ifndef FEATURE\nc\n#endif\nd";
    assert_eq!(preprocess(src, &defs).unwrap(), "a\nb\nd");
}

#[test]
fn preprocess_if_elif_else_chain() {
    let defs = defs_of(&[("TIER", ShaderDefValue::Int(2))]);
    let src = "#if TIER == 1\none\n#elif TIER == 2\ntwo\n#else\nother\n#endif";
    assert_eq!(preprocess(src, &defs).unwrap(), "two");
    let defs0 = defs_of(&[("TIER", ShaderDefValue::Int(9))]);
    assert_eq!(preprocess(src, &defs0).unwrap(), "other");
}

#[test]
fn preprocess_nested_and_inactive_parent_suppresses_children() {
    let defs = defs_of(&[("OUTER", ShaderDefValue::Bool(false)), ("INNER", ShaderDefValue::Bool(true))]);
    let src = "#if OUTER\n#if INNER\nx\n#endif\ny\n#endif\nz";
    assert_eq!(preprocess(src, &defs).unwrap(), "z");
}

#[test]
fn preprocess_passes_through_unknown_directives() {
    let defs = ShaderDefs::new();
    let src = "#import common/brdf\nfn main() {}";
    assert_eq!(preprocess(src, &defs).unwrap(), "#import common/brdf\nfn main() {}");
}

#[test]
fn preprocess_error_cases() {
    let defs = ShaderDefs::new();
    assert_eq!(
        preprocess("#endif", &defs),
        Err(PreprocessError::EndifWithoutIf { line: 1 }),
    );
    assert_eq!(
        preprocess("#else", &defs),
        Err(PreprocessError::ElseWithoutIf { line: 1 }),
    );
    assert_eq!(
        preprocess("#if 1\nx", &defs),
        Err(PreprocessError::UnterminatedConditional { line: 1 }),
    );
    assert_eq!(
        preprocess("#ifdef\nx\n#endif", &defs),
        Err(PreprocessError::MissingDefineName { line: 1 }),
    );
    assert!(matches!(
        preprocess("#if 1 /\n#endif", &defs),
        Err(PreprocessError::BadExpression { line: 1, .. })
    ));
}

// ----- module + compose -----------------------------------------------------

#[test]
fn module_parses_imports_and_strips_them_from_body() {
    let module = ShaderModule::new("main", "#import a\n#include \"b\"\nfn f() {}");
    assert_eq!(module.name(), "main");
    assert_eq!(module.imports(), ["a".to_string(), "b".to_string()]);
    assert_eq!(module.body(), "fn f() {}");
}

#[test]
fn compose_resolves_dependencies_in_post_order_with_dedup() {
    let mut composer = ShaderComposer::new();
    composer.add_module(ShaderModule::new("base", "// base")).unwrap();
    composer.add_module(ShaderModule::new("mid", "#import base\n// mid")).unwrap();
    composer
        .add_module(ShaderModule::new("root", "#import mid\n#import base\n// root"))
        .unwrap();
    assert_eq!(composer.len(), 3);
    assert!(composer.contains("mid"));
    let out = composer.compose("root", &ShaderDefs::new()).unwrap();
    // base before mid before root, each appearing exactly once.
    assert_eq!(out, "// base\n// mid\n// root");
}

#[test]
fn compose_runs_preprocessor_over_combined_source() {
    let defs = defs_of(&[("USE_PBR", ShaderDefValue::Bool(true))]);
    let mut composer = ShaderComposer::new();
    composer
        .add_module(ShaderModule::new("lib", "#ifdef USE_PBR\npbr\n#endif"))
        .unwrap();
    composer.add_module(ShaderModule::new("root", "#import lib\nmain")).unwrap();
    assert_eq!(composer.compose("root", &defs).unwrap(), "pbr\nmain");
}

#[test]
fn compose_detects_cycles_and_unknown_modules() {
    let mut composer = ShaderComposer::new();
    composer.add_module(ShaderModule::new("a", "#import b")).unwrap();
    composer.add_module(ShaderModule::new("b", "#import a")).unwrap();
    match composer.compose("a", &ShaderDefs::new()) {
        Err(ComposeError::ImportCycle(path)) => {
            assert_eq!(path.first().map(ToString::to_string), Some("a".to_string()));
            assert_eq!(path.last().map(ToString::to_string), Some("a".to_string()));
        }
        other => panic!("expected cycle, got {other:?}"),
    }
    assert_eq!(
        composer.compose("missing", &ShaderDefs::new()),
        Err(ComposeError::UnknownModule("missing".to_string())),
    );
}

#[test]
fn compose_rejects_duplicate_modules() {
    let mut composer = ShaderComposer::new();
    composer.add_module(ShaderModule::new("x", "a")).unwrap();
    assert_eq!(
        composer.add_module(ShaderModule::new("x", "b")),
        Err(ComposeError::DuplicateModule("x".to_string())),
    );
}
