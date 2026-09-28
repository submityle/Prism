//! Integration test against a real `slangc -reflection-json` fixture.
//!
//! The fixture was produced by `slangc 2026.18.3` from the closure proof-of-concept module.
//! Parsing it here proves the reflection parser and codegen handle actual
//! compiler output, not just hand-written samples.

use prism_render_slang::codegen::{generate, generate_struct, CodegenOptions};
use prism_render_slang::reflection::model::{FieldType, Scalar};
use prism_render_slang::reflection::parse_reflection;

const FIXTURE: &str = include_str!("fixtures/closure.reflect.json");

#[test]
fn parses_real_params_struct() {
    let model = parse_reflection(FIXTURE).expect("fixture parses");
    let params = model
        .struct_by_name("Params")
        .expect("Params struct present in fixture");

    // Layout observed from the fixture: vec3 fields are 16-byte aligned.
    assert_eq!(params.size, 64);
    assert_eq!(params.alignment, 16);

    let by_name = |name: &str| {
        params
            .fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("field {name} present"))
    };

    let base_color = by_name("baseColor");
    assert_eq!(base_color.offset, 0);
    assert_eq!(base_color.size, 12);
    assert_eq!(
        base_color.ty,
        FieldType::Vector {
            elem: Scalar::F32,
            count: 3
        }
    );

    let roughness = by_name("roughness");
    assert_eq!(roughness.offset, 12);
    assert_eq!(roughness.ty, FieldType::Scalar(Scalar::F32));

    assert_eq!(by_name("n").offset, 16);
    assert_eq!(by_name("l").offset, 32);
    assert_eq!(by_name("v").offset, 48);
}

#[test]
fn generates_layout_guarded_bindings() {
    let model = parse_reflection(FIXTURE).expect("fixture parses");
    let params = model.struct_by_name("Params").unwrap();
    let src = generate_struct(params, &CodegenOptions::default());

    // The generated struct must be repr(C, align(16)) and carry compile-time
    // guards on total size and each field offset.
    assert!(src.contains("#[repr(C, align(16))]"));
    assert!(src.contains("pub struct Params"));
    assert!(src.contains("size_of::<Params>() == 64"));
    assert!(src.contains("offset_of!(Params, base_color) == 0"));
    assert!(src.contains("offset_of!(Params, roughness) == 12"));
    assert!(src.contains("offset_of!(Params, n) == 16"));
    assert!(src.contains("offset_of!(Params, l) == 32"));
    assert!(src.contains("offset_of!(Params, v) == 48"));

    // Padding must appear where vec3 fields leave 4-byte gaps.
    assert!(src.contains("_pad"));
}

#[test]
fn whole_module_generation_is_deterministic() {
    let model = parse_reflection(FIXTURE).expect("fixture parses");
    let a = generate(&model, &CodegenOptions::default());
    let b = generate(&model, &CodegenOptions::default());
    assert_eq!(a, b, "codegen must be deterministic");
    assert!(a.contains("MATERIAL_ABI_VERSION"));
}
