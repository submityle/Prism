//! CPU/GPU mirror gate for the schema-backed fat surface.
//!
//! `GpuSurfaceParameters` and WESL `PrismSurfaceParameters` are separate
//! declarations because the shader source cannot import Rust types. They are
//! still one ABI: a field insertion, reorder, rename or vector/scalar change
//! must fail here before a packed surface can be decoded with shifted values.

use std::collections::BTreeMap;
use std::mem::size_of;

use prism_material_schema::{surface_schema, FieldType};
use prism_render_material::GpuSurfaceParameters;

fn strip_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(code, _)| code)
}

fn struct_body<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"))
        + marker.len();
    let end = source[start..]
        .find("\n}")
        .unwrap_or_else(|| panic!("missing closing brace for {marker}"))
        + start;
    &source[start..end]
}

fn rust_fields(source: &str) -> Vec<(String, String)> {
    struct_body(source, "pub struct GpuSurfaceParameters {")
        .lines()
        .filter_map(|line| {
            let line = strip_comment(line).trim();
            let line = line.strip_prefix("pub ")?;
            let (name, ty) = line.split_once(':')?;
            Some((
                name.trim().to_string(),
                ty.trim().trim_end_matches(',').to_string(),
            ))
        })
        .collect()
}

fn wesl_fields(source: &str) -> Vec<(String, String)> {
    let body = struct_body(source, "struct PrismSurfaceParameters {");
    let cleaned = body
        .lines()
        .map(strip_comment)
        .collect::<Vec<_>>()
        .join("\n");
    cleaned
        .split(',')
        .filter_map(|declaration| {
            let (name, ty) = declaration.trim().split_once(':')?;
            Some((name.trim().to_string(), ty.trim().to_string()))
        })
        .collect()
}

fn rust_storage_type(ty: &str) -> &'static str {
    match ty {
        "f32" => "f32",
        "[f32; 4]" => "vec4f",
        other => panic!("unsupported Rust surface ABI type {other:?}"),
    }
}

fn wesl_storage_type(ty: &str) -> &'static str {
    match ty {
        "f32" => "f32",
        "vec4<f32>" => "vec4f",
        other => panic!("unsupported WESL surface ABI type {other:?}"),
    }
}

fn storage_words(ty: &str) -> u32 {
    match ty {
        "f32" | "f16" => 1,
        "vec4f" => 4,
        other => panic!("unsupported schema surface ABI type {other:?}"),
    }
}

fn schema_surface_fields() -> BTreeMap<String, &'static str> {
    let schema = surface_schema();
    schema
        .core
        .fields
        .iter()
        .filter(|field| field.ty != FieldType::Pad)
        .map(|field| (field.surface_field().to_string(), field.ty.as_str()))
        .chain(schema.canonical_lobes().into_iter().flat_map(|lobe| {
            lobe.fields
                .iter()
                .filter(|field| field.ty != FieldType::Pad)
                .map(|field| (field.surface_field().to_string(), field.ty.as_str()))
        }))
        .collect()
}

#[test]
fn rust_and_wesl_surface_mirrors_match_in_order_and_type() {
    let rust = rust_fields(include_str!("../src/record.rs"));
    let wesl = wesl_fields(prism_render_material::shaders::MATERIAL_SHADER_SOURCE);
    assert!(!rust.is_empty());
    assert_eq!(
        rust.len(),
        wesl.len(),
        "CPU/GPU surface field count drifted"
    );
    for (index, ((rust_name, rust_ty), (wesl_name, wesl_ty))) in
        rust.iter().zip(wesl.iter()).enumerate()
    {
        assert_eq!(
            rust_name, wesl_name,
            "surface field {index} was reordered or renamed"
        );
        assert_eq!(
            rust_storage_type(rust_ty),
            wesl_storage_type(wesl_ty),
            "surface field {rust_name} changed its exact scalar/vector storage type"
        );
    }
}

#[test]
fn surface_mirrors_cover_exactly_the_schema_fields_and_bytes() {
    let schema = schema_surface_fields();
    let rust = rust_fields(include_str!("../src/record.rs"));
    let actual = rust
        .iter()
        .map(|(name, ty)| (name.clone(), rust_storage_type(ty)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(actual, schema, "fat surface mirror is not schema-complete");

    let words = schema.values().map(|ty| storage_words(ty)).sum::<u32>() as usize;
    assert_eq!(
        size_of::<GpuSurfaceParameters>(),
        words * size_of::<u32>(),
        "Rust fat surface size no longer matches schema field widths"
    );
}
