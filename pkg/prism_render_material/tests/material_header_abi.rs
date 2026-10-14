//! The `GpuMaterialHeader` ABI mirror gate (design doc §8.2, roadmap P1-b).
//!
//! The material header is declared once in Rust (`record.rs`) and mirrored by
//! six WESL shaders that cannot `import` the Rust type: `material.wesl`'s
//! canonical `PrismMaterialHeader`, plus the standalone `MaterialHeader`
//! mirrors in `opaque.wesl`, `transparent.wesl`, `visibility.wesl`,
//! `visibility_raster.wesl` and `material_classification.wesl`. WESL binds
//! storage-buffer struct fields
//! **positionally**, so a field added, removed or reordered in one place and not
//! the other silently shifts every subsequent word — the exact drift §8.2
//! demands be impossible.
//!
//! This gate is a pure source-text parity check, so it maintains itself: it
//! parses the Rust struct body and every WESL mirror body and asserts they
//! declare the same `u32` fields in the same order, and that the field count
//! matches `size_of::<GpuMaterialHeader>()`. Editing either side without the
//! other fails the build.

use std::mem::size_of;
use std::path::Path;

/// Extract `(name, type)` pairs from a `struct <name> { … }` body in WESL.
fn wesl_fields(source: &str, struct_name: &str) -> Vec<(String, String)> {
    let marker = format!("struct {struct_name} {{");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("{struct_name} not found in source"))
        + marker.len();
    let end = source[start..]
        .find("\n}")
        .unwrap_or_else(|| panic!("{struct_name} has no closing brace"))
        + start;
    fields_in_body(&source[start..end])
}

/// Extract `pub (name, type)` pairs from the `GpuMaterialHeader` Rust struct.
fn rust_fields(source: &str) -> Vec<(String, String)> {
    let marker = "pub struct GpuMaterialHeader {";
    let start = source.find(marker).expect("GpuMaterialHeader not found") + marker.len();
    let end = source[start..]
        .find("\n}")
        .expect("GpuMaterialHeader close")
        + start;
    let body = &source[start..end];
    let mut fields = Vec::new();
    for line in body.lines() {
        let line = strip_comment(line).trim();
        let Some(rest) = line.strip_prefix("pub ") else {
            continue;
        };
        let Some((name, ty)) = rest.split_once(':') else {
            continue;
        };
        fields.push((
            name.trim().to_string(),
            ty.trim().trim_end_matches(',').to_string(),
        ));
    }
    fields
}

/// Split a struct body into `name: type` declarations, ignoring comments.
fn fields_in_body(body: &str) -> Vec<(String, String)> {
    let cleaned: String = body
        .lines()
        .map(strip_comment)
        .collect::<Vec<_>>()
        .join("\n");
    let mut fields = Vec::new();
    for raw in cleaned.split(',') {
        let decl = raw.trim();
        if decl.is_empty() {
            continue;
        }
        let Some((name, ty)) = decl.split_once(':') else {
            continue;
        };
        let (name, ty) = (name.trim(), ty.trim());
        if name.is_empty() || ty.is_empty() {
            continue;
        }
        fields.push((name.to_string(), ty.to_string()));
    }
    fields
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(index) => &line[..index],
        None => line,
    }
}

/// Every WESL shader that mirrors the header, with its struct name.
const MIRRORS: [(&str, &str); 6] = [
    ("material.wesl", "PrismMaterialHeader"),
    ("opaque.wesl", "MaterialHeader"),
    ("transparent.wesl", "MaterialHeader"),
    ("visibility.wesl", "MaterialHeader"),
    ("visibility_raster.wesl", "MaterialHeader"),
    ("material_classification.wesl", "MaterialHeader"),
];

#[test]
fn every_wesl_material_header_mirrors_the_rust_struct_exactly() {
    let rust = rust_fields(include_str!("../src/record.rs"));
    assert!(!rust.is_empty(), "parsed no fields from the Rust struct");
    assert_eq!(
        rust.len() * size_of::<u32>(),
        size_of::<prism_render_material::GpuMaterialHeader>(),
        "the header is a dense array of u32 words, so its size must be 4x its field count"
    );
    for (name, ty) in &rust {
        assert_eq!(ty, "u32", "header field {name} must be a u32 word");
    }

    let shaders = Path::new(env!("CARGO_MANIFEST_DIR")).join("../prism_render_scene/src/shaders");
    for (file, struct_name) in MIRRORS {
        let source = std::fs::read_to_string(shaders.join(file))
            .unwrap_or_else(|error| panic!("read {file}: {error}"));
        let mirror = wesl_fields(&source, struct_name);
        assert_eq!(
            mirror, rust,
            "{file}'s {struct_name} mirror drifted from GpuMaterialHeader: WESL binds \
             storage-buffer fields positionally, so the names and order must match exactly"
        );
    }
}

#[test]
fn abi7_execution_binding_is_five_dense_words_with_full_permutation_width() {
    let header = prism_render_material::GpuMaterialHeader {
        execution_plan_index: 7,
        execution_program_index: 11,
        execution_package_index: 13,
        execution_permutation_low: 0x7654_3210,
        execution_permutation_high: 0xfedc_ba98,
        ..Default::default()
    };
    assert_eq!(
        header.execution_binding_words(),
        [7, 11, 13, 0x7654_3210, 0xfedc_ba98]
    );
    assert_eq!(
        header.execution_permutation_low as u64
            | (u64::from(header.execution_permutation_high) << 32),
        0xfedc_ba98_7654_3210
    );
    assert_eq!(
        std::mem::size_of::<prism_render_material::GpuMaterialHeader>(),
        29 * std::mem::size_of::<u32>()
    );
}
