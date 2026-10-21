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

use std::borrow::Cow;
use std::mem::size_of;
use std::path::Path;

/// Extract `(name, type)` pairs from a `struct <name> { … }` body in WESL.
fn wesl_fields(source: &str, struct_name: &str) -> Vec<(String, String)> {
    let marker = format!("struct {struct_name} {{");
    assert_eq!(
        source.matches(marker.as_str()).count(),
        1,
        "{struct_name} must have exactly one declaration in the WESL source"
    );
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

fn mirror_source(file: &str) -> Cow<'static, str> {
    if file == "material.wesl" {
        return Cow::Borrowed(prism_render_material::shaders::MATERIAL_SHADER_SOURCE);
    }
    if file == "material_classification.wesl" {
        return Cow::Borrowed(
            prism_render_material::shaders::MATERIAL_CLASSIFICATION_SHADER_SOURCE,
        );
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../prism_render_scene/src/shaders")
        .join(file);
    Cow::Owned(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
}

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

    for (file, struct_name) in MIRRORS {
        let source = mirror_source(file);
        let mirror = wesl_fields(&source, struct_name);
        assert_eq!(
            mirror, rust,
            "{file}'s {struct_name} mirror drifted from GpuMaterialHeader: WESL binds \
             storage-buffer fields positionally, so the names and order must match exactly"
        );
    }
}

#[test]
fn dynamic_channel_header_words_follow_schema_order() {
    let rust = rust_fields(include_str!("../src/record.rs"));
    let expected = prism_render_material::SURFACE_CHANNELS
        .iter()
        .map(|channel| format!("dynamic_{}", channel.id))
        .collect::<Vec<_>>();
    let actual = rust
        .iter()
        .filter_map(|(name, _)| name.strip_prefix("dynamic_").map(|_| name.clone()))
        .collect::<Vec<_>>();

    assert_eq!(
        actual, expected,
        "Rust dynamic header words drifted from schema order"
    );
    for (index, channel) in prism_render_material::SURFACE_CHANNELS.iter().enumerate() {
        assert_eq!(
            channel.header_word, index as u32,
            "{} header word",
            channel.id
        );
    }

    for (file, struct_name) in MIRRORS {
        let source = mirror_source(file);
        let mirror = wesl_fields(&source, struct_name);
        let dynamic = mirror
            .iter()
            .filter_map(|(name, _)| name.strip_prefix("dynamic_").map(|_| name.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            dynamic, expected,
            "{file} dynamic channel order drifted from schema"
        );
    }
}

#[test]
fn abi8_execution_binding_replaces_legacy_specialization_packing() {
    let rust = rust_fields(include_str!("../src/record.rs"));
    let execution_fields = rust
        .iter()
        .filter(|(name, _)| name.starts_with("execution_"))
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        execution_fields,
        [
            "execution_plan_index",
            "execution_program_index",
            "execution_package_index",
            "execution_permutation_low",
            "execution_permutation_high",
        ]
    );
    assert!(
        rust.iter()
            .all(|(name, _)| !name.contains("specialization")),
        "legacy SpecializationId words must not re-enter the material ABI"
    );

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

    let unbound = prism_render_material::fallback_material_header(0);
    assert_eq!(
        unbound.execution_binding_words(),
        [prism_render_material::GpuMaterialHeader::EXECUTION_BINDING_SENTINEL; 5]
    );
    assert!(!unbound.has_execution_binding());
    assert!(header.has_execution_binding());
}

#[test]
fn material_classification_consumes_execution_identity_not_surface_payload() {
    let source = prism_render_material::shaders::MATERIAL_CLASSIFICATION_SHADER_SOURCE;

    for required in [
        "header.execution_plan_index",
        "header.execution_program_index",
        "header.execution_package_index",
        "header.execution_permutation_low",
        "header.execution_permutation_high",
        "execution_plans",
        "execution_programs",
        "execution_packages",
        "response_buckets",
        "visibility_metadata",
    ] {
        assert!(
            source.contains(required),
            "classification lost `{required}`"
        );
    }
    for forbidden in [
        "PrismSurfaceParameters",
        "material_unpack",
        "parameter_heap",
        "header.parameter_offset",
        "header.parameter_size",
        "header.closure_mask",
        "header.lobe_mask",
    ] {
        assert!(
            !source.contains(forbidden),
            "classification must not derive work identity from Surface payload via `{forbidden}`"
        );
    }

    let region_bit = prism_render_material::MaterialFeatureFlags::REGION_ILLUMINATION
        .0
        .trailing_zeros();
    let blend_bit = prism_render_material::MaterialFeatureFlags::ILLUMINATION_BLEND
        .0
        .trailing_zeros();
    assert!(source.contains(&format!(
        "const FEATURE_REGION_ILLUMINATION: u32 = 1u << {region_bit}u;"
    )));
    assert!(source.contains(&format!(
        "const FEATURE_ILLUMINATION_BLEND: u32 = 1u << {blend_bit}u;"
    )));
}
