//! S2 freshness gate for the schema-driven code generators (design doc §17.5.8).
//!
//! Stage S2 flips the schema from a parity *checker* into the *generator* of
//! two committed, hand-reviewed twins:
//!
//! * the GPU word-heap decoder
//!   `prism_render_scene/src/shaders/material_unpack.wesl`, and
//! * the CPU layout facts `prism_render_material/src/surface_layout.rs`.
//!
//! Both are pure functions of `schema/surface.toml`, so the committed files
//! must be byte-identical to a fresh regeneration. If anyone edits a generated
//! file by hand, or edits the schema/emitter without re-running the generator,
//! this test fails with a clear "run the codegen" message instead of letting
//! stale generated code ship. Together with the S1 `schema_parity` gate (which
//! proves the hand-written `surface.rs` image equals the schema for all 2^N
//! masks), the schema -> Rust layout -> WESL chain stays transitively exact.

use prism_material_schema::codegen::{
    emit_material_classification, emit_rust_layout, emit_schema_contract, emit_wesl_unpack,
};
use prism_material_schema::surface_schema;
use prism_render_material::{ClosureKind, SURFACE_LOBE_CLOSURES};

/// Committed GPU decoder, read at compile time so the test has no runtime path
/// dependency. Relative to this test file in `prism_render_material/tests/`.
const COMMITTED_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/material_unpack.wesl");

/// Committed CPU layout module (lives next to the typed structs it guards).
const COMMITTED_RUST_LAYOUT: &str = include_str!("../src/surface_layout.rs");
const COMMITTED_SCHEMA_CONTRACT: &str =
    include_str!("../../prism_material_schema/src/generated_contract.rs");
const COMMITTED_CLASSIFICATION: &str =
    include_str!("../../prism_render_scene/src/shaders/material_classification.wesl");
const HAND_WRITTEN_SURFACE: &str = include_str!("../src/surface.rs");

#[test]
fn committed_wesl_matches_generator() {
    let fresh = emit_wesl_unpack(&surface_schema());
    assert_eq!(
        COMMITTED_WESL, fresh,
        "material_unpack.wesl is stale; regenerate with \
         `cargo run -p prism_material_schema --bin prism-material-codegen -- write`"
    );
}

#[test]
fn committed_rust_layout_matches_generator() {
    let fresh = emit_rust_layout(&surface_schema());
    assert_eq!(
        COMMITTED_RUST_LAYOUT, fresh,
        "surface_layout.rs is stale; regenerate with \
         `cargo run -p prism_material_schema --bin prism-material-codegen -- write`"
    );
}

#[test]
fn committed_schema_contract_matches_generator() {
    let fresh = emit_schema_contract(&surface_schema());
    assert_eq!(
        COMMITTED_SCHEMA_CONTRACT, fresh,
        "generated_contract.rs is stale; regenerate with `cargo run -p prism_material_schema --bin prism-material-codegen -- write`"
    );
}

#[test]
fn committed_classification_metadata_matches_generator() {
    let schema = surface_schema();
    let fresh = emit_material_classification(&schema, COMMITTED_CLASSIFICATION);
    assert_eq!(
        COMMITTED_CLASSIFICATION, fresh,
        "material_classification.wesl is stale; regenerate with `cargo run -p prism_material_schema --bin prism-material-codegen -- write`"
    );
}

#[test]
fn generated_field_directory_covers_exactly_non_padding_schema_fields() {
    use prism_material_schema::FieldType;
    use prism_render_material::SURFACE_FIELDS;

    let schema = surface_schema();
    let expected = schema
        .core
        .fields
        .iter()
        .filter(|field| field.ty != FieldType::Pad)
        .map(|field| {
            (
                None,
                field.name.as_str(),
                field.surface_field(),
                field.ty.as_str(),
                field
                    .consumed_by
                    .iter()
                    .map(|step| step.as_str())
                    .collect::<Vec<_>>(),
            )
        })
        .chain(schema.canonical_lobes().into_iter().flat_map(|lobe| {
            lobe.fields
                .iter()
                .filter(|field| field.ty != FieldType::Pad)
                .map(move |field| {
                    (
                        Some(lobe.id.as_str()),
                        field.name.as_str(),
                        field.surface_field(),
                        field.ty.as_str(),
                        field
                            .consumed_by
                            .iter()
                            .map(|step| step.as_str())
                            .collect::<Vec<_>>(),
                    )
                })
        }))
        .collect::<Vec<_>>();
    let actual = SURFACE_FIELDS
        .iter()
        .map(|field| {
            (
                field.lobe,
                field.name,
                field.surface_field,
                field.storage_type,
                field.consumed_by.to_vec(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(actual
        .iter()
        .all(|(_, name, _, _, _)| !name.starts_with("_pad")));
}

#[test]
fn generated_unpack_covers_every_schema_default_once() {
    let schema = surface_schema();
    let generated = emit_wesl_unpack(&schema);
    let start = generated
        .find("fn prism_neutral_surface()")
        .expect("generated neutral-surface function");
    let end = generated[start..]
        .find("fn prism_unpack_surface")
        .map(|offset| start + offset)
        .expect("generated unpack function");
    let actual = generated[start..end]
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("s.")?;
            let (name, _) = line.split_once(" = ")?;
            Some(name.to_string())
        })
        .collect::<Vec<_>>();
    let mut expected = schema
        .core
        .fields
        .iter()
        .filter(|field| field.ty != prism_material_schema::FieldType::Pad)
        .map(|field| field.surface_field().to_string())
        .chain(schema.canonical_lobes().into_iter().flat_map(|lobe| {
            lobe.fields
                .iter()
                .filter(|field| field.ty != prism_material_schema::FieldType::Pad)
                .map(|field| field.surface_field().to_string())
        }))
        .collect::<Vec<_>>();
    expected.sort();
    let mut actual_sorted = actual;
    actual_sorted.sort();
    assert_eq!(actual_sorted, expected);
}

#[test]
fn lobe_mask_derived_layout_is_generated() {
    let fresh = emit_rust_layout(&surface_schema());
    for generated_fact in [
        "pub const EMISSION: Self = Self(1u32 << 0);",
        "pub const KHR_REGION: Self = Self(",
        "pub const EXTENSION_REGION: Self =\n        Self(",
        "pub const COUNT: u32 = 10;",
        "pub const fn all() -> Self",
        "pub(crate) const fn canonical()",
    ] {
        assert!(
            fresh.contains(generated_fact),
            "generated layout missing `{generated_fact}`"
        );
        assert!(
            !HAND_WRITTEN_SURFACE.contains(generated_fact),
            "derived LobeMask fact `{generated_fact}` must not be hand-written"
        );
    }
}

#[test]
fn handwritten_surface_has_no_schema_derived_layout_definitions() {
    let schema = surface_schema();
    let implementation_start = HAND_WRITTEN_SURFACE
        .find("impl LobeMask {")
        .map(|offset| offset + "impl LobeMask {".len())
        .expect("hand-written LobeMask implementation");
    let implementation_end = HAND_WRITTEN_SURFACE[implementation_start..]
        .find("\n}\n\n/// Which of the two disjoint bit regions")
        .map(|offset| implementation_start + offset)
        .expect("LobeMask implementation end");
    let lobe_impl = &HAND_WRITTEN_SURFACE[implementation_start..implementation_end];
    let has_hand_written_associated_const = lobe_impl.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("pub const ") && !line.starts_with("pub const fn ")
    });
    assert!(
        !has_hand_written_associated_const,
        "LobeMask associated constants must come from generated surface_layout.rs"
    );
    assert!(
        !lobe_impl.contains("1u32 <<"),
        "LobeMask bit arithmetic must not be hand-written"
    );
    for lobe in schema.canonical_lobes() {
        let declaration = format!("pub const {}: Self", lobe.mask_const);
        assert!(
            !HAND_WRITTEN_SURFACE.contains(&declaration),
            "derived lobe bit `{declaration}` must not be hand-written"
        );
    }
    for derived_name in [
        "SURFACE_CORE_WORDS",
        "SURFACE_LOBE_WORDS_BY_SLOT",
        "SURFACE_LOBE_COUNT",
        "SURFACE_MAX_WORDS",
    ] {
        assert!(
            !HAND_WRITTEN_SURFACE.contains(&format!("pub const {derived_name}")),
            "derived layout constant `{derived_name}` must be generated"
        );
    }
}

#[test]
fn generated_layout_exports_schema_versions_and_field_metadata() {
    let schema = surface_schema();
    assert_eq!(
        prism_render_material::SURFACE_SCHEMA_VERSION,
        schema.schema_version
    );
    assert_eq!(
        prism_render_material::SURFACE_ABI_VERSION,
        schema.abi_version
    );

    let fresh = emit_rust_layout(&schema);
    for generated_fact in [
        "pub const SURFACE_SCHEMA_VERSION: u32 = 3;",
        "pub const SURFACE_ABI_VERSION: u32 = 8;",
        "pub mask_bits: u32,",
        "pub surface_field: &'static str,",
        "pub storage_type: &'static str,",
        "pub pack_slot: u32,",
        "pub words: u32,",
        "pub bit_region: &'static str,",
        "pub consumed_by: &'static [&'static str],",
    ] {
        assert!(
            fresh.contains(generated_fact),
            "generated layout missing `{generated_fact}`"
        );
    }
}

#[test]
fn rust_layout_keeps_singleton_tables_rustfmt_stable() {
    let fresh = emit_rust_layout(&surface_schema());
    let singleton_start = fresh
        .find("pub const SURFACE_CACHE_AUX: [CacheAuxLayout; 1] = [")
        .expect("generated layout must contain the cache-aux table");
    let singleton_end = fresh[singleton_start..]
        .find("];\n")
        .map(|offset| singleton_start + offset + 3)
        .expect("generated cache-aux table must be terminated");
    let singleton = &fresh[singleton_start..singleton_end];

    assert!(singleton.contains("[CacheAuxLayout {"));
    assert!(!singleton.contains("[\n    CacheAuxLayout {"));
}

#[test]
fn every_declared_lobe_closure_has_a_lobe() {
    let schema = surface_schema();
    assert_eq!(SURFACE_LOBE_CLOSURES.len(), schema.lobes.len());
    let mut seen = std::collections::BTreeSet::new();
    for &(id, implementation) in &SURFACE_LOBE_CLOSURES {
        assert!(
            seen.insert(id),
            "duplicate implementation relationship for {id}"
        );
        let lobe = schema
            .lobes
            .iter()
            .find(|lobe| lobe.id == id)
            .expect("implementation must reference a declared lobe");
        assert_eq!(lobe.closure.as_deref(), Some(implementation));
    }
    assert!(schema
        .lobes
        .iter()
        .all(|lobe| seen.contains(lobe.id.as_str())));
}

#[test]
fn closure_authoring_discriminants_remain_stable() {
    let kinds = [
        ClosureKind::Diffuse,
        ClosureKind::Conductor,
        ClosureKind::Dielectric,
        ClosureKind::ClearCoat,
        ClosureKind::Sheen,
        ClosureKind::Subsurface,
        ClosureKind::Transmission,
        ClosureKind::Emission,
        ClosureKind::Hair,
        ClosureKind::Volume,
        ClosureKind::Npr,
        ClosureKind::Custom,
        ClosureKind::Mix,
        ClosureKind::Layer,
    ];
    for (bit, kind) in kinds.into_iter().enumerate() {
        assert_eq!(kind as usize, bit, "authoring closure ABI changed");
    }
}

#[test]
fn unknown_closure_implementation_fails_rust_compilation() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Compile the actual authoring enum and the actual generated reference
    // table. This intentionally does not keep a second spelling registry.
    let ir = include_str!("../src/ir.rs");
    let enum_start = ir.find("pub enum ClosureKind {").expect("authoring enum");
    let enum_end = enum_start + ir[enum_start..].find('}').expect("enum end") + 1;
    let declaration = &ir[enum_start..enum_end];
    let temp = std::env::temp_dir().join(format!(
        "prism-closure-compile-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos()
    ));
    std::fs::create_dir(&temp).expect("create isolated compiler output");
    struct CompilerOutput(std::path::PathBuf);
    impl Drop for CompilerOutput {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let output_guard = CompilerOutput(temp);
    let compile = |schema: &prism_material_schema::SurfaceSchema| {
        let layout = emit_rust_layout(schema);
        let table_start = layout
            .find("const _: &[crate::ir::ClosureKind] = &[")
            .expect("generated implementation type check");
        let table_end =
            table_start + layout[table_start..].find("];\n").expect("type check end") + 3;
        let source = format!(
            "mod ir {{ {declaration} }}\n{}",
            &layout[table_start..table_end]
        );
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let mut child = Command::new(rustc)
            .args([
                "--crate-name",
                "closure_gate",
                "--crate-type",
                "lib",
                "--emit=metadata",
                "--out-dir",
            ])
            .arg(&output_guard.0)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run Rust compiler for implementation gate");
        child
            .stdin
            .take()
            .expect("compiler stdin")
            .write_all(source.as_bytes())
            .expect("send generated implementation references");
        child.wait_with_output().expect("compiler result")
    };
    let mut schema = surface_schema();
    let valid = compile(&schema);
    assert!(
        valid.status.success(),
        "valid implementation table failed: {}",
        String::from_utf8_lossy(&valid.stderr)
    );
    schema.lobes[0].closure = Some("MissingClosureImplementation".into());
    let invalid = compile(&schema);
    assert!(!invalid.status.success(), "unknown implementation compiled");
    let diagnostic = String::from_utf8_lossy(&invalid.stderr);
    assert!(
        diagnostic.contains("E0599") && diagnostic.contains("MissingClosureImplementation"),
        "must fail on missing enum implementation, not an unrelated error: {diagnostic}"
    );
}
