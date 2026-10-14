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

use prism_material_schema::codegen::{emit_rust_layout, emit_wesl_unpack};
use prism_material_schema::surface_schema;

/// Committed GPU decoder, read at compile time so the test has no runtime path
/// dependency. Relative to this test file in `prism_render_material/tests/`.
const COMMITTED_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/material_unpack.wesl");

/// Committed CPU layout module (lives next to the typed structs it guards).
const COMMITTED_RUST_LAYOUT: &str = include_str!("../src/surface_layout.rs");
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
