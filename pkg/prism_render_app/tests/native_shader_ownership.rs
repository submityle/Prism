//! Native shader sources must not be resolved through the retired scene host.
//!
//! This is a source-boundary gate, not a replacement for shader compilation or
//! GPU validation. It prevents the native render package from silently
//! reintroducing the legacy ownership path after the migration is complete.

#![forbid(unsafe_code)]

const CARGO_MANIFEST: &str = include_str!("../Cargo.toml");
const BUILD_SCRIPT: &str = include_str!("../build.rs");
const COMPOSITE_SHADER: &str = include_str!("../src/shaders/composite_material.wesl");
const LIGHTING_GPU: &str = include_str!("../src/lighting_gpu.rs");

#[test]
fn native_render_package_owns_its_shader_inputs() {
    for (label, source) in [
        ("Cargo.toml", CARGO_MANIFEST),
        ("build.rs", BUILD_SCRIPT),
        ("composite_material.wesl", COMPOSITE_SHADER),
        ("lighting_gpu.rs", LIGHTING_GPU),
    ] {
        assert!(
            !source.contains("prism_render_scene"),
            "native shader ownership boundary still references the retired scene host: {label}"
        );
    }
}
