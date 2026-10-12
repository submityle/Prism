//! Device-free WGSL parse and validation gates for every VT compute shader.
//!
//! Real-device parity remains responsible for execution correctness. This
//! gate catches syntax, builtin, and type/validation regressions even on hosts
//! without a usable adapter, so a skipped GPU test cannot make a broken shader
//! appear green.

use wgpu::naga::valid::{Capabilities, ValidationFlags, Validator};

const SHADERS: &[(&str, &str)] = &[
    (
        "page_lookup",
        include_str!("../src/shaders/page_lookup.wgsl"),
    ),
    (
        "page_resolve",
        include_str!("../src/shaders/page_resolve.wgsl"),
    ),
    (
        "feedback_decode",
        include_str!("../src/shaders/feedback_decode.wgsl"),
    ),
];

#[test]
fn every_virtual_texture_shader_parses_and_validates_without_a_device() {
    for &(name, source) in SHADERS {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("WGSL shader `{name}` failed to parse: {error}"));
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        validator
            .validate(&module)
            .unwrap_or_else(|error| panic!("WGSL shader `{name}` failed validation: {error:?}"));
    }
}
