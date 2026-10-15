//! Device-free shader parsing/validation gate.

#[cfg(feature = "gpu")]
#[test]
fn proxy_visibility_shader_parses_and_validates() {
    let source = include_str!("../src/shaders/proxy_visibility.wgsl");
    let module =
        wgpu::naga::front::wgsl::parse_str(source).expect("world visibility WGSL must parse");
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    );
    validator
        .validate(&module)
        .expect("world visibility WGSL must validate");
}
