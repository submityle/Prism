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

#[cfg(feature = "gpu")]
#[test]
fn world_shaders_validate_and_match_declared_binding_counts() {
    for (name, source, storage, uniform) in [
        (
            "density",
            include_str!("../src/shaders/proxy_density.wgsl"),
            4,
            2,
        ),
        (
            "density compact",
            include_str!("../src/shaders/proxy_compact_density.wgsl"),
            4,
            1,
        ),
        ("dither", prism_world_gpu::DENSITY_DITHER_WGSL, 0, 0),
        (
            "integer scatter",
            include_str!("../src/shaders/scatter.wgsl"),
            3,
            1,
        ),
        (
            "integer scatter author field",
            include_str!("../src/shaders/scatter_field.wgsl"),
            5,
            1,
        ),
    ] {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name} parse: {error}"));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} validation: {error}"));
        let count = |predicate: fn(wgpu::naga::AddressSpace) -> bool| {
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| predicate(variable.space))
                .count()
        };
        assert_eq!(
            count(|space| matches!(space, wgpu::naga::AddressSpace::Storage { .. })),
            storage,
            "{name} storage ABI"
        );
        assert_eq!(
            count(|space| space == wgpu::naga::AddressSpace::Uniform),
            uniform,
            "{name} uniform ABI"
        );
    }
}
