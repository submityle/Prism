//! Device-free shader parsing/validation gate.

#[cfg(feature = "gpu")]
#[test]
fn deletion_scatter_uses_core_integer_features_and_exact_binding_access() {
    let source = include_str!("../src/shaders/scatter_deletion.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("deletion parse: {}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("deletion shader requires no optional capabilities");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    let load = wgpu::naga::StorageAccess::LOAD;
    let store = load | wgpu::naga::StorageAccess::STORE;
    let mut bindings = std::collections::BTreeMap::new();
    for (_, var) in module.global_variables.iter() {
        if let Some(binding) = var.binding {
            assert_eq!(binding.group, 0);
            assert!(bindings.insert(binding.binding, var.space).is_none());
        }
    }
    assert_eq!(bindings.len(), 7);
    assert_eq!(bindings[&0], wgpu::naga::AddressSpace::Uniform);
    for binding in 1..=6 {
        assert_eq!(
            bindings[&binding],
            wgpu::naga::AddressSpace::Storage {
                access: if binding <= 4 { load } else { store },
            }
        );
    }
    for (_, ty) in module.types.iter() {
        match ty.inner {
            wgpu::naga::TypeInner::Scalar(scalar)
            | wgpu::naga::TypeInner::Atomic(scalar)
            | wgpu::naga::TypeInner::Vector { scalar, .. } => {
                assert_ne!(scalar.kind, wgpu::naga::ScalarKind::Float);
                assert!(scalar.width <= 4);
            }
            _ => {}
        }
    }
}

#[cfg(feature = "terrain_scatter")]
#[test]
fn heightfield_scatter_uses_only_core_integer_features_and_six_storage_bindings() {
    let source = include_str!("../src/shaders/scatter_heightfield.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("heightfield parse: {}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("heightfield shader requires no optional capabilities");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    for (space, expected) in [
        (wgpu::naga::AddressSpace::Uniform, 1),
        (
            wgpu::naga::AddressSpace::Storage {
                access: wgpu::naga::StorageAccess::LOAD,
            },
            3,
        ),
        (
            wgpu::naga::AddressSpace::Storage {
                access: wgpu::naga::StorageAccess::LOAD | wgpu::naga::StorageAccess::STORE,
            },
            3,
        ),
    ] {
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, var)| var.space == space)
                .count(),
            expected
        );
    }
    for (_, ty) in module.types.iter() {
        match ty.inner {
            wgpu::naga::TypeInner::Scalar(scalar)
            | wgpu::naga::TypeInner::Vector { scalar, .. } => {
                assert_ne!(scalar.kind, wgpu::naga::ScalarKind::Float);
                assert!(scalar.width <= 4);
            }
            _ => {}
        }
    }
}

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
        (
            "full-key scatter deletions",
            include_str!("../src/shaders/scatter_deletion.wgsl"),
            6,
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
