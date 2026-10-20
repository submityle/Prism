//! Device-free shader parsing/validation gate.

#[cfg(feature = "terrain_scatter")]
#[test]
fn placement_reset_uses_two_read_write_storage_bindings_and_core_u32_atomics() {
    use wgpu::naga::{AddressSpace, ScalarKind, StorageAccess, TypeInner};

    let source = include_str!("../src/shaders/heightfield_placement_reset.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("placement reset: {}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("placement reset: {error}"));

    let [entry] = module.entry_points.as_slice() else {
        panic!("placement reset must expose exactly one entry point");
    };
    assert_eq!(entry.name, "reset_main");
    assert_eq!(entry.workgroup_size, [1, 1, 1]);

    let mut bindings = std::collections::BTreeMap::new();
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = variable.binding {
            assert_eq!(binding.group, 0);
            assert!(bindings.insert(binding.binding, variable).is_none());
        }
        assert_ne!(variable.space, AddressSpace::Uniform);
    }
    assert_eq!(bindings.len(), 2);
    for binding in 0..=1 {
        assert_eq!(
            bindings[&binding].space,
            AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            }
        );
    }
    for (_, ty) in module.types.iter() {
        match ty.inner {
            TypeInner::Scalar(scalar)
            | TypeInner::Atomic(scalar)
            | TypeInner::Vector { scalar, .. } => {
                assert_ne!(scalar.kind, ScalarKind::Float);
                assert!(scalar.width <= 4);
            }
            _ => {}
        }
    }
}

#[cfg(feature = "terrain_scatter")]
#[test]
fn placement_shaders_match_core_integer_binding_and_record_abi() {
    use wgpu::naga::{AddressSpace, ScalarKind, StorageAccess, TypeInner};
    let load = StorageAccess::LOAD;
    let store = load | StorageAccess::STORE;
    for (name, source, uniform_size, accesses, entries) in [
        (
            "placement source",
            include_str!("../src/shaders/heightfield_placement_source.wgsl"),
            192,
            vec![store],
            vec!["placement_main"],
        ),
        (
            "placement projection",
            include_str!("../src/shaders/heightfield_placement_project.wgsl"),
            128,
            vec![store, load, store, store, store, store],
            vec!["project_main", "finalize_main"],
        ),
    ] {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(module.entry_points.len(), entries.len(), "{name}");
        for (entry, expected) in module.entry_points.iter().zip(entries) {
            assert_eq!(entry.name, expected, "{name}");
            assert_eq!(entry.workgroup_size, [64, 1, 1], "{name}");
        }
        let mut bindings = std::collections::BTreeMap::new();
        for (_, variable) in module.global_variables.iter() {
            if let Some(binding) = variable.binding {
                assert_eq!(binding.group, 0, "{name}");
                assert!(bindings.insert(binding.binding, variable).is_none());
            }
        }
        assert_eq!(bindings.len(), accesses.len() + 1, "{name}");
        assert_eq!(bindings[&0].space, AddressSpace::Uniform, "{name}");
        match module.types[bindings[&0].ty].inner {
            TypeInner::Struct { span, .. } => assert_eq!(span, uniform_size, "{name}"),
            _ => panic!("{name}: uniform must be a struct"),
        }
        for (index, access) in accesses.into_iter().enumerate() {
            assert_eq!(
                bindings[&(index as u32 + 1)].space,
                AddressSpace::Storage { access },
                "{name}"
            );
        }
        match module.types[bindings[&1].ty].inner {
            TypeInner::Array { stride, .. } => assert_eq!(stride, 64, "{name} record stride"),
            _ => panic!("{name}: records must be an array"),
        }
        if name == "placement projection" {
            match module.types[bindings[&3].ty].inner {
                TypeInner::Array { stride, .. } => assert_eq!(stride, 64, "sample stride"),
                _ => panic!("samples must be an array"),
            }
        }
        for (_, ty) in module.types.iter() {
            match ty.inner {
                TypeInner::Scalar(scalar)
                | TypeInner::Atomic(scalar)
                | TypeInner::Vector { scalar, .. } => {
                    assert_ne!(scalar.kind, ScalarKind::Float, "{name}");
                    assert!(scalar.width <= 4, "{name}");
                }
                _ => {}
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn field_brush_hierarchy_uses_core_integer_types_and_the_existing_binding_abi() {
    let source = include_str!("../src/shaders/scatter_field.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("field parse: {}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("indexed field requires no optional capabilities");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    let mut bindings = std::collections::BTreeMap::new();
    for (_, var) in module.global_variables.iter() {
        if let Some(binding) = var.binding {
            assert_eq!(binding.group, 0);
            assert!(bindings.insert(binding.binding, var.space).is_none());
        }
    }
    assert_eq!(bindings.len(), 6);
    assert_eq!(bindings[&0], wgpu::naga::AddressSpace::Uniform);
    for binding in 1..=5 {
        let mut access = wgpu::naga::StorageAccess::LOAD;
        if binding >= 3 {
            access |= wgpu::naga::StorageAccess::STORE;
        }
        assert_eq!(
            bindings[&binding],
            wgpu::naga::AddressSpace::Storage { access }
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

#[cfg(feature = "gpu")]
#[test]
fn sdf_lookup_uses_sorted_integer_key_and_packed_sample_abi() {
    let source = include_str!("../src/shaders/sdf_lookup.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("SDF lookup parse: {}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("SDF lookup validation: {error}"));
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "main");
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    let load = wgpu::naga::StorageAccess::LOAD;
    let store = load | wgpu::naga::StorageAccess::STORE;
    let mut bindings = std::collections::BTreeMap::new();
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = variable.binding {
            assert_eq!(binding.group, 0);
            assert!(bindings.insert(binding.binding, variable).is_none());
        }
    }
    assert_eq!(bindings.len(), 6);
    assert_eq!(bindings[&0].space, wgpu::naga::AddressSpace::Uniform);
    for binding in 1..=5 {
        assert_eq!(
            bindings[&binding].space,
            wgpu::naga::AddressSpace::Storage {
                access: if binding == 5 { store } else { load },
            }
        );
    }
    match module.types[bindings[&0].ty].inner {
        wgpu::naga::TypeInner::Struct { span, .. } => assert_eq!(span, 16),
        _ => panic!("SDF lookup params must be a 16-byte uniform struct"),
    }
    for binding in 1..=2 {
        match module.types[bindings[&binding].ty].inner {
            wgpu::naga::TypeInner::Array { stride, .. } => assert_eq!(stride, 16),
            _ => panic!("SDF key/meta bindings must use 16-byte records"),
        }
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
fn placement_deletion_uses_the_independent_56_byte_key_abi() {
    let source = include_str!("../src/shaders/heightfield_placement_deletion.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("placement deletion parse: {error}"));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("placement deletion requires no optional capabilities");
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    let mut bindings = std::collections::BTreeMap::new();
    for (_, variable) in module.global_variables.iter() {
        if let Some(binding) = variable.binding {
            assert_eq!(binding.group, 0);
            assert!(bindings.insert(binding.binding, variable).is_none());
        }
    }
    assert_eq!(bindings.len(), 7);
    assert_eq!(bindings[&0].space, wgpu::naga::AddressSpace::Uniform);
    let load = wgpu::naga::StorageAccess::LOAD;
    let store = load | wgpu::naga::StorageAccess::STORE;
    for binding in 1..=6 {
        assert_eq!(
            bindings[&binding].space,
            wgpu::naga::AddressSpace::Storage {
                access: if binding <= 4 { load } else { store },
            }
        );
    }
    match module.types[bindings[&4].ty].inner {
        wgpu::naga::TypeInner::Array { stride, .. } => assert_eq!(stride, 56),
        _ => panic!("placement deletion keys must have a 56-byte stride"),
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
