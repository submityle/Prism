use crate::*;
use alloc::collections::BTreeMap;
use prism_render_architecture::gpu_scene::GpuCompletionValue;

fn record(
    handle: prism_render_architecture::abi::GenerationalHandle,
    revision: u32,
) -> MaterialRecord {
    MaterialRecord {
        handle,
        revision,
        domain: MaterialDomain::Surface,
        render_class: MaterialRenderClass::Opaque,
        illumination: Illumination::Lit,
        features: MaterialFeatureFlags::default(),
        closure_mask: 1,
        surface: GpuSurfaceParameters::default(),
        textures: Vec::new(),
        custom_program: None,
    }
}

#[test]
fn registry_rejects_stale_revision_and_reuses_after_completion() {
    let mut registry = MaterialRegistry::new(4);
    let first = registry.allocate().unwrap();
    registry.publish(record(first, 1)).unwrap();
    assert!(matches!(
        registry.publish(record(first, 1)),
        Err(MaterialRegistryError::StaleRevision { .. })
    ));
    registry.retire(first, GpuCompletionValue(7)).unwrap();
    assert_eq!(registry.reclaim_completed(GpuCompletionValue(6)), 0);
    assert_eq!(registry.reclaim_completed(GpuCompletionValue(7)), 1);
    let second = registry.allocate().unwrap();
    assert_eq!(second.index, first.index);
    assert_ne!(second.generation, first.generation);
}

#[test]
fn graph_normalization_removes_dead_nodes_and_classifies_npr() {
    let input = MaterialNodeId(1);
    let output = MaterialNodeId(2);
    let dead = MaterialNodeId(3);
    let graph = MaterialGraph {
        nodes: BTreeMap::from([
            (
                input,
                MaterialNode::Constant(MaterialValue::Vector([1.0; 4])),
            ),
            (
                output,
                MaterialNode::Closure {
                    kind: ClosureKind::Npr,
                    inputs: vec![input],
                },
            ),
            (dead, MaterialNode::Constant(MaterialValue::Scalar(9.0))),
        ]),
        output: Some(output),
    };
    let normalized = graph.normalize().unwrap();
    assert_eq!(normalized.nodes.len(), 2);
    // NPR is now the orthogonal `Stylized` illumination axis, not a render
    // class flattened into the blend family.
    assert_eq!(normalized.render_class, MaterialRenderClass::Opaque);
    assert_eq!(normalized.illumination, Illumination::Stylized);
    assert_eq!(
        normalized.specialization_id,
        SpecializationId::new(
            Illumination::Stylized,
            normalized.closure_mask,
            MaterialRenderClass::Opaque as u32
        )
    );
}

#[test]
fn graph_normalization_rejects_over_deep_closure_slab() {
    use alloc::collections::BTreeMap;
    // Build a chain of Layer nodes deeper than MAX_CLOSURE_SLAB_DEPTH.
    let leaf = MaterialNodeId(0);
    let mut nodes = BTreeMap::from([(
        leaf,
        MaterialNode::Closure {
            kind: ClosureKind::Diffuse,
            inputs: vec![MaterialNodeId(100)],
        },
    )]);
    nodes.insert(
        MaterialNodeId(100),
        MaterialNode::Constant(MaterialValue::Scalar(1.0)),
    );
    let mut prev = leaf;
    let mut last = leaf;
    for i in 1..=(MAX_CLOSURE_SLAB_DEPTH + 1) {
        let id = MaterialNodeId(1000 + i);
        nodes.insert(
            id,
            MaterialNode::Closure {
                kind: ClosureKind::Layer,
                inputs: vec![prev],
            },
        );
        prev = id;
        last = id;
    }
    let graph = MaterialGraph {
        nodes,
        output: Some(last),
    };
    assert!(matches!(
        graph.normalize(),
        Err(MaterialValidationError::ClosureSlabTooDeep { .. })
    ));
}

#[cfg(feature = "bevy")]
#[test]
fn standard_material_bridge_preserves_surface_classification() {
    struct Resolver;
    impl StandardMaterialTextureResolver for Resolver {
        fn resolve(
            &mut self,
            _: bevy_asset::AssetId<bevy_image::Image>,
            semantic: TextureSemantic,
        ) -> GpuMaterialTexture {
            GpuMaterialTexture {
                index: semantic as u32 + 1,
                generation: 1,
                semantic: semantic as u32,
                sampler_index: 1,
            }
        }
    }
    let material = bevy_pbr::StandardMaterial {
        alpha_mode: bevy_material::AlphaMode::Mask(0.37),
        double_sided: true,
        clearcoat: 0.5,
        ..Default::default()
    };
    let handle = prism_render_architecture::abi::GenerationalHandle {
        index: 2,
        generation: 1,
    };
    let record = import_standard_material(handle, 4, &material, &mut Resolver);
    assert_eq!(record.render_class, MaterialRenderClass::MaskedTwoSided);
    // Clearcoat is a closure lobe, not a shading model: illumination stays Lit
    // and the clearcoat closure bit is set.
    assert_eq!(record.illumination, Illumination::Lit);
    assert_ne!(
        record.closure_mask & (1 << ClosureKind::ClearCoat as u32),
        0
    );
    assert_eq!(record.surface.alpha_cutoff, 0.37);
    assert!(record.features.contains(MaterialFeatureFlags::DOUBLE_SIDED));
}

/// The OpenPBR Surface importer must map an external material's dielectric F0
/// onto `specular_ior` exactly (design doc §4.4): Bevy's `reflectance` is the
/// `0.16 * r^2` remap, so `0.5` has to land on the `1.5` default and `0.0` on a
/// neutral interface, with no per-format 1:1 translation layer in between.
#[cfg(feature = "bevy")]
#[test]
fn openpbr_importer_maps_reflectance_to_a_faithful_ior() {
    let to_f0 = |ior: f32| {
        let r = (ior - 1.0) / (ior + 1.0);
        r * r
    };
    let default_like = bevy_pbr::StandardMaterial {
        reflectance: 0.5,
        ..Default::default()
    };
    let imported = import_openpbr_surface(&default_like, 0.5);
    assert!((imported.specular_ior - 1.5).abs() < 1.0e-6);
    assert!((to_f0(imported.specular_ior) - 0.04).abs() < 1.0e-6);

    // A zero reflectance is a fully transparent interface (F0 = 0), which the
    // inverse remap expresses as an index of exactly one.
    let no_specular = bevy_pbr::StandardMaterial {
        reflectance: 0.0,
        ..Default::default()
    };
    let imported = import_openpbr_surface(&no_specular, 0.5);
    assert!((imported.specular_ior - 1.0).abs() < 1.0e-6);
    assert!(to_f0(imported.specular_ior) < 1.0e-9);

    // The strongest authored reflectance stays inside the OpenPBR IOR range.
    let full = bevy_pbr::StandardMaterial {
        reflectance: 1.0,
        ..Default::default()
    };
    let imported = import_openpbr_surface(&full, 0.5);
    assert!(imported.specular_ior >= 1.0 && imported.specular_ior <= 3.0);
    assert!((to_f0(imported.specular_ior) - 0.16).abs() < 1.0e-5);
}

#[test]
fn registry_tracks_sparse_rows_and_rejects_texture_overflow() {
    let mut registry = MaterialRegistry::new(8);
    let handle = registry.allocate().unwrap();
    let mut value = record(handle, 1);
    value.textures = vec![GpuMaterialTexture::default(); MAX_MATERIAL_TEXTURES + 1];
    assert!(matches!(
        registry.publish(value),
        Err(MaterialRegistryError::TooManyTextures { .. })
    ));
    registry.publish(record(handle, 1)).unwrap();
    assert_eq!(registry.take_dirty(), vec![0, handle.index]);
    assert!(registry.take_dirty().is_empty());
}

#[test]
fn slot_zero_is_a_live_principled_fallback() {
    let registry = MaterialRegistry::new(8);
    let (headers, parameters, _) = registry.gpu_tables();
    assert_eq!(headers[0].generation, 0);
    assert_eq!(headers[0].active, 1);
    assert_eq!(headers[0].illumination, Illumination::Lit as u32);
    assert_eq!(parameters[0], GpuSurfaceParameters::default());
}

#[test]
fn gpu_material_rows_match_the_shader_abi() {
    assert_eq!(size_of::<GpuMaterialHeader>(), 80);
    // The v5 OpenPBR field completion (coat/fuzz/base/specular/subsurface/
    // transmission groups, including coat_darkening, scatter_anisotropy and
    // transmission_scatter) sized the fat authoring view, with trailing pad
    // words keeping it a 16-byte multiple.
    assert_eq!(size_of::<GpuSurfaceParameters>(), 288);
    assert_eq!(size_of::<GpuMaterialTexture>(), 16);
    assert_eq!(size_of::<GpuMaterialHeader>() % 16, 0);
    assert_eq!(size_of::<GpuSurfaceParameters>() % 16, 0);
    assert_eq!(size_of::<GpuMaterialTexture>() % 16, 0);
}
