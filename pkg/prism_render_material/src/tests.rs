use crate::*;
use alloc::collections::BTreeMap;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_architecture::render_fabric::RenderDomain;

fn record(
    handle: prism_render_architecture::abi::GenerationalHandle,
    revision: u32,
) -> MaterialRecord {
    MaterialRecord {
        handle,
        revision,
        domain: MaterialDomain::Surface,
        render_domain: RenderDomain::Surface,
        render_class: MaterialRenderClass::Opaque,
        response_plan: ShadingPlan::default(),
        features: MaterialFeatureFlags::default(),
        closure_mask: 1,
        surface: GpuSurfaceParameters::default(),
        textures: Vec::new(),
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
        plan: None,
    };
    let normalized = graph.normalize().unwrap();
    assert_eq!(normalized.nodes.len(), 2);
    assert_eq!(normalized.render_class, MaterialRenderClass::Opaque);
    assert_eq!(
        normalized.plan,
        ShadingPlan::single_builtin(BuiltinProgram::Npr)
    );
}

#[test]
fn graph_normalization_accepts_an_arbitrarily_deep_closure_slab() {
    use alloc::collections::BTreeMap;
    // ABI v5 / P1-c deletes the bounded closure slab (design §6.1): the bake
    // layer resolves *any* number of `Layer`/`Mix` nodes into a texture-space
    // blend, so a chain far deeper than the old `MAX_CLOSURE_SLAB_DEPTH` of 4
    // must normalize without a cap instead of erroring.
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
    for i in 1..=64u32 {
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
        plan: None,
    };
    let normalized = graph
        .normalize()
        .expect("a 64-deep layer chain is unbounded");
    assert!(normalized
        .features
        .contains(MaterialFeatureFlags::COMPLEX_CLOSURE));
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
            GpuMaterialTexture::non_virtual(semantic as u32 + 1, 1, semantic as u32, 1)
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
    // Clearcoat is a closure lobe, not a response-program identity.
    assert_eq!(record.response_plan, ShadingPlan::default());
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
fn registry_tracks_sparse_rows_and_accepts_an_unbounded_texture_set() {
    // ABI v5 removed the fixed `MAX_MATERIAL_TEXTURES` slot cap (§8.2): a
    // material may reference any number of bindless texture slots, so a record
    // with far more rows than the old 8-slot window must publish cleanly.
    let mut registry = MaterialRegistry::new(8);
    let handle = registry.allocate().unwrap();
    let mut value = record(handle, 1);
    value.textures = (0..64)
        .map(|index| GpuMaterialTexture::non_virtual(index, 1, 0, index))
        .collect();
    registry.publish(value).unwrap();
    let (headers, _, textures) = registry.gpu_tables();
    assert_eq!(textures.len(), 64, "every authored texture row is retained");
    assert_eq!(headers[handle.index as usize].texture_offset, 0);
    assert_eq!(headers[handle.index as usize].texture_count, 64);

    registry.publish(record(handle, 2)).unwrap();
    assert_eq!(registry.take_dirty(), vec![0, handle.index]);
    assert!(registry.take_dirty().is_empty());
}

#[test]
fn slot_zero_is_a_live_principled_fallback() {
    let registry = MaterialRegistry::new(8);
    let (headers, parameters, _) = registry.gpu_tables();
    assert_eq!(headers[0].generation, 0);
    assert_eq!(headers[0].is_active, 1);
    assert_eq!(headers[0].render_class, MaterialRenderClass::Opaque as u32);
    assert_eq!(parameters[0], GpuSurfaceParameters::default());
}

#[test]
fn gpu_material_rows_match_the_shader_abi() {
    // ABI8 retains the data/dynamic words, adds the explicit domain, and keeps
    // five dense execution words; legacy specialization words are removed.
    // `extensions_{used,required}_{low,high}`, so the header carries the
    // material's whole registry digest and a load-time negotiator can project it
    // against a target's capabilities without re-deriving the catalogue.
    // …plus the four runtime dynamic channel words of §17.1
    // (`dynamic_wetness/snow/damage/dissolve`), which a runtime state change
    // rewrites in place.
    assert_eq!(size_of::<GpuMaterialHeader>(), 116);
    // The v5 OpenPBR field completion (coat/fuzz/base/specular/subsurface/
    // transmission groups, including coat_darkening, scatter_anisotropy and
    // transmission_scatter) sized the fat authoring view, with trailing pad
    // words keeping it a 16-byte multiple.
    assert_eq!(size_of::<GpuSurfaceParameters>(), 320);
    assert_eq!(size_of::<GpuMaterialTexture>(), 32);
    // A storage-buffer array requires only the element's natural alignment
    // (4 for a struct of `u32`s); the header was never padded to 16, and
    // the header is a dense 29-word u32 array.
    assert_eq!(size_of::<GpuMaterialHeader>() % 4, 0);
    assert_eq!(size_of::<GpuSurfaceParameters>() % 16, 0);
    assert_eq!(size_of::<GpuMaterialTexture>() % 16, 0);
}

#[test]
fn non_virtual_texture_rows_are_explicitly_disabled() {
    let row = GpuMaterialTexture::non_virtual(4, 7, 2, 4);
    assert_eq!(row.virtual_texture_id, NO_VIRTUAL_TEXTURE);
    assert_eq!(row.virtual_texture_width, 0);
    assert_eq!(row.virtual_texture_height, 0);
    assert_eq!(row.virtual_texture_page_size, 0);
}

#[test]
fn virtual_texture_metadata_has_a_stable_row_layout() {
    let row = GpuMaterialTexture::with_virtual_texture(4, 7, 2, 4, 19, 4096, 2048, 128);
    assert_eq!(row.virtual_texture_id, 19);
    assert_eq!(row.virtual_texture_width, 4096);
    assert_eq!(row.virtual_texture_height, 2048);
    assert_eq!(row.virtual_texture_page_size, 128);
    assert_ne!(row.virtual_texture_id, NO_VIRTUAL_TEXTURE);
}

/// The importer must split Bevy's authored emission radiance into the
/// specification's `emission_color * emission_luminance` without changing the
/// product, so a converted material renders identically.
#[cfg(feature = "bevy")]
#[test]
fn openpbr_importer_preserves_the_emission_radiance() {
    let material = bevy_pbr::StandardMaterial {
        emissive: bevy_color::LinearRgba::new(2.0, 1.0, 0.5, 1.0),
        ..Default::default()
    };
    let imported = import_openpbr_surface(&material, 0.5);
    assert!((imported.emission_luminance - 2.0).abs() < 1.0e-6);
    for (channel, expected) in [2.0f32, 1.0, 0.5].into_iter().enumerate() {
        let radiance = imported.emission_color[channel] * imported.emission_luminance;
        assert!(
            (radiance - expected).abs() < 1.0e-5,
            "channel {channel}: {radiance} vs {expected}"
        );
    }
}
