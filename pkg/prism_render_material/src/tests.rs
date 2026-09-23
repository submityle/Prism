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
        shading_model: MaterialShadingModel::Principled,
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
    assert_eq!(normalized.render_class, MaterialRenderClass::NprOpaque);
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
    let record = lower_standard_material(handle, 4, &material, &mut Resolver);
    assert_eq!(record.render_class, MaterialRenderClass::MaskedTwoSided);
    assert_eq!(record.shading_model, MaterialShadingModel::ClearCoat);
    assert_eq!(record.surface.alpha_cutoff, 0.37);
    assert!(record.features.contains(MaterialFeatureFlags::DOUBLE_SIDED));
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
    assert_eq!(registry.take_dirty(), vec![handle.index]);
    assert!(registry.take_dirty().is_empty());
}
