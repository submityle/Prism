//! Device-local production of render pass descriptors from a frozen feature selection.
//!
//! `FeatureSelection` deliberately contains semantic choices only.  This module is
//! the app-side seam where a host supplies the real shader package/module
//! declarations and pass state, and where those declarations become resident
//! driver-local modules plus the immutable pass table consumed by both direct and
//! bin queueing.

use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;

use prism_render_architecture::render_fabric::{
    FeatureId, FeatureSelection, PassKey, SelectedFeature, ShaderPermutationBinding, VariantId,
};
use prism_render_architecture::shader_package::{
    PermutationError, PermutationIndex, PermutationKey, PermutationSpace,
};
use prism_render_driver::RenderDevice;
use prism_render_material::MaterialRecord;

use crate::draw::DrawFunctionId;
use crate::mesh_pipelines::{ResponseModuleCache, ResponseModuleCacheError, ResponseModuleSpec};
use crate::pipeline_pass::{
    PassDescriptor, PassDescriptorClass, PassDescriptorTable, PassRole, PassSetError,
};
use crate::scene::SceneRegistry;
use crate::sets::RenderPhase;

/// One host-owned binding from a selected feature variant to a real shader
/// package/module and one pass-table entry.
#[derive(Clone, Debug)]
pub struct FeaturePassBinding {
    /// Semantic feature selected by the render fabric.
    pub feature_id: FeatureId,
    /// Semantic variant selected by the render fabric.
    pub variant_id: VariantId,
    /// `None` is the ordinary response/surface pass.  For a variant declaring
    /// semantic pass templates, every template must be named exactly once.
    /// Semantic pass template this binding implements.
    pub pass_key: Option<PassKey>,
    /// Must equal the selected variant's semantic binding, when present.
    /// Shader permutation declaration and selected key.
    pub permutation: Option<ShaderPermutationBinding>,
    /// Real shader package/module declaration to make resident.
    pub module: ResponseModuleSpec,
    /// Driver pass role.
    pub role: PassRole,
    /// Queue phase for this pass.
    pub render_phase: RenderPhase,
    /// Draw function used by this pass, when it differs from the queue default.
    pub draw: Option<DrawFunctionId>,
    /// Coarse deterministic sorting class.
    pub sort_class: PassDescriptorClass,
}

/// A fully resident, immutable pass set produced from one selected feature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeaturePassSet {
    /// Semantic feature represented by this set.
    pub feature_id: FeatureId,
    /// Semantic variant represented by this set.
    pub variant_id: VariantId,
    passes: Vec<PassDescriptor>,
}

impl FeaturePassSet {
    /// Returns the immutable driver-local pass descriptors.
    #[must_use]
    pub fn passes(&self) -> &[PassDescriptor] {
        &self.passes
    }

    /// Installs this exact descriptor set for a material.  Scene direct and
    /// bin paths subsequently resolve the same deduplicated table entry.
    pub fn install(
        &self,
        registry: &mut SceneRegistry,
        material: &MaterialRecord,
    ) -> Result<u32, FeatureMappingError> {
        registry
            .install_material_pass_set(material.clone(), self.passes.clone())
            .map_err(FeatureMappingError::PassSet)
    }
}

/// Errors raised while turning semantic selection into executable render state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeatureMappingError {
    /// The requested feature was not selected.
    MissingSelection(FeatureId),
    /// A host binding names a different selected variant.
    SelectionBindingMismatch {
        /// Feature being mapped.
        feature: FeatureId,
        /// Variant selected by the feature resolver.
        selected: VariantId,
        /// Variant named by the host binding.
        binding: VariantId,
    },
    /// A selected semantic pass has no host binding.
    MissingPassBinding {
        /// Feature being mapped.
        feature: FeatureId,
        /// Selected variant.
        variant: VariantId,
        /// Missing pass, when known.
        pass: Option<PassKey>,
    },
    /// A host binding names a pass not declared by the selected variant.
    UnexpectedPassBinding {
        /// Feature being mapped.
        feature: FeatureId,
        /// Selected variant.
        variant: VariantId,
        /// Unexpected pass.
        pass: PassKey,
    },
    /// A semantic pass was bound more than once.
    DuplicatePassBinding {
        /// Feature being mapped.
        feature: FeatureId,
        /// Selected variant.
        variant: VariantId,
        /// Duplicated pass, when known.
        pass: Option<PassKey>,
    },
    /// The host binding and selected variant disagree on permutation identity.
    PermutationBindingMismatch {
        /// Feature being mapped.
        feature: FeatureId,
        /// Selected variant.
        variant: VariantId,
    },
    /// The selected key is invalid for its declared space.
    Permutation(PermutationError),
    /// Shader module residency rejected the declaration.
    Module(ResponseModuleCacheError),
    /// The produced pass set violates the app pass-table shape.
    PassSet(PassSetError),
}

impl fmt::Display for FeatureMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "feature-to-render mapping error: {self:?}")
    }
}

impl std::error::Error for FeatureMappingError {}

/// Produces resident shader modules and pass descriptors from one frozen
/// `FeatureSelection`.  The mapper holds no second selection and therefore
/// cannot make direct and bin paths disagree about the chosen implementation.
pub struct FeatureSelectionMapper<'a, D: RenderDevice + ?Sized> {
    device: &'a D,
    modules: &'a mut ResponseModuleCache,
    cache: Option<&'a mut FeatureMappingCache>,
}

/// Render-world cache for validated feature-to-pass mappings.
///
/// The cache is deliberately separate from [`ResponseModuleCache`]: module
/// residency is device state, while this cache is the immutable scene binding
/// from one frozen selection and host package declaration to one pass set.
#[derive(Clone, Debug, Default)]
pub struct FeatureMappingCache {
    entries: BTreeMap<Vec<u8>, FeaturePassSet>,
    hits: u64,
    misses: u64,
}

/// Mapping-cache counters useful to render diagnostics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FeatureMappingCacheStats {
    /// Number of scene mapping requests served by the cache.
    pub hits: u64,
    /// Number of scene mapping requests that required validation and residency.
    pub misses: u64,
}

impl FeatureMappingCache {
    /// Returns cache counters without mutating the cache.
    #[must_use]
    pub const fn stats(&self) -> FeatureMappingCacheStats {
        FeatureMappingCacheStats {
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Drops all mappings after a device/module-cache replacement.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of resident semantic mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache has no resident mappings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<'a, D: RenderDevice + ?Sized> FeatureSelectionMapper<'a, D> {
    /// Creates a mapper using the device and module cache owned by the render app.
    #[must_use]
    pub fn new(device: &'a D, modules: &'a mut ResponseModuleCache) -> Self {
        Self {
            device,
            modules,
            cache: None,
        }
    }

    /// Creates a mapper backed by a render-world mapping cache.
    #[must_use]
    pub fn with_cache(
        device: &'a D,
        modules: &'a mut ResponseModuleCache,
        cache: &'a mut FeatureMappingCache,
    ) -> Self {
        Self {
            device,
            modules,
            cache: Some(cache),
        }
    }

    /// Maps one selected feature to resident modules and pass descriptors.
    pub fn map_feature(
        &mut self,
        selection: &FeatureSelection,
        feature_id: FeatureId,
        bindings: &[FeaturePassBinding],
    ) -> Result<FeaturePassSet, FeatureMappingError> {
        let selected = selection
            .feature(feature_id)
            .ok_or(FeatureMappingError::MissingSelection(feature_id))?;
        let relevant: Vec<_> = bindings
            .iter()
            .filter(|binding| binding.feature_id == feature_id)
            .collect();
        if relevant.is_empty() {
            return Err(FeatureMappingError::MissingPassBinding {
                feature: feature_id,
                variant: selected.variant_id(),
                pass: None,
            });
        }
        for binding in &relevant {
            if binding.variant_id != selected.variant_id() {
                return Err(FeatureMappingError::SelectionBindingMismatch {
                    feature: feature_id,
                    selected: selected.variant_id(),
                    binding: binding.variant_id,
                });
            }
        }

        let expected_passes = selected.passes();
        let expected_keys: Vec<_> = expected_passes.iter().map(|pass| pass.id).collect();
        if expected_passes.is_empty() {
            if relevant.len() != 1 || relevant[0].pass_key.is_some() {
                return Err(FeatureMappingError::UnexpectedPassBinding {
                    feature: feature_id,
                    variant: selected.variant_id(),
                    pass: relevant
                        .iter()
                        .find_map(|binding| binding.pass_key)
                        .unwrap_or(PassKey(0)),
                });
            }
        } else {
            let mut seen = Vec::new();
            for binding in &relevant {
                let Some(pass) = binding.pass_key else {
                    return Err(FeatureMappingError::MissingPassBinding {
                        feature: feature_id,
                        variant: selected.variant_id(),
                        pass: None,
                    });
                };
                if !expected_keys.contains(&pass) {
                    return Err(FeatureMappingError::UnexpectedPassBinding {
                        feature: feature_id,
                        variant: selected.variant_id(),
                        pass,
                    });
                }
                if seen.contains(&pass) {
                    return Err(FeatureMappingError::DuplicatePassBinding {
                        feature: feature_id,
                        variant: selected.variant_id(),
                        pass: Some(pass),
                    });
                }
                seen.push(pass);
            }
            if seen.len() != expected_keys.len() {
                let missing = expected_keys.into_iter().find(|key| !seen.contains(key));
                return Err(FeatureMappingError::MissingPassBinding {
                    feature: feature_id,
                    variant: selected.variant_id(),
                    pass: missing,
                });
            }
        }

        let ordered: Vec<_> = if expected_passes.is_empty() {
            relevant
        } else {
            expected_keys
                .iter()
                .map(|pass| {
                    relevant
                        .iter()
                        .find(|binding| binding.pass_key == Some(*pass))
                        .copied()
                        .expect("pass binding validated above")
                })
                .collect()
        };

        // The semantic mapping depends on the selected variant and the
        // relevant host declarations, not on the other features in the
        // device-wide closure. Keep the hot key compact and injective; shader
        // source descriptors are deliberately checked again by `intern` on a
        // hit so an ABI/source drift cannot be hidden by this cache.
        let cache_key = mapping_cache_key(selected, &ordered);
        if let Some(cache) = self.cache.as_deref_mut() {
            if let Some(cached) = cache.entries.get(&cache_key).cloned() {
                for binding in &ordered {
                    self.modules
                        .intern(self.device, binding.module.clone())
                        .map_err(FeatureMappingError::Module)?;
                }
                cache.hits = cache.hits.saturating_add(1);
                return Ok(cached);
            }
            cache.misses = cache.misses.saturating_add(1);
        }

        let mut passes = Vec::with_capacity(ordered.len());
        for binding in ordered {
            let permutation = resolve_permutation(selected, binding)?;
            let module_id = self
                .modules
                .intern(self.device, binding.module.clone())
                .map_err(FeatureMappingError::Module)?;
            passes.push(PassDescriptor {
                role: binding.role,
                execution_family: binding.module.execution_family,
                response_module_id: module_id,
                permutation_index: permutation,
                fallback_permutation: binding.module.full_lobe_fallback,
                render_phase: binding.render_phase,
                draw: binding.draw,
                sort_class: binding.sort_class,
            });
        }
        PassDescriptorTable::validate(&passes).map_err(FeatureMappingError::PassSet)?;
        let pass_set = FeaturePassSet {
            feature_id,
            variant_id: selected.variant_id(),
            passes,
        };
        if let Some(cache) = self.cache.as_deref_mut() {
            cache.entries.insert(cache_key, pass_set.clone());
        }
        Ok(pass_set)
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value);
}

fn mapping_cache_key(selected: &SelectedFeature, bindings: &[&FeaturePassBinding]) -> Vec<u8> {
    let mut key = Vec::with_capacity(192);
    push_u32(&mut key, selected.feature_id().0);
    push_u32(&mut key, selected.variant_id().0);
    push_u64(&mut key, bindings.len() as u64);
    for binding in bindings {
        push_u32(&mut key, binding.feature_id.0);
        push_u32(&mut key, binding.variant_id.0);
        match binding.pass_key {
            Some(pass) => {
                key.push(1);
                push_u32(&mut key, pass.0);
            }
            None => key.push(0),
        }
        key.push(binding.role as u8);
        key.push(binding.module.execution_family as u8);
        key.push(binding.render_phase.index() as u8);
        key.push(binding.module.builtin as u8);
        match binding.draw {
            Some(draw) => {
                key.push(1);
                push_u32(&mut key, draw.0);
            }
            None => key.push(0),
        }
        key.push(binding.sort_class.get());
        push_bytes(&mut key, binding.module.identity.package.as_str().as_bytes());
        key.extend_from_slice(&binding.module.identity.digest.0);
        push_bytes(&mut key, binding.module.identity.entry_point.as_bytes());
        key.extend_from_slice(&binding.module.identity.binding_layout_digest);
        push_u64(&mut key, binding.module.permutations.axis_count() as u64);
        for axis in 0..binding.module.permutations.axis_count() {
            push_u64(
                &mut key,
                binding
                    .module
                    .permutations
                    .axis_options(axis)
                    .expect("axis index is bounded") as u64,
            );
        }
        match binding.module.full_lobe_fallback {
            Some(index) => {
                key.push(1);
                push_u64(&mut key, index.get());
            }
            None => key.push(0),
        }
        match binding.permutation {
            Some(permutation) => {
                key.push(1);
                for coordinate in permutation.key.coords() {
                    push_u32(&mut key, *coordinate);
                }
            }
            None => key.push(0),
        }
    }
    key
}

fn resolve_permutation(
    selected: &SelectedFeature,
    binding: &FeaturePassBinding,
) -> Result<PermutationIndex, FeatureMappingError> {
    let selected_permutation = selected.variant().shader_permutation;
    if selected_permutation != binding.permutation {
        return Err(FeatureMappingError::PermutationBindingMismatch {
            feature: selected.feature_id(),
            variant: selected.variant_id(),
        });
    }
    let empty_space = PermutationSpace::new([]);
    let empty_key = PermutationKey::new([]);
    let (space, key) = match binding.permutation {
        Some(permutation) => (permutation.space, permutation.key),
        None => (&empty_space, &empty_key),
    };
    if binding.module.permutations != *space {
        return Err(FeatureMappingError::PermutationBindingMismatch {
            feature: selected.feature_id(),
            variant: selected.variant_id(),
        });
    }
    space
        .checked_index_of(key)
        .map_err(FeatureMappingError::Permutation)
}

#[cfg(test)]
mod tests {
    use super::*;

    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::backend::capability::CapabilitySet;
    use prism_render_architecture::render_fabric::{
        FeatureCatalog, FeatureDefinition, FeatureKind, FeatureVariant, FrozenRoots,
        RenderQualityTier, ServiceSet,
    };
    use prism_render_architecture::shader_package::{
        ArtifactDigest, ExecutionFamily, ModuleIdentity, ShaderPackageId,
    };
    use prism_render_driver::{
        DeviceDescriptor, Instance, PrimitiveTopology, RequestAdapterOptions,
    };
    use prism_render_driver_null::NullInstance;
    use prism_render_material::fallback_material_record;

    const FEATURE: FeatureId = FeatureId(0x9000);
    const VARIANT: VariantId = VariantId(0x9001);

    fn static_ref<T>(value: T) -> &'static T {
        Box::leak(Box::new(value))
    }

    fn selection() -> (
        FeatureSelection,
        &'static PermutationSpace,
        &'static PermutationKey,
    ) {
        let space = static_ref(PermutationSpace::new([2, 3]));
        let key = static_ref(PermutationKey::new([1, 2]));
        let variant = FeatureVariant {
            id: VARIANT,
            rank: 0,
            min_quality: RenderQualityTier::Low,
            required_caps: CapabilitySet::EMPTY,
            requires_services_extra: ServiceSet::EMPTY,
            passes: &[],
            shader_permutation: Some(ShaderPermutationBinding { space, key }),
        };
        let definition = static_ref(FeatureDefinition {
            id: FEATURE,
            kind: FeatureKind::Response,
            domain: None,
            requires_services: ServiceSet::EMPTY,
            variants: static_ref([variant]),
        });
        let definitions = static_ref([*definition]);
        let catalog = FeatureCatalog::new(definitions, &[]).unwrap();
        let roots = FrozenRoots::from_build_profile(&catalog, &[], &[FEATURE]).unwrap();
        (
            FeatureSelection::resolve(
                &catalog,
                &roots,
                CapabilitySet::EMPTY,
                RenderQualityTier::Low,
            )
            .unwrap(),
            space,
            key,
        )
    }

    fn null_device() -> prism_render_driver_null::NullDevice {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .unwrap();
        instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0
    }

    fn spec(descriptor: prism_render_driver::ShaderModuleDescriptor) -> ResponseModuleSpec {
        ResponseModuleSpec {
            execution_family: ExecutionFamily::FusedBuiltin,
            identity: ModuleIdentity {
                package: ShaderPackageId::new("test.feature.response"),
                digest: ArtifactDigest([0x90; 32]),
                entry_point: "main".into(),
                binding_layout_digest: [0x91; 32],
            },
            permutations: PermutationSpace::new([2, 3]),
            full_lobe_fallback: None,
            descriptor,
            builtin: true,
        }
    }

    fn binding(
        variant: VariantId,
        permutation: Option<ShaderPermutationBinding>,
        module: ResponseModuleSpec,
    ) -> FeaturePassBinding {
        FeaturePassBinding {
            feature_id: FEATURE,
            variant_id: variant,
            pass_key: None,
            permutation,
            module,
            role: PassRole::Surface,
            render_phase: RenderPhase::Opaque,
            draw: None,
            sort_class: PassDescriptorClass::SURFACE,
        }
    }

    #[test]
    fn maps_real_shader_module_and_reuses_cache_identity() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let binding = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        let mut mapper = FeatureSelectionMapper::new(&device, &mut modules);
        let first = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        let second = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        assert_eq!(first.passes()[0].permutation_index.get(), 5);
        assert_eq!(first.passes(), second.passes());
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn rejects_variant_and_permutation_mismatches() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let module = spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
            "@compute @workgroup_size(1) fn main() {}",
        ));
        let mut mapper = FeatureSelectionMapper::new(&device, &mut modules);
        assert!(matches!(
            mapper.map_feature(
                &selection,
                FEATURE,
                &[binding(
                    VariantId(0xdead),
                    Some(ShaderPermutationBinding { space, key }),
                    module.clone()
                )]
            ),
            Err(FeatureMappingError::SelectionBindingMismatch { .. })
        ));
        let wrong_space = static_ref(PermutationSpace::new([2, 2]));
        assert!(matches!(
            mapper.map_feature(
                &selection,
                FEATURE,
                &[binding(
                    VARIANT,
                    Some(ShaderPermutationBinding { space, key }),
                    ResponseModuleSpec {
                        permutations: (*wrong_space).clone(),
                        ..module
                    },
                )]
            ),
            Err(FeatureMappingError::PermutationBindingMismatch { .. })
        ));
    }

    #[test]
    fn rejects_changed_descriptor_for_same_module_identity() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let first_spec = spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
            "@compute @workgroup_size(1) fn main() {}",
        ));
        let changed_spec = spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
            "@compute @workgroup_size(1) fn main() { let x: u32 = 1u; }",
        ));
        let mut mapper = FeatureSelectionMapper::new(&device, &mut modules);
        mapper
            .map_feature(
                &selection,
                FEATURE,
                &[binding(
                    VARIANT,
                    Some(ShaderPermutationBinding { space, key }),
                    first_spec,
                )],
            )
            .unwrap();
        assert!(matches!(
            mapper.map_feature(
                &selection,
                FEATURE,
                &[binding(
                    VARIANT,
                    Some(ShaderPermutationBinding { space, key }),
                    changed_spec,
                )]
            ),
            Err(FeatureMappingError::Module(
                ResponseModuleCacheError::DescriptorChanged(_)
            ))
        ));
    }

    #[test]
    fn installed_passes_are_shared_by_direct_and_bin_resolution() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let binding = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        let set = FeatureSelectionMapper::new(&device, &mut modules)
            .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        let material = fallback_material_record(GenerationalHandle::new(4, 1), 1);
        let mut registry = SceneRegistry::new();
        let table_index = set.install(&mut registry, &material).unwrap();
        registry.insert_mesh_class(
            7,
            crate::MeshClass::new(Vec::new(), PrimitiveTopology::TriangleList),
        );
        let key = prism_render_visibility::DrawBinKey {
            geometry: GenerationalHandle::new(1, 1),
            lod_or_cluster: 0,
            pipeline_class: table_index,
            vertex_buffer_class: 7,
            index_buffer_class: 0,
            indexed: true,
            primitive_kind: prism_render_architecture::geometry::GeometryPrimitiveKind::Indexed,
            pass_mask: 0,
        };
        let bin = registry.bin_variants(&key, None, 1).unwrap();
        assert_eq!(bin.len(), 1);
        assert_eq!(
            bin[0],
            set.passes()[0].mesh_variant(
                PrimitiveTopology::TriangleList,
                Some(crate::RENDER_TARGET_FORMAT),
                None,
                1
            )
        );
        assert_eq!(registry.pass_descriptors(table_index), Some(set.passes()));
    }
}
