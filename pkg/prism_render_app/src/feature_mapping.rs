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
    AccessKind, FeatureId, FeatureSelection, PassKey, PassTemplate, QueueClass, RenderQualityTier,
    ResourceInput, ResourceKey, ResourceOwner, ResourceScope, ResourceUse, SelectedFeature,
    ShaderPermutationBinding, VariantId,
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

    /// Invalidates every mapping after a device or response-module cache
    /// recovery. Counters are retained so diagnostics can observe the
    /// invalidation boundary without losing the preceding hit/miss history.
    pub fn invalidate(&mut self) {
        self.entries.clear();
    }

    /// Compatibility alias for callers that used the cache as a plain
    /// clearable container. New recovery code should call [`Self::invalidate`]
    /// so the device/module lifecycle is explicit at the call site.
    pub fn clear(&mut self) {
        self.invalidate();
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
        let cache_key = mapping_cache_key(selection, selected, &ordered);
        if let Some(cache) = self.cache.as_deref_mut() {
            if let Some(cached) = cache.entries.get(&cache_key).cloned() {
                let mut valid = cached.passes().len() == ordered.len();
                if valid {
                    for (binding, pass) in ordered.iter().zip(cached.passes()) {
                        let permutation = resolve_permutation(selected, binding)?;
                        let module_id = self
                            .modules
                            .intern(self.device, binding.module.clone())
                            .map_err(FeatureMappingError::Module)?;
                        // A module cache may have been replaced rather than
                        // rebuilt in place. Do not return descriptors carrying
                        // module IDs from the old cache generation.
                        if module_id != pass.response_module_id
                            || permutation != pass.permutation_index
                        {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    cache.hits = cache.hits.saturating_add(1);
                    return Ok(cached);
                }
                cache.entries.remove(&cache_key);
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

fn mapping_cache_key(
    selection: &FeatureSelection,
    selected: &SelectedFeature,
    bindings: &[&FeaturePassBinding],
) -> Vec<u8> {
    // The mapper is device-local, but the cache key must still describe the
    // complete frozen selection. In particular, an author may replace a
    // variant declaration while retaining its numeric feature/variant IDs.
    // Encoding the public semantic identity prevents that replacement from
    // reusing a stale pass set.
    let mut key = Vec::with_capacity(512);
    push_u32(&mut key, 0x504d_4348); // versioned "PMCH" key format
    push_u64(&mut key, selection.roots().ids().len() as u64);
    for id in selection.roots().ids() {
        push_u32(&mut key, id.0);
    }
    push_u64(&mut key, selection.roots().subsystems().len() as u64);
    for id in selection.roots().subsystems() {
        push_u32(&mut key, id.0);
    }
    push_u64(&mut key, selection.roots().responses().len() as u64);
    for id in selection.roots().responses() {
        push_u32(&mut key, id.0);
    }
    push_u32(&mut key, selection.device_caps().bits());
    push_u8(&mut key, quality_tag(selection.quality()));
    push_ids(&mut key, selection.closure());
    push_u64(&mut key, selection.selected().len() as u64);
    for feature in selection.selected() {
        push_selected_feature_identity(&mut key, feature);
    }
    let services: Vec<_> = selection.services().collect();
    push_u64(&mut key, services.len() as u64);
    for provider in services {
        push_u32(&mut key, provider.service().0);
        push_u32(&mut key, provider.feature().0);
        push_resource_keys(&mut key, provider.output().resources);
    }

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
        push_bytes(
            &mut key,
            binding.module.identity.package.as_str().as_bytes(),
        );
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
                push_permutation_space(&mut key, permutation.space);
                push_permutation_key(&mut key, permutation.key);
            }
            None => key.push(0),
        }
    }
    key
}

fn push_u8(bytes: &mut Vec<u8>, value: u8) {
    bytes.push(value);
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_ids(bytes: &mut Vec<u8>, ids: &[FeatureId]) {
    push_u64(bytes, ids.len() as u64);
    for id in ids {
        push_u32(bytes, id.0);
    }
}

fn push_selected_feature_identity(bytes: &mut Vec<u8>, selected: &SelectedFeature) {
    push_u32(bytes, selected.feature_id().0);
    push_u32(bytes, selected.variant_id().0);
    let variant = selected.variant();
    push_u16(bytes, variant.rank);
    push_u8(bytes, quality_tag(variant.min_quality));
    push_u32(bytes, variant.required_caps.bits());
    push_service_ids(bytes, variant.requires_services_extra.as_slice());
    push_pass_templates(bytes, variant.passes);
    push_shader_permutation(bytes, variant.shader_permutation);
}

fn push_service_ids(
    bytes: &mut Vec<u8>,
    services: &[prism_render_architecture::render_fabric::ServiceId],
) {
    let mut sorted = services.to_vec();
    sorted.sort_unstable();
    push_u64(bytes, sorted.len() as u64);
    for service in sorted {
        push_u32(bytes, service.0);
    }
}

fn push_pass_templates(bytes: &mut Vec<u8>, passes: &[PassTemplate]) {
    push_u64(bytes, passes.len() as u64);
    for pass in passes {
        push_u32(bytes, pass.id.0);
        push_u8(bytes, queue_tag(pass.queue));
        push_u64(bytes, pass.accesses.len() as u64);
        for access in pass.accesses {
            push_resource_use(bytes, access);
        }
        push_u64(bytes, pass.depends_on.len() as u64);
        for dependency in pass.depends_on {
            push_u32(bytes, dependency.0);
        }
    }
}

fn push_resource_use(bytes: &mut Vec<u8>, usage: &ResourceUse) {
    push_resource_key(bytes, usage.key);
    push_u8(bytes, access_tag(usage.access));
    push_u8(bytes, scope_tag(usage.scope));
    match usage.frame_lag {
        Some(lag) => {
            push_u8(bytes, 1);
            push_u32(bytes, lag.frames());
        }
        None => push_u8(bytes, 0),
    }
    match usage.input {
        ResourceInput::Automatic => push_u8(bytes, 0),
        ResourceInput::Initial => push_u8(bytes, 1),
        ResourceInput::ProducedBy(pass) => {
            push_u8(bytes, 2);
            push_u32(bytes, pass.0);
        }
    }
}

fn push_resource_keys(bytes: &mut Vec<u8>, keys: &[ResourceKey]) {
    push_u64(bytes, keys.len() as u64);
    for key in keys {
        push_resource_key(bytes, *key);
    }
}

fn push_resource_key(bytes: &mut Vec<u8>, key: ResourceKey) {
    match key.owner() {
        ResourceOwner::Engine => {
            push_u8(bytes, 0);
        }
        ResourceOwner::Feature(feature) => {
            push_u8(bytes, 1);
            push_u32(bytes, feature.0);
        }
    }
    push_u16(bytes, key.slot());
}

fn push_shader_permutation(bytes: &mut Vec<u8>, permutation: Option<ShaderPermutationBinding>) {
    match permutation {
        Some(permutation) => {
            push_u8(bytes, 1);
            push_permutation_space(bytes, permutation.space);
            push_permutation_key(bytes, permutation.key);
        }
        None => push_u8(bytes, 0),
    }
}

fn push_permutation_space(bytes: &mut Vec<u8>, space: &PermutationSpace) {
    push_u64(bytes, space.axis_count() as u64);
    for axis in 0..space.axis_count() {
        push_u32(bytes, space.axis_options(axis).expect("bounded axis"));
    }
}

fn push_permutation_key(bytes: &mut Vec<u8>, key: &PermutationKey) {
    push_u64(bytes, key.coords().len() as u64);
    for coordinate in key.coords() {
        push_u32(bytes, *coordinate);
    }
}

fn quality_tag(quality: RenderQualityTier) -> u8 {
    match quality {
        RenderQualityTier::Low => 0,
        RenderQualityTier::Medium => 1,
        RenderQualityTier::High => 2,
        RenderQualityTier::Ultra => 3,
    }
}

fn queue_tag(queue: QueueClass) -> u8 {
    match queue {
        QueueClass::Graphics => 0,
        QueueClass::Compute => 1,
        QueueClass::Transfer => 2,
    }
}

fn access_tag(access: AccessKind) -> u8 {
    match access {
        AccessKind::SampledRead => 0,
        AccessKind::StorageRead => 1,
        AccessKind::StorageWrite => 2,
        AccessKind::ColorAttachment => 3,
        AccessKind::DepthAttachment => 4,
        AccessKind::IndirectRead => 5,
        AccessKind::TransferRead => 6,
        AccessKind::TransferWrite => 7,
    }
}

fn scope_tag(scope: ResourceScope) -> u8 {
    match scope {
        ResourceScope::Whole => 0,
    }
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
        selection_with_rank(0)
    }

    fn selection_with_rank(
        rank: u16,
    ) -> (
        FeatureSelection,
        &'static PermutationSpace,
        &'static PermutationKey,
    ) {
        let space = static_ref(PermutationSpace::new([2, 3]));
        let key = static_ref(PermutationKey::new([1, 2]));
        let variant = FeatureVariant {
            id: VARIANT,
            rank,
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
        let mut mapping_cache = FeatureMappingCache::default();
        let mut mapper =
            FeatureSelectionMapper::with_cache(&device, &mut modules, &mut mapping_cache);
        let first = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        let second = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        assert_eq!(first.passes()[0].permutation_index.get(), 5);
        assert_eq!(first.passes(), second.passes());
        assert_eq!(modules.len(), 1);
        assert_eq!(mapping_cache.len(), 1);
        assert_eq!(
            mapping_cache.stats(),
            FeatureMappingCacheStats { hits: 1, misses: 1 }
        );
    }

    #[test]
    fn explicit_invalidate_forces_rebuild_without_resetting_counters() {
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
        let mut mapping_cache = FeatureMappingCache::default();
        {
            let mut mapper =
                FeatureSelectionMapper::with_cache(&device, &mut modules, &mut mapping_cache);
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
        }
        mapping_cache.invalidate();
        assert!(mapping_cache.is_empty());
        {
            let mut mapper =
                FeatureSelectionMapper::with_cache(&device, &mut modules, &mut mapping_cache);
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
        }
        assert_eq!(mapping_cache.len(), 1);
        assert_eq!(mapping_cache.stats().hits, 1);
        assert_eq!(mapping_cache.stats().misses, 2);
    }

    #[test]
    fn selection_identity_drift_does_not_reuse_mapping() {
        let (low_selection, space, key) = selection();
        let high_selection = {
            let definition = static_ref(FeatureDefinition {
                id: FEATURE,
                kind: FeatureKind::Response,
                domain: None,
                requires_services: ServiceSet::EMPTY,
                variants: static_ref([FeatureVariant {
                    id: VARIANT,
                    rank: 0,
                    min_quality: RenderQualityTier::Low,
                    required_caps: CapabilitySet::EMPTY,
                    requires_services_extra: ServiceSet::EMPTY,
                    passes: &[],
                    shader_permutation: Some(ShaderPermutationBinding { space, key }),
                }]),
            });
            let definitions = static_ref([*definition]);
            let catalog = FeatureCatalog::new(definitions, &[]).unwrap();
            let roots = FrozenRoots::from_build_profile(&catalog, &[], &[FEATURE]).unwrap();
            FeatureSelection::resolve(
                &catalog,
                &roots,
                CapabilitySet::EMPTY,
                RenderQualityTier::High,
            )
            .unwrap()
        };
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let binding = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        let mut mapping_cache = FeatureMappingCache::default();
        let mut mapper =
            FeatureSelectionMapper::with_cache(&device, &mut modules, &mut mapping_cache);
        mapper
            .map_feature(&low_selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        mapper
            .map_feature(&high_selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        assert_eq!(mapping_cache.stats().hits, 0);
        assert_eq!(mapping_cache.stats().misses, 2);
    }

    #[test]
    fn module_identity_drift_is_a_distinct_mapping_and_module() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let first = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        let mut changed_spec = first.module.clone();
        changed_spec.identity.digest = ArtifactDigest([0x92; 32]);
        let changed = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            changed_spec,
        );
        let mut mapping_cache = FeatureMappingCache::default();
        let mut mapper =
            FeatureSelectionMapper::with_cache(&device, &mut modules, &mut mapping_cache);
        let first_set = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&first))
            .unwrap();
        let changed_set = mapper
            .map_feature(&selection, FEATURE, core::slice::from_ref(&changed))
            .unwrap();
        assert_ne!(
            first_set.passes()[0].response_module_id,
            changed_set.passes()[0].response_module_id
        );
        assert_eq!(modules.len(), 2);
        assert_eq!(mapping_cache.stats().hits, 0);
        assert_eq!(mapping_cache.stats().misses, 2);
    }

    #[test]
    fn cache_reuses_then_explicitly_invalidates_mapping() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let mut cache = FeatureMappingCache::default();
        let binding = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        {
            let mut mapper = FeatureSelectionMapper::with_cache(&device, &mut modules, &mut cache);
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
        }
        assert_eq!(
            cache.stats(),
            FeatureMappingCacheStats { hits: 1, misses: 1 }
        );
        assert_eq!(cache.len(), 1);

        cache.invalidate();
        assert!(cache.is_empty());
        {
            let mut mapper = FeatureSelectionMapper::with_cache(&device, &mut modules, &mut cache);
            mapper
                .map_feature(&selection, FEATURE, core::slice::from_ref(&binding))
                .unwrap();
        }
        assert_eq!(
            cache.stats(),
            FeatureMappingCacheStats { hits: 1, misses: 2 }
        );
    }

    #[test]
    fn cache_key_rejects_selection_declaration_identity_drift() {
        let (first_selection, space, key) = selection_with_rank(0);
        let (second_selection, _, _) = selection_with_rank(1);
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let mut cache = FeatureMappingCache::default();
        let binding = binding(
            VARIANT,
            Some(ShaderPermutationBinding { space, key }),
            spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                "@compute @workgroup_size(1) fn main() {}",
            )),
        );
        let mut mapper = FeatureSelectionMapper::with_cache(&device, &mut modules, &mut cache);
        mapper
            .map_feature(&first_selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        mapper
            .map_feature(&second_selection, FEATURE, core::slice::from_ref(&binding))
            .unwrap();
        assert_eq!(
            cache.stats(),
            FeatureMappingCacheStats { hits: 0, misses: 2 }
        );
    }

    #[test]
    fn failed_install_does_not_publish_material_or_pass_set() {
        let material = fallback_material_record(GenerationalHandle::new(8, 1), 1);
        let mut registry = SceneRegistry::new();
        let invalid = FeaturePassSet {
            feature_id: FEATURE,
            variant_id: VARIANT,
            passes: vec![PassDescriptor {
                role: PassRole::ResponseLeft,
                execution_family: ExecutionFamily::SplitResponse,
                response_module_id: prism_render_architecture::shader_package::ModuleId(7),
                permutation_index: PermutationIndex::ZERO,
                fallback_permutation: None,
                render_phase: RenderPhase::Opaque,
                draw: Some(DrawFunctionId(0)),
                sort_class: PassDescriptorClass::SURFACE,
            }],
        };
        assert!(matches!(
            invalid.install(&mut registry, &material),
            Err(FeatureMappingError::PassSet(PassSetError::InvalidShape))
        ));
        assert_eq!(registry.material_count(), 0);
        assert_eq!(registry.material_pass_index(material.handle), None);
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

    #[test]
    fn failed_feature_install_does_not_publish_material_or_pass_binding() {
        let (selection, space, key) = selection();
        let device = null_device();
        let mut modules = ResponseModuleCache::new();
        let mut registry = SceneRegistry::new();
        let material = fallback_material_record(GenerationalHandle::new(8, 1), 1);
        let result = registry.insert_material_with_feature_selection(
            &device,
            &mut modules,
            material.clone(),
            &selection,
            FEATURE,
            &[binding(
                VariantId(0xdead),
                Some(ShaderPermutationBinding { space, key }),
                spec(prism_render_driver::ShaderModuleDescriptor::wgsl(
                    "@compute @workgroup_size(1) fn main() {}",
                )),
            )],
        );
        assert!(matches!(
            result,
            Err(FeatureMappingError::SelectionBindingMismatch { .. })
        ));
        assert_eq!(registry.material_count(), 0);
        assert!(registry.material(material.handle).is_none());
        assert!(registry.material_pass_index(material.handle).is_none());
        assert_eq!(registry.feature_mapping_cache_len(), 0);
    }
}
