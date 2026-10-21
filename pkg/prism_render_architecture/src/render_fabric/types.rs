use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;
use core::num::NonZeroU32;

use super::selection::{FeatureSelection, SelectedFeature};

pub use crate::backend::capability::CapabilitySet;
pub use crate::frame_graph::{AccessKind, QueueClass};
use crate::shader_package::{
    ArtifactDigest, ModuleIdentity, PermutationError, PermutationIndex, PermutationKey,
    PermutationSpace, PipelineStateHash, PsoCacheKey, ShaderPackageId,
};

/// Author-supplied, persistent semantic identity; never assigned by a catalog.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FeatureId(pub u32);

/// Globally unique, explicitly authored implementation identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariantId(pub u32);

/// Globally stable pass-template identity, independent of frame-local pass IDs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PassKey(pub u32);

/// Explicit semantic service identity. No built-in or engine-provider exceptions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ServiceId(pub u32);

/// One reusable static set representation for both hard and soft requirements.
/// Repeated entries are rejected by the catalog; declaration order is irrelevant.
#[derive(Clone, Copy, Debug)]
pub struct ServiceSet(&'static [ServiceId]);

impl PartialEq for ServiceSet {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().all(|service| other.0.contains(service))
            && other.0.iter().all(|service| self.0.contains(service))
    }
}

impl Eq for ServiceSet {}

impl ServiceSet {
    pub const EMPTY: Self = Self(&[]);

    pub const fn new(services: &'static [ServiceId]) -> Self {
        Self(services)
    }

    pub const fn as_slice(self) -> &'static [ServiceId] {
        self.0
    }

    pub fn contains(self, service: ServiceId) -> bool {
        self.0.contains(&service)
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn is_superset(self, other: Self) -> bool {
        other.0.iter().all(|&service| self.contains(service))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResourceOwner {
    Engine,
    Feature(FeatureId),
}

/// Namespaced semantic key. Declarations and references are separate tables.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceKey {
    owner: ResourceOwner,
    slot: u16,
}

impl ResourceKey {
    pub const fn engine(slot: u16) -> Self {
        Self {
            owner: ResourceOwner::Engine,
            slot,
        }
    }

    pub const fn of(owner: FeatureId, slot: u16) -> Self {
        Self {
            owner: ResourceOwner::Feature(owner),
            slot,
        }
    }

    pub const fn owner(self) -> ResourceOwner {
        self.owner
    }
    pub const fn slot(self) -> u16 {
        self.slot
    }
}

/// Whole-resource access only until the authoritative graph supports subresources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceScope {
    Whole,
}

/// Positive history distance. The parent integration must map this to persistent
/// frame-versioned reads, never to same-frame dependency edges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameLag(NonZeroU32);

impl FrameLag {
    pub const fn new(frames: u32) -> Option<Self> {
        match NonZeroU32::new(frames) {
            Some(frames) => Some(Self(frames)),
            None => None,
        }
    }

    pub const fn frames(self) -> u32 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceInput {
    /// Reads the unique selected writer, or imported initial contents when no
    /// writer exists. Writes consume initial contents. Multiple writers require
    /// explicit inputs; registration order never selects a producer.
    Automatic,
    /// Version zero, including an explicitly imported history frame.
    Initial,
    /// The output of a selected pass writing this same current-frame resource.
    ProducedBy(PassKey),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceUse {
    pub key: ResourceKey,
    pub access: AccessKind,
    pub scope: ResourceScope,
    pub frame_lag: Option<FrameLag>,
    pub input: ResourceInput,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderDomain {
    Surface,
    Particle,
    Volume,
    Hair,
    Decal,
}

/// Response axis shared by domain renderers. It is orthogonal to
/// [`RenderDomain`] and to blend/raster state; a domain chooses where it is
/// drawn, while this value chooses how its lighting response is evaluated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResponseProgram {
    Pbr,
    Npr,
    Custom,
}

/// Device/configuration quality, not a per-frame budget or measurement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RenderQualityTier {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceOutput {
    pub service: ServiceId,
    pub resources: &'static [ResourceKey],
}

/// Only services can publish a semantic interface, enforced by enum shape.
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::FeatureKind;
/// let _ = FeatureKind::Subsystem { provides: &[] };
/// ```
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::FeatureKind;
/// let _ = FeatureKind::Response { provides: &[] };
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureKind {
    Subsystem,
    Service { provides: &'static [ServiceOutput] },
    Response,
}

impl FeatureKind {
    pub const fn provides(self) -> &'static [ServiceOutput] {
        match self {
            Self::Service { provides } => provides,
            Self::Subsystem | Self::Response => &[],
        }
    }

    pub const fn is_service(self) -> bool {
        matches!(self, Self::Service { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassTemplate {
    pub id: PassKey,
    pub accesses: &'static [ResourceUse],
    pub queue: QueueClass,
    /// Semantic control prerequisites, retained even without resource hazards.
    pub depends_on: &'static [PassKey],
}

/// Bind an existing shader permutation system, without another variant enum.
/// `None` on a feature variant means it requires no shader permutation binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderPermutationBinding {
    pub space: &'static PermutationSpace,
    pub key: &'static PermutationKey,
}

impl ShaderPermutationBinding {
    /// Resolves the selected key against the exact space declared by the
    /// variant.  Callers must use this result when constructing a pipeline
    /// identity; a dense index supplied independently is not accepted by the
    /// Render Fabric binding layer.
    pub fn checked_index(&self) -> Result<PermutationIndex, PermutationError> {
        self.space.checked_index_of(self.key)
    }
}

/// The pass identity exported by a Render Fabric variant's shader package.
///
/// `PassTemplate` is the semantic declaration consumed by the fabric compiler;
/// this owned value is the package-side copy of the same identity. Keeping the
/// copy explicit lets package loading reject a stale or incorrectly paired pass
/// table before it reaches a driver-local cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassDescriptorIdentity {
    pub key: PassKey,
    pub accesses: &'static [ResourceUse],
    pub queue: QueueClass,
    pub depends_on: &'static [PassKey],
}

impl PassDescriptorIdentity {
    #[must_use]
    pub const fn from_template(template: &'static PassTemplate) -> Self {
        Self {
            key: template.id,
            accesses: template.accesses,
            queue: template.queue,
            depends_on: template.depends_on,
        }
    }

    #[must_use]
    pub fn matches_template(&self, template: &PassTemplate) -> bool {
        self.key == template.id
            && self.queue == template.queue
            && self.accesses == template.accesses
            && self.depends_on == template.depends_on
    }
}

/// Driver-independent module declaration carried by a Render Fabric package
/// binding. `ModuleId` is intentionally absent: it is allocated by the driver
/// interner and must never become part of a portable feature selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderModuleBinding {
    pub identity: ModuleIdentity,
    pub permutations: PermutationSpace,
    pub full_lobe_fallback: Option<PermutationIndex>,
}

/// Complete package-side identity for one selected feature variant.
///
/// This is the single bridge between semantic feature selection and executable
/// shader declarations. The package, module declaration, permutation binding,
/// and pass descriptor identities are validated together, so a caller cannot
/// accidentally combine a key from one package with a module or pass table from
/// another package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderPackageBinding {
    pub package: ShaderPackageId,
    pub module: ShaderModuleBinding,
    pub permutation: Option<ShaderPermutationBinding>,
    pub pass_descriptors: Vec<PassDescriptorIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderBindingError {
    PackageModuleMismatch {
        package: ShaderPackageId,
        module_package: ShaderPackageId,
    },
    PermutationBindingMismatch {
        variant: VariantId,
    },
    Permutation(PermutationError),
    ModulePermutationMismatch {
        variant: VariantId,
    },
    PassDescriptorCountMismatch {
        variant: VariantId,
        expected: usize,
        actual: usize,
    },
    PassDescriptorMismatch {
        variant: VariantId,
        pass: PassKey,
    },
    CacheIdentityChanged {
        variant: VariantId,
    },
    CacheEntryMissing {
        variant: VariantId,
    },
    CacheIdentityMismatch {
        variant: VariantId,
    },
    MissingVariantBinding {
        variant: VariantId,
    },
    UnexpectedVariantBinding {
        variant: VariantId,
    },
    BuiltinVariantUnavailable {
        variant: VariantId,
    },
}

impl fmt::Display for ShaderBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render fabric shader binding error: {self:?}")
    }
}

impl std::error::Error for ShaderBindingError {}

impl ShaderPackageBinding {
    /// Validates the complete package declaration against one semantic variant.
    pub fn validate_for(&self, variant: &FeatureVariant) -> Result<(), ShaderBindingError> {
        if self.package != self.module.identity.package {
            return Err(ShaderBindingError::PackageModuleMismatch {
                package: self.package,
                module_package: self.module.identity.package,
            });
        }

        if self.permutation != variant.shader_permutation {
            return Err(ShaderBindingError::PermutationBindingMismatch {
                variant: variant.id,
            });
        }

        let expected_space = variant
            .shader_permutation
            .map(|binding| binding.space.clone())
            .unwrap_or_default();
        if self.module.permutations != expected_space {
            return Err(ShaderBindingError::ModulePermutationMismatch {
                variant: variant.id,
            });
        }
        if let Some(permutation) = self.permutation {
            permutation
                .checked_index()
                .map_err(ShaderBindingError::Permutation)?;
        }
        if let Some(fallback) = self.module.full_lobe_fallback {
            self.module
                .permutations
                .check_index(fallback.get())
                .map_err(ShaderBindingError::Permutation)?;
        }

        if self.pass_descriptors.len() != variant.passes.len() {
            return Err(ShaderBindingError::PassDescriptorCountMismatch {
                variant: variant.id,
                expected: variant.passes.len(),
                actual: self.pass_descriptors.len(),
            });
        }
        for (descriptor, template) in self.pass_descriptors.iter().zip(variant.passes) {
            if !descriptor.matches_template(template) {
                return Err(ShaderBindingError::PassDescriptorMismatch {
                    variant: variant.id,
                    pass: template.id,
                });
            }
        }
        Ok(())
    }

    /// Returns the dense index after validating the package's selected key.
    pub fn permutation_index(&self) -> Result<PermutationIndex, ShaderBindingError> {
        match self.permutation {
            Some(permutation) => permutation
                .checked_index()
                .map_err(ShaderBindingError::Permutation),
            None => PermutationSpace::new([])
                .check_index(0)
                .map_err(ShaderBindingError::Permutation),
        }
    }

    /// Stable cold identity used by the package binding cache.
    #[must_use]
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut bytes = Vec::new();
        append_len_prefixed(&mut bytes, self.package.as_str().as_bytes());
        append_len_prefixed(&mut bytes, self.module.identity.package.as_str().as_bytes());
        append_len_prefixed(&mut bytes, self.module.identity.entry_point.as_bytes());
        bytes.extend_from_slice(&self.module.identity.digest.0);
        bytes.extend_from_slice(&self.module.identity.binding_layout_digest);
        bytes.extend_from_slice(&(self.module.permutations.axis_count() as u32).to_le_bytes());
        for axis in 0..self.module.permutations.axis_count() {
            bytes.extend_from_slice(
                &self
                    .module
                    .permutations
                    .axis_options(axis)
                    .expect("axis index is bounded by axis_count")
                    .to_le_bytes(),
            );
        }
        if let Some(permutation) = self.permutation {
            bytes.push(1);
            bytes.extend_from_slice(&(permutation.key.coords().len() as u32).to_le_bytes());
            for coordinate in permutation.key.coords() {
                bytes.extend_from_slice(&coordinate.to_le_bytes());
            }
        } else {
            bytes.push(0);
        }
        if let Some(fallback) = self.module.full_lobe_fallback {
            bytes.push(1);
            bytes.extend_from_slice(&fallback.get().to_le_bytes());
        } else {
            bytes.push(0);
        }
        bytes.extend_from_slice(&(self.pass_descriptors.len() as u32).to_le_bytes());
        for descriptor in &self.pass_descriptors {
            bytes.extend_from_slice(&descriptor.key.0.to_le_bytes());
            bytes.push(queue_tag(descriptor.queue));
            bytes.extend_from_slice(&(descriptor.accesses.len() as u32).to_le_bytes());
            for access in descriptor.accesses {
                match access.key.owner() {
                    ResourceOwner::Engine => bytes.push(0),
                    ResourceOwner::Feature(feature) => {
                        bytes.push(1);
                        bytes.extend_from_slice(&feature.0.to_le_bytes());
                    }
                }
                bytes.extend_from_slice(&access.key.slot().to_le_bytes());
                bytes.push(access_tag(access.access));
                bytes.push(scope_tag(access.scope));
                match access.frame_lag {
                    None => bytes.push(0),
                    Some(lag) => {
                        bytes.push(1);
                        bytes.extend_from_slice(&lag.frames().to_le_bytes());
                    }
                }
                match access.input {
                    ResourceInput::Automatic => bytes.push(0),
                    ResourceInput::Initial => bytes.push(1),
                    ResourceInput::ProducedBy(pass) => {
                        bytes.push(2);
                        bytes.extend_from_slice(&pass.0.to_le_bytes());
                    }
                }
            }
            bytes.extend_from_slice(&(descriptor.depends_on.len() as u32).to_le_bytes());
            for dependency in descriptor.depends_on {
                bytes.extend_from_slice(&dependency.0.to_le_bytes());
            }
        }
        ArtifactDigest::from_bytes(&bytes)
    }
}

fn append_len_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value);
}

const fn queue_tag(queue: QueueClass) -> u8 {
    match queue {
        QueueClass::Graphics => 0,
        QueueClass::Compute => 1,
        QueueClass::Transfer => 2,
    }
}

const fn access_tag(access: AccessKind) -> u8 {
    match access {
        AccessKind::SampledRead => 0,
        AccessKind::StorageRead => 1,
        AccessKind::StorageWrite => 2,
        AccessKind::ColorAttachment => 3,
        AccessKind::DepthAttachment => 4,
        AccessKind::IndirectRead => 5,
        AccessKind::TransferRead => 6,
        AccessKind::TransferWrite => 7,
        AccessKind::UniformRead => 8,
        AccessKind::IndexRead => 9,
        AccessKind::DepthAttachmentRead => 10,
    }
}

const fn scope_tag(scope: ResourceScope) -> u8 {
    match scope {
        ResourceScope::Whole => 0,
    }
}

impl FeatureVariant {
    /// Returns the canonical package declaration for a built-in variant.
    ///
    /// User-defined variants must provide their own package declaration and
    /// use [`ShaderPackageBinding::validate_for`] before registration.
    pub fn builtin_shader_package(&self) -> Result<ShaderPackageBinding, ShaderBindingError> {
        super::builtins::builtin_shader_package_binding(self)
    }
}

impl SelectedFeature {
    /// Completes the package binding for this already-selected variant.
    ///
    /// The returned value carries the selected feature/variant identity and the
    /// validated package, module, permutation and pass descriptors together.
    pub fn bind_shader_package(
        &self,
        binding: ShaderPackageBinding,
    ) -> Result<SelectedFeatureShaderBinding, ShaderBindingError> {
        binding.validate_for(self.variant())?;
        Ok(SelectedFeatureShaderBinding {
            feature: self.feature_id(),
            variant: self.variant_id(),
            binding,
        })
    }

    /// Binds the canonical package declaration for a built-in variant.
    pub fn bind_builtin_shader_package(
        &self,
    ) -> Result<SelectedFeatureShaderBinding, ShaderBindingError> {
        self.bind_shader_package(self.variant().builtin_shader_package()?)
    }
}

/// Result of binding one selected feature. It is immutable and owns the full
/// package declaration, so consumers cannot later substitute only its key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedFeatureShaderBinding {
    feature: FeatureId,
    variant: VariantId,
    binding: ShaderPackageBinding,
}

impl SelectedFeatureShaderBinding {
    pub const fn feature_id(&self) -> FeatureId {
        self.feature
    }

    pub const fn variant_id(&self) -> VariantId {
        self.variant
    }

    pub fn package(&self) -> ShaderPackageId {
        self.binding.package
    }

    pub fn module(&self) -> &ShaderModuleBinding {
        &self.binding.module
    }

    pub fn permutation(&self) -> Option<ShaderPermutationBinding> {
        self.binding.permutation
    }

    pub fn permutation_index(&self) -> Result<PermutationIndex, ShaderBindingError> {
        self.binding.permutation_index()
    }

    pub fn pass_descriptors(&self) -> &[PassDescriptorIdentity] {
        &self.binding.pass_descriptors
    }

    pub fn identity_digest(&self) -> ArtifactDigest {
        self.binding.identity_digest()
    }

    /// Produces the cold PSO identity for this already-bound selection.
    ///
    /// This is intentionally the only architecture-side path from a selected
    /// package binding to a `PsoCacheKey`: the package owns the permutation
    /// space and the selected key, so consumers cannot substitute a different
    /// dense index while retaining the same feature/variant identity.
    pub fn pso_cache_key(
        &self,
        state: PipelineStateHash,
    ) -> Result<PsoCacheKey, ShaderBindingError> {
        Ok(PsoCacheKey::new(
            self.package(),
            self.permutation_index()?.get(),
            state,
        ))
    }

    pub fn binding(&self) -> &ShaderPackageBinding {
        &self.binding
    }
}

/// All shader package bindings derived from one immutable semantic selection.
///
/// The selection is retained alongside the package bindings so a production
/// consumer can prove that it is using the same frozen result used by feature
/// admission.  No method on this type performs feature resolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenShaderSelection {
    selection: FeatureSelection,
    bindings: Vec<SelectedFeatureShaderBinding>,
    identity: ArtifactDigest,
}

impl FrozenShaderSelection {
    pub fn selection(&self) -> &FeatureSelection {
        &self.selection
    }

    pub fn bindings(&self) -> &[SelectedFeatureShaderBinding] {
        &self.bindings
    }

    pub fn binding(&self, feature: FeatureId) -> Option<&SelectedFeatureShaderBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.feature_id() == feature)
    }

    /// Stable identity for cache generations and diagnostics.  The encoding is
    /// independent of declaration/registration order because the selection is
    /// already emitted in the deterministic resolver order and bindings are
    /// appended in that same order.
    pub fn identity_digest(&self) -> ArtifactDigest {
        self.identity
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl FeatureSelection {
    /// Binds every selected variant to a package declaration.  The map must
    /// contain exactly the selected variants: missing or extra entries are
    /// rejected so a consumer cannot silently fabricate a package/permutation
    /// for a feature that was not part of the frozen selection.
    pub fn bind_shader_packages(
        &self,
        packages: &BTreeMap<VariantId, ShaderPackageBinding>,
    ) -> Result<FrozenShaderSelection, ShaderBindingError> {
        let selected_variants: BTreeMap<_, _> = self
            .selected()
            .iter()
            .map(|selected| (selected.variant_id(), selected))
            .collect();
        for variant in selected_variants.keys() {
            if !packages.contains_key(variant) {
                return Err(ShaderBindingError::MissingVariantBinding { variant: *variant });
            }
        }
        for variant in packages.keys() {
            if !selected_variants.contains_key(variant) {
                return Err(ShaderBindingError::UnexpectedVariantBinding { variant: *variant });
            }
        }

        let mut bindings = Vec::with_capacity(self.selected().len());
        for selected in self.selected() {
            bindings.push(
                selected.bind_shader_package(
                    packages
                        .get(&selected.variant_id())
                        .expect("validated selected variant binding")
                        .clone(),
                )?,
            );
        }
        Ok(FrozenShaderSelection::new(self.clone(), bindings))
    }

    /// Binds the canonical package declaration for every selected built-in
    /// variant.  This is a convenience over [`Self::bind_shader_packages`],
    /// not a second selection or package resolver.
    pub fn bind_builtin_shader_packages(
        &self,
    ) -> Result<FrozenShaderSelection, ShaderBindingError> {
        let mut bindings = Vec::with_capacity(self.selected().len());
        for selected in self.selected() {
            bindings.push(selected.bind_builtin_shader_package()?);
        }
        Ok(FrozenShaderSelection::new(self.clone(), bindings))
    }
}

impl FrozenShaderSelection {
    fn new(selection: FeatureSelection, bindings: Vec<SelectedFeatureShaderBinding>) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PRISM-FROZEN-SHADER-SELECTION-V1");
        bytes.extend_from_slice(&(selection.roots().ids().len() as u64).to_le_bytes());
        for id in selection.roots().ids() {
            bytes.extend_from_slice(&id.0.to_le_bytes());
        }
        bytes.extend_from_slice(&selection.device_caps().bits().to_le_bytes());
        bytes.push(quality_tag(selection.quality()));
        bytes.extend_from_slice(&(bindings.len() as u64).to_le_bytes());
        for binding in &bindings {
            bytes.extend_from_slice(&binding.feature_id().0.to_le_bytes());
            bytes.extend_from_slice(&binding.variant_id().0.to_le_bytes());
            bytes.extend_from_slice(&binding.identity_digest().0);
        }
        let identity = ArtifactDigest::from_bytes(&bytes);
        Self {
            selection,
            bindings,
            identity,
        }
    }
}

const fn quality_tag(quality: RenderQualityTier) -> u8 {
    match quality {
        RenderQualityTier::Low => 0,
        RenderQualityTier::Medium => 1,
        RenderQualityTier::High => 2,
        RenderQualityTier::Ultra => 3,
    }
}

/// Result of inserting or reusing one immutable variant binding in the cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderBindingCacheStatus {
    Inserted,
    Reused,
}

/// A small device/profile-local cache that refuses silent shader identity drift.
/// A changed package/module/pass declaration must be explicitly invalidated
/// before a new declaration can be admitted for the same semantic variant.
#[derive(Clone, Debug, Default)]
pub struct ShaderBindingCache {
    entries: BTreeMap<VariantId, ShaderPackageBinding>,
}

impl ShaderBindingCache {
    pub fn register(
        &mut self,
        variant: &FeatureVariant,
        binding: ShaderPackageBinding,
    ) -> Result<ShaderBindingCacheStatus, ShaderBindingError> {
        binding.validate_for(variant)?;
        match self.entries.get(&variant.id) {
            None => {
                self.entries.insert(variant.id, binding);
                Ok(ShaderBindingCacheStatus::Inserted)
            }
            Some(existing) if existing == &binding => Ok(ShaderBindingCacheStatus::Reused),
            Some(_) => Err(ShaderBindingError::CacheIdentityChanged {
                variant: variant.id,
            }),
        }
    }

    pub fn invalidate(
        &mut self,
        variant: VariantId,
        expected: &ShaderPackageBinding,
    ) -> Result<(), ShaderBindingError> {
        let Some(existing) = self.entries.get(&variant) else {
            return Err(ShaderBindingError::CacheEntryMissing { variant });
        };
        if existing != expected {
            return Err(ShaderBindingError::CacheIdentityMismatch { variant });
        }
        self.entries.remove(&variant);
        Ok(())
    }

    pub fn get(&self, variant: VariantId) -> Option<&ShaderPackageBinding> {
        self.entries.get(&variant)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureVariant {
    pub id: VariantId,
    pub rank: u16,
    pub min_quality: RenderQualityTier,
    pub required_caps: CapabilitySet,
    pub requires_services_extra: ServiceSet,
    pub passes: &'static [PassTemplate],
    pub shader_permutation: Option<ShaderPermutationBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureDefinition {
    pub id: FeatureId,
    pub kind: FeatureKind,
    pub domain: Option<RenderDomain>,
    pub requires_services: ServiceSet,
    pub variants: &'static [FeatureVariant],
    /// Disjoint groups of mutually exclusive execution paths. Selection freezes
    /// all templates; a frame/stage may admit at most one member of each group.
    /// A variant contains either every member of a group or none of them.
    pub pass_alternatives: &'static [&'static [PassKey]],
}

impl FeatureDefinition {
    pub const fn subsystem(
        id: FeatureId,
        domain: RenderDomain,
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Subsystem,
            domain: Some(domain),
            requires_services,
            variants,
            pass_alternatives: &[],
        }
    }

    pub const fn service(
        id: FeatureId,
        provides: &'static [ServiceOutput],
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Service { provides },
            domain: None,
            requires_services,
            variants,
            pass_alternatives: &[],
        }
    }

    pub const fn response(
        id: FeatureId,
        requires_services: ServiceSet,
        variants: &'static [FeatureVariant],
    ) -> Self {
        Self {
            id,
            kind: FeatureKind::Response,
            domain: None,
            requires_services,
            variants,
            pass_alternatives: &[],
        }
    }

    pub const fn with_pass_alternatives(mut self, groups: &'static [&'static [PassKey]]) -> Self {
        self.pass_alternatives = groups;
        self
    }

    pub(crate) fn passes_are_alternatives(&self, first: PassKey, second: PassKey) -> bool {
        first != second
            && self
                .pass_alternatives
                .iter()
                .any(|group| group.contains(&first) && group.contains(&second))
    }
}
