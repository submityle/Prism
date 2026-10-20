//! GPU resource lifetime for a compiled Ember graph.
//!
//! The architecture crate deliberately stops at validated WGSL and a typed
//! ABI. This module is the next M2 boundary: it turns that contract into
//! device buffers, bind-group layouts/groups, a pipeline layout, and one
//! compute pipeline per generated entry point. Module-owned storage buffers
//! and textures remain caller-owned; parameter blocks and dispatch counts are
//! owned here because their layout and initial contents come from the compiled
//! graph itself.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use prism_render_architecture::particle::attributes::{
    AttributeLayoutPlan, AttributeSemantic, PlannedAttribute,
};
use prism_render_architecture::particle::compiler::{
    AttributeBufferRole, CompiledGraph, PreparedParameterBlock,
};
use prism_render_architecture::particle::modules::{ResourceAccess, ResourceDecl, ResourceKind};
use prism_render_driver as rhi;
use prism_render_graph::{
    BufferDesc, BufferHandle, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    TextureHandle,
};
use rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupId, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutId, BindingResource, BindingType, BufferBindingType,
    BufferDescriptor, BufferId, BufferState, BufferUsages, ComputePipelineDescriptor,
    ComputePipelineId, PipelineLayoutDescriptor, PipelineLayoutId, RenderDevice, RenderQueue,
    ShaderModuleDescriptor, ShaderModuleId, ShaderSource, ShaderStages, TextureId,
    TextureSampleType, TextureState, TextureViewDimension, TextureViewId,
};

/// A stable compiled binding slot used by caller-owned Ember resources.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompiledGraphGpuResourceSlot {
    /// Compiled bind-group index.
    pub group: u32,
    /// Compiled binding index.
    pub binding: u32,
}

impl CompiledGraphGpuResourceSlot {
    /// Creates a stable slot from a compiled group and binding.
    #[must_use]
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }
}

/// Monotonic revision of one caller-owned Ember resource slot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompiledGraphGpuRevision(u64);

impl CompiledGraphGpuRevision {
    /// Creates a revision received from an external resource owner.
    #[must_use]
    pub const fn from_value(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric revision used for stale-resource detection.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// The concrete resource a caller owns for one compiled Ember slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuExternalResource {
    /// A storage buffer used by a compiled module.
    StorageBuffer {
        /// Existing caller-owned buffer.
        buffer: BufferId,
        /// Optional byte range exposed to the shader.
        size: Option<u64>,
    },
    /// A curve lookup buffer used by a compiled module.
    CurveBuffer {
        /// Existing caller-owned buffer.
        buffer: BufferId,
        /// Optional byte range exposed to the shader.
        size: Option<u64>,
    },
    /// An existing caller-owned texture and view.
    Texture {
        /// Existing caller-owned texture.
        texture: TextureId,
        /// Existing caller-owned sampled view.
        view: TextureViewId,
    },
    /// An existing caller-owned sampled texture view without its texture.
    TextureView {
        /// Existing caller-owned view.
        view: TextureViewId,
    },
}

impl CompiledGraphGpuExternalResource {
    /// Returns the resource category used in diagnostics and validation.
    #[must_use]
    pub const fn kind(self) -> CompiledGraphGpuExternalKind {
        match self {
            Self::StorageBuffer { .. } => CompiledGraphGpuExternalKind::StorageBuffer,
            Self::CurveBuffer { .. } => CompiledGraphGpuExternalKind::CurveBuffer,
            Self::Texture { .. } => CompiledGraphGpuExternalKind::Texture,
            Self::TextureView { .. } => CompiledGraphGpuExternalKind::TextureView,
        }
    }
}

/// The concrete category of a caller-owned Ember resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompiledGraphGpuExternalKind {
    /// A shader storage buffer.
    StorageBuffer,
    /// A curve lookup buffer.
    CurveBuffer,
    /// A texture together with a sampled view.
    Texture,
    /// A sampled view without the underlying texture object.
    TextureView,
}

/// A caller-owned resource together with its stable slot and revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledGraphGpuBinding {
    /// Stable compiled group/binding slot.
    pub slot: CompiledGraphGpuResourceSlot,
    /// Monotonic resource revision.
    pub revision: CompiledGraphGpuRevision,
    /// Caller-owned resource currently published at the slot.
    pub resource: CompiledGraphGpuExternalResource,
}

impl CompiledGraphGpuBinding {
    const fn slot(self) -> (u32, u32) {
        (self.slot.group, self.slot.binding)
    }
}

/// A deterministic registry of caller-owned Ember GPU resources.
#[derive(Clone, Default)]
pub struct CompiledGraphGpuResourceRegistry {
    slots: BTreeMap<CompiledGraphGpuResourceSlot, CompiledGraphGpuBinding>,
    next_revision: u64,
}

/// A registry mutation can fail only when its revision counter is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuRegistryError {
    /// No further monotonic revision can be minted.
    RevisionExhausted,
    /// An exact-revision insertion would replace an existing slot.
    DuplicateSlot {
        /// Stable slot already present in the registry.
        slot: CompiledGraphGpuResourceSlot,
    },
}

impl CompiledGraphGpuResourceRegistry {
    /// Creates an empty caller-owned resource registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
            next_revision: 0,
        }
    }

    /// Returns the number of published caller-owned slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns whether no caller-owned slots are published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Looks up the current binding at a stable slot.
    #[must_use]
    pub fn get(&self, slot: CompiledGraphGpuResourceSlot) -> Option<&CompiledGraphGpuBinding> {
        self.slots.get(&slot)
    }

    /// Returns deterministic slot order for diagnostics and extraction.
    pub fn iter(&self) -> impl Iterator<Item = &CompiledGraphGpuBinding> {
        self.slots.values()
    }

    /// Returns a deterministic owned snapshot without minting revisions.
    #[must_use]
    pub fn snapshot(&self) -> Vec<CompiledGraphGpuBinding> {
        self.slots.values().copied().collect()
    }

    /// Publishes or replaces a caller-owned resource at `slot`.
    pub fn set(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        resource: CompiledGraphGpuExternalResource,
    ) -> Result<CompiledGraphGpuRevision, CompiledGraphGpuRegistryError> {
        let revision = self
            .next_revision
            .checked_add(1)
            .ok_or(CompiledGraphGpuRegistryError::RevisionExhausted)?;
        self.next_revision = revision;
        let revision = CompiledGraphGpuRevision::from_value(revision);
        self.slots.insert(
            slot,
            CompiledGraphGpuBinding {
                slot,
                revision,
                resource,
            },
        );
        Ok(revision)
    }

    /// Inserts an already revisioned binding without minting a new revision.
    ///
    /// This is the render-extraction path: the main world may carry a
    /// published slot/revision pair across the boundary, and the render-side
    /// registry can reproduce it exactly. The monotonic counter is advanced
    /// to the maximum imported revision so later [`Self::set`] calls remain
    /// newer than every imported snapshot.
    pub fn insert_exact_revision(
        &mut self,
        binding: CompiledGraphGpuBinding,
    ) -> Result<(), CompiledGraphGpuRegistryError> {
        if self.slots.contains_key(&binding.slot) {
            return Err(CompiledGraphGpuRegistryError::DuplicateSlot { slot: binding.slot });
        }
        self.next_revision = self.next_revision.max(binding.revision.value());
        self.slots.insert(binding.slot, binding);
        Ok(())
    }

    /// Atomically replaces the registry with an existing revisioned snapshot.
    ///
    /// The input is validated completely before the live map is replaced, so
    /// a duplicate slot cannot leave a partially extracted registry. No
    /// revision is minted; future mutations continue above the greatest
    /// revision found in either the previous registry or this snapshot.
    pub fn replace_snapshot(
        &mut self,
        snapshot: &[CompiledGraphGpuBinding],
    ) -> Result<(), CompiledGraphGpuRegistryError> {
        let mut next = BTreeMap::new();
        let mut maximum = self.next_revision;
        for binding in snapshot {
            if next.insert(binding.slot, *binding).is_some() {
                return Err(CompiledGraphGpuRegistryError::DuplicateSlot { slot: binding.slot });
            }
            maximum = maximum.max(binding.revision.value());
        }
        self.slots = next;
        self.next_revision = maximum;
        Ok(())
    }

    /// Publishes a storage buffer and returns its new slot revision.
    pub fn set_storage_buffer(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        buffer: BufferId,
        size: Option<u64>,
    ) -> Result<CompiledGraphGpuRevision, CompiledGraphGpuRegistryError> {
        self.set(
            slot,
            CompiledGraphGpuExternalResource::StorageBuffer { buffer, size },
        )
    }

    /// Publishes a curve buffer and returns its new slot revision.
    pub fn set_curve_buffer(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        buffer: BufferId,
        size: Option<u64>,
    ) -> Result<CompiledGraphGpuRevision, CompiledGraphGpuRegistryError> {
        self.set(
            slot,
            CompiledGraphGpuExternalResource::CurveBuffer { buffer, size },
        )
    }

    /// Publishes a texture and sampled view and returns its new slot revision.
    pub fn set_texture(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        texture: TextureId,
        view: TextureViewId,
    ) -> Result<CompiledGraphGpuRevision, CompiledGraphGpuRegistryError> {
        self.set(
            slot,
            CompiledGraphGpuExternalResource::Texture { texture, view },
        )
    }

    /// Publishes a sampled view and returns its new slot revision.
    pub fn set_texture_view(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
        view: TextureViewId,
    ) -> Result<CompiledGraphGpuRevision, CompiledGraphGpuRegistryError> {
        self.set(slot, CompiledGraphGpuExternalResource::TextureView { view })
    }

    /// Removes a caller-owned slot without taking ownership of its resource.
    pub fn remove(
        &mut self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<CompiledGraphGpuBinding> {
        self.slots.remove(&slot)
    }
}

/// A frame-graph import description for a caller-owned Ember resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuGraphImport {
    /// Import a caller-owned buffer into the frame graph.
    Buffer {
        /// Symbolic buffer description used by graph validation.
        desc: BufferDesc,
        /// Buffer state on entry to the frame.
        entry_state: BufferState,
    },
    /// Import a caller-owned texture and its default view into the frame graph.
    Texture {
        /// Symbolic texture description used by graph validation.
        desc: TextureDesc,
        /// Texture state on entry to the frame.
        entry_state: TextureState,
        /// Texture state required on frame exit.
        exit_state: TextureState,
        /// Number of mip levels tracked by the graph.
        mip_levels: u32,
        /// Number of array layers tracked by the graph.
        array_layers: u32,
    },
}

/// A failure while importing a registry slot into a frame graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuImportError {
    /// The requested slot is not currently published.
    MissingSlot { slot: CompiledGraphGpuResourceSlot },
    /// A buffer import was requested for a texture binding.
    ExpectedTexture { slot: CompiledGraphGpuResourceSlot },
    /// A texture import was requested for a buffer binding.
    ExpectedBuffer { slot: CompiledGraphGpuResourceSlot },
    /// A texture view-only slot cannot be imported without its texture object.
    TextureObjectRequired { slot: CompiledGraphGpuResourceSlot },
    /// The graph descriptor is smaller than the bound buffer range.
    BufferSizeTooSmall {
        slot: CompiledGraphGpuResourceSlot,
        required: u64,
        provided: u64,
    },
}

/// A result of refreshing caller-owned bind groups from a resource registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuExternalUpdate {
    /// The registry revisions and resource ids are unchanged.
    Unchanged,
    /// Bind groups were rebuilt for the changed slots.
    Rebuilt { changed_slots: usize },
}

/// A failure found while resolving a compiled graph to device resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuError {
    /// The graph supplied to a resident GPU state is not the graph used to
    /// create that state.
    /// The graph passed by a caller is not the graph used to create this
    /// resident state. Both values are included so stale render extraction is
    /// diagnosable without relying on object identity.
    GraphAbiMismatch {
        /// ABI fingerprint captured when the state was created.
        expected: CompiledGraphGpuAbiFingerprint,
        /// ABI fingerprint supplied by the caller.
        actual: CompiledGraphGpuAbiFingerprint,
    },
    /// The caller supplied a malformed compiled graph.
    InvalidGraphAbi(CompiledGraphGpuAbiError),
    /// Two caller-owned entries claimed one compiled slot.
    DuplicateExternalBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
    },
    /// A compiled non-uniform resource has no caller-owned resource.
    MissingExternalBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// An external entry does not correspond to a compiled non-uniform resource.
    UnexpectedExternalBinding {
        /// Supplied group index.
        group: u32,
        /// Supplied binding index.
        binding: u32,
    },
    /// A resource kind needs a buffer but the caller supplied a texture view.
    ExpectedBuffer {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A resource kind needs a texture view but the caller supplied a buffer.
    ExpectedTexture {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A resource kind is not representable by this bridge.
    UnsupportedResource {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// The generated graph has a uniform resource without a typed block.
    MissingParameterBlock {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A registry revision is older than the revision already resident in the
    /// GPU state and cannot safely replace it.
    StaleExternalRevision {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Revision currently resident in the GPU state.
        expected: CompiledGraphGpuRevision,
        /// Revision supplied by the registry.
        actual: CompiledGraphGpuRevision,
    },
    /// A curve/storage resource category does not match the compiled binding.
    ExternalResourceKindMismatch {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
        /// Resource category required by the compiled graph.
        expected: ResourceKind,
        /// Resource category supplied by the caller.
        actual: CompiledGraphGpuExternalKind,
    },
    /// The device cannot represent the persistent Ember attribute pool.
    AttributeBufferLimit {
        /// Attribute semantic whose per-copy binding is too large.
        semantic: AttributeSemantic,
        /// Required binding size in bytes.
        required: u64,
        /// Device binding-size limit in bytes.
        limit: u64,
    },
    /// The device cannot represent the complete persistent Ember pool.
    AttributePoolLimit {
        /// Required allocation size in bytes.
        required: u64,
        /// Device allocation-size limit in bytes.
        limit: u64,
    },
    /// The graph uses more bind groups than the device supports.
    BindGroupLimit {
        /// Number of bind groups required by the graph.
        required: u64,
        /// Device limit.
        limit: u32,
    },
    /// One bind group contains more entries than the device supports.
    BindingCountLimit {
        /// Bind-group index.
        group: u32,
        /// Number of entries required by the graph.
        required: u64,
        /// Device limit.
        limit: u32,
    },
    /// A uniform parameter block exceeds the device binding-size limit.
    UniformBufferLimit {
        /// Parameter resource name.
        resource: String,
        /// Required binding size.
        required: u64,
        /// Device binding-size limit.
        limit: u64,
    },
}

/// A stable fingerprint of every field that participates in the compiled
/// Ember graph ABI and generated executable.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompiledGraphGpuAbiFingerprint(u64);

impl CompiledGraphGpuAbiFingerprint {
    /// Returns the numeric ABI fingerprint.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A malformed compiled graph rejected before any device or frame-graph work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuAbiError {
    /// The generated entry-point, stage-order, and dispatch arrays disagree.
    EntryPointCountMismatch {
        /// Number of generated entry points.
        entry_points: usize,
        /// Number of stage-order entries.
        stage_order: usize,
        /// Number of dispatch records.
        dispatches: usize,
    },
    /// A generated entry point name is empty.
    EmptyEntryPoint { index: usize },
    /// Two generated stages use the same entry-point name.
    DuplicateEntryPoint { index: usize, name: String },
    /// A dispatch record does not correspond to its scheduled stage.
    DispatchStageMismatch {
        /// Dispatch array index.
        dispatch: usize,
        /// Stage index expected from the schedule.
        expected: usize,
        /// Stage index supplied by the dispatch record.
        actual: usize,
    },
    /// A dispatch record names a different generated entry point.
    DispatchEntryPointMismatch {
        /// Dispatch array index.
        dispatch: usize,
        /// Generated entry-point name expected at that position.
        expected: String,
        /// Name supplied by the dispatch record.
        actual: String,
    },
    /// A stage index cannot be represented by the dispatch-count ABI.
    DispatchStageIndexOverflow { dispatch: usize },
    /// Two graph resources claim one group/binding slot.
    DuplicateBinding { group: u32, binding: u32 },
    /// A graph resource name is declared at more than one slot.
    DuplicateResourceName { name: String },
    /// The reserved dispatch-count binding has the wrong ABI declaration.
    InvalidDispatchBinding {
        /// Supplied group index.
        group: u32,
        /// Supplied binding index.
        binding: u32,
    },
    /// Dispatch records require the reserved dispatch-count binding.
    MissingDispatchBinding,
    /// A planned attribute semantic appears more than once.
    DuplicateAttributeSemantic { semantic: AttributeSemantic },
    /// A planned attribute has an invalid number of physical copies.
    InvalidAttributeCopies {
        /// Attribute semantic.
        semantic: AttributeSemantic,
        /// Supplied copy count.
        copies: u32,
    },
    /// A planned attribute's size/offset metadata disagrees with its format.
    AttributeLayoutMismatch { semantic: AttributeSemantic },
    /// The sum of planned ranges does not equal the declared pool size.
    AttributePoolSizeMismatch { expected: u64, actual: u64 },
    /// A generated attribute binding has no matching planned allocation.
    MissingAttributeLayout { semantic: AttributeSemantic },
    /// A generated attribute binding is not in group zero.
    AttributeBindingGroup {
        semantic: AttributeSemantic,
        group: u32,
    },
    /// A generated attribute binding has an incompatible role or multiplicity.
    AttributeBindingRoleMismatch { semantic: AttributeSemantic },
    /// A uniform resource has no corresponding parameter block.
    MissingParameterBlock { resource: String },
    /// A parameter block has no corresponding uniform resource.
    UnexpectedParameterBlock { resource: String },
    /// A parameter resource has more than one parameter block.
    DuplicateParameterBlock { resource: String },
    /// A parameter block has an invalid declared payload size.
    ParameterBlockSizeMismatch {
        /// Parameter resource name.
        resource: String,
        /// Declared block size.
        expected: usize,
        /// Default payload size.
        actual: usize,
    },
    /// A parameter field's default payload does not match its type.
    ParameterFieldSizeMismatch { resource: String, field: String },
    /// A parameter field lies outside its containing block.
    ParameterFieldOutOfBounds { resource: String, field: String },
    /// A parameter field name appears more than once in one block.
    DuplicateParameterField { resource: String, field: String },
    /// Checked layout arithmetic overflowed before a GPU allocation.
    LayoutArithmeticOverflow,
    /// A runtime attribute binding requests a texture layout for another kind.
    InvalidTextureBindingKind { group: u32, binding: u32 },
}

/// A failure while updating compiled parameter buffers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParameterUploadError {
    /// One parameter resource was supplied more than once.
    DuplicateResource {
        /// Parameter resource name.
        resource: String,
    },
    /// The update refers to no compiled parameter block.
    UnknownResource {
        /// Parameter resource name.
        resource: String,
    },
    /// The supplied bytes do not match the compiled block size.
    SizeMismatch {
        /// Parameter resource name.
        resource: String,
        /// Compiled block size.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    /// The update's binding metadata disagrees with the compiled block.
    BindingMismatch {
        /// Parameter resource name.
        resource: String,
        /// Compiled group index.
        expected_group: u32,
        /// Compiled binding index.
        expected_binding: u32,
        /// Supplied group index.
        actual_group: u32,
        /// Supplied binding index.
        actual_binding: u32,
    },
    /// A complete frame update omitted a compiled parameter block.
    MissingResource {
        /// Parameter resource name.
        resource: String,
    },
}

/// A graph-owned handle corresponding to one compiled Ember binding.
///
/// The real GPU object is still owned by the caller or by
/// [`CompiledGraphGpuState`]. The handle only lets the frame graph declare the
/// same resource access that the compiled bind group will use at execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompiledGraphGpuGraphBinding {
    /// A buffer imported or created in the caller's frame graph.
    Buffer {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Revision of the caller-owned resource used by this graph handle.
        revision: CompiledGraphGpuRevision,
        /// Frame-graph buffer handle for the same real buffer bound to Ember.
        handle: BufferHandle,
    },
    /// A texture imported or created in the caller's frame graph.
    Texture {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Revision of the caller-owned resource used by this graph handle.
        revision: CompiledGraphGpuRevision,
        /// Frame-graph texture handle for the same real view bound to Ember.
        handle: TextureHandle,
    },
}

impl CompiledGraphGpuGraphBinding {
    const fn slot(self) -> (u32, u32) {
        match self {
            Self::Buffer { group, binding, .. } | Self::Texture { group, binding, .. } => {
                (group, binding)
            }
        }
    }

    const fn revision(self) -> CompiledGraphGpuRevision {
        match self {
            Self::Buffer { revision, .. } | Self::Texture { revision, .. } => revision,
        }
    }
}

/// A failure while declaring a compiled graph in a frame graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompiledGraphGraphError {
    /// The graph supplied for recording is not the graph used to create the
    /// resident GPU state.
    /// The graph passed by a caller is not the graph used to create this
    /// resident state.
    GraphAbiMismatch {
        /// ABI fingerprint captured when the state was created.
        expected: CompiledGraphGpuAbiFingerprint,
        /// ABI fingerprint supplied by the caller.
        actual: CompiledGraphGpuAbiFingerprint,
    },
    /// The caller supplied a malformed compiled graph.
    InvalidGraphAbi(CompiledGraphGpuAbiError),
    /// A graph resource was supplied more than once for one binding slot.
    DuplicateBinding { group: u32, binding: u32 },
    /// A compiled non-uniform binding has no graph handle.
    MissingBinding {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Compiled resource name.
        name: String,
    },
    /// A graph resource does not correspond to a compiled binding.
    UnexpectedBinding { group: u32, binding: u32 },
    /// The caller supplied a texture handle where a buffer was required.
    ExpectedBuffer {
        group: u32,
        binding: u32,
        name: String,
    },
    /// The caller supplied a buffer handle where a texture was required.
    ExpectedTexture {
        group: u32,
        binding: u32,
        name: String,
    },
    /// The runtime count array does not address every compiled stage index.
    DispatchCountLength { expected: usize, actual: usize },
    /// The state and compiled graph disagree about generated stage pipelines.
    PipelineCountMismatch { expected: usize, actual: usize },
    /// A frame-graph binding was made from an older or newer external slot
    /// revision than the bind groups currently use.
    ExternalRevisionMismatch {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
        /// Revision currently used by the state.
        expected: CompiledGraphGpuRevision,
        /// Revision attached to the graph handle.
        actual: CompiledGraphGpuRevision,
    },
    /// A caller-owned texture needs an explicit frame-graph description.
    ///
    /// The RHI texture view alone does not carry the symbolic format, extent,
    /// mip, and state information required by `RenderGraph::import_texture`.
    /// Callers must use `CompiledGraphGpuState::import_external_to_graph` with
    /// a concrete `CompiledGraphGpuGraphImport` before recording the graph.
    ExternalTextureImportRequired {
        /// Compiled group index.
        group: u32,
        /// Compiled binding index.
        binding: u32,
    },
    /// A caller-provided external-resource import descriptor was rejected.
    ExternalImport(CompiledGraphGpuImportError),
}

/// A failure while updating the compiled graph's persistent attribute pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttributeUploadError {
    /// The write range exceeds the compiled SoA pool.
    OutOfBounds {
        offset: u64,
        size: usize,
        capacity: u64,
    },
    /// The resident state no longer contains the layout needed to translate
    /// a logical upload into its aligned physical pool range.
    InvalidLayout { semantic: AttributeSemantic },
}

struct ParameterBuffer {
    resource: String,
    group: u32,
    binding: u32,
    size: u64,
    buffer: BufferId,
}

struct ResolvedEntry {
    binding: u32,
    layout: BindGroupLayoutEntry,
    resource: BindingResource,
}

/// GPU resources and pipelines for one compiled Ember graph.
pub struct CompiledGraphGpuState {
    compiled_graph: CompiledGraph,
    graph_abi_fingerprint: CompiledGraphGpuAbiFingerprint,
    attribute_layout: AttributeLayoutPlan,
    shader: ShaderModuleId,
    pipeline_layout: PipelineLayoutId,
    bind_group_layouts: Vec<BindGroupLayoutId>,
    bind_groups: [Vec<BindGroupId>; 2],
    compute_pipelines: Vec<ComputePipelineId>,
    parameter_buffers: Vec<ParameterBuffer>,
    external_bindings: Vec<CompiledGraphGpuBinding>,
    dispatch_counts: Option<BufferId>,
    dispatch_count_slots: usize,
    attribute_pool: Option<BufferId>,
    attribute_pool_size: u64,
}

impl CompiledGraphGpuState {
    /// Creates all device-owned resources and uploads compiled defaults.
    pub fn new(
        device: &impl RenderDevice,
        queue: &impl RenderQueue,
        graph: &CompiledGraph,
        registry: &CompiledGraphGpuResourceRegistry,
    ) -> Result<Self, CompiledGraphGpuError> {
        validate_compiled_graph_abi(graph).map_err(CompiledGraphGpuError::InvalidGraphAbi)?;
        let external_bindings = validate_external_registry(graph, registry)?;

        let limits = device.capabilities().limits;
        let attribute_layout = aligned_attribute_layout(
            &graph.attribute_layout,
            u64::from(limits.min_storage_buffer_offset_alignment),
        )
        .map_err(CompiledGraphGpuError::InvalidGraphAbi)?;
        let group_count = max_group(graph).map_or(0, |group| group as u64 + 1);
        if group_count > u64::from(limits.max_bind_groups) {
            return Err(CompiledGraphGpuError::BindGroupLimit {
                required: group_count,
                limit: limits.max_bind_groups,
            });
        }
        for group in 0..group_count as u32 {
            let required = graph
                .attribute_bindings
                .iter()
                .filter(|binding| binding.group == group)
                .count()
                + graph
                    .bindings
                    .iter()
                    .filter(|binding| binding.group == group)
                    .count();
            if required as u64 > u64::from(limits.max_bindings_per_bind_group) {
                return Err(CompiledGraphGpuError::BindingCountLimit {
                    group,
                    required: required as u64,
                    limit: limits.max_bindings_per_bind_group,
                });
            }
        }
        for attribute in &attribute_layout.attributes {
            let per_copy = attribute.byte_size / u64::from(attribute.copies);
            if per_copy > limits.max_storage_buffer_binding_size {
                return Err(CompiledGraphGpuError::AttributeBufferLimit {
                    semantic: attribute.semantic,
                    required: per_copy,
                    limit: limits.max_storage_buffer_binding_size,
                });
            }
        }
        if attribute_layout.total_bytes > limits.max_buffer_size {
            return Err(CompiledGraphGpuError::AttributePoolLimit {
                required: attribute_layout.total_bytes,
                limit: limits.max_buffer_size,
            });
        }
        for block in &graph.parameter_blocks {
            if u64::from(block.size) > limits.max_uniform_buffer_binding_size {
                return Err(CompiledGraphGpuError::UniformBufferLimit {
                    resource: block.resource.clone(),
                    required: u64::from(block.size),
                    limit: limits.max_uniform_buffer_binding_size,
                });
            }
        }

        let group_count = group_count as usize;
        let attribute_pool_size = attribute_layout.total_bytes;
        let attribute_pool = (attribute_pool_size != 0).then(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some(String::from("ember-attribute-pool")),
                size: attribute_pool_size,
                usage: BufferUsages::STORAGE
                    .union(BufferUsages::COPY_DST)
                    .union(BufferUsages::COPY_SRC),
                mapped_at_creation: false,
            })
        });
        let mut parameter_buffers = Vec::with_capacity(graph.parameter_blocks.len());
        for block in &graph.parameter_blocks {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(format!("ember-parameter-{}", block.resource)),
                size: block.size as u64,
                usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            });
            queue.write_buffer(buffer, 0, &block.default_bytes);
            parameter_buffers.push(ParameterBuffer {
                resource: block.resource.clone(),
                group: block.group,
                binding: block.binding,
                size: block.size as u64,
                buffer,
            });
        }

        let dispatch_count_slots = graph
            .dispatches
            .iter()
            .map(|dispatch| dispatch.stage_index + 1)
            .max()
            .unwrap_or(0);
        let dispatch_counts = if dispatch_count_slots == 0 {
            None
        } else {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(String::from("ember-dispatch-counts")),
                size: dispatch_count_slots as u64 * 4,
                usage: BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
                mapped_at_creation: false,
            });
            let mut bytes = vec![0; dispatch_count_slots * 4];
            for dispatch in &graph.dispatches {
                let offset = dispatch.stage_index * 4;
                bytes[offset..offset + 4].copy_from_slice(&dispatch.default_count.to_le_bytes());
            }
            queue.write_buffer(buffer, 0, &bytes);
            Some(buffer)
        };

        let mut bind_group_layouts = Vec::with_capacity(group_count);
        for group in 0..group_count as u32 {
            let entries = group_entries(
                graph,
                &external_bindings,
                &parameter_buffers,
                dispatch_counts,
                attribute_pool,
                &attribute_layout,
                0,
                group,
            )?;
            let mut layout_entries: Vec<_> = entries.iter().map(|entry| entry.layout).collect();
            layout_entries.sort_by_key(|entry| entry.binding);
            bind_group_layouts.push(device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(format!("ember-graph-group-{group}")),
                entries: layout_entries,
            }));
        }
        let mut bind_groups = [
            Vec::with_capacity(group_count),
            Vec::with_capacity(group_count),
        ];
        for frame_index in 0..2u64 {
            for group in 0..group_count as u32 {
                let entries = group_entries(
                    graph,
                    &external_bindings,
                    &parameter_buffers,
                    dispatch_counts,
                    attribute_pool,
                    &attribute_layout,
                    frame_index,
                    group,
                )?;
                let mut bind_entries: Vec<_> = entries
                    .into_iter()
                    .map(|entry| BindGroupEntry {
                        binding: entry.binding,
                        resource: entry.resource,
                    })
                    .collect();
                bind_entries.sort_by_key(|entry| entry.binding);
                bind_groups[frame_index as usize].push(device.create_bind_group(
                    &BindGroupDescriptor {
                        label: Some(format!("ember-graph-group-{group}-frame-{frame_index}")),
                        layout: Some(bind_group_layouts[group as usize]),
                        entries: bind_entries,
                    },
                ));
            }
        }

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("ember-graph-pipeline-layout")),
            bind_group_layouts: bind_group_layouts.clone(),
            push_constant_ranges: Vec::new(),
        });
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some(String::from("ember-compiled-graph")),
            source: ShaderSource::Wgsl(Cow::Owned(graph.source.clone())),
        });
        let mut compute_pipelines = Vec::with_capacity(graph.entry_points.len());
        for entry_point in &graph.entry_points {
            compute_pipelines.push(device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(format!("ember-graph-{entry_point}")),
                layout: Some(pipeline_layout),
                module: shader,
                entry_point: Cow::Owned(entry_point.clone()),
            }));
        }

        Ok(Self {
            compiled_graph: graph.clone(),
            graph_abi_fingerprint: compiled_graph_abi_fingerprint(graph),
            attribute_layout,
            shader,
            pipeline_layout,
            bind_group_layouts,
            bind_groups,
            compute_pipelines,
            parameter_buffers,
            external_bindings,
            dispatch_counts,
            dispatch_count_slots,
            attribute_pool,
            attribute_pool_size,
        })
    }

    /// Returns the caller-owned bindings resident in the current bind groups.
    #[must_use]
    pub fn external_bindings(&self) -> &[CompiledGraphGpuBinding] {
        &self.external_bindings
    }

    /// Looks up the caller-owned binding resident at one stable slot.
    #[must_use]
    pub fn external_binding(
        &self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<CompiledGraphGpuBinding> {
        self.external_bindings
            .iter()
            .find(|binding| binding.slot == slot)
            .copied()
    }

    /// Returns the revision and concrete bind-group resource for a slot.
    ///
    /// This is the observable binding view used by render code that needs to
    /// inspect or cache a bind-group entry without taking ownership of the
    /// caller's buffer, texture, or view.
    #[must_use]
    pub fn external_bind_group_resource(
        &self,
        slot: CompiledGraphGpuResourceSlot,
    ) -> Option<(CompiledGraphGpuRevision, BindingResource)> {
        self.external_binding(slot)
            .map(|binding| (binding.revision, binding_resource(binding.resource)))
    }

    /// Produces a graph buffer binding carrying the resident slot revision.
    ///
    /// The actual graph resource must already have been imported by the
    /// caller, because the graph owns its virtual handle.
    #[must_use]
    pub fn external_graph_buffer_binding(
        &self,
        slot: CompiledGraphGpuResourceSlot,
        handle: BufferHandle,
    ) -> Option<CompiledGraphGpuGraphBinding> {
        let binding = self.external_binding(slot)?;
        matches!(
            binding.resource,
            CompiledGraphGpuExternalResource::StorageBuffer { .. }
                | CompiledGraphGpuExternalResource::CurveBuffer { .. }
        )
        .then_some(CompiledGraphGpuGraphBinding::Buffer {
            group: slot.group,
            binding: slot.binding,
            revision: binding.revision,
            handle,
        })
    }

    /// Produces a graph texture binding carrying the resident slot revision.
    ///
    /// This also supports view-only slots whose texture was imported by a
    /// separate owner that already knows the underlying texture object.
    #[must_use]
    pub fn external_graph_texture_binding(
        &self,
        slot: CompiledGraphGpuResourceSlot,
        handle: TextureHandle,
    ) -> Option<CompiledGraphGpuGraphBinding> {
        let binding = self.external_binding(slot)?;
        matches!(
            binding.resource,
            CompiledGraphGpuExternalResource::Texture { .. }
                | CompiledGraphGpuExternalResource::TextureView { .. }
        )
        .then_some(CompiledGraphGpuGraphBinding::Texture {
            group: slot.group,
            binding: slot.binding,
            revision: binding.revision,
            handle,
        })
    }

    /// Imports one caller-owned registry slot into the frame graph.
    pub fn import_external_to_graph(
        &self,
        graph: &mut RenderGraph,
        slot: CompiledGraphGpuResourceSlot,
        name: impl Into<String>,
        import: CompiledGraphGpuGraphImport,
    ) -> Result<CompiledGraphGpuGraphBinding, CompiledGraphGpuImportError> {
        let binding = self
            .external_binding(slot)
            .ok_or(CompiledGraphGpuImportError::MissingSlot { slot })?;
        match (binding.resource, import) {
            (
                CompiledGraphGpuExternalResource::StorageBuffer { buffer, size }
                | CompiledGraphGpuExternalResource::CurveBuffer { buffer, size },
                CompiledGraphGpuGraphImport::Buffer { desc, entry_state },
            ) => {
                if let Some(required) = size {
                    if desc.size < required {
                        return Err(CompiledGraphGpuImportError::BufferSizeTooSmall {
                            slot,
                            required,
                            provided: desc.size,
                        });
                    }
                }
                let handle = graph.import_buffer(
                    name,
                    desc,
                    ImportedBuffer {
                        buffer,
                        entry_state,
                    },
                );
                Ok(CompiledGraphGpuGraphBinding::Buffer {
                    group: slot.group,
                    binding: slot.binding,
                    revision: binding.revision,
                    handle,
                })
            }
            (
                CompiledGraphGpuExternalResource::Texture { texture, view },
                CompiledGraphGpuGraphImport::Texture {
                    desc,
                    entry_state,
                    exit_state,
                    mip_levels,
                    array_layers,
                },
            ) => {
                let handle = graph.import_texture(
                    name,
                    desc,
                    ImportedTexture {
                        texture,
                        default_view: view,
                        entry_state,
                        exit_state,
                        mip_levels: mip_levels.max(1),
                        array_layers: array_layers.max(1),
                    },
                );
                Ok(CompiledGraphGpuGraphBinding::Texture {
                    group: slot.group,
                    binding: slot.binding,
                    revision: binding.revision,
                    handle,
                })
            }
            (
                CompiledGraphGpuExternalResource::TextureView { .. },
                CompiledGraphGpuGraphImport::Texture { .. },
            ) => Err(CompiledGraphGpuImportError::TextureObjectRequired { slot }),
            (
                CompiledGraphGpuExternalResource::Texture { .. }
                | CompiledGraphGpuExternalResource::TextureView { .. },
                CompiledGraphGpuGraphImport::Buffer { .. },
            ) => Err(CompiledGraphGpuImportError::ExpectedTexture { slot }),
            (
                CompiledGraphGpuExternalResource::StorageBuffer { .. }
                | CompiledGraphGpuExternalResource::CurveBuffer { .. },
                CompiledGraphGpuGraphImport::Texture { .. },
            ) => Err(CompiledGraphGpuImportError::ExpectedBuffer { slot }),
        }
    }

    /// Rebuilds caller-facing bind groups after a newer registry publication.
    ///
    /// The state never destroys caller-owned resources. Only its bind groups
    /// are replaced, and a registry revision older than the resident revision
    /// is rejected before any GPU object is changed.
    pub fn refresh_external_resources(
        &mut self,
        device: &impl RenderDevice,
        graph: &CompiledGraph,
        registry: &CompiledGraphGpuResourceRegistry,
    ) -> Result<CompiledGraphGpuExternalUpdate, CompiledGraphGpuError> {
        validate_compiled_graph_abi(graph).map_err(CompiledGraphGpuError::InvalidGraphAbi)?;
        let actual_fingerprint = compiled_graph_abi_fingerprint(graph);
        if actual_fingerprint != self.graph_abi_fingerprint || &self.compiled_graph != graph {
            return Err(CompiledGraphGpuError::GraphAbiMismatch {
                expected: self.graph_abi_fingerprint,
                actual: actual_fingerprint,
            });
        }
        let next_bindings = validate_external_registry(graph, registry)?;
        for current in &self.external_bindings {
            let Some(next) = next_bindings
                .iter()
                .find(|binding| binding.slot == current.slot)
            else {
                return Err(CompiledGraphGpuError::MissingExternalBinding {
                    group: current.slot.group,
                    binding: current.slot.binding,
                    name: String::from("resident external binding"),
                });
            };
            if next.revision < current.revision {
                return Err(CompiledGraphGpuError::StaleExternalRevision {
                    group: current.slot.group,
                    binding: current.slot.binding,
                    expected: current.revision,
                    actual: next.revision,
                });
            }
        }
        if next_bindings == self.external_bindings {
            return Ok(CompiledGraphGpuExternalUpdate::Unchanged);
        }

        let group_count = self.bind_group_layouts.len();
        let mut replacement = [
            Vec::with_capacity(group_count),
            Vec::with_capacity(group_count),
        ];
        for frame_index in 0..2u64 {
            for group in 0..group_count as u32 {
                let entries = group_entries(
                    graph,
                    &next_bindings,
                    &self.parameter_buffers,
                    self.dispatch_counts,
                    self.attribute_pool,
                    &self.attribute_layout,
                    frame_index,
                    group,
                )?;
                let mut bind_entries: Vec<_> = entries
                    .into_iter()
                    .map(|entry| BindGroupEntry {
                        binding: entry.binding,
                        resource: entry.resource,
                    })
                    .collect();
                bind_entries.sort_by_key(|entry| entry.binding);
                replacement[frame_index as usize].push(device.create_bind_group(
                    &BindGroupDescriptor {
                        label: Some(format!("ember-graph-group-{group}-frame-{frame_index}")),
                        layout: Some(self.bind_group_layouts[group as usize]),
                        entries: bind_entries,
                    },
                ));
            }
        }
        for bind_groups in &self.bind_groups {
            for bind_group in bind_groups {
                device.destroy_bind_group(*bind_group);
            }
        }
        self.bind_groups = replacement;
        let changed_slots = self
            .external_bindings
            .iter()
            .zip(next_bindings.iter())
            .filter(|(current, next)| current != next)
            .count();
        self.external_bindings = next_bindings;
        Ok(CompiledGraphGpuExternalUpdate::Rebuilt { changed_slots })
    }

    /// Uploads a complete prepared parameter set without rebuilding bind groups.
    pub fn update_parameters(
        &self,
        queue: &impl RenderQueue,
        updates: &[PreparedParameterBlock],
    ) -> Result<(), ParameterUploadError> {
        for (index, update) in updates.iter().enumerate() {
            if updates[..index]
                .iter()
                .any(|previous| previous.resource == update.resource)
            {
                return Err(ParameterUploadError::DuplicateResource {
                    resource: update.resource.clone(),
                });
            }
            let Some(buffer) = self
                .parameter_buffers
                .iter()
                .find(|buffer| buffer.resource == update.resource)
            else {
                return Err(ParameterUploadError::UnknownResource {
                    resource: update.resource.clone(),
                });
            };
            if (update.group, update.binding) != (buffer.group, buffer.binding) {
                return Err(ParameterUploadError::BindingMismatch {
                    resource: update.resource.clone(),
                    expected_group: buffer.group,
                    expected_binding: buffer.binding,
                    actual_group: update.group,
                    actual_binding: update.binding,
                });
            }
            if update.bytes.len() != buffer.size as usize {
                return Err(ParameterUploadError::SizeMismatch {
                    resource: update.resource.clone(),
                    expected: buffer.size as usize,
                    actual: update.bytes.len(),
                });
            }
            queue.write_buffer(buffer.buffer, 0, &update.bytes);
        }
        for buffer in &self.parameter_buffers {
            if !updates
                .iter()
                .any(|update| update.resource == buffer.resource)
            {
                return Err(ParameterUploadError::MissingResource {
                    resource: buffer.resource.clone(),
                });
            }
        }
        Ok(())
    }

    /// Uploads the runtime stage counts used by generated dispatch guards.
    pub fn update_dispatch_counts(
        &self,
        queue: &impl RenderQueue,
        counts: &[u32],
    ) -> Result<(), &'static str> {
        let Some(buffer) = self.dispatch_counts else {
            return if counts.is_empty() {
                Ok(())
            } else {
                Err("compiled graph has no dispatch-count buffer")
            };
        };
        if counts.len() != self.dispatch_count_slots {
            return Err("dispatch-count length does not match compiled stages");
        }
        let mut bytes = Vec::with_capacity(counts.len() * 4);
        for count in counts {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        queue.write_buffer(buffer, 0, &bytes);
        Ok(())
    }

    /// Returns a state-owned parameter buffer and its compiled binding slot.
    ///
    /// Callers use this to import the buffer into the frame graph before
    /// [`Self::append_to_graph`] declares the uniform read.
    #[must_use]
    pub fn parameter_buffer(&self, resource: &str) -> Option<(u32, u32, BufferId, u64)> {
        self.parameter_buffers
            .iter()
            .find(|parameter| parameter.resource == resource)
            .map(|parameter| {
                (
                    parameter.group,
                    parameter.binding,
                    parameter.buffer,
                    parameter.size,
                )
            })
    }

    /// Returns the state-owned dispatch-count storage buffer.
    #[must_use]
    pub const fn dispatch_counts_buffer(&self) -> Option<BufferId> {
        self.dispatch_counts
    }

    /// Returns the number of stage-index slots in the dispatch-count ABI.
    #[must_use]
    pub const fn dispatch_count_slots(&self) -> usize {
        self.dispatch_count_slots
    }

    /// Returns the persistent SoA attribute pool, when the graph has live
    /// attributes.
    #[must_use]
    pub const fn attribute_pool_buffer(&self) -> Option<BufferId> {
        self.attribute_pool
    }

    /// Returns the total byte size of the persistent SoA attribute pool.
    #[must_use]
    pub const fn attribute_pool_size(&self) -> u64 {
        self.attribute_pool_size
    }

    /// Uploads a byte range into the persistent SoA attribute pool.
    pub fn upload_attribute_bytes(
        &self,
        queue: &impl RenderQueue,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), AttributeUploadError> {
        let logical_capacity = self.compiled_graph.attribute_layout.total_bytes;
        let end = offset.saturating_add(bytes.len() as u64);
        if end > logical_capacity {
            return Err(AttributeUploadError::OutOfBounds {
                offset,
                size: bytes.len(),
                capacity: logical_capacity,
            });
        }
        if let Some(buffer) = self.attribute_pool {
            let end = offset + bytes.len() as u64;
            let logical = &self.compiled_graph.attribute_layout;
            for planned in &logical.attributes {
                let logical_per_copy = u64::from(planned.stride) * u64::from(logical.capacity);
                let Some(aligned) = self.attribute_layout.get(planned.semantic) else {
                    return Err(AttributeUploadError::InvalidLayout {
                        semantic: planned.semantic,
                    });
                };
                let physical_per_copy = aligned.byte_size / u64::from(aligned.copies);
                for copy in 0..planned.copies {
                    let logical_start = planned.byte_offset + u64::from(copy) * logical_per_copy;
                    let logical_end = logical_start + logical_per_copy;
                    let overlap_start = offset.max(logical_start);
                    let overlap_end = end.min(logical_end);
                    if overlap_start >= overlap_end {
                        continue;
                    }
                    let source_start = (overlap_start - offset) as usize;
                    let source_end = (overlap_end - offset) as usize;
                    let destination = aligned.byte_offset
                        + u64::from(copy) * physical_per_copy
                        + (overlap_start - logical_start);
                    queue.write_buffer(buffer, destination, &bytes[source_start..source_end]);
                }
            }
        } else if !bytes.is_empty() {
            return Err(AttributeUploadError::OutOfBounds {
                offset,
                size: bytes.len(),
                capacity: logical_capacity,
            });
        }
        Ok(())
    }

    /// Declares and records all compiled stage dispatches in a frame graph.
    ///
    /// `bindings` must contain one graph handle for every compiled resource
    /// binding, including the state-owned parameter and dispatch-count buffers.
    /// The handles must refer to the same real resources used when this state
    /// was created. This method records commands only; callers still upload
    /// parameters/counts and execute the returned graph through the runtime.
    pub fn append_to_graph(
        &self,
        compiled: &CompiledGraph,
        graph: &mut RenderGraph,
        bindings: &[CompiledGraphGpuGraphBinding],
        stage_counts: &[u32],
    ) -> Result<(), CompiledGraphGraphError> {
        self.append_to_graph_for_frame(compiled, graph, bindings, stage_counts, 0)
    }

    /// Variant of [`Self::append_to_graph`] selecting the ping-pong side for a
    /// specific frame parity.
    pub fn append_to_graph_for_frame(
        &self,
        compiled: &CompiledGraph,
        graph: &mut RenderGraph,
        bindings: &[CompiledGraphGpuGraphBinding],
        stage_counts: &[u32],
        frame_index: u64,
    ) -> Result<(), CompiledGraphGraphError> {
        if let Err(error) = validate_compiled_graph_abi(compiled) {
            return Err(CompiledGraphGraphError::InvalidGraphAbi(error));
        }
        let actual_fingerprint = compiled_graph_abi_fingerprint(compiled);
        if actual_fingerprint != self.graph_abi_fingerprint || &self.compiled_graph != compiled {
            return Err(CompiledGraphGraphError::GraphAbiMismatch {
                expected: self.graph_abi_fingerprint,
                actual: actual_fingerprint,
            });
        }
        if self.compute_pipelines.len() != compiled.dispatches.len() {
            return Err(CompiledGraphGraphError::PipelineCountMismatch {
                expected: compiled.dispatches.len(),
                actual: self.compute_pipelines.len(),
            });
        }
        if stage_counts.len() != self.dispatch_count_slots {
            return Err(CompiledGraphGraphError::DispatchCountLength {
                expected: self.dispatch_count_slots,
                actual: stage_counts.len(),
            });
        }

        let accesses = graph_accesses(compiled, bindings, &self.external_bindings)?;
        for (pipeline_index, dispatch) in compiled.dispatches.iter().enumerate() {
            let workgroups = stage_counts
                .get(dispatch.stage_index)
                .copied()
                .unwrap_or(0)
                .saturating_add(63)
                / 64;
            let pipeline = self.compute_pipelines[pipeline_index];
            let bind_groups = self.bind_groups[(frame_index & 1) as usize].clone();
            let entry_name = dispatch.entry_point.clone();
            let iterations = dispatch.iterations;
            let accesses_for_setup = accesses.clone();
            graph.add_compute_pass(
                format!("ember-{entry_name}"),
                PassFlags::SIDE_EFFECT,
                move |builder| {
                    for access in &accesses_for_setup {
                        access.declare(builder);
                    }
                    move |context| {
                        context.set_compute_pipeline(pipeline);
                        for (group, bind_group) in bind_groups.iter().copied().enumerate() {
                            context.set_compute_bind_group(group as u32, bind_group, Vec::new());
                        }
                        for _ in 0..iterations {
                            if workgroups != 0 {
                                context.dispatch(workgroups, 1, 1);
                            }
                        }
                    }
                },
            );
        }
        Ok(())
    }

    /// Returns the generated shader module.
    #[must_use]
    pub const fn shader(&self) -> ShaderModuleId {
        self.shader
    }

    /// Returns the pipeline layout shared by generated stage pipelines.
    #[must_use]
    pub const fn pipeline_layout(&self) -> PipelineLayoutId {
        self.pipeline_layout
    }

    /// Returns a group layout by compiled group index.
    #[must_use]
    pub fn bind_group_layout(&self, group: u32) -> Option<BindGroupLayoutId> {
        self.bind_group_layouts.get(group as usize).copied()
    }

    /// Returns a bind group by compiled group index.
    #[must_use]
    pub fn bind_group(&self, group: u32) -> Option<BindGroupId> {
        self.bind_groups[0].get(group as usize).copied()
    }

    /// Returns a bind group for the selected ping-pong frame parity.
    #[must_use]
    pub fn bind_group_for_frame(&self, group: u32, frame_index: u64) -> Option<BindGroupId> {
        self.bind_groups[(frame_index & 1) as usize]
            .get(group as usize)
            .copied()
    }

    /// Returns a generated compute pipeline by entry-point order.
    #[must_use]
    pub fn compute_pipeline(&self, index: usize) -> Option<ComputePipelineId> {
        self.compute_pipelines.get(index).copied()
    }

    /// Releases every resource created by [`Self::new`].
    pub fn destroy(self, device: &impl RenderDevice) {
        for pipeline in self.compute_pipelines {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_shader_module(self.shader);
        device.destroy_pipeline_layout(self.pipeline_layout);
        for bind_groups in self.bind_groups {
            for bind_group in bind_groups {
                device.destroy_bind_group(bind_group);
            }
        }
        for layout in self.bind_group_layouts {
            device.destroy_bind_group_layout(layout);
        }
        for parameter in self.parameter_buffers {
            device.destroy_buffer(parameter.buffer);
        }
        if let Some(dispatch_counts) = self.dispatch_counts {
            device.destroy_buffer(dispatch_counts);
        }
        if let Some(attribute_pool) = self.attribute_pool {
            device.destroy_buffer(attribute_pool);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GraphAccess {
    Uniform(BufferHandle),
    StorageRead(BufferHandle),
    StorageWrite(BufferHandle),
    Sample(TextureHandle),
}

impl GraphAccess {
    fn declare(self, builder: &mut prism_render_graph::PassBuilder<'_>) {
        match self {
            Self::Uniform(handle) => {
                let _ = builder.read_uniform(handle);
            }
            Self::StorageRead(handle) => {
                let _ = builder.read_storage_buffer(handle);
            }
            Self::StorageWrite(handle) => {
                let _ = builder.write_storage_buffer(handle);
            }
            Self::Sample(handle) => {
                let _ = builder.sample(handle);
            }
        }
    }
}

fn graph_accesses(
    compiled: &CompiledGraph,
    bindings: &[CompiledGraphGpuGraphBinding],
    external: &[CompiledGraphGpuBinding],
) -> Result<Vec<GraphAccess>, CompiledGraphGraphError> {
    for (index, entry) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|previous| previous.slot() == entry.slot())
        {
            let (group, binding) = entry.slot();
            return Err(CompiledGraphGraphError::DuplicateBinding { group, binding });
        }
    }

    for entry in bindings {
        if let Some(expected) = external
            .iter()
            .find(|binding| binding.slot() == entry.slot())
        {
            if expected.revision != entry.revision() {
                let (group, binding) = entry.slot();
                return Err(CompiledGraphGraphError::ExternalRevisionMismatch {
                    group,
                    binding,
                    expected: expected.revision,
                    actual: entry.revision(),
                });
            }
        }
    }

    let mut accesses = Vec::new();
    let mut expected = Vec::new();
    for attribute in &compiled.attribute_bindings {
        expected.push((
            attribute.group,
            attribute.binding,
            attribute.name.as_str(),
            ResourceKind::StorageBuffer,
            attribute.access,
        ));
        let entry = bindings
            .iter()
            .find(|entry| entry.slot() == (attribute.group, attribute.binding))
            .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                group: attribute.group,
                binding: attribute.binding,
                name: attribute.name.clone(),
            })?;
        let CompiledGraphGpuGraphBinding::Buffer { handle, .. } = entry else {
            return Err(CompiledGraphGraphError::ExpectedBuffer {
                group: attribute.group,
                binding: attribute.binding,
                name: attribute.name.clone(),
            });
        };
        accesses.push(match attribute.access {
            ResourceAccess::ReadOnly => GraphAccess::StorageRead(*handle),
            ResourceAccess::ReadWrite => GraphAccess::StorageWrite(*handle),
        });
    }
    for binding in &compiled.bindings {
        if binding.name == "ember_dispatch_counts" {
            expected.push((
                binding.group,
                binding.binding,
                binding.name,
                binding.kind,
                binding.access,
            ));
            let entry = bindings
                .iter()
                .find(|entry| entry.slot() == (binding.group, binding.binding))
                .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                })?;
            let CompiledGraphGpuGraphBinding::Buffer { handle, .. } = entry else {
                return Err(CompiledGraphGraphError::ExpectedBuffer {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            };
            accesses.push(GraphAccess::StorageRead(*handle));
            continue;
        }
        expected.push((
            binding.group,
            binding.binding,
            binding.name,
            binding.kind,
            binding.access,
        ));
        let entry = bindings
            .iter()
            .find(|entry| entry.slot() == (binding.group, binding.binding))
            .ok_or_else(|| CompiledGraphGraphError::MissingBinding {
                group: binding.group,
                binding: binding.binding,
                name: String::from(binding.name),
            })?;
        match (binding.kind, entry) {
            (ResourceKind::Uniform, CompiledGraphGpuGraphBinding::Buffer { handle, .. }) => {
                accesses.push(GraphAccess::Uniform(*handle));
            }
            (
                ResourceKind::Curve | ResourceKind::StorageBuffer,
                CompiledGraphGpuGraphBinding::Buffer { handle, .. },
            ) => {
                accesses.push(match binding.access {
                    ResourceAccess::ReadOnly => GraphAccess::StorageRead(*handle),
                    ResourceAccess::ReadWrite => GraphAccess::StorageWrite(*handle),
                });
            }
            (
                ResourceKind::Texture2d | ResourceKind::Texture3d,
                CompiledGraphGpuGraphBinding::Texture { handle, .. },
            ) => accesses.push(GraphAccess::Sample(*handle)),
            (ResourceKind::Uniform, _) => {
                return Err(CompiledGraphGraphError::ExpectedBuffer {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            }
            (_, _) => {
                return Err(CompiledGraphGraphError::ExpectedTexture {
                    group: binding.group,
                    binding: binding.binding,
                    name: String::from(binding.name),
                });
            }
        }
    }
    for entry in bindings {
        if !expected
            .iter()
            .any(|(group, binding, _, _, _)| (*group, *binding) == entry.slot())
        {
            let (group, binding) = entry.slot();
            return Err(CompiledGraphGraphError::UnexpectedBinding { group, binding });
        }
    }
    Ok(accesses)
}

fn max_group(graph: &CompiledGraph) -> Option<u32> {
    graph
        .attribute_bindings
        .iter()
        .map(|binding| binding.group)
        .chain(graph.bindings.iter().map(|binding| binding.group))
        .max()
}

fn validate_external_registry(
    graph: &CompiledGraph,
    registry: &CompiledGraphGpuResourceRegistry,
) -> Result<Vec<CompiledGraphGpuBinding>, CompiledGraphGpuError> {
    let requirements: Vec<_> = graph
        .bindings
        .iter()
        .filter(|binding| {
            binding.name != "ember_dispatch_counts" && binding.kind != ResourceKind::Uniform
        })
        .collect();

    for entry in registry.iter() {
        let Some(requirement) = requirements.iter().find(|requirement| {
            (requirement.group, requirement.binding) == (entry.slot.group, entry.slot.binding)
        }) else {
            return Err(CompiledGraphGpuError::UnexpectedExternalBinding {
                group: entry.slot.group,
                binding: entry.slot.binding,
            });
        };
        validate_external_resource_kind(requirement, entry.resource)?;
    }

    let mut resolved = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let slot = CompiledGraphGpuResourceSlot::new(requirement.group, requirement.binding);
        let Some(entry) = registry.get(slot) else {
            return Err(CompiledGraphGpuError::MissingExternalBinding {
                group: requirement.group,
                binding: requirement.binding,
                name: String::from(requirement.name),
            });
        };
        resolved.push(*entry);
    }
    Ok(resolved)
}

fn validate_external_resource_kind(
    requirement: &ResourceDecl,
    resource: CompiledGraphGpuExternalResource,
) -> Result<(), CompiledGraphGpuError> {
    let slot = (requirement.group, requirement.binding);
    let actual = resource.kind();
    let valid = match requirement.kind {
        ResourceKind::StorageBuffer => matches!(
            resource,
            CompiledGraphGpuExternalResource::StorageBuffer { .. }
        ),
        ResourceKind::Curve => {
            matches!(
                resource,
                CompiledGraphGpuExternalResource::CurveBuffer { .. }
            )
        }
        ResourceKind::Texture2d | ResourceKind::Texture3d => matches!(
            resource,
            CompiledGraphGpuExternalResource::Texture { .. }
                | CompiledGraphGpuExternalResource::TextureView { .. }
        ),
        ResourceKind::Uniform => false,
    };
    if valid {
        return Ok(());
    }
    if matches!(
        requirement.kind,
        ResourceKind::StorageBuffer | ResourceKind::Curve
    ) && matches!(
        actual,
        CompiledGraphGpuExternalKind::Texture | CompiledGraphGpuExternalKind::TextureView
    ) {
        return Err(CompiledGraphGpuError::ExpectedBuffer {
            group: slot.0,
            binding: slot.1,
            name: String::from(requirement.name),
        });
    }
    if matches!(
        requirement.kind,
        ResourceKind::Texture2d | ResourceKind::Texture3d
    ) && matches!(
        actual,
        CompiledGraphGpuExternalKind::StorageBuffer | CompiledGraphGpuExternalKind::CurveBuffer
    ) {
        return Err(CompiledGraphGpuError::ExpectedTexture {
            group: slot.0,
            binding: slot.1,
            name: String::from(requirement.name),
        });
    }
    Err(CompiledGraphGpuError::ExternalResourceKindMismatch {
        group: slot.0,
        binding: slot.1,
        name: String::from(requirement.name),
        expected: requirement.kind,
        actual,
    })
}

fn aligned_attribute_layout(
    layout: &AttributeLayoutPlan,
    alignment: u64,
) -> Result<AttributeLayoutPlan, CompiledGraphGpuAbiError> {
    let alignment = alignment.max(1);
    let mut attributes = Vec::with_capacity(layout.attributes.len());
    let mut offset = 0u64;
    for attribute in &layout.attributes {
        let logical_per_copy = u64::from(attribute.stride) * u64::from(layout.capacity);
        let physical_per_copy = align_up(logical_per_copy, alignment)
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        offset = align_up(offset, alignment)
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        let byte_size = physical_per_copy
            .checked_mul(u64::from(attribute.copies))
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        attributes.push(PlannedAttribute {
            semantic: attribute.semantic,
            format: attribute.format,
            stride: attribute.stride,
            copies: attribute.copies,
            byte_offset: offset,
            byte_size,
        });
        offset = offset
            .checked_add(byte_size)
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
    }
    Ok(AttributeLayoutPlan {
        capacity: layout.capacity,
        attributes,
        total_bytes: offset,
    })
}

const fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn validate_compiled_graph_abi(graph: &CompiledGraph) -> Result<(), CompiledGraphGpuAbiError> {
    if graph.entry_points.len() != graph.stage_order.len()
        || graph.entry_points.len() != graph.dispatches.len()
    {
        return Err(CompiledGraphGpuAbiError::EntryPointCountMismatch {
            entry_points: graph.entry_points.len(),
            stage_order: graph.stage_order.len(),
            dispatches: graph.dispatches.len(),
        });
    }

    for (index, entry_point) in graph.entry_points.iter().enumerate() {
        if entry_point.is_empty() {
            return Err(CompiledGraphGpuAbiError::EmptyEntryPoint { index });
        }
        if graph.entry_points[..index]
            .iter()
            .any(|previous| previous == entry_point)
        {
            return Err(CompiledGraphGpuAbiError::DuplicateEntryPoint {
                index,
                name: entry_point.clone(),
            });
        }
        let dispatch = &graph.dispatches[index];
        if dispatch.stage_index != graph.stage_order[index] {
            return Err(CompiledGraphGpuAbiError::DispatchStageMismatch {
                dispatch: index,
                expected: graph.stage_order[index],
                actual: dispatch.stage_index,
            });
        }
        if dispatch.entry_point != *entry_point {
            return Err(CompiledGraphGpuAbiError::DispatchEntryPointMismatch {
                dispatch: index,
                expected: entry_point.clone(),
                actual: dispatch.entry_point.clone(),
            });
        }
        if dispatch.stage_index == usize::MAX {
            return Err(CompiledGraphGpuAbiError::DispatchStageIndexOverflow { dispatch: index });
        }
    }
    for (index, stage_index) in graph.stage_order.iter().copied().enumerate() {
        if graph.stage_order[..index]
            .iter()
            .any(|previous| *previous == stage_index)
        {
            return Err(CompiledGraphGpuAbiError::DispatchStageMismatch {
                dispatch: index,
                expected: stage_index,
                actual: stage_index,
            });
        }
    }

    let mut slots = Vec::with_capacity(
        graph
            .bindings
            .len()
            .saturating_add(graph.attribute_bindings.len()),
    );
    let mut resource_names: Vec<&str> = Vec::with_capacity(graph.bindings.len());
    let mut dispatch_binding = None;
    for binding in &graph.bindings {
        if slots.contains(&(binding.group, binding.binding)) {
            return Err(CompiledGraphGpuAbiError::DuplicateBinding {
                group: binding.group,
                binding: binding.binding,
            });
        }
        slots.push((binding.group, binding.binding));
        if resource_names.contains(&binding.name) {
            return Err(CompiledGraphGpuAbiError::DuplicateResourceName {
                name: String::from(binding.name),
            });
        }
        resource_names.push(binding.name);
        if binding.name == "ember_dispatch_counts" {
            if dispatch_binding.is_some()
                || binding.group != 1
                || binding.binding != 0
                || binding.kind != ResourceKind::StorageBuffer
                || binding.access != ResourceAccess::ReadOnly
            {
                return Err(CompiledGraphGpuAbiError::InvalidDispatchBinding {
                    group: binding.group,
                    binding: binding.binding,
                });
            }
            dispatch_binding = Some(binding);
        }
    }
    if !graph.dispatches.is_empty() && dispatch_binding.is_none() {
        return Err(CompiledGraphGpuAbiError::MissingDispatchBinding);
    }
    if graph.dispatches.is_empty() && dispatch_binding.is_some() {
        return Err(CompiledGraphGpuAbiError::InvalidDispatchBinding {
            group: 1,
            binding: 0,
        });
    }

    validate_attribute_layout(&graph.attribute_layout)?;
    for attribute in &graph.attribute_bindings {
        if attribute.group != 0 {
            return Err(CompiledGraphGpuAbiError::AttributeBindingGroup {
                semantic: attribute.semantic,
                group: attribute.group,
            });
        }
        if slots.contains(&(attribute.group, attribute.binding)) {
            return Err(CompiledGraphGpuAbiError::DuplicateBinding {
                group: attribute.group,
                binding: attribute.binding,
            });
        }
        slots.push((attribute.group, attribute.binding));
        let Some(planned) = graph.attribute_layout.get(attribute.semantic) else {
            return Err(CompiledGraphGpuAbiError::MissingAttributeLayout {
                semantic: attribute.semantic,
            });
        };
        if planned.format != attribute.format {
            return Err(CompiledGraphGpuAbiError::AttributeLayoutMismatch {
                semantic: attribute.semantic,
            });
        }
        let role_is_valid = match (planned.copies, attribute.role) {
            (1, AttributeBufferRole::Single) => true,
            (2, AttributeBufferRole::Previous | AttributeBufferRole::Current) => true,
            _ => false,
        };
        if !role_is_valid {
            return Err(CompiledGraphGpuAbiError::AttributeBindingRoleMismatch {
                semantic: attribute.semantic,
            });
        }
        let same_semantic = graph
            .attribute_bindings
            .iter()
            .filter(|other| other.semantic == attribute.semantic);
        let role_count = same_semantic
            .clone()
            .filter(|other| other.role == attribute.role)
            .count();
        if role_count > 1 {
            return Err(CompiledGraphGpuAbiError::AttributeBindingRoleMismatch {
                semantic: attribute.semantic,
            });
        }
    }
    for planned in &graph.attribute_layout.attributes {
        let bindings: Vec<_> = graph
            .attribute_bindings
            .iter()
            .filter(|binding| binding.semantic == planned.semantic)
            .collect();
        let valid = match planned.copies {
            1 => bindings.len() == 1 && bindings[0].role == AttributeBufferRole::Single,
            2 => {
                bindings.len() == 2
                    && bindings
                        .iter()
                        .filter(|binding| binding.role == AttributeBufferRole::Previous)
                        .count()
                        == 1
                    && bindings
                        .iter()
                        .filter(|binding| binding.role == AttributeBufferRole::Current)
                        .count()
                        == 1
            }
            _ => false,
        };
        if !valid {
            return Err(CompiledGraphGpuAbiError::AttributeBindingRoleMismatch {
                semantic: planned.semantic,
            });
        }
    }

    validate_parameter_blocks(graph)
}

fn validate_attribute_layout(layout: &AttributeLayoutPlan) -> Result<(), CompiledGraphGpuAbiError> {
    let mut expected_offset = 0u64;
    for (index, attribute) in layout.attributes.iter().enumerate() {
        if layout.attributes[..index]
            .iter()
            .any(|previous| previous.semantic == attribute.semantic)
        {
            return Err(CompiledGraphGpuAbiError::DuplicateAttributeSemantic {
                semantic: attribute.semantic,
            });
        }
        if attribute.copies != 1 && attribute.copies != 2 {
            return Err(CompiledGraphGpuAbiError::InvalidAttributeCopies {
                semantic: attribute.semantic,
                copies: attribute.copies,
            });
        }
        let expected_stride = attribute.format.std430_stride();
        let expected_per_copy = u64::from(expected_stride)
            .checked_mul(u64::from(layout.capacity))
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        let expected_size = expected_per_copy
            .checked_mul(u64::from(attribute.copies))
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        let expected_end = attribute
            .byte_offset
            .checked_add(expected_size)
            .ok_or(CompiledGraphGpuAbiError::LayoutArithmeticOverflow)?;
        if attribute.stride != expected_stride
            || attribute.byte_size != expected_size
            || attribute.byte_offset != expected_offset
        {
            return Err(CompiledGraphGpuAbiError::AttributeLayoutMismatch {
                semantic: attribute.semantic,
            });
        }
        expected_offset = expected_end;
    }
    if layout.total_bytes != expected_offset {
        return Err(CompiledGraphGpuAbiError::AttributePoolSizeMismatch {
            expected: expected_offset,
            actual: layout.total_bytes,
        });
    }
    Ok(())
}

fn validate_parameter_blocks(graph: &CompiledGraph) -> Result<(), CompiledGraphGpuAbiError> {
    for (index, block) in graph.parameter_blocks.iter().enumerate() {
        if graph.parameter_blocks[..index]
            .iter()
            .any(|previous| previous.resource == block.resource)
        {
            return Err(CompiledGraphGpuAbiError::DuplicateParameterBlock {
                resource: block.resource.clone(),
            });
        }
        if block.size == 0 || block.default_bytes.len() != block.size as usize {
            return Err(CompiledGraphGpuAbiError::ParameterBlockSizeMismatch {
                resource: block.resource.clone(),
                expected: block.size as usize,
                actual: block.default_bytes.len(),
            });
        }
        for (field_index, field) in block.fields.iter().enumerate() {
            if block.fields[..field_index]
                .iter()
                .any(|previous| previous.name == field.name)
            {
                return Err(CompiledGraphGpuAbiError::DuplicateParameterField {
                    resource: block.resource.clone(),
                    field: field.name.clone(),
                });
            }
            if field.default_bytes.len() != field.ty.size() as usize {
                return Err(CompiledGraphGpuAbiError::ParameterFieldSizeMismatch {
                    resource: block.resource.clone(),
                    field: field.name.clone(),
                });
            }
            let Some(end) = field.offset.checked_add(field.ty.size()) else {
                return Err(CompiledGraphGpuAbiError::ParameterFieldOutOfBounds {
                    resource: block.resource.clone(),
                    field: field.name.clone(),
                });
            };
            if end > block.size {
                return Err(CompiledGraphGpuAbiError::ParameterFieldOutOfBounds {
                    resource: block.resource.clone(),
                    field: field.name.clone(),
                });
            }
        }
        let Some(binding) = graph.bindings.iter().find(|binding| {
            binding.kind == ResourceKind::Uniform
                && binding.name == block.resource
                && binding.group == block.group
                && binding.binding == block.binding
        }) else {
            return Err(CompiledGraphGpuAbiError::UnexpectedParameterBlock {
                resource: block.resource.clone(),
            });
        };
        let _ = binding;
    }
    for binding in graph
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::Uniform)
    {
        let matches = graph
            .parameter_blocks
            .iter()
            .filter(|block| {
                block.resource == binding.name
                    && block.group == binding.group
                    && block.binding == binding.binding
            })
            .count();
        if matches == 0 {
            return Err(CompiledGraphGpuAbiError::MissingParameterBlock {
                resource: String::from(binding.name),
            });
        }
        if matches > 1 {
            return Err(CompiledGraphGpuAbiError::DuplicateParameterBlock {
                resource: String::from(binding.name),
            });
        }
    }
    for block in &graph.parameter_blocks {
        if !graph.bindings.iter().any(|binding| {
            binding.kind == ResourceKind::Uniform
                && binding.name == block.resource
                && binding.group == block.group
                && binding.binding == block.binding
        }) {
            return Err(CompiledGraphGpuAbiError::UnexpectedParameterBlock {
                resource: block.resource.clone(),
            });
        }
    }
    Ok(())
}

fn compiled_graph_abi_fingerprint(graph: &CompiledGraph) -> CompiledGraphGpuAbiFingerprint {
    let mut hasher = FingerprintHasher::new();
    hasher.str_value(&graph.source);
    hasher.usize(graph.stage_order.len());
    for value in &graph.stage_order {
        hasher.usize(*value);
    }
    hasher.usize(graph.entry_points.len());
    for value in &graph.entry_points {
        hasher.str_value(value);
    }
    hasher.usize(graph.dispatches.len());
    for dispatch in &graph.dispatches {
        hasher.usize(dispatch.stage_index);
        hasher.iteration_domain(dispatch.domain);
        hasher.u32(dispatch.default_count);
        hasher.u32(dispatch.iterations);
        hasher.str_value(&dispatch.entry_point);
    }
    hasher.usize(graph.bindings.len());
    for binding in &graph.bindings {
        hasher.str_value(binding.name);
        hasher.resource_kind(binding.kind);
        hasher.resource_access(binding.access);
        hasher.attribute_format(binding.format);
        hasher.u32(binding.group);
        hasher.u32(binding.binding);
    }
    hasher.usize(graph.parameter_blocks.len());
    for block in &graph.parameter_blocks {
        hasher.str_value(&block.resource);
        hasher.u32(block.group);
        hasher.u32(block.binding);
        hasher.u32(block.size);
        hasher.bytes(&block.default_bytes);
        hasher.usize(block.fields.len());
        for field in &block.fields {
            hasher.str_value(&field.name);
            hasher.parameter_type(field.ty);
            hasher.u32(field.offset);
            hasher.bytes(&field.default_bytes);
        }
    }
    hasher.u32(graph.attribute_layout.capacity);
    hasher.usize(graph.attribute_layout.attributes.len());
    for attribute in &graph.attribute_layout.attributes {
        hasher.attribute_semantic(attribute.semantic);
        hasher.attribute_format(attribute.format);
        hasher.u32(attribute.stride);
        hasher.u32(attribute.copies);
        hasher.u64(attribute.byte_offset);
        hasher.u64(attribute.byte_size);
    }
    hasher.u64(graph.attribute_layout.total_bytes);
    hasher.usize(graph.attribute_bindings.len());
    for binding in &graph.attribute_bindings {
        hasher.attribute_semantic(binding.semantic);
        hasher.attribute_format(binding.format);
        hasher.attribute_buffer_role(binding.role);
        hasher.str_value(&binding.name);
        hasher.u32(binding.group);
        hasher.u32(binding.binding);
        hasher.resource_access(binding.access);
    }
    CompiledGraphGpuAbiFingerprint(hasher.finish())
}

struct FingerprintHasher {
    state: u64,
}

impl FingerprintHasher {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    const fn new() -> Self {
        Self {
            state: Self::OFFSET,
        }
    }

    fn byte(&mut self, value: u8) {
        self.state ^= u64::from(value);
        self.state = self.state.wrapping_mul(Self::PRIME);
    }

    fn raw_bytes(&mut self, values: &[u8]) {
        for value in values {
            self.byte(*value);
        }
    }

    fn bytes(&mut self, values: &[u8]) {
        self.raw_bytes(&(values.len() as u64).to_le_bytes());
        self.raw_bytes(values);
    }

    fn u32(&mut self, value: u32) {
        self.raw_bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw_bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn str_value(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn attribute_semantic(&mut self, value: AttributeSemantic) {
        match value {
            AttributeSemantic::Custom(id) => {
                self.byte(19);
                self.u32(id);
            }
            other => self.byte(other.ordinal() as u8),
        }
    }

    fn attribute_format(
        &mut self,
        value: prism_render_architecture::particle::attributes::AttributeFormat,
    ) {
        self.byte(match value {
            prism_render_architecture::particle::attributes::AttributeFormat::F32 => 0,
            prism_render_architecture::particle::attributes::AttributeFormat::U32 => 1,
            prism_render_architecture::particle::attributes::AttributeFormat::Vec2 => 2,
            prism_render_architecture::particle::attributes::AttributeFormat::Vec3 => 3,
            prism_render_architecture::particle::attributes::AttributeFormat::Vec4 => 4,
        });
    }

    fn resource_kind(&mut self, value: ResourceKind) {
        self.byte(match value {
            ResourceKind::Uniform => 0,
            ResourceKind::Curve => 1,
            ResourceKind::Texture2d => 2,
            ResourceKind::Texture3d => 3,
            ResourceKind::StorageBuffer => 4,
        });
    }

    fn resource_access(&mut self, value: ResourceAccess) {
        self.byte(match value {
            ResourceAccess::ReadOnly => 0,
            ResourceAccess::ReadWrite => 1,
        });
    }

    fn parameter_type(
        &mut self,
        value: prism_render_architecture::particle::modules::ParameterType,
    ) {
        self.byte(match value {
            prism_render_architecture::particle::modules::ParameterType::F32 => 0,
            prism_render_architecture::particle::modules::ParameterType::U32 => 1,
            prism_render_architecture::particle::modules::ParameterType::I32 => 2,
            prism_render_architecture::particle::modules::ParameterType::Vec2 => 3,
            prism_render_architecture::particle::modules::ParameterType::Vec3 => 4,
            prism_render_architecture::particle::modules::ParameterType::Vec4 => 5,
        });
    }

    fn attribute_buffer_role(&mut self, value: AttributeBufferRole) {
        self.byte(match value {
            AttributeBufferRole::Single => 0,
            AttributeBufferRole::Previous => 1,
            AttributeBufferRole::Current => 2,
        });
    }

    fn iteration_domain(&mut self, value: prism_render_architecture::particle::IterationDomain) {
        match value {
            prism_render_architecture::particle::IterationDomain::PerParticle => self.byte(0),
            prism_render_architecture::particle::IterationDomain::PerNeighborCell => self.byte(1),
            prism_render_architecture::particle::IterationDomain::PerGridVoxel => self.byte(2),
            prism_render_architecture::particle::IterationDomain::PerConstraint => self.byte(3),
            prism_render_architecture::particle::IterationDomain::PerEvent => self.byte(4),
            prism_render_architecture::particle::IterationDomain::Custom(value) => {
                self.byte(5);
                self.u32(value);
            }
        }
    }

    const fn finish(self) -> u64 {
        self.state
    }
}

fn group_entries(
    graph: &CompiledGraph,
    external: &[CompiledGraphGpuBinding],
    parameter_buffers: &[ParameterBuffer],
    dispatch_counts: Option<BufferId>,
    attribute_pool: Option<BufferId>,
    attribute_layout: &AttributeLayoutPlan,
    frame_index: u64,
    group: u32,
) -> Result<Vec<ResolvedEntry>, CompiledGraphGpuError> {
    let mut entries = Vec::new();
    for attribute in graph
        .attribute_bindings
        .iter()
        .filter(|binding| binding.group == group)
    {
        let Some(pool) = attribute_pool else {
            return Err(CompiledGraphGpuError::InvalidGraphAbi(
                CompiledGraphGpuAbiError::MissingAttributeLayout {
                    semantic: attribute.semantic,
                },
            ));
        };
        let Some(planned) = attribute_layout.get(attribute.semantic) else {
            return Err(CompiledGraphGpuError::InvalidGraphAbi(
                CompiledGraphGpuAbiError::MissingAttributeLayout {
                    semantic: attribute.semantic,
                },
            ));
        };
        let per_copy = u64::from(planned.stride) * u64::from(attribute_layout.capacity);
        let physical_copy_span = planned.byte_size / u64::from(planned.copies);
        let physical_copy = attribute.physical_copy(frame_index) as u64;
        let offset = planned.byte_offset + physical_copy * physical_copy_span;
        entries.push(ResolvedEntry {
            binding: attribute.binding,
            layout: storage_layout_entry(attribute.binding, attribute.access),
            resource: BindingResource::Buffer {
                buffer: pool,
                offset,
                size: Some(per_copy),
            },
        });
    }
    for binding in graph
        .bindings
        .iter()
        .filter(|binding| binding.group == group)
    {
        if binding.name == "ember_dispatch_counts" {
            let Some(buffer) = dispatch_counts else {
                continue;
            };
            entries.push(ResolvedEntry {
                binding: binding.binding,
                layout: storage_layout_entry(binding.binding, binding.access),
                resource: BindingResource::Buffer {
                    buffer,
                    offset: 0,
                    size: None,
                },
            });
            continue;
        }
        match binding.kind {
            ResourceKind::Uniform => {
                let Some(parameter) = parameter_buffers.iter().find(|parameter| {
                    parameter.resource == binding.name
                        && parameter.group == binding.group
                        && parameter.binding == binding.binding
                }) else {
                    return Err(CompiledGraphGpuError::MissingParameterBlock {
                        group,
                        binding: binding.binding,
                        name: String::from(binding.name),
                    });
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: BindGroupLayoutEntry::uniform(binding.binding, ShaderStages::COMPUTE),
                    resource: BindingResource::Buffer {
                        buffer: parameter.buffer,
                        offset: 0,
                        size: Some(parameter.size),
                    },
                });
            }
            ResourceKind::Curve | ResourceKind::StorageBuffer => {
                let Some(external) = external
                    .iter()
                    .find(|entry| entry.slot() == (group, binding.binding))
                else {
                    return Err(CompiledGraphGpuError::MissingExternalBinding {
                        group,
                        binding: binding.binding,
                        name: String::from(binding.name),
                    });
                };
                let (buffer, size) = match external.resource {
                    CompiledGraphGpuExternalResource::StorageBuffer { buffer, size }
                    | CompiledGraphGpuExternalResource::CurveBuffer { buffer, size } => {
                        (buffer, size)
                    }
                    CompiledGraphGpuExternalResource::Texture { .. }
                    | CompiledGraphGpuExternalResource::TextureView { .. } => {
                        return Err(CompiledGraphGpuError::ExpectedBuffer {
                            group,
                            binding: binding.binding,
                            name: String::from(binding.name),
                        });
                    }
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: storage_layout_entry(binding.binding, binding.access),
                    resource: BindingResource::Buffer {
                        buffer,
                        offset: 0,
                        size,
                    },
                });
            }
            ResourceKind::Texture2d | ResourceKind::Texture3d => {
                let Some(external) = external
                    .iter()
                    .find(|entry| entry.slot() == (group, binding.binding))
                else {
                    return Err(CompiledGraphGpuError::MissingExternalBinding {
                        group,
                        binding: binding.binding,
                        name: String::from(binding.name),
                    });
                };
                let view = match external.resource {
                    CompiledGraphGpuExternalResource::Texture { view, .. }
                    | CompiledGraphGpuExternalResource::TextureView { view } => view,
                    CompiledGraphGpuExternalResource::StorageBuffer { .. }
                    | CompiledGraphGpuExternalResource::CurveBuffer { .. } => {
                        return Err(CompiledGraphGpuError::ExpectedTexture {
                            group,
                            binding: binding.binding,
                            name: String::from(binding.name),
                        });
                    }
                };
                entries.push(ResolvedEntry {
                    binding: binding.binding,
                    layout: texture_layout_entry(binding)
                        .map_err(CompiledGraphGpuError::InvalidGraphAbi)?,
                    resource: BindingResource::TextureView(view),
                });
            }
        }
    }
    Ok(entries)
}

fn storage_layout_entry(binding: u32, access: ResourceAccess) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: access == ResourceAccess::ReadOnly,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_layout_entry(
    binding: &ResourceDecl,
) -> Result<BindGroupLayoutEntry, CompiledGraphGpuAbiError> {
    let view_dimension = match binding.kind {
        ResourceKind::Texture2d => TextureViewDimension::D2,
        ResourceKind::Texture3d => TextureViewDimension::D3,
        _ => {
            return Err(CompiledGraphGpuAbiError::InvalidTextureBindingKind {
                group: binding.group,
                binding: binding.binding,
            });
        }
    };
    Ok(BindGroupLayoutEntry {
        binding: binding.binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float,
            view_dimension,
            multisampled: false,
        },
        count: None,
    })
}

fn binding_resource(resource: CompiledGraphGpuExternalResource) -> BindingResource {
    match resource {
        CompiledGraphGpuExternalResource::StorageBuffer { buffer, size }
        | CompiledGraphGpuExternalResource::CurveBuffer { buffer, size } => {
            BindingResource::Buffer {
                buffer,
                offset: 0,
                size,
            }
        }
        CompiledGraphGpuExternalResource::Texture { view, .. }
        | CompiledGraphGpuExternalResource::TextureView { view } => {
            BindingResource::TextureView(view)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::particle::attributes::{
        AttributeAccess, AttributeFormat, AttributeLayoutPlan, AttributeSemantic, AttributeUsage,
    };
    use prism_render_architecture::particle::compiler::{
        AttributeBufferRole, CompiledAttributeBinding, CompiledDispatch, CompiledParameterBlock,
        CompiledParameterField,
    };
    use prism_render_architecture::particle::modules::{ParameterType, ResourceDecl};
    use prism_render_architecture::particle::IterationDomain;
    use prism_render_driver::Instance;
    use prism_render_driver_null::NullInstance;
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{BufferDesc, ImportedBuffer, TextureDesc};

    fn graph_with_one_parameter() -> CompiledGraph {
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: vec![ResourceDecl::uniform("params", 2, 0)],
            parameter_blocks: vec![CompiledParameterBlock {
                resource: String::from("params"),
                group: 2,
                binding: 0,
                fields: vec![CompiledParameterField {
                    name: String::from("gain"),
                    ty: ParameterType::F32,
                    offset: 0,
                    default_bytes: 1.0f32.to_bits().to_le_bytes().to_vec(),
                }],
                size: 16,
                default_bytes: {
                    let mut bytes = vec![0u8; 16];
                    bytes[..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
                    bytes
                },
            }],
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        }
    }

    fn graph_with_ping_pong_attribute() -> CompiledGraph {
        let layout = AttributeLayoutPlan::build(
            4,
            &[AttributeUsage::new(
                AttributeSemantic::Position,
                AttributeFormat::Vec3,
                AttributeAccess::READ_PREV.union(AttributeAccess::WRITE),
            )],
        );
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: Vec::new(),
            parameter_blocks: Vec::new(),
            attribute_layout: layout,
            attribute_bindings: vec![
                CompiledAttributeBinding {
                    semantic: AttributeSemantic::Position,
                    format: AttributeFormat::Vec3,
                    role: AttributeBufferRole::Previous,
                    name: String::from("position_previous"),
                    group: 0,
                    binding: 0,
                    access: ResourceAccess::ReadOnly,
                },
                CompiledAttributeBinding {
                    semantic: AttributeSemantic::Position,
                    format: AttributeFormat::Vec3,
                    role: AttributeBufferRole::Current,
                    name: String::from("position_current"),
                    group: 0,
                    binding: 1,
                    access: ResourceAccess::ReadWrite,
                },
            ],
        }
    }

    fn graph_with_external_binding(binding: ResourceDecl) -> CompiledGraph {
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: vec![binding],
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        }
    }

    #[test]
    fn external_registry_clone_and_snapshot_preserve_revisions() {
        let storage_slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let curve_slot = CompiledGraphGpuResourceSlot::new(0, 1);
        let texture_slot = CompiledGraphGpuResourceSlot::new(0, 2);
        let view_slot = CompiledGraphGpuResourceSlot::new(0, 3);
        let mut registry = CompiledGraphGpuResourceRegistry::new();
        let storage_revision = registry
            .set_storage_buffer(storage_slot, BufferId::from_parts(1, 1), Some(64))
            .expect("storage revision");
        let curve_revision = registry
            .set_curve_buffer(curve_slot, BufferId::from_parts(2, 1), Some(128))
            .expect("curve revision");
        let texture_revision = registry
            .set_texture(
                texture_slot,
                TextureId::from_parts(3, 1),
                TextureViewId::from_parts(4, 1),
            )
            .expect("texture revision");
        let view_revision = registry
            .set_texture_view(view_slot, TextureViewId::from_parts(5, 1))
            .expect("texture view revision");

        assert_eq!(storage_revision.value(), 1);
        assert_eq!(curve_revision.value(), 2);
        assert_eq!(texture_revision.value(), 3);
        assert_eq!(view_revision.value(), 4);

        let snapshot = registry.snapshot();
        let clone = registry.clone();
        assert_eq!(clone.snapshot(), snapshot);
        assert_eq!(clone.get(storage_slot).unwrap().revision, storage_revision);
        assert_eq!(clone.get(curve_slot).unwrap().revision, curve_revision);
        assert_eq!(clone.get(texture_slot).unwrap().revision, texture_revision);
        assert_eq!(clone.get(view_slot).unwrap().revision, view_revision);

        let mut extracted = CompiledGraphGpuResourceRegistry::new();
        extracted
            .replace_snapshot(&snapshot)
            .expect("published snapshot");
        assert_eq!(extracted.snapshot(), snapshot);
        let next_revision = extracted
            .set_texture_view(
                CompiledGraphGpuResourceSlot::new(1, 0),
                TextureViewId::from_parts(6, 1),
            )
            .expect("revision after snapshot")
            .value();
        assert_eq!(next_revision, 5);

        let exact_binding = registry.get(curve_slot).copied().unwrap();
        let mut exact = CompiledGraphGpuResourceRegistry::new();
        exact
            .insert_exact_revision(exact_binding)
            .expect("exact revision insertion");
        assert_eq!(
            exact
                .set_storage_buffer(
                    CompiledGraphGpuResourceSlot::new(2, 0),
                    BufferId::from_parts(7, 1),
                    None,
                )
                .expect("revision after exact insertion")
                .value(),
            curve_revision.value() + 1
        );
    }

    #[test]
    fn external_registry_rejects_duplicate_exact_slots_atomically() {
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let mut registry = CompiledGraphGpuResourceRegistry::new();
        registry
            .set_storage_buffer(slot, BufferId::from_parts(8, 1), None)
            .expect("initial resource");
        let binding = registry.get(slot).copied().unwrap();

        assert_eq!(
            registry.insert_exact_revision(binding),
            Err(CompiledGraphGpuRegistryError::DuplicateSlot { slot })
        );

        let before = registry.snapshot();
        assert_eq!(
            registry.replace_snapshot(&[binding, binding]),
            Err(CompiledGraphGpuRegistryError::DuplicateSlot { slot })
        );
        assert_eq!(registry.snapshot(), before);
    }

    #[test]
    fn external_registry_reports_missing_and_wrong_resource_kinds() {
        let buffer_graph = graph_with_external_binding(ResourceDecl::buffer("particles", 0, 0));
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        assert_eq!(
            validate_external_registry(&buffer_graph, &CompiledGraphGpuResourceRegistry::new()),
            Err(CompiledGraphGpuError::MissingExternalBinding {
                group: 0,
                binding: 0,
                name: String::from("particles"),
            })
        );

        let mut wrong_kind = CompiledGraphGpuResourceRegistry::new();
        wrong_kind
            .set_texture_view(slot, TextureViewId::from_parts(9, 1))
            .expect("texture view revision");
        assert_eq!(
            validate_external_registry(&buffer_graph, &wrong_kind),
            Err(CompiledGraphGpuError::ExpectedBuffer {
                group: 0,
                binding: 0,
                name: String::from("particles"),
            })
        );

        let texture_graph = graph_with_external_binding(ResourceDecl::texture2d("field", 1, 2));
        let texture_slot = CompiledGraphGpuResourceSlot::new(1, 2);
        let mut wrong_texture_kind = CompiledGraphGpuResourceRegistry::new();
        wrong_texture_kind
            .set_storage_buffer(texture_slot, BufferId::from_parts(10, 1), None)
            .expect("storage revision");
        assert_eq!(
            validate_external_registry(&texture_graph, &wrong_texture_kind),
            Err(CompiledGraphGpuError::ExpectedTexture {
                group: 1,
                binding: 2,
                name: String::from("field"),
            })
        );
    }

    #[test]
    fn external_state_refreshes_bind_groups_and_rejects_stale_registry() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_external_binding(ResourceDecl::buffer("particles", 0, 0));
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let buffer_one = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("external-one")),
            size: 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut registry = CompiledGraphGpuResourceRegistry::new();
        let first_revision = registry
            .set_storage_buffer(slot, buffer_one, Some(64))
            .expect("first external revision");
        let stale_registry = registry.clone();
        let mut state = CompiledGraphGpuState::new(&device, &queue, &graph, &registry)
            .expect("external buffer state");

        let mut changed_graph = graph.clone();
        changed_graph.source.push_str("changed");
        assert_eq!(
            state.refresh_external_resources(&device, &changed_graph, &registry),
            Err(CompiledGraphGpuError::GraphAbiMismatch {
                expected: state.graph_abi_fingerprint,
                actual: compiled_graph_abi_fingerprint(&changed_graph),
            })
        );

        let buffer_two = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("external-two")),
            size: 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let second_revision = registry
            .set_storage_buffer(slot, buffer_two, Some(64))
            .expect("second external revision");
        assert!(second_revision.value() > first_revision.value());
        assert_eq!(
            state.refresh_external_resources(&device, &graph, &registry),
            Ok(CompiledGraphGpuExternalUpdate::Rebuilt { changed_slots: 1 })
        );
        assert_eq!(
            state.external_binding(slot).unwrap().revision,
            second_revision
        );
        assert_eq!(
            state.external_binding(slot).unwrap().resource,
            CompiledGraphGpuExternalResource::StorageBuffer {
                buffer: buffer_two,
                size: Some(64),
            }
        );
        assert_eq!(
            state.refresh_external_resources(&device, &graph, &stale_registry),
            Err(CompiledGraphGpuError::StaleExternalRevision {
                group: 0,
                binding: 0,
                expected: second_revision,
                actual: first_revision,
            })
        );

        state.destroy(&device);
        device.destroy_buffer(buffer_one);
        device.destroy_buffer(buffer_two);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn external_graph_imports_and_bindings_carry_the_published_revision() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_external_binding(ResourceDecl::buffer("particles", 0, 0));
        let slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("external-import")),
            size: 64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut registry = CompiledGraphGpuResourceRegistry::new();
        let revision = registry
            .set_storage_buffer(slot, buffer, Some(64))
            .expect("external revision");
        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &registry)
            .expect("external buffer state");
        assert_eq!(
            state.external_bind_group_resource(slot),
            Some((
                revision,
                BindingResource::Buffer {
                    buffer,
                    offset: 0,
                    size: Some(64),
                },
            ))
        );

        let mut render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let binding = state
            .import_external_to_graph(
                &mut render_graph,
                slot,
                "external-particles",
                CompiledGraphGpuGraphImport::Buffer {
                    desc: BufferDesc::new(64, BufferUsages::STORAGE),
                    entry_state: BufferState::initial(),
                },
            )
            .expect("buffer import");
        assert_eq!(binding.revision(), revision);
        let graph_handle = match binding {
            CompiledGraphGpuGraphBinding::Buffer { handle, .. } => handle,
            CompiledGraphGpuGraphBinding::Texture { .. } => unreachable!("buffer import"),
        };
        assert!(state
            .external_graph_buffer_binding(slot, graph_handle)
            .is_some());

        let stale = match binding {
            CompiledGraphGpuGraphBinding::Buffer {
                group,
                binding,
                handle,
                ..
            } => CompiledGraphGpuGraphBinding::Buffer {
                group,
                binding,
                revision: CompiledGraphGpuRevision::from_value(revision.value() - 1),
                handle,
            },
            CompiledGraphGpuGraphBinding::Texture { .. } => unreachable!("buffer import"),
        };
        assert_eq!(
            graph_accesses(&graph, &[stale], state.external_bindings()),
            Err(CompiledGraphGraphError::ExternalRevisionMismatch {
                group: 0,
                binding: 0,
                expected: revision,
                actual: CompiledGraphGpuRevision::from_value(revision.value() - 1),
            })
        );

        state.destroy(&device);
        device.destroy_buffer(buffer);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn external_texture_import_supports_texture_and_view_only_slots() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: vec![
                ResourceDecl::texture2d("field", 0, 0),
                ResourceDecl::texture2d("view_only", 0, 1),
            ],
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        };
        let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(1, 1),
            rhi::TextureFormat::Rgba8Unorm,
            rhi::TextureUsages::TEXTURE_BINDING,
        ));
        let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());
        let texture_slot = CompiledGraphGpuResourceSlot::new(0, 0);
        let view_slot = CompiledGraphGpuResourceSlot::new(0, 1);
        let mut registry = CompiledGraphGpuResourceRegistry::new();
        let texture_revision = registry
            .set_texture(texture_slot, texture, view)
            .expect("texture revision");
        let view_revision = registry
            .set_texture_view(view_slot, view)
            .expect("view revision");
        let state = CompiledGraphGpuState::new(&device, &queue, &graph, &registry)
            .expect("external texture state");
        let mut render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let imported = state
            .import_external_to_graph(
                &mut render_graph,
                texture_slot,
                "external-field",
                CompiledGraphGpuGraphImport::Texture {
                    desc: TextureDesc::color(rhi::TextureFormat::Rgba8Unorm),
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::initial(),
                    mip_levels: 1,
                    array_layers: 1,
                },
            )
            .expect("texture import");
        assert_eq!(imported.revision(), texture_revision);
        let view_handle = render_graph.import_texture(
            "view-only",
            TextureDesc::color(rhi::TextureFormat::Rgba8Unorm),
            ImportedTexture {
                texture,
                default_view: view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        assert!(state
            .external_graph_texture_binding(view_slot, view_handle)
            .is_some());
        assert_eq!(
            state.import_external_to_graph(
                &mut render_graph,
                view_slot,
                "external-view-only",
                CompiledGraphGpuGraphImport::Texture {
                    desc: TextureDesc::color(rhi::TextureFormat::Rgba8Unorm),
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::initial(),
                    mip_levels: 1,
                    array_layers: 1,
                },
            ),
            Err(CompiledGraphGpuImportError::TextureObjectRequired { slot: view_slot })
        );
        assert_eq!(
            state.external_binding(view_slot).unwrap().revision,
            view_revision
        );

        state.destroy(&device);
        device.destroy_texture_view(view);
        device.destroy_texture(texture);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn compiled_graph_state_owns_parameter_buffer_and_bindings() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_one_parameter();
        let state = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        )
        .expect("uniform resources are self-contained");
        assert!(state.bind_group_layout(2).is_some());
        assert!(state.bind_group(2).is_some());
        assert!(state.pipeline_layout().index() != u32::MAX);
        state
            .update_parameters(
                &queue,
                &[PreparedParameterBlock {
                    resource: String::from("params"),
                    group: 2,
                    binding: 0,
                    bytes: {
                        let mut bytes = vec![0u8; 16];
                        bytes[..4].copy_from_slice(&2.0f32.to_bits().to_le_bytes());
                        bytes
                    },
                }],
            )
            .expect("compiled parameter ABI update");
        assert!(device.live_resource_count() >= 5);
        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn parameter_update_rejects_incomplete_and_wrong_sized_payloads() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_one_parameter();
        let state = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        )
        .expect("uniform resources are self-contained");
        let error = state
            .update_parameters(&queue, &[])
            .expect_err("missing parameter must be observable");
        assert_eq!(
            error,
            ParameterUploadError::MissingResource {
                resource: String::from("params")
            }
        );
        let error = state
            .update_parameters(
                &queue,
                &[PreparedParameterBlock {
                    resource: String::from("params"),
                    group: 2,
                    binding: 0,
                    bytes: vec![0; 4],
                }],
            )
            .expect_err("wrong parameter size must be observable");
        assert_eq!(
            error,
            ParameterUploadError::SizeMismatch {
                resource: String::from("params"),
                expected: 16,
                actual: 4,
            }
        );
        state.destroy(&device);
    }

    #[test]
    fn compiled_graph_records_stage_dispatches_into_render_graph() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = CompiledGraph {
            source: String::new(),
            stage_order: vec![2],
            entry_points: vec![String::from("ember_stage_2")],
            dispatches: vec![CompiledDispatch {
                stage_index: 2,
                domain: IterationDomain::PerParticle,
                default_count: 1,
                iterations: 2,
                entry_point: String::from("ember_stage_2"),
            }],
            bindings: vec![ResourceDecl::buffer("ember_dispatch_counts", 1, 0)],
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        };
        let state = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        )
        .expect("dispatch count binding is state-owned");
        let mut changed_graph = graph.clone();
        changed_graph.source.push_str("changed");
        let mut mismatched_render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        assert_eq!(
            state.append_to_graph_for_frame(
                &changed_graph,
                &mut mismatched_render_graph,
                &[],
                &[0, 0, 0],
                0,
            ),
            Err(CompiledGraphGraphError::GraphAbiMismatch {
                expected: state.graph_abi_fingerprint,
                actual: compiled_graph_abi_fingerprint(&changed_graph),
            })
        );
        assert_eq!(state.dispatch_count_slots(), 3);
        state
            .update_dispatch_counts(&queue, &[0, 0, 65])
            .expect("counts are indexed by compiled stage index");

        let dispatch_buffer = state
            .dispatch_counts_buffer()
            .expect("compiled stage owns dispatch counts");
        let mut render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let dispatch_handle = render_graph.import_buffer(
            "ember-dispatch-counts",
            BufferDesc::new(12, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: dispatch_buffer,
                entry_state: BufferState::initial(),
            },
        );
        state
            .append_to_graph(
                &graph,
                &mut render_graph,
                &[CompiledGraphGpuGraphBinding::Buffer {
                    group: 1,
                    binding: 0,
                    revision: CompiledGraphGpuRevision::from_value(0),
                    handle: dispatch_handle,
                }],
                &[0, 0, 65],
            )
            .expect("compiled stages are declared in the frame graph");
        let recorded = render_graph
            .execute(&device, &queue)
            .expect("null RHI records the compute frame");
        assert_eq!(recorded.commands().passes.len(), 1);
        let prism_render_driver::Pass::Compute { commands, .. } = &recorded.commands().passes[0]
        else {
            panic!("compiled Ember stage must record a compute pass");
        };
        assert_eq!(
            commands
                .iter()
                .filter(|command| matches!(
                    command,
                    prism_render_driver::ComputeCommand::Dispatch { x: 2, y: 1, z: 1 }
                ))
                .count(),
            2
        );
        recorded.release(&device);
        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn malformed_compiled_graph_is_rejected_before_resource_resolution() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let mut graph = graph_with_ping_pong_attribute();
        graph.attribute_layout.attributes[0].copies = 0;
        let result = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        );
        assert!(matches!(
            result,
            Err(CompiledGraphGpuError::InvalidGraphAbi(
                CompiledGraphGpuAbiError::InvalidAttributeCopies {
                    semantic: AttributeSemantic::Position,
                    copies: 0,
                }
            ))
        ));
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn attribute_pool_uses_layout_ranges_and_ping_pong_bindings() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let graph = graph_with_ping_pong_attribute();
        assert_eq!(graph.attribute_layout.total_bytes, 128);

        let state = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        )
        .expect("attribute pool is state-owned");
        let pool = state
            .attribute_pool_buffer()
            .expect("live attributes own a persistent pool");
        assert_eq!(state.attribute_pool_size(), 512);
        state
            .upload_attribute_bytes(&queue, 64, &[7; 64])
            .expect("an in-range copy upload succeeds");
        assert_eq!(
            state
                .upload_attribute_bytes(&queue, 120, &[7; 9])
                .expect_err("an upload past the pool must be rejected"),
            AttributeUploadError::OutOfBounds {
                offset: 120,
                size: 9,
                capacity: 128,
            }
        );

        let frame_zero = group_entries(
            &graph,
            &[],
            &[],
            None,
            Some(pool),
            &state.attribute_layout,
            0,
            0,
        )
        .expect("frame-zero attribute bindings");
        let frame_one = group_entries(
            &graph,
            &[],
            &[],
            None,
            Some(pool),
            &state.attribute_layout,
            1,
            0,
        )
        .expect("frame-one attribute bindings");
        assert_eq!(
            frame_zero[0].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 256,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_zero[1].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 0,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_one[0].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 0,
                size: Some(64),
            }
        );
        assert_eq!(
            frame_one[1].resource,
            BindingResource::Buffer {
                buffer: pool,
                offset: 256,
                size: Some(64),
            }
        );
        assert_ne!(
            state.bind_group_for_frame(0, 0),
            state.bind_group_for_frame(0, 1)
        );

        state.destroy(&device);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn physical_gpu_ember_compute_modifies_attribute_pool() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
                );
                eprintln!("SKIPPED Ember physical GPU test: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
                );
                eprintln!("SKIPPED Ember physical GPU test: software/virtual adapter");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("physical GPU adapter was found but device creation failed: {error}");
            }
        };

        let graph = CompiledGraph {
            source: String::from(
                "@group(0) @binding(0) var<storage, read_write> ember_attr_age: array<f32>;\n\
                 @group(1) @binding(0) var<storage, read> ember_dispatch_counts: array<u32>;\n\
                 @compute @workgroup_size(64)\n\
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
                     if (id.x >= ember_dispatch_counts[0u]) { return; }\n\
                     ember_attr_age[id.x] = ember_attr_age[id.x] + 1.0;\n\
                 }",
            ),
            stage_order: vec![0],
            entry_points: vec![String::from("main")],
            dispatches: vec![CompiledDispatch {
                stage_index: 0,
                domain: IterationDomain::PerParticle,
                default_count: 1,
                iterations: 1,
                entry_point: String::from("main"),
            }],
            bindings: vec![ResourceDecl::buffer("ember_dispatch_counts", 1, 0)],
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(
                1,
                &[AttributeUsage::new(
                    AttributeSemantic::Age,
                    AttributeFormat::F32,
                    AttributeAccess::READ.union(AttributeAccess::WRITE),
                )],
            ),
            attribute_bindings: vec![CompiledAttributeBinding {
                semantic: AttributeSemantic::Age,
                format: AttributeFormat::F32,
                role: AttributeBufferRole::Single,
                name: String::from("ember_attr_age"),
                group: 0,
                binding: 0,
                access: ResourceAccess::ReadWrite,
            }],
        };
        let state = CompiledGraphGpuState::new(
            &device,
            &queue,
            &graph,
            &CompiledGraphGpuResourceRegistry::new(),
        )
        .expect("physical device accepts the one-particle Ember graph");
        let attribute_pool = state
            .attribute_pool_buffer()
            .expect("the graph owns an attribute pool");
        state
            .upload_attribute_bytes(&queue, 0, &1.5f32.to_le_bytes())
            .expect("attribute upload fits the logical layout");
        state
            .update_dispatch_counts(&queue, &[1])
            .expect("one dispatch count is accepted");

        let mut render_graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let attribute_handle = render_graph.import_buffer(
            "ember-attributes",
            BufferDesc::new(
                state.attribute_pool_size(),
                BufferUsages::STORAGE.union(BufferUsages::COPY_DST),
            ),
            ImportedBuffer {
                buffer: attribute_pool,
                entry_state: BufferState::initial(),
            },
        );
        let dispatch_buffer = state
            .dispatch_counts_buffer()
            .expect("the graph owns dispatch counts");
        let dispatch_handle = render_graph.import_buffer(
            "ember-dispatch-counts",
            BufferDesc::new(4, BufferUsages::STORAGE.union(BufferUsages::COPY_DST)),
            ImportedBuffer {
                buffer: dispatch_buffer,
                entry_state: BufferState::initial(),
            },
        );
        state
            .append_to_graph(
                &graph,
                &mut render_graph,
                &[
                    CompiledGraphGpuGraphBinding::Buffer {
                        group: 0,
                        binding: 0,
                        revision: CompiledGraphGpuRevision::from_value(0),
                        handle: attribute_handle,
                    },
                    CompiledGraphGpuGraphBinding::Buffer {
                        group: 1,
                        binding: 0,
                        revision: CompiledGraphGpuRevision::from_value(0),
                        handle: dispatch_handle,
                    },
                ],
                &[1],
            )
            .expect("Ember compute is declared in the frame graph");
        let recorded = render_graph
            .execute(&device, &queue)
            .expect("Ember frame graph executes on the physical GPU");
        assert!(device.poll_wait(), "the physical GPU remains healthy");

        let bytes = queue.read_buffer(attribute_pool, state.attribute_pool_size());
        let value = f32::from_le_bytes(bytes[..4].try_into().expect("four-byte attribute"));
        assert!(
            (value - 2.5).abs() <= 1.0e-6,
            "GPU Ember output was {value}"
        );

        recorded.release(&device);
        state.destroy(&device);
    }
}
