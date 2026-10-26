//! Unified recorder-owned execution through the frozen scene Render Fabric.
//!
//! Callers provide live Bevy handles or borrowed SceneRhi buffer/texture leases,
//! never unowned table-local IDs or semantic physical bindings. Imports use
//! the scene's `SceneRhi`. The selected active pass slice is compiled against
//! frozen `SceneFabric`, prepared by the existing graph runtime. Owned imports
//! remain live through completion (or an unconfirmed handoff). Borrowed entries
//! remain caller-owned, live through encoding, and never acquire ownership here.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy_render::{
    render_resource::{BindGroup, Buffer, ComputePipeline, Texture, TextureView},
    renderer::RenderContext,
};
use prism_render_architecture::render_fabric::{
    builtin_resource_templates, DomainPayload, FabricCompileError, FrameRenderItem,
    FrameResourceBinding, ObjectResourceAccess, ObjectResourceBinding, ObjectResourceIdentity,
    ObjectRuntimeImport, PassKey, PhysicalResource, ResourceAddress, ResourceKey, ResourceLayout,
    ResourceTemplate, RuntimeFeatureGraph, RuntimeImport, RuntimePassBinding,
    RuntimeResourceHandle,
};
use prism_render_driver::{
    BufferState, CompletionToken, Extent3d, FrameSync, RenderQueue as _, TextureState,
};
use prism_render_graph::{
    capacity::GraphLimits, ExecutionPlan, GraphAllocations, ImportedBuffer, ImportedTexture,
    RecordedGraph,
};
use thiserror::Error;
use wgpu_types::{
    BufferUsages as NativeBufferUsages, TextureDimension as NativeTextureDimension,
    TextureFormat as NativeTextureFormat, TextureUsages as NativeTextureUsages,
};

use crate::fabric_domains::gpu::{
    DomainBufferOwnerRef, DomainGpuResourceError, DomainRecordReadyResource, DomainRecordReadyView,
    DomainRuntimeResource,
};
use crate::gpu::{
    fabric::SceneFabric,
    rhi::{SceneRhi, SceneRhiTexture, SceneRhiTextureView},
};

mod graphics;
mod typed;
pub use typed::*;

/// A real external resource for a selected pass. Templates are validated
/// against the frozen catalog, including allocation-sized producer metadata.
pub enum SceneFabricResource<'a, 'lease> {
    Buffer {
        key: ResourceKey,
        frame: u64,
        version: u32,
        buffer: &'a Buffer,
        offset: u64,
    },
    /// Imports a live Bevy buffer while preserving the producer-authenticated
    /// state at the first selected pass.
    ImportedBuffer {
        key: ResourceKey,
        frame: u64,
        version: u32,
        buffer: &'a Buffer,
        offset: u64,
        entry_state: BufferState,
    },
    /// Reuses the producer's exact buffer table entry and subrange. Borrowing
    /// lasts through encoding; the caller retains ownership and completion policy.
    ExistingBuffer {
        key: ResourceKey,
        frame: u64,
        version: u32,
        owner: DomainBufferOwnerRef<'lease>,
        offset: u64,
        entry_state: BufferState,
    },
    Texture {
        key: ResourceKey,
        frame: u64,
        version: u32,
        texture: &'a Texture,
        /// Caller-certified full view for the canonical layout. The native API
        /// exposes its parent texture, but not the view's subresource descriptor.
        default_view: &'a TextureView,
        entry_state: TextureState,
        exit_state: TextureState,
        mip_levels: u32,
        array_layers: u32,
    },
    /// Reuses exact existing table entries, including history texture IDs.
    /// Both leases must belong to this SceneRhi, and the view must be a full
    /// canonical view of the texture (subresource descriptors are not exposed).
    /// Runtime borrowing retains the entries through encoding without minting
    /// replacement IDs. Frame/version remain the producer's original identity.
    ExistingTexture {
        key: ResourceKey,
        frame: u64,
        version: u32,
        texture: &'lease SceneRhiTexture,
        default_view: &'lease SceneRhiTextureView,
        entry_state: TextureState,
        exit_state: TextureState,
        mip_levels: u32,
        array_layers: u32,
    },
}

/// One object-owned resource, retaining its complete semantic identity. Layout
/// sizes may vary only when the catalog's explicit object-access policy permits
/// them; this metadata never changes the device's frozen feature selection.
pub struct SceneFabricObjectResource<'a, 'lease> {
    pub identity: ObjectResourceIdentity,
    pub resource: SceneFabricResource<'a, 'lease>,
    pub layout: ResourceLayout,
}

impl SceneFabricObjectResource<'_, '_> {
    fn address(&self) -> ResourceAddress {
        ResourceAddress::object(self.resource.key(), self.resource.frame(), self.identity)
    }
}

impl<'a, 'lease> SceneFabricObjectResource<'a, 'lease> {
    /// Borrows the authenticated domain owner without minting replacement RHI
    /// IDs or losing the object generation, history stream or semantic role.
    pub fn from_domain(
        resource: &'lease DomainRuntimeResource,
    ) -> Result<Self, DomainGpuResourceError> {
        let item = resource.declaration();
        let identity =
            ObjectResourceIdentity::new(item.object_id, item.history_key, item.payload_role());
        let binding = resource.frame_binding();
        let resource = match resource.record_ready()? {
            DomainRecordReadyResource::Buffer {
                owner,
                offset,
                entry_state,
                ..
            } => SceneFabricResource::ExistingBuffer {
                key: binding.key,
                frame: binding.frame,
                version: binding.version,
                owner,
                offset,
                entry_state,
            },
            DomainRecordReadyResource::Texture {
                texture,
                default_view,
                entry_state,
                exit_state,
                mip_levels,
                array_layers,
                ..
            } => SceneFabricResource::ExistingTexture {
                key: binding.key,
                frame: binding.frame,
                version: binding.version,
                texture,
                default_view,
                entry_state,
                exit_state,
                mip_levels,
                array_layers,
            },
        };
        Ok(Self {
            identity,
            resource,
            layout: binding.layout,
        })
    }
}

impl SceneFabricResource<'_, '_> {
    fn key(&self) -> ResourceKey {
        match self {
            Self::Buffer { key, .. }
            | Self::ImportedBuffer { key, .. }
            | Self::ExistingBuffer { key, .. }
            | Self::Texture { key, .. }
            | Self::ExistingTexture { key, .. } => *key,
        }
    }

    fn frame(&self) -> u64 {
        match self {
            Self::Buffer { frame, .. }
            | Self::ImportedBuffer { frame, .. }
            | Self::ExistingBuffer { frame, .. }
            | Self::Texture { frame, .. }
            | Self::ExistingTexture { frame, .. } => *frame,
        }
    }

    fn buffer_metadata(&self) -> Option<(&Buffer, u64)> {
        match self {
            Self::Buffer { buffer, offset, .. } | Self::ImportedBuffer { buffer, offset, .. } => {
                Some((buffer, *offset))
            }
            Self::ExistingBuffer { owner, offset, .. } => Some((owner.buffer(), *offset)),
            Self::Texture { .. } | Self::ExistingTexture { .. } => None,
        }
    }

    fn texture_metadata(&self) -> Option<(&Texture, &TextureView, u32, u32)> {
        match self {
            Self::Texture {
                texture,
                default_view,
                mip_levels,
                array_layers,
                ..
            } => Some((texture, default_view, *mip_levels, *array_layers)),
            Self::ExistingTexture {
                texture,
                default_view,
                mip_levels,
                array_layers,
                ..
            } => Some((
                texture.texture(),
                default_view.view(),
                *mip_levels,
                *array_layers,
            )),
            Self::Buffer { .. } | Self::ImportedBuffer { .. } | Self::ExistingBuffer { .. } => None,
        }
    }
}

fn validate_resource_origins(
    rhi: &SceneRhi,
    resources: &[SceneFabricResource<'_, '_>],
) -> Result<(), SceneFabricRuntimeError> {
    for resource in resources {
        if let SceneFabricResource::ExistingBuffer { key, owner, .. } = resource {
            if !owner.belongs_to(rhi) {
                return Err(SceneFabricRuntimeError::BufferOrigin { key: *key });
            }
        }
        if let SceneFabricResource::ExistingTexture {
            key,
            texture,
            default_view,
            ..
        } = resource
        {
            if !Arc::ptr_eq(texture.device(), rhi.device())
                || !Arc::ptr_eq(default_view.device(), rhi.device())
            {
                return Err(SceneFabricRuntimeError::TextureOrigin { key: *key });
            }
        }
    }
    Ok(())
}

fn validate_resource_frames(
    frame: u64,
    resources: &[SceneFabricResource<'_, '_>],
) -> Result<(), SceneFabricRuntimeError> {
    validate_frame_keys(
        frame,
        resources
            .iter()
            .map(|resource| (resource.key(), resource.frame())),
    )
}

fn validate_frame_keys(
    frame: u64,
    resources: impl IntoIterator<Item = (ResourceKey, u64)>,
) -> Result<(), SceneFabricRuntimeError> {
    let mut keys = BTreeSet::new();
    for (key, resource_frame) in resources {
        if !keys.insert((key, resource_frame)) {
            return Err(SceneFabricRuntimeError::DuplicateResource(key));
        }
        if resource_frame > frame {
            return Err(SceneFabricRuntimeError::FutureFrame {
                key,
                request: frame,
                resource: resource_frame,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn canonical_template(key: ResourceKey) -> Option<ResourceTemplate> {
    builtin_resource_templates()
        .iter()
        .find(|template| template.key == key)
        .copied()
}

/// One real native compute dispatch retained by the prepared Fabric runtime.
/// The caller certifies same-device provenance and that the bind group matches
/// the pipeline and declared resource accesses. Native handles do not expose
/// enough reflection to authenticate those contracts here.
pub struct SceneFabricDispatch<'a> {
    pub pipeline: &'a ComputePipeline,
    pub bind_group: &'a BindGroup,
    pub bind_group_index: u32,
    pub workgroups: [u32; 3],
    /// Written after pipeline/group binding and before this dispatch. The caller
    /// must cover the pipeline's required immediate range; native validation
    /// remains authoritative for layout compatibility/initialization coverage.
    pub push_constants: Option<SceneFabricPushConstants>,
}

/// One owned compute push-constant update, moved into the existing RHI command.
pub struct SceneFabricPushConstants {
    pub offset: u32,
    pub data: Vec<u8>,
}

/// All inputs for one recorder-owned active-pass slice, compiled and prepared
/// together. Every active pass has exactly one typed domain-batch recorder.
pub struct SceneFabricRequest<'a, 'lease> {
    pub frame: u64,
    pub extent: Extent3d,
    pub items: &'a [FrameRenderItem],
    pub active_passes: &'a [PassKey],
    pub fulfilled_dependencies: &'a [PassKey],
    pub resources: Vec<SceneFabricResource<'a, 'lease>>,
    pub recorders: Vec<SceneFabricPass<'a>>,
}

// Metadata-only compute cases exercise the production typed coverage gate
// without constructing native handles.
#[cfg(test)]
fn validate_compute_passes<C, D>(
    active_passes: &[PassKey],
    recorders: C,
) -> Result<(), SceneFabricRuntimeError>
where
    C: IntoIterator<Item = (PassKey, D)>,
    D: IntoIterator<Item = [u32; 3]>,
{
    typed::validate_metadata(
        active_passes,
        recorders
            .into_iter()
            .map(|(pass, dispatches)| (pass, typed::CommandCoverage::Compute(dispatches))),
    )
}

fn validate_compute_limits<C, D>(
    limits: &wgpu_types::Limits,
    recorders: C,
) -> Result<(), SceneFabricRuntimeError>
where
    C: IntoIterator<Item = (PassKey, D)>,
    D: IntoIterator<Item = (u32, [u32; 3])>,
{
    for (pass, commands) in recorders {
        for (dispatch, (group_index, workgroups)) in commands.into_iter().enumerate() {
            if group_index >= limits.max_bind_groups {
                return Err(SceneFabricRuntimeError::BindGroupIndex {
                    pass,
                    dispatch,
                    index: group_index,
                    limit: limits.max_bind_groups,
                });
            }
            if workgroups
                .iter()
                .any(|&count| count > limits.max_compute_workgroups_per_dimension)
            {
                return Err(SceneFabricRuntimeError::WorkgroupsLimit {
                    pass,
                    dispatch,
                    limit: limits.max_compute_workgroups_per_dimension,
                });
            }
        }
    }
    Ok(())
}

// Metadata-only preflight uses enabled device features/limits, not adapter
// capabilities. Lengths are checked without copying/allocating payloads.
fn validate_push_constants<C, D>(
    features: wgpu_types::Features,
    limit: u32,
    recorders: C,
) -> Result<(), SceneFabricRuntimeError>
where
    C: IntoIterator<Item = (PassKey, D)>,
    D: IntoIterator<Item = Option<(u32, usize)>>,
{
    for (pass, commands) in recorders {
        for (dispatch, constants) in commands.into_iter().enumerate() {
            let Some((offset, bytes)) = constants else {
                continue;
            };
            let alignment = wgpu_types::IMMEDIATE_DATA_ALIGNMENT;
            if !offset.is_multiple_of(alignment) || !bytes.is_multiple_of(alignment as usize) {
                return Err(SceneFabricRuntimeError::PushConstantAlignment {
                    pass,
                    dispatch,
                    offset,
                    bytes,
                });
            }
            let end = u32::try_from(bytes)
                .ok()
                .and_then(|size| offset.checked_add(size))
                .ok_or(SceneFabricRuntimeError::PushConstantRange {
                    pass,
                    dispatch,
                    offset,
                    bytes,
                })?;
            if !features.contains(wgpu_types::Features::IMMEDIATES) {
                return Err(SceneFabricRuntimeError::PushConstantsUnsupported { pass, dispatch });
            }
            if end > limit {
                return Err(SceneFabricRuntimeError::PushConstantLimit {
                    pass,
                    dispatch,
                    end,
                    limit,
                });
            }
        }
    }
    Ok(())
}

fn validate_templates(templates: &[ResourceTemplate]) -> Result<(), SceneFabricRuntimeError> {
    for template in templates {
        if template.layout.size().is_none() || !template.alignment.is_power_of_two() {
            return Err(FabricCompileError::InvalidLayout(template.key).into());
        }
    }
    Ok(())
}

// This gate deliberately has no RHI argument: every resource must pass before
// the first table entry is registered. Catalog/access validation still belongs
// to compile_active; its later errors unwind the import leases normally.
#[cfg(test)]
fn validate_resources(
    templates: &[ResourceTemplate],
    resources: &[SceneFabricResource<'_, '_>],
) -> Result<(), SceneFabricRuntimeError> {
    validate_templates(templates)?;
    let index: BTreeMap<_, _> = templates.iter().map(|entry| (entry.key, *entry)).collect();
    let instances = resources
        .iter()
        .map(|resource| {
            let key = resource.key();
            let template = index
                .get(&key)
                .copied()
                .ok_or(SceneFabricRuntimeError::MissingTemplate(key))?;
            Ok((resource, template))
        })
        .collect::<Result<Vec<_>, SceneFabricRuntimeError>>()?;
    validate_resource_instances(&instances)
}

// Validate the combined shared/object set, including native allocation aliases
// across distinct semantic keys and per-object suballocation sizes.
fn validate_resource_instances(
    instances: &[(&SceneFabricResource<'_, '_>, ResourceTemplate)],
) -> Result<(), SceneFabricRuntimeError> {
    let mut buffer_ranges = Vec::with_capacity(instances.len());
    let mut textures = BTreeMap::new();
    for &(resource, template) in instances {
        let key = resource.key();
        validate_templates(std::slice::from_ref(&template))?;
        match resource {
            SceneFabricResource::Buffer { .. }
            | SceneFabricResource::ImportedBuffer { .. }
            | SceneFabricResource::ExistingBuffer { .. } => {
                let (buffer, offset) = resource.buffer_metadata().expect("buffer resource");
                let ResourceLayout::Buffer { bytes, usage } = template.layout else {
                    return Err(SceneFabricRuntimeError::ResourceKind { key });
                };
                let end = offset
                    .checked_add(bytes)
                    .filter(|end| *end <= buffer.size())
                    .ok_or(SceneFabricRuntimeError::BufferRange {
                        key,
                        offset,
                        required: bytes,
                    })?;
                if !buffer.usage().contains(buffer_usage(usage)) {
                    return Err(SceneFabricRuntimeError::BufferRange {
                        key,
                        offset,
                        required: bytes,
                    });
                }
                if !offset.is_multiple_of(template.alignment) {
                    return Err(FabricCompileError::InvalidBinding {
                        resource: key,
                        frame: resource.frame(),
                    }
                    .into());
                }
                // Native handles, not Bevy wrapper IDs, identify allocations.
                buffer_ranges.push((buffer.deref(), offset, end, resource));
            }
            SceneFabricResource::Texture { .. } | SceneFabricResource::ExistingTexture { .. } => {
                let (texture, default_view, mip_levels, array_layers) =
                    resource.texture_metadata().expect("texture resource");
                let ResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels: required_mips,
                    samples,
                    usage,
                } = template.layout
                else {
                    return Err(SceneFabricRuntimeError::ResourceKind { key });
                };
                let layers = if dimension
                    == prism_render_architecture::render_fabric::TextureDimension::D3
                {
                    1
                } else {
                    extent[2]
                };
                if texture.width() != extent[0]
                    || texture.height() != extent[1]
                    || texture.depth_or_array_layers() != extent[2]
                    || Some(texture.format()) != texture_format(format)
                    || texture.mip_level_count() != required_mips
                    || texture.sample_count() != samples
                    || !texture.usage().contains(texture_usage(usage))
                    || texture.dimension() != texture_dimension(dimension)
                    || mip_levels != required_mips
                    || array_layers != layers
                    || default_view.deref().texture() != texture.deref()
                {
                    return Err(SceneFabricRuntimeError::TextureMetadata { key });
                }
                if let Some(previous) = textures.insert(texture.deref(), resource) {
                    return Err(overlapping_resources(previous, resource));
                }
            }
        }
    }
    // Sorting once avoids a quadratic alias scan when a domain contributes
    // thousands of object suballocations. Adjacent ranges suffice: any overlap
    // in an allocation has an overlapping adjacent pair in start-offset order.
    buffer_ranges.sort_unstable_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));
    for pair in buffer_ranges.windows(2) {
        let (first, start, end, resource) = pair[0];
        let (second, next_start, _, next_resource) = pair[1];
        if first == second && start < end && next_start < end {
            return Err(overlapping_resources(resource, next_resource));
        }
    }
    Ok(())
}

fn overlapping_resources(
    first: &SceneFabricResource<'_, '_>,
    second: &SceneFabricResource<'_, '_>,
) -> SceneFabricRuntimeError {
    FabricCompileError::OverlappingBindings {
        first: first.key(),
        first_frame: first.frame(),
        second: second.key(),
        second_frame: second.frame(),
    }
    .into()
}

// Validate the logical device's enabled binding limits, not adapter maxima or
// the whole allocation size. Run before any import mutates the resource table.
#[cfg(test)]
fn validate_buffer_limits(
    limits: &wgpu_types::Limits,
    templates: &[ResourceTemplate],
    resources: &[SceneFabricResource<'_, '_>],
) -> Result<(), SceneFabricRuntimeError> {
    for resource in resources {
        if let Some((_, offset)) = resource.buffer_metadata() {
            let key = resource.key();
            let template = templates
                .iter()
                .find(|entry| entry.key == key)
                .ok_or(SceneFabricRuntimeError::MissingTemplate(key))?;
            validate_buffer_binding_limits(limits, key, template.layout, offset)?;
        }
    }
    Ok(())
}

fn validate_buffer_binding_limits(
    limits: &wgpu_types::Limits,
    key: ResourceKey,
    layout: ResourceLayout,
    offset: u64,
) -> Result<(), SceneFabricRuntimeError> {
    use prism_render_architecture::render_fabric::BufferUsages;
    let ResourceLayout::Buffer { bytes, usage } = layout else {
        return Err(SceneFabricRuntimeError::ResourceKind { key });
    };
    for (flag, name, alignment, limit) in [
        (
            BufferUsages::STORAGE,
            "storage",
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_binding_size,
        ),
        (
            BufferUsages::UNIFORM,
            "uniform",
            limits.min_uniform_buffer_offset_alignment,
            limits.max_uniform_buffer_binding_size,
        ),
    ] {
        if !usage.contains(flag) {
            continue;
        }
        if alignment == 0 || !offset.is_multiple_of(u64::from(alignment)) {
            return Err(SceneFabricRuntimeError::BufferBindingAlignment {
                key,
                offset,
                usage: name,
                alignment,
            });
        }
        if bytes > limit {
            return Err(SceneFabricRuntimeError::BufferBindingSize {
                key,
                bytes,
                usage: name,
                limit,
            });
        }
    }
    Ok(())
}

enum PendingImport {
    Buffer {
        buffer: prism_render_driver::BufferId,
        entry_state: BufferState,
    },
    Texture(ImportedTexture),
}

#[derive(Debug, Error)]
pub enum SceneFabricRuntimeError {
    #[error("typed command in pass {pass:?} at command {command} is invalid: {reason}")]
    Command {
        pass: PassKey,
        command: usize,
        reason: &'static str,
    },
    #[error("pass {pass:?} command {command} has no compatible declared access {access:?}")]
    CommandAccess {
        pass: PassKey,
        command: usize,
        access: SceneFabricAccess,
    },
    #[error("domain view frame {actual} does not match request frame {expected}")]
    DomainFrameMismatch { expected: u64, actual: u64 },
    #[error("authenticated domain declaration {identity:?} does not match this frame")]
    DomainDeclarationMismatch { identity: ObjectResourceIdentity },
    #[error("invalid Scene Fabric payload upload: {0}")]
    PayloadUpload(&'static str),
    #[error("resource {key:?} is from future frame {resource}, request frame is {request}")]
    FutureFrame {
        key: ResourceKey,
        request: u64,
        resource: u64,
    },
    #[error("duplicate scene Render Fabric resource {0:?}")]
    DuplicateResource(ResourceKey),
    #[error("scene Render Fabric active pass slice is empty")]
    EmptyActivePasses,
    #[error("scene Render Fabric pass {0:?} has no dispatch commands")]
    EmptyDispatches(PassKey),
    #[error(
        "scene Render Fabric pass {pass:?} dispatch {dispatch} has a zero workgroup dimension"
    )]
    ZeroWorkgroups { pass: PassKey, dispatch: usize },
    #[error(
        "scene Render Fabric pass {pass:?} dispatch {dispatch} exceeds workgroup limit {limit}"
    )]
    WorkgroupsLimit {
        pass: PassKey,
        dispatch: usize,
        limit: u32,
    },
    #[error("scene Render Fabric pass {pass:?} dispatch {dispatch} bind-group index {index} exceeds slot count {limit}")]
    BindGroupIndex {
        pass: PassKey,
        dispatch: usize,
        index: u32,
        limit: u32,
    },
    #[error("scene Render Fabric pass {pass:?} dispatch {dispatch} push constants require enabled IMMEDIATES")]
    PushConstantsUnsupported { pass: PassKey, dispatch: usize },
    #[error("scene Render Fabric pass {pass:?} dispatch {dispatch} push constants offset {offset} or length {bytes} is not 4-byte aligned")]
    PushConstantAlignment {
        pass: PassKey,
        dispatch: usize,
        offset: u32,
        bytes: usize,
    },
    #[error("scene Render Fabric pass {pass:?} dispatch {dispatch} push constants overflow at offset {offset} with length {bytes}")]
    PushConstantRange {
        pass: PassKey,
        dispatch: usize,
        offset: u32,
        bytes: usize,
    },
    #[error("scene Render Fabric pass {pass:?} dispatch {dispatch} push constants end {end} exceeds enabled device limit {limit}")]
    PushConstantLimit {
        pass: PassKey,
        dispatch: usize,
        end: u32,
        limit: u32,
    },
    #[error("scene Render Fabric frame extent has a zero dimension")]
    EmptyExtent,
    #[error("scene Render Fabric pass {0:?} cannot execute typed compute commands on its queue")]
    NonComputePass(PassKey),
    #[error("no canonical built-in resource template exists for {0:?}")]
    MissingTemplate(ResourceKey),
    #[error("resource {key:?} kind does not match its canonical template")]
    ResourceKind { key: ResourceKey },
    #[error(
        "buffer {key:?} does not cover its canonical {required}-byte range at offset {offset}"
    )]
    BufferRange {
        key: ResourceKey,
        offset: u64,
        required: u64,
    },
    #[error("buffer {key:?} belongs to a different SceneRhi table")]
    BufferOrigin { key: ResourceKey },
    #[error(
        "buffer {key:?} offset {offset} does not satisfy enabled {usage} alignment {alignment}"
    )]
    BufferBindingAlignment {
        key: ResourceKey,
        offset: u64,
        usage: &'static str,
        alignment: u32,
    },
    #[error("buffer {key:?} binding size {bytes} exceeds enabled {usage} limit {limit}")]
    BufferBindingSize {
        key: ResourceKey,
        bytes: u64,
        usage: &'static str,
        limit: u64,
    },
    #[error("texture {key:?} metadata does not match its canonical template")]
    TextureMetadata { key: ResourceKey },
    #[error("texture {key:?} or its view belongs to a different SceneRhi table")]
    TextureOrigin { key: ResourceKey },
    #[error(transparent)]
    DomainGpu(#[from] DomainGpuResourceError),
    #[error(transparent)]
    Compile(#[from] FabricCompileError),
    #[error(transparent)]
    Runtime(#[from] prism_render_architecture::render_fabric::FabricRuntimeError),
}

/// Prepared real SceneFabric runtime. The private leases cannot be dropped
/// before completion is confirmed or ownership transfers to the host callback.
pub struct SceneFabricRuntime<'a> {
    graph: RuntimeFeatureGraph,
    device: Arc<prism_render_driver_wgpu::WgpuDevice>,
    _buffer_leases: Vec<crate::gpu::rhi::SceneRhiImportedBuffer>,
    _texture_leases: Vec<crate::gpu::rhi::SceneRhiTexture>,
    _view_leases: Vec<crate::gpu::rhi::SceneRhiTextureView>,
    _borrowed_buffer_leases: Vec<DomainBufferOwnerRef<'a>>,
    _borrowed_texture_leases: Vec<(&'a SceneRhiTexture, &'a SceneRhiTextureView)>,
    _pipeline_leases: Vec<crate::gpu::rhi::SceneRhiImportedComputePipeline>,
    _render_pipeline_leases: Vec<crate::gpu::rhi::SceneRhiImportedRenderPipeline>,
    _bind_group_leases: Vec<crate::gpu::rhi::SceneRhiImportedBindGroup>,
    frame: u64,
}

/// Cancels only commands that have never been submitted or handed to a host.
/// Native encoding unwinds before this guard, so its partial encoder is gone
/// before the graph's physical allocations are released.
struct SceneFabricUnsubmittedGraph<'a> {
    recorded: Option<RecordedGraph>,
    device: &'a prism_render_driver_wgpu::WgpuDevice,
}

impl<'a> SceneFabricUnsubmittedGraph<'a> {
    fn new(recorded: RecordedGraph, device: &'a prism_render_driver_wgpu::WgpuDevice) -> Self {
        Self {
            recorded: Some(recorded),
            device,
        }
    }

    fn recorded(&self) -> &RecordedGraph {
        self.recorded.as_ref().expect("unsubmitted graph")
    }

    fn into_recorded(mut self) -> RecordedGraph {
        self.recorded.take().expect("unsubmitted graph")
    }
}

impl Drop for SceneFabricUnsubmittedGraph<'_> {
    fn drop(&mut self) {
        if let Some(recorded) = self.recorded.take() {
            recorded.release(self.device);
        }
    }
}

/// Installed immediately before entering native submission. If that call
/// unwinds, no completion token proves whether the queue accepted the work.
/// Preserve the entire ownership bundle, including its device and import
/// leases, instead of applying the unsubmitted-frame cancellation policy.
struct SceneFabricSubmissionHandoff<T> {
    resources: Option<T>,
}

impl<T> SceneFabricSubmissionHandoff<T> {
    fn complete(mut self) {
        drop(self.resources.take());
    }
}

impl<T> Drop for SceneFabricSubmissionHandoff<T> {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
    }
}

impl<'a> SceneFabricRuntime<'a> {
    /// Frame whose graph and imported resources were validated and prepared.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Existing graph submission plan selected from the frozen scene catalog.
    pub fn plan(&self) -> &prism_render_graph::ExecutionPlan {
        self.graph.plan()
    }

    /// Uploads the complete contents of an imported buffer payload validated
    /// for this active slice. The compiled semantic binding, not a caller ID,
    /// selects the exact buffer and byte offset. This enqueues a queue write;
    /// subsequent submission on this same bridge orders it before the passes.
    ///
    /// Existing imports may instead be populated by their external producer.
    /// This does not infer a byte representation for inline value payloads or
    /// populate textures. Callers certify native handles belong to this device.
    pub fn upload_payload(
        &mut self,
        rhi: &SceneRhi,
        item: &FrameRenderItem,
        data: &[u8],
    ) -> Result<(), SceneFabricRuntimeError> {
        use SceneFabricRuntimeError::PayloadUpload;
        if !Arc::ptr_eq(&self.device, rhi.device()) {
            return Err(PayloadUpload("foreign SceneRhi table"));
        }
        let declaration = self.graph.declaration();
        if !declaration.contains_item(item) {
            return Err(PayloadUpload("item was not validated for this slice"));
        }
        let DomainPayload::Resource(payload) = item.payload else {
            return Err(PayloadUpload("item is not a resource payload"));
        };
        if data.is_empty() || data.len() as u64 != payload.size || data.len() % 4 != 0 {
            return Err(PayloadUpload(
                "upload must cover the complete aligned payload",
            ));
        }
        let identity = ObjectResourceIdentity::new(item.object_id, item.history_key, payload.role);
        let object = declaration.object_resource_id(payload.resource, self.frame, identity);
        let shared = declaration.resource_id(payload.resource, self.frame);
        let id = match (object, shared) {
            (Some(id), None) | (None, Some(id)) => id,
            (Some(_), Some(_)) => return Err(PayloadUpload("ambiguous resource scope")),
            (None, None) => return Err(PayloadUpload("payload has no current external binding")),
        };
        let binding = declaration.resources()[id.0 as usize]
            .external
            .ok_or(PayloadUpload("payload has no current external binding"))?;
        let PhysicalResource::Buffer { id, offset } = binding.physical else {
            return Err(PayloadUpload("payload is not an imported buffer"));
        };
        let native = self.device.wgpu_buffer(id);
        if offset % 4 != 0
            || !native.usage().contains(NativeBufferUsages::COPY_DST)
            || offset
                .checked_add(payload.size)
                .is_none_or(|end| end > native.size())
        {
            return Err(PayloadUpload(
                "buffer cannot accept this aligned queue upload",
            ));
        }
        rhi.queue().write_buffer(id, offset, data);
        Ok(())
    }

    /// Records, encodes and hands the command buffer to Bevy's host context.
    /// Submission/completion remain host-owned. Owned import leases follow this
    /// exact native command buffer until its completion callback; abandoning an
    /// unconfirmed handoff preserves the leases instead of recycling table IDs.
    ///
    /// # Panics
    /// Panics before recording if `rhi` is not the original import bridge.
    pub fn record_into(self, rhi: &SceneRhi, context: &mut RenderContext) -> RecordedGraph {
        let (native, recorded) = self.encode(rhi);
        context.add_command_buffer(native);
        recorded
    }

    /// Encodes with every import lease alive, returning graph allocation
    /// ownership separately for the caller's actual completion policy.
    pub(crate) fn encode(
        self,
        rhi: &SceneRhi,
    ) -> (bevy_render::render_resource::CommandBuffer, RecordedGraph) {
        self.encode_with(rhi, |recorded| rhi.queue().encode(recorded.commands()))
    }

    fn encode_with(
        self,
        rhi: &SceneRhi,
        encode: impl FnOnce(&RecordedGraph) -> bevy_render::render_resource::CommandBuffer,
    ) -> (bevy_render::render_resource::CommandBuffer, RecordedGraph) {
        self.assert_origin(rhi);
        let recorded = SceneFabricUnsubmittedGraph::new(
            self.graph.record(rhi.device().as_ref()),
            rhi.device().as_ref(),
        );
        let native = encode(recorded.recorded());
        // This callback is attached to this command buffer, not to earlier
        // queue work. Dropping a callback without invoking it cannot prove
        // completion (including device loss), so the handoff guard preserves
        // the table and every owned import on that path.
        let handoff = SceneFabricSubmissionHandoff {
            resources: Some((
                self.device,
                self._buffer_leases,
                self._texture_leases,
                self._view_leases,
                self._pipeline_leases,
                self._render_pipeline_leases,
                self._bind_group_leases,
            )),
        };
        native.on_submitted_work_done(move || handoff.complete());
        (native, recorded.into_recorded())
    }

    /// Records and submits through the same adopted SceneRhi queue, retaining
    /// runtime-owned resources until the queue confirms completion. Existing
    /// buffer/texture/view entries are only borrowed through encoding, not transferred
    /// to the submission. Their native handles are retained by queued work;
    /// callers still own history publication and reuse/completion policy.
    ///
    /// The returned owner is the Scene Fabric completion boundary. Call
    /// [`SceneFabricSubmission::wait_for_completion`] before reading back,
    /// publishing history, or dropping the owner. If the backend cannot
    /// confirm completion, the owner is returned unchanged so the caller can
    /// retry or apply its device-loss policy without releasing GPU-owned data.
    ///
    /// # Panics
    /// Panics before recording if `rhi` is not the original import bridge.
    /// Encoding panics cancel unsubmitted allocations. Once native submission
    /// starts, a panic instead preserves all resources because acceptance and
    /// completion are unknown without the returned token.
    pub fn submit_tracked(self, rhi: &SceneRhi) -> SceneFabricSubmission {
        self.submit_tracked_with(
            rhi,
            |recorded| rhi.queue().encode(recorded.commands()),
            |native| -> CompletionToken { Box::new(rhi.queue().wgpu_queue().submit([native])) },
        )
    }

    fn submit_tracked_with(
        self,
        rhi: &SceneRhi,
        encode: impl FnOnce(&RecordedGraph) -> bevy_render::render_resource::CommandBuffer,
        submit: impl FnOnce(bevy_render::render_resource::CommandBuffer) -> CompletionToken,
    ) -> SceneFabricSubmission {
        self.assert_origin(rhi);
        let SceneFabricRuntime {
            graph,
            device,
            _buffer_leases,
            _texture_leases,
            _view_leases,
            _borrowed_buffer_leases,
            _borrowed_texture_leases,
            _pipeline_leases,
            _render_pipeline_leases,
            _bind_group_leases,
            frame,
        } = self;
        let recorded = SceneFabricUnsubmittedGraph::new(
            graph.record(rhi.device().as_ref()),
            rhi.device().as_ref(),
        );
        let native = encode(recorded.recorded());
        let (commands, plan, allocations) = recorded.into_recorded().into_parts();
        // Encoding succeeded without submitting. Transfer every owned resource
        // before entering a call that might accept work and then panic.
        let mut handoff = SceneFabricSubmissionHandoff {
            resources: Some(SceneFabricSubmissionResources {
                _commands: commands,
                allocations,
                device,
                _buffer_leases,
                _texture_leases,
                _view_leases,
                _pipeline_leases,
                _render_pipeline_leases,
                _bind_group_leases,
            }),
        };
        let completion = submit(native);
        // Caller-owned entries need only survive native encoding. Native
        // command buffers retain their handles; this never retires those IDs.
        drop(_borrowed_texture_leases);
        drop(_borrowed_buffer_leases);
        SceneFabricSubmission {
            completion,
            plan,
            queue: rhi.queue().clone(),
            resources: handoff.resources.take(),
            frame,
        }
    }

    fn assert_origin(&self, rhi: &SceneRhi) {
        // Even two bridges adopted from the same native device have distinct
        // tables and may mint colliding IDs. Reject before any graph lookup.
        assert!(
            Arc::ptr_eq(&self.device, rhi.device()),
            "SceneFabric runtime must use its original SceneRhi resource table"
        );
    }
}

/// The move-only owner of one submitted Scene Fabric frame.
///
/// This owner deliberately keeps the command packet, graph allocations and
/// SceneRhi import leases together. Submission/encoding is not completion:
/// callers must retain this value until [`Self::wait_for_completion`] returns
/// `Ok(())`. A failed wait returns the same owner so no GPU-owned allocation or
/// imported table entry is retired before completion is confirmed.
/// Caller-owned `ExistingBuffer` / `ExistingTexture` entries are not transferred
/// to this owner.
/// Dropping a pending owner blocks on its original submission as a safety net.
/// If that wait cannot confirm completion, resources are deliberately leaked
/// (including their device/table owner), never prematurely retired. Explicit
/// waiting is preferred so the host can handle device loss and retry itself.
#[must_use = "retain until queue completion, then call wait_for_completion"]
pub struct SceneFabricSubmission {
    completion: CompletionToken,
    plan: ExecutionPlan,
    queue: prism_render_driver_wgpu::WgpuQueue,
    resources: Option<SceneFabricSubmissionResources>,
    frame: u64,
}

struct SceneFabricSubmissionResources {
    _commands: prism_render_driver::CommandBuffer,
    allocations: GraphAllocations,
    device: Arc<prism_render_driver_wgpu::WgpuDevice>,
    _buffer_leases: Vec<crate::gpu::rhi::SceneRhiImportedBuffer>,
    _texture_leases: Vec<crate::gpu::rhi::SceneRhiTexture>,
    _view_leases: Vec<crate::gpu::rhi::SceneRhiTextureView>,
    _pipeline_leases: Vec<crate::gpu::rhi::SceneRhiImportedComputePipeline>,
    _render_pipeline_leases: Vec<crate::gpu::rhi::SceneRhiImportedRenderPipeline>,
    _bind_group_leases: Vec<crate::gpu::rhi::SceneRhiImportedBindGroup>,
}

impl std::fmt::Debug for SceneFabricSubmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneFabricSubmission")
            .field("frame", &self.frame)
            .field("pending", &self.resources.is_some())
            .finish_non_exhaustive()
    }
}

impl SceneFabricSubmission {
    /// Frame whose graph was submitted.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Exact execution plan used for this submission.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Graph-created allocations retained until completion.
    pub fn allocations(&self) -> &GraphAllocations {
        &self
            .resources
            .as_ref()
            .expect("pending submission")
            .allocations
    }

    /// Waits for this submission and retires all Scene Fabric ownership.
    ///
    /// On success, graph allocations are released on the creating device
    /// before imported SceneRhi leases are dropped. On failure, the original
    /// owner is returned intact and remains responsible for the pending GPU
    /// work.
    /// A foreign bridge is also rejected without consuming any ownership.
    pub fn wait_for_completion(mut self, rhi: &SceneRhi) -> Result<(), Self> {
        let resources = self.resources.as_ref().expect("pending submission");
        if !Arc::ptr_eq(&resources.device, rhi.device())
            || !FrameSync::wait_for(&self.queue, self.completion.as_ref())
        {
            return Err(self);
        }
        self.release_completed();
        Ok(())
    }

    fn release_completed(&mut self) {
        let resources = self.resources.take().expect("pending submission");
        resources.allocations.release(resources.device.as_ref());
    }
}

impl Drop for SceneFabricSubmission {
    fn drop(&mut self) {
        if self.resources.is_none() {
            return;
        }
        if FrameSync::wait_for(&self.queue, self.completion.as_ref()) {
            self.release_completed();
        } else if let Some(resources) = self.resources.take() {
            // There is no confirmed retirement boundary, including device-loss
            // failure. Preserve allocations and import leases rather than
            // running their destructors while the GPU may still own the work.
            std::mem::forget(resources);
        }
    }
}

/// Compile, import and prepare selected passes on the frozen SceneFabric.
/// Physical IDs are minted only by `SceneRhi` or reused from borrowed leases;
/// mismatched or absent bindings are rejected by the authoritative
/// compile/runtime validators.
pub(crate) fn prepare<'a>(
    scene_fabric: &SceneFabric,
    rhi: &SceneRhi,
    request: SceneFabricRequest<'_, 'a>,
) -> Result<SceneFabricRuntime<'a>, SceneFabricRuntimeError> {
    prepare_with_templates(scene_fabric, rhi, request, &[])
}

fn scene_templates(
    scene_fabric: &SceneFabric,
    producer_templates: &[ResourceTemplate],
) -> Result<Vec<ResourceTemplate>, SceneFabricRuntimeError> {
    let mut templates = builtin_resource_templates().to_vec();
    if scene_fabric
        .0
        .selection()
        .feature(crate::shading::taa::fabric::TAA_FEATURE_ID)
        .is_some()
    {
        // Catalog validation covers every declaration, including inactive TAA
        // alternatives. Supply their layouts without inventing frame imports;
        // allocation-sized producer metadata is merged below.
        merge_templates(
            &mut templates,
            &crate::shading::taa::fabric::taa_templates(bevy_math::UVec2::ONE),
        )?;
    }
    if scene_fabric
        .0
        .selection()
        .feature(crate::shading::composition::COMPOSITION_FEATURE_ID)
        .is_some()
    {
        merge_templates(
            &mut templates,
            &crate::shading::composition::default_composition_templates(),
        )?;
    }
    merge_templates(&mut templates, producer_templates)?;
    Ok(templates)
}

fn merge_templates(
    templates: &mut Vec<ResourceTemplate>,
    replacements: &[ResourceTemplate],
) -> Result<(), SceneFabricRuntimeError> {
    let mut indices: BTreeMap<_, _> = templates
        .iter()
        .enumerate()
        .map(|(index, template)| (template.key, index))
        .collect();
    let mut seen = BTreeSet::new();
    for template in replacements {
        if !seen.insert(template.key) {
            return Err(FabricCompileError::DuplicateTemplate(template.key).into());
        }
        if let Some(&index) = indices.get(&template.key) {
            templates[index] = *template;
        } else {
            indices.insert(template.key, templates.len());
            templates.push(*template);
        }
    }
    Ok(())
}

/// Producer metadata changes allocation sizes, never the frozen selection or
/// selected pass declarations. The catalog compiler remains authoritative.
pub(crate) fn prepare_with_templates<'a>(
    scene_fabric: &SceneFabric,
    rhi: &SceneRhi,
    request: SceneFabricRequest<'_, 'a>,
    producer_templates: &[ResourceTemplate],
) -> Result<SceneFabricRuntime<'a>, SceneFabricRuntimeError> {
    prepare_with_objects(
        scene_fabric,
        rhi,
        request,
        producer_templates,
        &[],
        Vec::new(),
    )
}

/// Joins authenticated domain owners to this exact semantic frame. Identity
/// membership alone is insufficient: payload metadata and motion must also
/// match. Original leases stay borrowed until native encoding is complete.
pub(crate) fn prepare_with_domains<'a>(
    scene_fabric: &SceneFabric,
    rhi: &SceneRhi,
    request: SceneFabricRequest<'_, 'a>,
    producer_templates: &[ResourceTemplate],
    object_accesses: &[ObjectResourceAccess],
    domain_views: &[&'a DomainRecordReadyView],
) -> Result<SceneFabricRuntime<'a>, SceneFabricRuntimeError> {
    let resources = collect_domain_resources(request.frame, request.items, domain_views)?;
    prepare_with_objects(
        scene_fabric,
        rhi,
        request,
        producer_templates,
        object_accesses,
        resources,
    )
}

fn collect_domain_resources<'a>(
    frame: u64,
    items: &[FrameRenderItem],
    domain_views: &[&'a DomainRecordReadyView],
) -> Result<Vec<SceneFabricObjectResource<'a, 'a>>, SceneFabricRuntimeError> {
    let mut declarations = BTreeMap::new();
    for item in items {
        if item.domain != item.object_id.domain() {
            return Err(FabricCompileError::DomainIdentityMismatch {
                object: item.object_id,
                declared: item.domain,
            }
            .into());
        }
        let identity =
            ObjectResourceIdentity::new(item.object_id, item.history_key, item.payload_role());
        if declarations.insert(identity, item).is_some() {
            return Err(FabricCompileError::DuplicatePayloadRole {
                object: item.object_id,
                domain: item.domain,
                history_key: item.history_key,
                role: item.payload_role(),
            }
            .into());
        }
    }
    let mut resources = Vec::new();
    for &view in domain_views {
        if view.frame() != frame {
            return Err(SceneFabricRuntimeError::DomainFrameMismatch {
                expected: frame,
                actual: view.frame(),
            });
        }
        for resource in view.resources() {
            let item = resource.declaration();
            let identity =
                ObjectResourceIdentity::new(item.object_id, item.history_key, item.payload_role());
            if declarations.get(&identity).copied() != Some(&item) {
                return Err(SceneFabricRuntimeError::DomainDeclarationMismatch { identity });
            }
            resources.push(SceneFabricObjectResource::from_domain(resource)?);
        }
    }
    Ok(resources)
}

/// Uses the same compiler, runtime and recorder ownership as shared resources.
/// Object identities expand resource accesses, not pass registration or feature
/// selection. All external objects must be authenticated by the frame payloads.
pub(crate) fn prepare_with_objects<'a>(
    scene_fabric: &SceneFabric,
    rhi: &SceneRhi,
    request: SceneFabricRequest<'_, 'a>,
    producer_templates: &[ResourceTemplate],
    object_accesses: &[ObjectResourceAccess],
    object_resources: Vec<SceneFabricObjectResource<'_, 'a>>,
) -> Result<SceneFabricRuntime<'a>, SceneFabricRuntimeError> {
    typed::validate_passes(request.active_passes, &request.recorders)?;

    validate_compute_limits(
        &rhi.device().wgpu_device().limits(),
        request.recorders.iter().map(|recorder| {
            (
                recorder.pass(),
                recorder
                    .dispatches()
                    .iter()
                    .map(|dispatch| (dispatch.bind_group_index, dispatch.workgroups)),
            )
        }),
    )?;
    validate_push_constants(
        rhi.device().wgpu_device().features(),
        rhi.device().wgpu_device().limits().max_immediate_size,
        request.recorders.iter().map(|recorder| {
            (
                recorder.pass(),
                recorder.dispatches().iter().map(|dispatch| {
                    dispatch
                        .push_constants
                        .as_ref()
                        .map(|constants| (constants.offset, constants.data.len()))
                }),
            )
        }),
    )?;
    if request.extent.width == 0
        || request.extent.height == 0
        || request.extent.depth_or_array_layers == 0
    {
        return Err(SceneFabricRuntimeError::EmptyExtent);
    }
    validate_resource_frames(request.frame, &request.resources)?;
    let templates = scene_templates(scene_fabric, producer_templates)?;
    validate_templates(&templates)?;
    let template_index: BTreeMap<_, _> = templates
        .iter()
        .map(|template| (template.key, *template))
        .collect();
    validate_resource_origins(rhi, &request.resources)?;
    let mut object_addresses = BTreeSet::new();
    for object in &object_resources {
        let address = object.address();
        if !object_addresses.insert(address) {
            return Err(FabricCompileError::DuplicateObjectBinding(address).into());
        }
        if address.frame > request.frame {
            return Err(SceneFabricRuntimeError::FutureFrame {
                key: address.key,
                request: request.frame,
                resource: address.frame,
            });
        }
        validate_resource_origins(rhi, std::slice::from_ref(&object.resource))?;
    }
    let mut instances = Vec::with_capacity(request.resources.len() + object_resources.len());
    for resource in &request.resources {
        let template = template_index
            .get(&resource.key())
            .copied()
            .ok_or(SceneFabricRuntimeError::MissingTemplate(resource.key()))?;
        instances.push((resource, template, None));
    }
    for object in &object_resources {
        let mut template = template_index.get(&object.resource.key()).copied().ok_or(
            SceneFabricRuntimeError::MissingTemplate(object.resource.key()),
        )?;
        template.layout = object.layout;
        instances.push((&object.resource, template, Some(object.identity)));
    }
    validate_resource_instances(
        &instances
            .iter()
            .map(|&(resource, template, _)| (resource, template))
            .collect::<Vec<_>>(),
    )?;
    let limits = rhi.device().wgpu_device().limits();
    for &(resource, template, _) in &instances {
        if let Some((_, offset)) = resource.buffer_metadata() {
            validate_buffer_binding_limits(&limits, resource.key(), template.layout, offset)?;
        }
    }
    let mut bindings = Vec::with_capacity(request.resources.len());
    let mut object_bindings = Vec::with_capacity(object_resources.len());
    let mut pending_imports = Vec::with_capacity(instances.len());
    let mut buffer_leases = Vec::new();
    let mut texture_leases = Vec::new();
    let mut view_leases = Vec::new();
    let mut borrowed_buffer_leases = Vec::new();
    let mut borrowed_texture_leases = Vec::new();

    for (resource, template, identity) in instances {
        let key = resource.key();
        let binding_start = bindings.len();

        match resource {
            SceneFabricResource::Buffer { .. } | SceneFabricResource::ImportedBuffer { .. } => {
                let (frame, version, buffer, offset, entry_state) = match resource {
                    SceneFabricResource::Buffer {
                        frame,
                        version,
                        buffer,
                        offset,
                        ..
                    } => (*frame, *version, *buffer, *offset, BufferState::initial()),
                    SceneFabricResource::ImportedBuffer {
                        frame,
                        version,
                        buffer,
                        offset,
                        entry_state,
                        ..
                    } => (*frame, *version, *buffer, *offset, *entry_state),
                    _ => unreachable!("buffer arm already matched"),
                };
                let lease = rhi.import_buffer(buffer);
                bindings.push(FrameResourceBinding {
                    key,
                    frame,
                    version,
                    physical: PhysicalResource::Buffer {
                        id: lease.id(),
                        offset,
                    },
                    layout: template.layout,
                    alignment: template.alignment,
                });
                pending_imports.push((
                    key,
                    frame,
                    PendingImport::Buffer {
                        buffer: lease.id(),
                        entry_state,
                    },
                    None,
                ));
                buffer_leases.push(lease);
            }
            SceneFabricResource::ExistingBuffer {
                frame,
                version,
                owner,
                offset,
                entry_state,
                ..
            } => {
                bindings.push(FrameResourceBinding {
                    key,
                    frame: *frame,
                    version: *version,
                    physical: PhysicalResource::Buffer {
                        id: owner.id(),
                        offset: *offset,
                    },
                    layout: template.layout,
                    alignment: template.alignment,
                });
                pending_imports.push((
                    key,
                    *frame,
                    PendingImport::Buffer {
                        buffer: owner.id(),
                        entry_state: *entry_state,
                    },
                    None,
                ));
                borrowed_buffer_leases.push(*owner);
            }
            SceneFabricResource::Texture {
                frame,
                version,
                texture,
                default_view,
                entry_state,
                exit_state,
                mip_levels,
                array_layers,
                ..
            } => {
                let lease = rhi.import_texture(texture);
                let view_lease = rhi.import_texture_view(default_view);
                bindings.push(FrameResourceBinding {
                    key,
                    frame: *frame,
                    version: *version,
                    physical: PhysicalResource::Texture(lease.id()),
                    layout: template.layout,
                    alignment: template.alignment,
                });
                pending_imports.push((
                    key,
                    *frame,
                    PendingImport::Texture(ImportedTexture {
                        texture: lease.id(),
                        default_view: view_lease.id(),
                        entry_state: *entry_state,
                        exit_state: *exit_state,
                        mip_levels: *mip_levels,
                        array_layers: *array_layers,
                    }),
                    None,
                ));
                texture_leases.push(lease);
                view_leases.push(view_lease);
            }
            SceneFabricResource::ExistingTexture {
                frame,
                version,
                texture,
                default_view,
                entry_state,
                exit_state,
                mip_levels,
                array_layers,
                ..
            } => {
                bindings.push(FrameResourceBinding {
                    key,
                    frame: *frame,
                    version: *version,
                    physical: PhysicalResource::Texture(texture.id()),
                    layout: template.layout,
                    alignment: template.alignment,
                });
                pending_imports.push((
                    key,
                    *frame,
                    PendingImport::Texture(ImportedTexture {
                        texture: texture.id(),
                        default_view: default_view.id(),
                        entry_state: *entry_state,
                        exit_state: *exit_state,
                        mip_levels: *mip_levels,
                        array_layers: *array_layers,
                    }),
                    None,
                ));
                borrowed_texture_leases.push((*texture, *default_view));
            }
        }
        let binding = bindings[binding_start];
        let pending = pending_imports
            .last_mut()
            .expect("resource emitted an import");
        // Attach the complete address before lowering table-local IDs. Shared
        // and object bindings never resolve through each other's namespace.
        pending.3 = identity;
        if let Some(identity) = identity {
            bindings.pop();
            object_bindings.push(ObjectResourceBinding { identity, binding });
        }
    }

    let compiled = scene_fabric.0.compile_active_instanced(
        &templates,
        object_accesses,
        request.frame,
        &bindings,
        &object_bindings,
        request.items,
        request.active_passes,
        request.fulfilled_dependencies,
        GraphLimits::default(),
    )?;
    graphics::validate(&compiled, &request.recorders, rhi)?;

    // Resolve IDs with the exact (semantic key, frame) identity. History
    // resources legitimately refer to a previous frame of the same key.
    let mut imports = Vec::with_capacity(pending_imports.len());
    let mut object_imports = Vec::with_capacity(object_bindings.len());
    for (key, frame, import, identity) in pending_imports {
        let address = ResourceAddress {
            key,
            frame,
            identity,
        };
        let id = compiled
            .resource_id_at(address)
            .ok_or(FabricCompileError::MissingBinding {
                resource: key,
                frame,
            })?;
        let imported = match import {
            PendingImport::Buffer {
                buffer,
                entry_state,
            } => RuntimeImport::Buffer {
                resource: id,
                imported: ImportedBuffer {
                    buffer,
                    entry_state,
                },
            },
            PendingImport::Texture(imported) => RuntimeImport::Texture {
                resource: id,
                imported,
            },
        };
        if identity.is_some() {
            let binding = compiled.resources()[id.0 as usize]
                .external
                .expect("external object import has a validated binding");
            object_imports.push(ObjectRuntimeImport {
                address,
                binding,
                imported,
            });
        } else {
            imports.push(imported);
        }
    }

    let dispatch_count = request
        .recorders
        .iter()
        .map(|recorder| recorder.dispatches().len())
        .sum();
    let mut pipeline_leases = Vec::with_capacity(dispatch_count);
    let mut render_pipeline_leases = Vec::new();
    let mut bind_group_leases = Vec::with_capacity(dispatch_count);
    let mut recorders = Vec::with_capacity(request.recorders.len());
    for recorder in request.recorders {
        match recorder {
            SceneFabricPass::Compute { pass, dispatches } => {
                let mut commands = Vec::with_capacity(dispatches.len());
                for dispatch in dispatches {
                    let pipeline = rhi.import_compute_pipeline(dispatch.pipeline);
                    let bind_group = rhi.import_bind_group(dispatch.bind_group);
                    commands.push((
                        pipeline.id(),
                        bind_group.id(),
                        dispatch.bind_group_index,
                        dispatch.workgroups,
                        dispatch.push_constants,
                    ));
                    pipeline_leases.push(pipeline);
                    bind_group_leases.push(bind_group);
                }
                recorders.push(RuntimePassBinding::new(pass, move |context| {
                    for (pipeline, group, group_index, workgroups, push_constants) in commands {
                        let commands = context
                            .commands()
                            .set_compute_pipeline(pipeline)
                            .set_compute_bind_group(group_index, group, Vec::new());
                        if let Some(constants) = push_constants {
                            commands.set_compute_push_constants(constants.offset, constants.data);
                        }
                        commands.dispatch(workgroups[0], workgroups[1], workgroups[2]);
                    }
                }));
            }
            SceneFabricPass::Graphics { pass, draws } => {
                let mut commands = Vec::with_capacity(draws.len());
                for draw in draws {
                    let pipeline = rhi.import_render_pipeline(draw.pipeline);
                    let mut groups = Vec::with_capacity(draw.bind_groups.len());
                    for group in draw.bind_groups {
                        let imported = rhi.import_bind_group(group.group);
                        groups.push((group.index, imported.id(), group.dynamic_offsets));
                        bind_group_leases.push(imported);
                    }
                    commands.push(graphics::RecordedDraw {
                        pipeline: pipeline.id(),
                        groups,
                        vertices: draw.vertex_buffers,
                        index: draw.index_buffer,
                        command: draw.command,
                    });
                    render_pipeline_leases.push(pipeline);
                }
                recorders.push(RuntimePassBinding::new(pass, move |context| {
                    graphics::record(context, commands);
                }));
            }
            SceneFabricPass::Transfer {
                pass,
                texture_copies,
            } => {
                let mut commands = Vec::with_capacity(texture_copies.len());
                for copy in texture_copies {
                    commands.push((
                        copy.source,
                        copy.source_mip_level,
                        copy.source_origin,
                        copy.destination,
                        copy.destination_mip_level,
                        copy.destination_origin,
                        copy.extent,
                    ));
                }
                recorders.push(RuntimePassBinding::new(pass, move |context| {
                    for (src_access, src_mip, src_origin, dst_access, dst_mip, dst_origin, extent) in commands {
                        let source = context
                            .access(src_access.identity, src_access.ordinal)
                            .expect("validated transfer source access is present at runtime");
                        let RuntimeResourceHandle::Texture(src) = source.handle else {
                            unreachable!("transfer source texture type was validated before runtime preparation");
                        };
                        let destination = context
                            .access(dst_access.identity, dst_access.ordinal)
                            .expect("validated transfer destination access is present at runtime");
                        let RuntimeResourceHandle::Texture(dst) = destination.handle else {
                            unreachable!("transfer destination texture type was validated before runtime preparation");
                        };
                        context.commands().copy_texture_to_texture(
                            src, src_mip, src_origin, dst, dst_mip, dst_origin, extent,
                        );
                    }
                }));
            }
        }
    }
    // Stable-key bindings may arrive in any order. The existing runtime maps
    // them to the compiled pass order; no local executor or scheduling exists.
    let graph = compiled.prepare_runtime_with_object_imports(
        request.extent,
        &imports,
        &object_imports,
        recorders,
    )?;

    Ok(SceneFabricRuntime {
        graph,
        device: Arc::clone(rhi.device()),
        _buffer_leases: buffer_leases,
        _texture_leases: texture_leases,
        _view_leases: view_leases,
        _borrowed_buffer_leases: borrowed_buffer_leases,
        _borrowed_texture_leases: borrowed_texture_leases,
        _pipeline_leases: pipeline_leases,
        _render_pipeline_leases: render_pipeline_leases,
        _bind_group_leases: bind_group_leases,
        frame: request.frame,
    })
}

fn buffer_usage(
    usage: prism_render_architecture::render_fabric::BufferUsages,
) -> NativeBufferUsages {
    let mut mapped = NativeBufferUsages::empty();
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::COPY_SRC) {
        mapped |= NativeBufferUsages::COPY_SRC;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::COPY_DST) {
        mapped |= NativeBufferUsages::COPY_DST;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::INDEX) {
        mapped |= NativeBufferUsages::INDEX;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::VERTEX) {
        mapped |= NativeBufferUsages::VERTEX;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::UNIFORM) {
        mapped |= NativeBufferUsages::UNIFORM;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::STORAGE) {
        mapped |= NativeBufferUsages::STORAGE;
    }
    if usage.contains(prism_render_architecture::render_fabric::BufferUsages::INDIRECT) {
        mapped |= NativeBufferUsages::INDIRECT;
    }
    mapped
}

fn texture_dimension(
    dimension: prism_render_architecture::render_fabric::TextureDimension,
) -> NativeTextureDimension {
    match dimension {
        prism_render_architecture::render_fabric::TextureDimension::D1 => {
            NativeTextureDimension::D1
        }
        prism_render_architecture::render_fabric::TextureDimension::D2 => {
            NativeTextureDimension::D2
        }
        prism_render_architecture::render_fabric::TextureDimension::D3 => {
            NativeTextureDimension::D3
        }
    }
}

fn texture_usage(
    usage: prism_render_architecture::render_fabric::TextureUsages,
) -> NativeTextureUsages {
    let mut mapped = NativeTextureUsages::empty();
    if usage.contains(prism_render_architecture::render_fabric::TextureUsages::COPY_SRC) {
        mapped |= NativeTextureUsages::COPY_SRC;
    }
    if usage.contains(prism_render_architecture::render_fabric::TextureUsages::COPY_DST) {
        mapped |= NativeTextureUsages::COPY_DST;
    }
    if usage.contains(prism_render_architecture::render_fabric::TextureUsages::TEXTURE_BINDING) {
        mapped |= NativeTextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(prism_render_architecture::render_fabric::TextureUsages::STORAGE_BINDING) {
        mapped |= NativeTextureUsages::STORAGE_BINDING;
    }
    if usage.contains(prism_render_architecture::render_fabric::TextureUsages::RENDER_ATTACHMENT) {
        mapped |= NativeTextureUsages::RENDER_ATTACHMENT;
    }
    mapped
}

fn texture_format(
    format: prism_render_architecture::render_fabric::TextureFormat,
) -> Option<NativeTextureFormat> {
    use prism_render_architecture::render_fabric::TextureFormat as F;
    match format {
        F::R8Unorm => Some(NativeTextureFormat::R8Unorm),
        F::R8Uint => Some(NativeTextureFormat::R8Uint),
        F::R16Float => Some(NativeTextureFormat::R16Float),
        F::R32Float => Some(NativeTextureFormat::R32Float),
        F::R32Uint => Some(NativeTextureFormat::R32Uint),
        F::Rg8Unorm => Some(NativeTextureFormat::Rg8Unorm),
        F::Rg16Float => Some(NativeTextureFormat::Rg16Float),
        F::Rg32Float => Some(NativeTextureFormat::Rg32Float),
        F::Rgba8Unorm => Some(NativeTextureFormat::Rgba8Unorm),
        F::Rgba8UnormSrgb => Some(NativeTextureFormat::Rgba8UnormSrgb),
        F::Bgra8Unorm => Some(NativeTextureFormat::Bgra8Unorm),
        F::Bgra8UnormSrgb => Some(NativeTextureFormat::Bgra8UnormSrgb),
        F::Rgb10a2Unorm => Some(NativeTextureFormat::Rgb10a2Unorm),
        F::Rg11b10Float => Some(NativeTextureFormat::Rg11b10Ufloat),
        F::Rgba16Float => Some(NativeTextureFormat::Rgba16Float),
        F::Rgba32Float => Some(NativeTextureFormat::Rgba32Float),
        F::Depth32Float => Some(NativeTextureFormat::Depth32Float),
        F::Depth24PlusStencil8 => Some(NativeTextureFormat::Depth24PlusStencil8),
        F::Rgba32Uint => Some(NativeTextureFormat::Rgba32Uint),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod domain_bridge_gpu_tests;
    mod graphics_gpu_tests;
    mod particle_bridge_gpu_tests;
    mod transfer_gpu_tests;
    use prism_render_architecture::{
        backend::capability::CapabilitySet,
        render_fabric::{
            builtin_catalog, builtin_roots, DeviceFabric, RenderQualityTier, WATER_SURFACE_RESOURCE,
        },
    };

    #[test]
    fn buffer_binding_limits_use_enabled_alignment_and_exact_binding_size() {
        use prism_render_architecture::render_fabric::BufferUsages;
        let key = WATER_SURFACE_RESOURCE;
        let layout = ResourceLayout::Buffer {
            bytes: 64,
            usage: BufferUsages::STORAGE,
        };
        for alignment in [16, 64, 256, 1024] {
            let limits = wgpu_types::Limits {
                min_storage_buffer_offset_alignment: alignment,
                max_storage_buffer_binding_size: 64,
                ..Default::default()
            };
            validate_buffer_binding_limits(&limits, key, layout, u64::from(alignment)).unwrap();
            assert!(matches!(
                validate_buffer_binding_limits(&limits, key, layout, u64::from(alignment) + 1),
                Err(SceneFabricRuntimeError::BufferBindingAlignment {
                    usage: "storage",
                    ..
                })
            ));
            assert!(matches!(
                validate_buffer_binding_limits(
                    &limits,
                    key,
                    ResourceLayout::Buffer {
                        bytes: 65,
                        usage: BufferUsages::STORAGE
                    },
                    0
                ),
                Err(SceneFabricRuntimeError::BufferBindingSize {
                    bytes: 65,
                    limit: 64,
                    ..
                })
            ));
        }
    }

    #[test]
    fn buffer_binding_limits_check_uniform_and_storage_independently() {
        use prism_render_architecture::render_fabric::BufferUsages;
        let key = WATER_SURFACE_RESOURCE;
        let limits = wgpu_types::Limits {
            min_storage_buffer_offset_alignment: 64,
            min_uniform_buffer_offset_alignment: 256,
            max_storage_buffer_binding_size: 512,
            max_uniform_buffer_binding_size: 128,
            ..Default::default()
        };
        let both = BufferUsages::STORAGE | BufferUsages::UNIFORM;
        assert!(matches!(
            validate_buffer_binding_limits(
                &limits,
                key,
                ResourceLayout::Buffer {
                    bytes: 64,
                    usage: both
                },
                64
            ),
            Err(SceneFabricRuntimeError::BufferBindingAlignment {
                usage: "uniform",
                ..
            })
        ));
        assert!(matches!(
            validate_buffer_binding_limits(
                &limits,
                key,
                ResourceLayout::Buffer {
                    bytes: 256,
                    usage: both
                },
                256
            ),
            Err(SceneFabricRuntimeError::BufferBindingSize {
                usage: "uniform",
                ..
            })
        ));
        validate_buffer_binding_limits(
            &limits,
            key,
            ResourceLayout::Buffer {
                bytes: 128,
                usage: both,
            },
            256,
        )
        .unwrap();
        validate_buffer_binding_limits(
            &limits,
            key,
            ResourceLayout::Buffer {
                bytes: 256,
                usage: BufferUsages::STORAGE,
            },
            64,
        )
        .unwrap();
    }

    #[test]
    fn buffer_binding_limits_reject_zero_alignment_only_for_bound_usages() {
        use prism_render_architecture::render_fabric::BufferUsages;
        let limits = wgpu_types::Limits {
            min_storage_buffer_offset_alignment: 0,
            min_uniform_buffer_offset_alignment: 0,
            ..Default::default()
        };
        for usage in [BufferUsages::STORAGE, BufferUsages::UNIFORM] {
            assert!(matches!(
                validate_buffer_binding_limits(
                    &limits,
                    WATER_SURFACE_RESOURCE,
                    ResourceLayout::Buffer { bytes: 16, usage },
                    0
                ),
                Err(SceneFabricRuntimeError::BufferBindingAlignment { alignment: 0, .. })
            ));
        }
        validate_buffer_binding_limits(
            &limits,
            WATER_SURFACE_RESOURCE,
            ResourceLayout::Buffer {
                bytes: 16,
                usage: BufferUsages::COPY_SRC,
            },
            4,
        )
        .unwrap();
    }

    #[test]
    fn submission_handoff_preserves_unknown_ownership_and_transfers_on_success() {
        use std::{
            cell::Cell,
            panic::{catch_unwind, AssertUnwindSafe},
        };

        struct DropProbe<'a>(&'a Cell<u32>);
        impl Drop for DropProbe<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Cell::new(0);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _handoff = SceneFabricSubmissionHandoff {
                resources: Some(DropProbe(&drops)),
            };
            panic!("submission acceptance is unknown");
        }))
        .is_err());
        assert_eq!(
            drops.get(),
            0,
            "unknown submissions must not retire resources"
        );

        let mut handoff = SceneFabricSubmissionHandoff {
            resources: Some(DropProbe(&drops)),
        };
        let resources = handoff.resources.take().unwrap();
        drop(handoff);
        assert_eq!(drops.get(), 0, "successful handoff transfers ownership");
        drop(resources);
        assert_eq!(drops.get(), 1);

        SceneFabricSubmissionHandoff {
            resources: Some(DropProbe(&drops)),
        }
        .complete();
        assert_eq!(drops.get(), 2, "confirmed callback retires exactly once");
    }

    #[test]
    fn compute_coverage_accepts_unsorted_recorders_and_multiple_dispatches() {
        let active = [PassKey(9), PassKey(2)];
        assert!(validate_compute_passes(
            &active,
            [
                (PassKey(2), vec![[1, 2, 3], [4, 5, 6]]),
                (PassKey(9), vec![[1, 1, 1]]),
            ],
        )
        .is_ok());
    }

    #[test]
    fn compute_coverage_rejects_empty_duplicate_missing_and_unexpected_passes() {
        use prism_render_architecture::render_fabric::FabricRuntimeError as E;
        let a = PassKey(2);
        let b = PassKey(9);
        let one = |pass| (pass, vec![[1, 1, 1]]);
        assert!(matches!(
            validate_compute_passes(&[], Vec::<(PassKey, Vec<[u32; 3]>)>::new()),
            Err(SceneFabricRuntimeError::EmptyActivePasses)
        ));
        assert!(matches!(
            validate_compute_passes(&[a, a], [one(a)]),
            Err(SceneFabricRuntimeError::Compile(FabricCompileError::DuplicateActivePass(key)))
                if key == a
        ));
        assert!(matches!(
            validate_compute_passes(&[a, b], [one(a), one(a)]),
            Err(SceneFabricRuntimeError::Runtime(E::DuplicateRecorder(key))) if key == a
        ));
        assert!(matches!(
            validate_compute_passes(&[a, b], [one(a)]),
            Err(SceneFabricRuntimeError::Runtime(E::MissingRecorder(key))) if key == b
        ));
        assert!(matches!(
            validate_compute_passes(&[a], [one(a), one(b)]),
            Err(SceneFabricRuntimeError::Runtime(E::UnexpectedRecorder(key))) if key == b
        ));
        assert!(matches!(
            validate_compute_passes(&[a], Vec::<(PassKey, Vec<[u32; 3]>)>::new()),
            Err(SceneFabricRuntimeError::Runtime(E::MissingRecorder(key))) if key == a
        ));
    }

    #[test]
    fn compute_commands_reject_empty_later_pass_and_every_zero_dimension() {
        let a = PassKey(2);
        let b = PassKey(9);
        assert!(matches!(
            validate_compute_passes(&[a, b], [(a, vec![[1, 1, 1]]), (b, vec![])]),
            Err(SceneFabricRuntimeError::EmptyDispatches(key)) if key == b
        ));
        for dimension in 0..3 {
            let mut workgroups = [1, 1, 1];
            workgroups[dimension] = 0;
            assert!(matches!(
                validate_compute_passes(
                    &[a, b],
                    [(a, vec![[1, 1, 1]]), (b, vec![[1, 1, 1], workgroups])],
                ),
                Err(SceneFabricRuntimeError::ZeroWorkgroups { pass, dispatch: 1 }) if pass == b
            ));
        }
    }

    #[test]
    fn compute_limits_check_every_dispatch_axis_and_bind_group_slot() {
        let limits = wgpu::Limits {
            max_bind_groups: 3,
            max_compute_workgroups_per_dimension: 8,
            ..Default::default()
        };
        let a = PassKey(2);
        let b = PassKey(9);
        assert!(validate_compute_limits(&limits, [(a, [(2, [8, 8, 8])])]).is_ok());
        for axis in 0..3 {
            let mut groups = [1, 1, 1];
            groups[axis] = 9;
            assert!(matches!(
                validate_compute_limits(
                    &limits,
                    [(a, vec![(0, [1, 1, 1])]), (b, vec![(0, [1, 1, 1]), (0, groups)])],
                ),
                Err(SceneFabricRuntimeError::WorkgroupsLimit {
                    pass, dispatch: 1, limit: 8,
                }) if pass == b
            ));
        }
        for index in [3, u32::MAX] {
            assert!(matches!(
                validate_compute_limits(&limits, [(b, [(0, [1, 1, 1]), (index, [1, 1, 1])])]),
                Err(SceneFabricRuntimeError::BindGroupIndex {
                    pass, dispatch: 1, index: bad, limit: 3,
                }) if pass == b && bad == index
            ));
        }
    }

    #[test]
    fn push_constants_allow_none_without_feature_and_exact_taa_range() {
        let pass = PassKey(9);
        assert!(
            validate_push_constants(wgpu_types::Features::empty(), 0, [(pass, [None, None])])
                .is_ok()
        );
        let enabled = wgpu_types::Features::IMMEDIATES;
        assert!(validate_push_constants(enabled, 20, [(pass, [Some((0, 20))])]).is_ok());
        assert!(validate_push_constants(enabled, 24, [(pass, [Some((4, 20))])]).is_ok());
        // Match native alignment/range rules; an empty update is not banned.
        assert!(validate_push_constants(enabled, 20, [(pass, [Some((20, 0))])]).is_ok());
    }

    #[test]
    fn push_constants_reject_unaligned_offset_or_length_with_dispatch_context() {
        let a = PassKey(2);
        let b = PassKey(9);
        for (offset, bytes) in [(1, 20), (2, 20), (3, 20), (0, 1), (0, 2), (0, 3), (4, 19)] {
            assert!(matches!(
                validate_push_constants(wgpu_types::Features::IMMEDIATES, 64,
                    [(a, vec![None]), (b, vec![Some((0, 20)), Some((offset, bytes))])]),
                Err(SceneFabricRuntimeError::PushConstantAlignment { pass, dispatch: 1, offset: bad_offset, bytes: bad_bytes })
                    if pass == b && bad_offset == offset && bad_bytes == bytes
            ));
        }
    }

    #[test]
    fn push_constants_require_enabled_device_feature_and_limit() {
        let pass = PassKey(9);
        assert!(matches!(
            validate_push_constants(
                wgpu_types::Features::empty(),
                128,
                [(pass, [None, Some((0, 20))])]
            ),
            Err(SceneFabricRuntimeError::PushConstantsUnsupported {
                pass: PassKey(9),
                dispatch: 1
            })
        ));
        for (offset, bytes, limit) in [(0, 20, 0), (0, 20, 16), (4, 20, 20), (24, 0, 20)] {
            assert!(matches!(
                validate_push_constants(wgpu_types::Features::IMMEDIATES, limit, [(pass, [None, Some((offset, bytes))])]),
                Err(SceneFabricRuntimeError::PushConstantLimit { pass: PassKey(9), dispatch: 1, end, limit: actual })
                    if end == offset + bytes as u32 && actual == limit
            ));
        }
    }

    #[test]
    fn push_constants_reject_u32_length_and_end_overflow_without_allocating() {
        let pass = PassKey(9);
        let enabled = wgpu_types::Features::IMMEDIATES;
        assert!(matches!(
            validate_push_constants(enabled, u32::MAX, [(pass, [Some((u32::MAX - 3, 4))])]),
            Err(SceneFabricRuntimeError::PushConstantRange {
                pass: PassKey(9),
                dispatch: 0,
                ..
            })
        ));
        if let Some(bytes) = (u32::MAX as usize).checked_add(1) {
            assert!(matches!(
                validate_push_constants(enabled, u32::MAX, [(pass, [Some((0, bytes))])]),
                Err(SceneFabricRuntimeError::PushConstantRange { pass: PassKey(9), dispatch: 0, bytes: actual, .. })
                    if actual == bytes
            ));
        }
    }

    #[test]
    fn resource_template_preflight_rejects_invalid_layout_and_alignment() {
        let valid = canonical_template(WATER_SURFACE_RESOURCE).unwrap();
        assert!(validate_templates(&[valid]).is_ok());
        for alignment in [0, 3, u64::MAX] {
            let invalid = ResourceTemplate { alignment, ..valid };
            assert!(matches!(
                validate_templates(&[valid, invalid]),
                Err(SceneFabricRuntimeError::Compile(FabricCompileError::InvalidLayout(key)))
                    if key == valid.key
            ));
        }
        let invalid = ResourceTemplate {
            layout: ResourceLayout::Buffer {
                bytes: 0,
                usage: prism_render_driver::BufferUsages::STORAGE,
            },
            ..valid
        };
        assert!(matches!(
            validate_resources(&[invalid], &[]),
            Err(SceneFabricRuntimeError::Compile(FabricCompileError::InvalidLayout(key)))
                if key == valid.key
        ));
    }

    #[test]
    fn resource_frames_reject_future_and_duplicate_bindings_but_allow_history() {
        assert!(validate_frame_keys(9, [(WATER_SURFACE_RESOURCE, 8)]).is_ok());
        assert!(matches!(
            validate_frame_keys(9, [(WATER_SURFACE_RESOURCE, 10)]),
            Err(SceneFabricRuntimeError::FutureFrame { .. })
        ));
        assert!(matches!(
            validate_frame_keys(
                9,
                [(WATER_SURFACE_RESOURCE, 8), (WATER_SURFACE_RESOURCE, 8)]
            ),
            Err(SceneFabricRuntimeError::DuplicateResource(
                WATER_SURFACE_RESOURCE
            ))
        ));
    }

    #[test]
    fn unknown_semantic_binding_has_no_canonical_template() {
        assert!(canonical_template(ResourceKey::engine(u16::MAX)).is_none());
    }

    #[test]
    fn shared_scene_templates_cover_composition_and_keep_unrelated_resources() {
        let scene = crate::shading::composition::scene_fabric(CapabilitySet::WEBGPU_BASELINE);
        let composition = crate::shading::composition::default_composition_templates();
        let mut resized = composition[0];
        let ResourceLayout::Buffer { ref mut bytes, .. } = resized.layout else {
            panic!("left response must be a buffer");
        };
        *bytes *= 6;
        let templates = scene_templates(&scene, &[resized]).unwrap();
        scene
            .0
            .compile_active(&templates, 0, &[], &[], &[], &[], GraphLimits::default())
            .expect("complete shared catalog, including inactive TAA declarations");
        assert_eq!(
            templates.iter().find(|entry| entry.key == resized.key),
            Some(&resized)
        );
        for template in composition.iter().skip(1) {
            assert_eq!(
                templates.iter().find(|entry| entry.key == template.key),
                Some(template)
            );
        }
        for template in builtin_resource_templates() {
            if !composition.iter().any(|entry| entry.key == template.key) {
                assert_eq!(
                    templates.iter().find(|entry| entry.key == template.key),
                    Some(template)
                );
            }
        }
        for template in crate::shading::taa::fabric::taa_templates(bevy_math::UVec2::ONE) {
            if !composition.iter().any(|entry| entry.key == template.key) {
                assert_eq!(
                    templates.iter().find(|entry| entry.key == template.key),
                    Some(&template)
                );
            }
        }
        assert!(matches!(
            scene_templates(&scene, &[resized, resized]),
            Err(SceneFabricRuntimeError::Compile(FabricCompileError::DuplicateTemplate(key)))
                if key == resized.key
        ));
    }

    #[test]
    fn frozen_water_compile_rejects_missing_real_bindings() {
        let catalog = builtin_catalog().expect("builtin catalog");
        let roots = builtin_roots(&catalog).expect("builtin roots");
        let fabric = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .expect("frozen builtin selection");
        let result = fabric.compile_active(
            builtin_resource_templates(),
            4,
            &[],
            &[],
            &[PassKey(0x5000)],
            &[],
            GraphLimits::default(),
        );
        assert!(matches!(
            result,
            Err(FabricCompileError::MissingBinding { .. })
        ));
    }

    mod multipass_gpu {
        use super::*;
        use bevy_platform::future::block_on;
        use bevy_render::renderer::{RenderAdapter, RenderDevice, RenderQueue};
        use prism_render_architecture::{
            render_fabric::ResourceLifetime,
            render_fabric::{
                AccessKind, FeatureCatalog, FeatureDefinition, FeatureId, FeatureVariant,
                FrozenRoots, PassTemplate, QueueClass, RenderDomain, ResourceInput, ResourceScope,
                ResourceUse, ServiceSet, VariantId,
            },
        };
        use prism_render_driver::{Adapter as _, BufferUsages};
        use prism_render_driver_wgpu::{
            require_hardware_gpu, AdapterEvidence, WgpuAdapter, WgpuInstance,
        };

        const FEATURE: FeatureId = FeatureId(0xff00);
        const OUTPUT: ResourceKey = ResourceKey::of(FEATURE, 0);
        // Numeric, catalog, active-list and recorder order intentionally oppose
        // the real resource/control dependency: seed must execute first.
        const SEED: PassKey = PassKey(0xff02);
        const UPDATE: PassKey = PassKey(0xff01);
        const ACTIVE: &[PassKey] = &[UPDATE, SEED];

        pub(super) fn hardware_context(
        ) -> Option<(wgpu::Device, wgpu::Queue, RenderAdapter, RenderDevice)> {
            hardware_context_with_immediates(0)
        }

        fn hardware_context_with_immediates(
            immediate_size: u32,
        ) -> Option<(wgpu::Device, wgpu::Queue, RenderAdapter, RenderDevice)> {
            let instance = WgpuInstance::new();
            let Some(adapter) =
                instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
            else {
                assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: no adapter");
                eprintln!("SKIPPED SceneFabric multipass GPU test: no adapter");
                return None;
            };
            let evidence = AdapterEvidence::from_info(WgpuAdapter::new(adapter.clone()).info());
            if !evidence.is_hardware() {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
                );
                eprintln!("SKIPPED SceneFabric multipass GPU test: {evidence}");
                return None;
            }
            eprintln!("SceneFabric multipass REAL HARDWARE GPU: {evidence}");
            let required_features = if immediate_size == 0 {
                wgpu::Features::empty()
            } else {
                wgpu::Features::IMMEDIATES
            };
            if !adapter.features().contains(required_features)
                || adapter.limits().max_immediate_size < immediate_size
            {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1: required immediates unavailable"
                );
                eprintln!("SKIPPED SceneFabric push constants: required immediates unavailable");
                return None;
            }
            let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("fabric-multipass-host"),
                required_features,
                required_limits: wgpu::Limits {
                    max_immediate_size: immediate_size,
                    ..Default::default()
                },
                ..Default::default()
            }))
            .unwrap();
            let render_device = RenderDevice::new(device.clone());
            Some((device, queue, RenderAdapter::new(adapter), render_device))
        }

        fn table<T>(values: Vec<T>) -> &'static [T] {
            Box::leak(values.into_boxed_slice())
        }

        fn scene(update_queue: QueueClass) -> (SceneFabric, ResourceTemplate) {
            scene_with_access(update_queue, AccessKind::StorageWrite)
        }

        fn scene_with_access(
            update_queue: QueueClass,
            access: AccessKind,
        ) -> (SceneFabric, ResourceTemplate) {
            let initial = ResourceUse {
                key: OUTPUT,
                access,
                scope: ResourceScope::Whole,
                frame_lag: None,
                input: ResourceInput::Initial,
            };
            let feature = FeatureDefinition::subsystem(
                FEATURE,
                RenderDomain::Surface,
                ServiceSet::EMPTY,
                table(vec![FeatureVariant {
                    id: VariantId(0xff00),
                    rank: 0,
                    min_quality: RenderQualityTier::Low,
                    required_caps: CapabilitySet::EMPTY,
                    requires_services_extra: ServiceSet::EMPTY,
                    passes: table(vec![
                        PassTemplate {
                            id: UPDATE,
                            accesses: table(vec![ResourceUse {
                                input: ResourceInput::ProducedBy(SEED),
                                ..initial
                            }]),
                            queue: update_queue,
                            depends_on: &[SEED],
                        },
                        PassTemplate {
                            id: SEED,
                            accesses: table(vec![initial]),
                            queue: QueueClass::Compute,
                            depends_on: &[],
                        },
                    ]),
                    shader_permutation: None,
                }]),
            );
            // Exercise the actual helper's builtin template contract, not a
            // standalone runtime/executor with fabricated physical IDs.
            let builtin = builtin_catalog().unwrap();
            let builtin_roots = builtin_roots(&builtin).unwrap();
            let mut definitions: Vec<_> = builtin.features().copied().collect();
            definitions.push(feature);
            let mut resources: Vec<_> = builtin.resources().collect();
            resources.push(OUTPUT);
            let catalog = FeatureCatalog::new(table(definitions), table(resources)).unwrap();
            let mut subsystems = builtin_roots.subsystems().to_vec();
            subsystems.push(FEATURE);
            let roots = FrozenRoots::from_build_profile(
                &catalog,
                table(subsystems),
                table(builtin_roots.responses().to_vec()),
            )
            .unwrap();
            let scene = SceneFabric(
                DeviceFabric::new(
                    catalog,
                    roots,
                    CapabilitySet::WEBGPU_BASELINE,
                    RenderQualityTier::High,
                )
                .unwrap(),
            );
            let template = ResourceTemplate {
                key: OUTPUT,
                lifetime: ResourceLifetime::Imported,
                layout: ResourceLayout::Buffer {
                    bytes: 16,
                    usage: BufferUsages::STORAGE,
                },
                alignment: 16,
            };
            (scene, template)
        }

        fn read_output(device: &wgpu::Device, queue: &wgpu::Queue, output: &Buffer) -> u32 {
            read_word_at(device, queue, output, 0)
        }

        pub(super) fn read_word_at(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            output: &Buffer,
            offset: u64,
        ) -> u32 {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-multipass-readback"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(output, offset, &staging, 0, 4);
            queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
                result.expect("fabric multipass readback map");
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("fabric multipass queue completion");
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
            drop(mapped);
            staging.unmap();
            value
        }

        #[test]
        fn object_suballocations_upload_and_execute_by_exact_identity() {
            use prism_render_architecture::{
                gpu_scene::SceneHandle,
                motion::MotionSource,
                render_fabric::{ObjectLayoutPolicy, PayloadRole, RenderObjectId, ResourcePayload},
            };
            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let (scene, template) = scene(QueueClass::Compute);
            let role = PayloadRole(0);
            let identities = [
                ObjectResourceIdentity::new(
                    RenderObjectId::Surface(SceneHandle::new(7, 3)),
                    11,
                    role,
                ),
                ObjectResourceIdentity::new(
                    RenderObjectId::Surface(SceneHandle::new(7, 4)),
                    11,
                    role,
                ),
                ObjectResourceIdentity::new(
                    RenderObjectId::Surface(SceneHandle::new(7, 3)),
                    12,
                    role,
                ),
                ObjectResourceIdentity::new(
                    RenderObjectId::Surface(SceneHandle::new(8, 3)),
                    11,
                    role,
                ),
            ];
            let layouts = [16, 32, 16, 32].map(|bytes| ResourceLayout::Buffer {
                bytes,
                usage: BufferUsages::STORAGE,
            });
            let items: Vec<_> = identities
                .iter()
                .zip(layouts)
                .map(|(identity, layout)| FrameRenderItem {
                    object_id: identity.object(),
                    domain: identity.domain(),
                    history_key: identity.history_key(),
                    motion_source: MotionSource::Rigid,
                    payload: DomainPayload::Resource(ResourcePayload {
                        resource: OUTPUT,
                        lifetime: template.lifetime,
                        size: layout.size().unwrap(),
                        alignment: template.alignment,
                        layout,
                        role,
                    }),
                })
                .collect();
            let stride = u64::from(device.limits().min_storage_buffer_offset_alignment).max(64);
            let output = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-exact-object-suballocations"),
                size: stride * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let sentinel = 0xdead_beefu32;
            queue.write_buffer(
                &output,
                0,
                &vec![sentinel.to_le_bytes(); stride as usize].concat(),
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-object-isolation-kernels"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> values: array<u32>;
                     @compute @workgroup_size(1) fn seed() { values[0] = values[1]; }
                     @compute @workgroup_size(1) fn update() {
                         values[0] = values[0] * values[2] + values[3] + arrayLength(&values);
                     }"
                    .into(),
                ),
            });
            let pipeline = |entry| {
                ComputePipeline::from(device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: None,
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    },
                ))
            };
            let seed = pipeline("seed");
            let update = pipeline("update");
            let groups = |pipeline: &ComputePipeline| {
                identities
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        BindGroup::from(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("fabric-object-subrange-group"),
                            layout: &pipeline.get_bind_group_layout(0),
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &output,
                                    offset: stride * index as u64,
                                    size: std::num::NonZeroU64::new(layouts[index].size().unwrap()),
                                }),
                            }],
                        }))
                    })
                    .collect::<Vec<_>>()
            };
            let seed_groups = groups(&seed);
            let update_groups = groups(&update);
            let accesses = [SEED, UPDATE].map(|pass| ObjectResourceAccess {
                pass,
                ordinal: 0,
                role,
                layout: ObjectLayoutPolicy::Sized { max_bytes: 32 },
            });
            let request = || SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &items,
                active_passes: ACTIVE,
                fulfilled_dependencies: &[],
                resources: Vec::new(),
                recorders: [
                    (UPDATE, &update, &update_groups),
                    (SEED, &seed, &seed_groups),
                ]
                .into_iter()
                .map(|(pass, pipeline, groups)| SceneFabricPass::Compute {
                    pass,
                    dispatches: groups
                        .iter()
                        .map(|bind_group| SceneFabricDispatch {
                            pipeline,
                            bind_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        })
                        .collect(),
                })
                .collect(),
            };
            let objects = || {
                identities
                    .iter()
                    .enumerate()
                    .map(|(index, &identity)| SceneFabricObjectResource {
                        identity,
                        layout: layouts[index],
                        resource: SceneFabricResource::Buffer {
                            key: OUTPUT,
                            frame: 4,
                            version: 10 + index as u32,
                            buffer: &output,
                            offset: stride * index as u64,
                        },
                    })
                    .collect::<Vec<_>>()
            };
            let mut missing = objects();
            missing.pop();
            assert!(
                prepare_with_objects(&scene, &rhi, request(), &[template], &accesses, missing)
                    .is_err()
            );
            let mut duplicate = objects();
            duplicate.push(objects().remove(0));
            assert!(matches!(
                prepare_with_objects(&scene, &rhi, request(), &[template], &accesses, duplicate),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::DuplicateObjectBinding(_)
                ))
            ));
            // Distinct Bevy wrapper IDs must not hide native allocation aliasing.
            let cloned_native = Buffer::from(output.deref().clone());
            let mut overlapping = objects();
            overlapping[1].resource = SceneFabricResource::Buffer {
                key: OUTPUT,
                frame: 4,
                version: 11,
                buffer: &cloned_native,
                offset: 0,
            };
            assert!(matches!(
                prepare_with_objects(&scene, &rhi, request(), &[template], &accesses, overlapping),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::OverlappingBindings { .. }
                ))
            ));
            for index in 0..4 {
                assert_eq!(
                    read_word_at(&device, &queue, &output, stride * index),
                    sentinel
                );
            }
            let mut runtime =
                prepare_with_objects(&scene, &rhi, request(), &[template], &accesses, objects())
                    .unwrap();
            assert!(runtime.graph.declaration().resource_id(OUTPUT, 4).is_none());
            assert_eq!(
                runtime
                    .graph
                    .declaration()
                    .passes()
                    .iter()
                    .map(|pass| pass.key())
                    .collect::<Vec<_>>(),
                [SEED, UPDATE]
            );
            let mut expected = Vec::new();
            for index in (0..4).rev() {
                let id = runtime
                    .graph
                    .declaration()
                    .object_resource_id(OUTPUT, 4, identities[index])
                    .unwrap();
                assert_eq!(
                    runtime.graph.declaration().resources()[id.0 as usize]
                        .external
                        .unwrap()
                        .version,
                    10 + index as u32
                );
                let size = layouts[index].size().unwrap() as usize;
                let mut words = vec![0u32; size / 4];
                words[..4].copy_from_slice(&[
                    999,
                    index as u32 + 7,
                    index as u32 + 2,
                    index as u32 + 3,
                ]);
                expected.push((index, words[1] * words[2] + words[3] + words.len() as u32));
                let bytes = words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>();
                assert!(runtime
                    .upload_payload(&rhi, &items[index], &bytes[..12])
                    .is_err());
                let mut wrong_role = items[index];
                if let DomainPayload::Resource(payload) = &mut wrong_role.payload {
                    payload.role = PayloadRole(9);
                }
                assert!(runtime.upload_payload(&rhi, &wrong_role, &bytes).is_err());
                runtime.upload_payload(&rhi, &items[index], &bytes).unwrap();
            }
            runtime
                .submit_tracked(&rhi)
                .wait_for_completion(&rhi)
                .expect("object execution completed");
            for (index, expected) in expected {
                assert_eq!(
                    read_word_at(&device, &queue, &output, stride * index as u64),
                    expected
                );
                assert_eq!(
                    read_word_at(
                        &device,
                        &queue,
                        &output,
                        stride * index as u64 + layouts[index].size().unwrap()
                    ),
                    sentinel
                );
            }
        }

        fn read_texture_output(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            texture: &wgpu::Texture,
        ) -> u32 {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-texture-lease-readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |result| {
                result.expect("fabric texture lease readback map");
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
            drop(mapped);
            staging.unmap();
            value
        }

        #[test]
        fn encode_panics_cancel_transients_before_submission_and_success_transfers_ownership() {
            use prism_render_architecture::render_fabric::{
                TextureDimension, TextureFormat, TextureUsages,
            };
            use std::{
                cell::Cell,
                panic::{catch_unwind, AssertUnwindSafe},
            };

            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            // These native handles satisfy prepare's import contracts. The
            // injected encoder below fails before any native commands run;
            // this test exercises ownership, not shader/access reflection.
            let scratch = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-cancel-scratch"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-cancel-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> output: u32;
                     @compute @workgroup_size(1) fn main() { output = 1u; }"
                        .into(),
                ),
            });
            let native_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fabric-cancel-pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let native_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fabric-cancel-group"),
                layout: &native_pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.as_entire_binding(),
                }],
            });
            let pipeline = ComputePipeline::from(native_pipeline);
            let group = BindGroup::from(native_group);
            let (scene, base_template) = scene(QueueClass::Compute);
            let request = || SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &[],
                active_passes: ACTIVE,
                fulfilled_dependencies: &[],
                resources: vec![],
                recorders: ACTIVE
                    .iter()
                    .map(|&pass| SceneFabricPass::Compute {
                        pass,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &pipeline,
                            bind_group: &group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    })
                    .collect(),
            };
            let assert_released = |buffers: &[prism_render_driver::BufferId],
                                   textures: &[(
                prism_render_driver::TextureId,
                prism_render_driver::TextureViewId,
            )]| {
                for &buffer in buffers {
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_buffer(buffer)))
                            .is_err()
                    );
                }
                for &(texture, view) in textures {
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_texture(texture)))
                            .is_err()
                    );
                    assert!(catch_unwind(AssertUnwindSafe(|| rhi
                        .device()
                        .wgpu_texture_view(view)))
                    .is_err());
                }
            };
            for layout in [
                base_template.layout,
                ResourceLayout::Texture {
                    extent: [1, 1, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Uint,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::STORAGE_BINDING,
                },
            ] {
                let template = ResourceTemplate {
                    lifetime: ResourceLifetime::Transient,
                    layout,
                    ..base_template
                };
                for submit in [false, true] {
                    let runtime =
                        prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
                    let pipelines: Vec<_> = runtime
                        ._pipeline_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect();
                    let groups: Vec<_> = runtime
                        ._bind_group_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect();
                    let mut buffers = Vec::new();
                    let mut textures = Vec::new();
                    let submit_called = Cell::new(false);
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        let fail_encode = |recorded: &RecordedGraph| -> bevy_render::render_resource::CommandBuffer {
                            buffers.extend(recorded.allocations().buffers().iter().map(|(_, _, id)| *id));
                            textures.extend(recorded.allocations().textures().iter().map(|(_, _, resource)| (resource.texture, resource.default_view)));
                            panic!("injected native encode failure after graph allocation");
                        };
                        if submit {
                            drop(runtime.submit_tracked_with(&rhi, fail_encode, |_| {
                                submit_called.set(true);
                                panic!("encode failure must not reach submission");
                            }));
                        } else {
                            drop(runtime.encode_with(&rhi, fail_encode));
                        }
                    })).is_err());
                    assert!(!submit_called.get());
                    assert_eq!(
                        buffers.len() + textures.len(),
                        1,
                        "must allocate before failing"
                    );
                    assert_released(&buffers, &textures);
                    for pipeline in pipelines {
                        assert!(catch_unwind(AssertUnwindSafe(|| rhi
                            .device()
                            .wgpu_compute_pipeline(pipeline)))
                        .is_err());
                    }
                    for group in groups {
                        assert!(catch_unwind(AssertUnwindSafe(|| rhi
                            .device()
                            .wgpu_bind_group(group)))
                        .is_err());
                    }
                }

                // Successful encoding must disarm cancellation, both for a
                // host-owned packet and for a tracked completion owner. Empty
                // native work isolates this test from shader binding policy.
                for submit in [false, true] {
                    let runtime =
                        prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
                    let mut buffers = Vec::new();
                    let mut textures = Vec::new();
                    let encode = |recorded: &RecordedGraph| {
                        buffers.extend(
                            recorded
                                .allocations()
                                .buffers()
                                .iter()
                                .map(|(_, _, id)| *id),
                        );
                        textures.extend(
                            recorded
                                .allocations()
                                .textures()
                                .iter()
                                .map(|(_, _, resource)| (resource.texture, resource.default_view)),
                        );
                        device.create_command_encoder(&Default::default()).finish()
                    };
                    if submit {
                        let submission = runtime.submit_tracked_with(
                            &rhi,
                            encode,
                            |native| -> CompletionToken { Box::new(queue.submit([native])) },
                        );
                        for &buffer in &buffers {
                            let _ = rhi.device().wgpu_buffer(buffer);
                        }
                        for &(texture, view) in &textures {
                            let _ = rhi.device().wgpu_texture(texture);
                            let _ = rhi.device().wgpu_texture_view(view);
                        }
                        submission
                            .wait_for_completion(&rhi)
                            .expect("native completion");
                    } else {
                        let (native, recorded) = runtime.encode_with(&rhi, encode);
                        for &buffer in &buffers {
                            let _ = rhi.device().wgpu_buffer(buffer);
                        }
                        for &(texture, view) in &textures {
                            let _ = rhi.device().wgpu_texture(texture);
                            let _ = rhi.device().wgpu_texture_view(view);
                        }
                        queue.submit([native]);
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        recorded.release(rhi.device().as_ref());
                    }
                    assert_eq!(buffers.len() + textures.len(), 1);
                    assert_released(&buffers, &textures);
                }
            }
        }

        #[test]
        fn frozen_multipass_compilation_requires_real_bindings_and_active_dependencies() {
            let (scene, template) = scene(QueueClass::Compute);
            let templates = scene_templates(&scene, &[template]).unwrap();
            let compile = |active: &[PassKey]| {
                scene
                    .0
                    .compile_active(&templates, 4, &[], &[], active, &[], GraphLimits::default())
            };
            assert!(matches!(
                compile(ACTIVE),
                Err(FabricCompileError::MissingBinding {
                    resource: OUTPUT,
                    frame: 4,
                })
            ));
            assert!(matches!(
                compile(&[UPDATE]),
                Err(FabricCompileError::MissingActiveDependency {
                    pass: UPDATE,
                    dependency: SEED,
                })
            ));
            assert!(matches!(
                compile(&[UPDATE, PassKey(u32::MAX)]),
                Err(FabricCompileError::ForeignActivePass(_))
            ));
        }

        #[test]
        fn native_resource_preflight_checks_ranges_aliases_views_and_d3_layers() {
            use prism_render_architecture::render_fabric::{
                TextureDimension, TextureFormat, TextureUsages,
            };
            let Some((device, _queue, _adapter, render_device)) = hardware_context() else {
                return;
            };
            let (_, template) = scene(QueueClass::Compute);
            let buffer = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-preflight-buffer"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            // A new Bevy wrapper ID must not hide a shared native allocation.
            let alias = Buffer::from(buffer.deref().clone());
            assert_ne!(buffer.id(), alias.id());
            let resource = |offset, frame| SceneFabricResource::Buffer {
                key: OUTPUT,
                frame,
                version: 0,
                buffer: &buffer,
                offset,
            };
            assert!(validate_resources(&[template], &[resource(48, 4)]).is_ok());
            for offset in [64, u64::MAX] {
                assert!(matches!(
                    validate_resources(&[template], &[resource(offset, 4)]),
                    Err(SceneFabricRuntimeError::BufferRange { key: OUTPUT, .. })
                ));
            }
            assert!(matches!(
                validate_resources(&[template], &[resource(1, 4)]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::InvalidBinding {
                        resource: OUTPUT,
                        frame: 4,
                    }
                ))
            ));
            let alias_resource = |offset| SceneFabricResource::Buffer {
                key: OUTPUT,
                frame: 3,
                version: 0,
                buffer: &alias,
                offset,
            };
            assert!(
                validate_resources(&[template], &[resource(0, 4), alias_resource(16)],).is_ok(),
                "adjacent ranges are not aliases"
            );
            assert!(matches!(
                validate_resources(&[template], &[resource(0, 4), alias_resource(0)]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::OverlappingBindings {
                        first: OUTPUT,
                        first_frame: 4,
                        second: OUTPUT,
                        second_frame: 3,
                    }
                ))
            ));
            let wider = ResourceTemplate {
                layout: ResourceLayout::Buffer {
                    bytes: 32,
                    usage: BufferUsages::STORAGE,
                },
                ..template
            };
            for resources in [
                [resource(0, 4), alias_resource(16)],
                [resource(16, 4), alias_resource(0)],
            ] {
                assert!(matches!(
                    validate_resources(&[wider], &resources),
                    Err(SceneFabricRuntimeError::Compile(
                        FabricCompileError::OverlappingBindings { .. }
                    ))
                ));
            }
            let needs_copy = ResourceTemplate {
                layout: ResourceLayout::Buffer {
                    bytes: 16,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                },
                ..template
            };
            assert!(matches!(
                validate_resources(&[needs_copy], &[resource(0, 4)]),
                Err(SceneFabricRuntimeError::BufferRange { key: OUTPUT, .. })
            ));
            assert!(matches!(
                validate_resources(&[], &[resource(0, 4)]),
                Err(SceneFabricRuntimeError::MissingTemplate(OUTPUT))
            ));

            for dimension in [TextureDimension::D2, TextureDimension::D3] {
                let texture_template = ResourceTemplate {
                    layout: ResourceLayout::Texture {
                        extent: [4, 4, 4],
                        dimension,
                        format: TextureFormat::Rgba8Unorm,
                        mip_levels: 2,
                        samples: 1,
                        usage: TextureUsages::TEXTURE_BINDING,
                    },
                    ..template
                };
                let descriptor = wgpu::TextureDescriptor {
                    label: Some("fabric-preflight-texture"),
                    size: wgpu::Extent3d {
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 4,
                    },
                    mip_level_count: 2,
                    sample_count: 1,
                    dimension: texture_dimension(dimension),
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                };
                let texture = Texture::from(device.create_texture(&descriptor));
                let alias = Texture::from(texture.deref().clone());
                assert_ne!(texture.id(), alias.id());
                let view = texture.create_view(&Default::default());
                let other = device.create_texture(&descriptor);
                let wrong_view = TextureView::from(other.create_view(&Default::default()));
                let layers = if dimension == TextureDimension::D3 {
                    1
                } else {
                    4
                };
                let resource = |texture, default_view, frame, mip_levels, array_layers| {
                    SceneFabricResource::Texture {
                        key: OUTPUT,
                        frame,
                        version: 0,
                        texture,
                        default_view,
                        entry_state: TextureState::initial(),
                        exit_state: TextureState::initial(),
                        mip_levels,
                        array_layers,
                    }
                };
                assert!(validate_resources(
                    &[texture_template],
                    &[resource(&texture, &view, 4, 2, layers)],
                )
                .is_ok());
                for (default_view, mips, layer_count) in [
                    (&wrong_view, 2, layers),
                    (&view, 1, layers),
                    (&view, 2, if layers == 1 { 4 } else { 1 }),
                ] {
                    assert!(matches!(
                        validate_resources(
                            &[texture_template],
                            &[resource(&texture, default_view, 4, mips, layer_count)],
                        ),
                        Err(SceneFabricRuntimeError::TextureMetadata { key: OUTPUT })
                    ));
                }
                assert!(matches!(
                    validate_resources(
                        &[texture_template],
                        &[
                            resource(&texture, &view, 4, 2, layers),
                            resource(&alias, &view, 3, 2, layers),
                        ]
                    ),
                    Err(SceneFabricRuntimeError::Compile(
                        FabricCompileError::OverlappingBindings {
                            first: OUTPUT,
                            first_frame: 4,
                            second: OUTPUT,
                            second_frame: 3,
                        }
                    ))
                ));
                assert!(matches!(
                    validate_resources(&[template], &[resource(&texture, &view, 4, 2, layers)]),
                    Err(SceneFabricRuntimeError::ResourceKind { key: OUTPUT })
                ));
                assert!(matches!(
                    validate_resources(
                        &[texture_template],
                        &[SceneFabricResource::Buffer {
                            key: OUTPUT,
                            frame: 4,
                            version: 0,
                            buffer: &buffer,
                            offset: 0,
                        }]
                    ),
                    Err(SceneFabricRuntimeError::ResourceKind { key: OUTPUT })
                ));
            }
        }

        #[test]
        fn texture_view_leases_unwind_on_preflight_compile_and_runtime_failures() {
            use prism_render_architecture::render_fabric::{
                FabricRuntimeError, TextureDimension, TextureFormat, TextureUsages,
            };
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let descriptor = wgpu::TextureDescriptor {
                label: Some("fabric-texture-lease-output"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };
            let texture = Texture::from(device.create_texture(&descriptor));
            queue.write_texture(
                texture.as_image_copy(),
                &999u32.to_le_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                descriptor.size,
            );
            let view = texture.create_view(&Default::default());
            let other = Texture::from(device.create_texture(&descriptor));
            let wrong_view = other.create_view(&Default::default());
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-texture-lease-kernels"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var output: texture_storage_2d<r32uint, write>;
                     @compute @workgroup_size(1) fn seed() {
                         textureStore(output, vec2i(0), vec4u(7u));
                     }
                     @compute @workgroup_size(1) fn update() {
                         textureStore(output, vec2i(0), vec4u(24u));
                     }"
                    .into(),
                ),
            });
            let kernel = |entry_point| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(entry_point),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                });
                (ComputePipeline::from(pipeline), BindGroup::from(group))
            };
            let (seed, seed_group) = kernel("seed");
            let (update, update_group) = kernel("update");
            let (scene, buffer_template) = scene(QueueClass::Compute);
            let template = ResourceTemplate {
                layout: ResourceLayout::Texture {
                    extent: [1, 1, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Uint,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::STORAGE_BINDING,
                },
                ..buffer_template
            };
            let request = || SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &[],
                active_passes: ACTIVE,
                fulfilled_dependencies: &[],
                resources: vec![SceneFabricResource::Texture {
                    key: OUTPUT,
                    frame: 4,
                    version: 0,
                    texture: &texture,
                    default_view: &view,
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::copy_src(),
                    mip_levels: 1,
                    array_layers: 1,
                }],
                recorders: vec![
                    SceneFabricPass::Compute {
                        pass: UPDATE,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &update,
                            bind_group: &update_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    },
                    SceneFabricPass::Compute {
                        pass: SEED,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &seed,
                            bind_group: &seed_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    },
                ],
            };

            // Independent registrations must survive every failed request.
            let survivor_texture = rhi.import_texture(&texture);
            let survivor_view = rhi.import_texture_view(&view);
            let probe_ids = || {
                let texture = rhi.import_texture(&texture);
                let view = rhi.import_texture_view(&view);
                (texture.id(), view.id())
            };
            let assert_recycled = |before: (
                prism_render_driver::TextureId,
                prism_render_driver::TextureViewId,
            ),
                                   imports: u32| {
                let after = probe_ids();
                assert_eq!(after.0.index(), before.0.index(), "texture slot leaked");
                assert_eq!(after.1.index(), before.1.index(), "view slot leaked");
                assert_eq!(after.0.generation(), before.0.generation() + imports + 1);
                assert_eq!(after.1.generation(), before.1.generation() + imports + 1);
            };
            // A valid first texture cannot import either table entry when a
            // later resource has a wrong-parent view or invalid mip metadata.
            for bad_view in [true, false] {
                let before = probe_ids();
                let mut invalid = request();
                invalid.resources.push(SceneFabricResource::Texture {
                    key: OUTPUT,
                    frame: 3,
                    version: 0,
                    texture: &texture,
                    default_view: if bad_view { &wrong_view } else { &view },
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::copy_src(),
                    mip_levels: if bad_view { 1 } else { 2 },
                    array_layers: 1,
                });
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, invalid, &[template]),
                    Err(SceneFabricRuntimeError::TextureMetadata { key: OUTPUT })
                ));
                assert_recycled(before, 0);
            }

            let before = probe_ids();
            let mut missing_dependency = request();
            missing_dependency.active_passes = &[UPDATE];
            missing_dependency.recorders.pop();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, missing_dependency, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::MissingActiveDependency {
                        pass: UPDATE,
                        dependency: SEED,
                    }
                ))
            ));
            assert_recycled(before, 1);

            let before = probe_ids();
            let (graphics_scene, _) = self::scene(QueueClass::Graphics);
            assert!(matches!(
                prepare_with_templates(&graphics_scene, &rhi, request(), &[template]),
                Err(SceneFabricRuntimeError::NonComputePass(UPDATE))
            ));
            assert_recycled(before, 1);

            // Catalog validation rejects compute attachments before runtime.
            // Exercise a genuine late import failure instead: compile valid
            // declarations, retain real resource/command leases, then reject
            // import metadata that disagrees with the compiled texture layout.
            // No recorder may run, and all four tables must unwind.
            let before = probe_ids();
            let pipeline_probe = rhi.import_compute_pipeline(&seed);
            let group_probe = rhi.import_bind_group(&seed_group);
            let before_pipeline = pipeline_probe.id();
            let before_group = group_probe.id();
            drop((pipeline_probe, group_probe));
            let prepare_invalid_import =
                || -> Result<RuntimeFeatureGraph, SceneFabricRuntimeError> {
                    let texture_lease = rhi.import_texture(&texture);
                    let view_lease = rhi.import_texture_view(&view);
                    let pipeline_lease = rhi.import_compute_pipeline(&seed);
                    let group_lease = rhi.import_bind_group(&seed_group);
                    let compiled = scene.0.compile_active(
                        &scene_templates(&scene, &[template])?,
                        4,
                        &[FrameResourceBinding {
                            key: OUTPUT,
                            frame: 4,
                            version: 0,
                            physical: PhysicalResource::Texture(texture_lease.id()),
                            layout: template.layout,
                            alignment: template.alignment,
                        }],
                        &[],
                        &[SEED],
                        &[],
                        GraphLimits::default(),
                    )?;
                    let resource = compiled.resource_id(OUTPUT, 4).unwrap();
                    let pipeline = pipeline_lease.id();
                    let group = group_lease.id();
                    Ok(compiled.prepare_runtime(
                        Extent3d::new_2d(1, 1),
                        &[RuntimeImport::Texture {
                            resource,
                            imported: ImportedTexture {
                                texture: texture_lease.id(),
                                default_view: view_lease.id(),
                                entry_state: TextureState::initial(),
                                exit_state: TextureState::copy_src(),
                                mip_levels: 2,
                                array_layers: 1,
                            },
                        }],
                        vec![RuntimePassBinding::new(SEED, move |context| {
                            context
                                .commands()
                                .set_compute_pipeline(pipeline)
                                .set_compute_bind_group(0, group, Vec::new())
                                .dispatch(1, 1, 1);
                            panic!("invalid imports must fail before recording");
                        })],
                    )?)
                };
            assert!(matches!(
                prepare_invalid_import(),
                Err(SceneFabricRuntimeError::Runtime(
                    FabricRuntimeError::InvalidImport(_)
                ))
            ));
            assert_recycled(before, 1);
            let pipeline_probe = rhi.import_compute_pipeline(&seed);
            let group_probe = rhi.import_bind_group(&seed_group);
            assert_eq!(pipeline_probe.id().index(), before_pipeline.index());
            assert_eq!(
                pipeline_probe.id().generation(),
                before_pipeline.generation() + 2
            );
            assert_eq!(group_probe.id().index(), before_group.index());
            assert_eq!(group_probe.id().generation(), before_group.generation() + 2);
            drop((pipeline_probe, group_probe));

            let imported_ids = |runtime: &SceneFabricRuntime| {
                (
                    runtime._texture_leases[0].id(),
                    runtime._view_leases[0].id(),
                    runtime
                        ._pipeline_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect::<Vec<_>>(),
                    runtime
                        ._bind_group_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect::<Vec<_>>(),
                )
            };
            let assert_released = |(texture, view, pipelines, groups): (
                prism_render_driver::TextureId,
                prism_render_driver::TextureViewId,
                Vec<prism_render_driver::ComputePipelineId>,
                Vec<prism_render_driver::BindGroupId>,
            )| {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| { rhi.device().wgpu_texture(texture) }))
                        .is_err()
                );
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    rhi.device().wgpu_texture_view(view)
                }))
                .is_err());
                for pipeline in pipelines {
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        rhi.device().wgpu_compute_pipeline(pipeline)
                    }))
                    .is_err());
                }
                for group in groups {
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        rhi.device().wgpu_bind_group(group)
                    }))
                    .is_err());
                }
            };
            let abandoned = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
            let ids = imported_ids(&abandoned);
            drop(abandoned);
            assert_released(ids);

            // A bridge from the same native device still owns a distinct ID
            // table. Neither rejection may record commands, leak our imports,
            // or release the foreign bridge's texture/view registrations.
            let foreign_rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let foreign_texture = foreign_rhi.import_texture(&texture);
            let foreign_view = foreign_rhi.import_texture_view(&view);
            for submit in [false, true] {
                let runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
                let ids = imported_ids(&runtime);
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    if submit {
                        drop(runtime.submit_tracked(&foreign_rhi));
                    } else {
                        drop(runtime.encode(&foreign_rhi));
                    }
                }))
                .is_err());
                assert_released(ids);
                assert_eq!(
                    foreign_rhi.device().wgpu_texture(foreign_texture.id()),
                    *texture.deref()
                );
                assert_eq!(
                    foreign_rhi.device().wgpu_texture_view(foreign_view.id()),
                    *view.deref()
                );
                assert_eq!(read_texture_output(&device, &queue, &texture), 999);
            }
            drop((foreign_texture, foreign_view));
            assert_eq!(
                rhi.device().wgpu_texture(survivor_texture.id()),
                *texture.deref()
            );
            assert_eq!(
                rhi.device().wgpu_texture_view(survivor_view.id()),
                *view.deref()
            );

            let runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
            let ids = imported_ids(&runtime);
            assert_eq!(read_texture_output(&device, &queue, &texture), 999);
            // The graph leases retain Bevy texture/view and command handles
            // after caller wrappers and independent registrations are gone.
            // Only a native texture clone remains for the host's readback.
            let readback_texture = texture.deref().clone();
            drop((
                survivor_texture,
                survivor_view,
                seed,
                seed_group,
                update,
                update_group,
            ));
            drop((texture, view));
            drop(shader);
            let (native, recorded) = runtime.encode(&rhi);
            // Host encoding is not completion. The command-buffer callback
            // retains the exact table entries until the submitted work ends.
            let _ = rhi.device().wgpu_texture(ids.0);
            let _ = rhi.device().wgpu_texture_view(ids.1);
            for &pipeline in &ids.2 {
                let _ = rhi.device().wgpu_compute_pipeline(pipeline);
            }
            for &group in &ids.3 {
                let _ = rhi.device().wgpu_bind_group(group);
            }
            assert!(recorded.allocations().is_empty(), "imported-only graph");
            queue.submit([native]);
            assert_eq!(read_texture_output(&device, &queue, &readback_texture), 24);
            assert_released(ids);
            drop(recorded);
        }

        #[test]
        fn existing_buffer_leases_preserve_identity_subranges_and_native_lifetime() {
            use crate::fabric_domains::gpu::DomainBufferOwner;
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let offset = u64::from(device.limits().min_storage_buffer_offset_alignment).max(16);
            let size = offset + 32;
            let sentinel = 0x5a17_c0de_u32;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-existing-buffer-kernels"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> output: array<u32>;
                     @compute @workgroup_size(1) fn seed() {
                         output[0] = 7u;
                         output[1] = arrayLength(&output);
                     }
                     @compute @workgroup_size(1) fn update() {
                         output[0] = output[0] * 3u + 3u;
                         output[2] = output[1] + 1u;
                     }"
                    .into(),
                ),
            });
            for allocated in [true, false] {
                let rhi =
                    SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
                let owner = if allocated {
                    DomainBufferOwner::Allocated(Arc::new(rhi.create_buffer(
                        &prism_render_driver::BufferDescriptor::new(
                            size,
                            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                        ),
                    )))
                } else {
                    let native = render_device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fabric-existing-imported-buffer"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    DomainBufferOwner::Imported(Arc::new(rhi.import_buffer(&native)))
                };
                let borrowed = match &owner {
                    DomainBufferOwner::Allocated(lease) => DomainBufferOwnerRef::Allocated(lease),
                    DomainBufferOwner::Imported(lease) => DomainBufferOwnerRef::Imported(lease),
                };
                let id = borrowed.id();
                let buffer = borrowed.buffer().clone();
                let initial: Vec<u8> = (0..size / 4).flat_map(|_| sentinel.to_le_bytes()).collect();
                queue.write_buffer(&buffer, 0, &initial);
                let kernel = |entry_point| {
                    let pipeline =
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(entry_point),
                            layout: None,
                            module: &shader,
                            entry_point: Some(entry_point),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(entry_point),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset,
                                size: std::num::NonZeroU64::new(16),
                            }),
                        }],
                    });
                    (ComputePipeline::from(pipeline), BindGroup::from(group))
                };
                let (seed, seed_group) = kernel("seed");
                let (update, update_group) = kernel("update");
                let (scene, template) = scene(QueueClass::Compute);
                let request = || SceneFabricRequest {
                    frame: 4,
                    extent: Extent3d::new_2d(1, 1),
                    items: &[],
                    active_passes: ACTIVE,
                    fulfilled_dependencies: &[],
                    resources: vec![SceneFabricResource::ExistingBuffer {
                        key: OUTPUT,
                        frame: 4,
                        version: 7,
                        owner: borrowed,
                        offset,
                        entry_state: BufferState::copy_dst(),
                    }],
                    recorders: vec![
                        SceneFabricPass::Compute {
                            pass: UPDATE,
                            dispatches: vec![SceneFabricDispatch {
                                pipeline: &update,
                                bind_group: &update_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: None,
                            }],
                        },
                        SceneFabricPass::Compute {
                            pass: SEED,
                            dispatches: vec![SceneFabricDispatch {
                                pipeline: &seed,
                                bind_group: &seed_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: None,
                            }],
                        },
                    ],
                };
                let foreign =
                    SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
                let foreign_lease = foreign.import_buffer(&buffer);
                assert_eq!(id, foreign_lease.id(), "IDs collide across distinct tables");
                let mut invalid = request();
                if let SceneFabricResource::ExistingBuffer { owner, .. } = &mut invalid.resources[0]
                {
                    *owner = DomainBufferOwnerRef::Imported(&foreign_lease);
                }
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, invalid, &[template]),
                    Err(SceneFabricRuntimeError::BufferOrigin { key: OUTPUT })
                ));

                // A fresh imported ID and Bevy wrapper cannot hide a native alias.
                let alias = Buffer::from(buffer.deref().clone());
                let alias_lease = rhi.import_buffer(&alias);
                assert_ne!(id, alias_lease.id());
                for raw in [true, false] {
                    for reverse in [true, false] {
                        let candidate = |alias_offset| {
                            if raw {
                                SceneFabricResource::Buffer {
                                    key: OUTPUT,
                                    frame: 3,
                                    version: 1,
                                    buffer: &alias,
                                    offset: alias_offset,
                                }
                            } else {
                                SceneFabricResource::ExistingBuffer {
                                    key: OUTPUT,
                                    frame: 3,
                                    version: 1,
                                    owner: DomainBufferOwnerRef::Imported(&alias_lease),
                                    offset: alias_offset,
                                    entry_state: BufferState::initial(),
                                }
                            }
                        };
                        let mut invalid = request();
                        invalid.resources.push(candidate(offset));
                        if reverse {
                            invalid.resources.reverse();
                        }
                        assert!(matches!(
                            prepare_with_templates(&scene, &rhi, invalid, &[template]),
                            Err(SceneFabricRuntimeError::Compile(
                                FabricCompileError::OverlappingBindings { .. }
                            ))
                        ));
                        let mut disjoint = request();
                        disjoint.resources.push(candidate(offset + 16));
                        if reverse {
                            disjoint.resources.reverse();
                        }
                        assert!(validate_resources(&[template], &disjoint.resources).is_ok());
                    }
                }
                if offset > 16 {
                    let mut invalid = request();
                    if let SceneFabricResource::ExistingBuffer { offset, .. } =
                        &mut invalid.resources[0]
                    {
                        *offset = 16;
                    }
                    assert!(matches!(
                        prepare_with_templates(&scene, &rhi, invalid, &[template]),
                        Err(SceneFabricRuntimeError::BufferBindingAlignment {
                            key: OUTPUT,
                            usage: "storage",
                            ..
                        })
                    ));
                }
                // Keep other live entries so the free-slot generation probe is exact.
                let probe = rhi.import_buffer(&buffer);
                let before = probe.id();
                drop(probe);
                let runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
                assert!(runtime._buffer_leases.is_empty());
                assert_eq!(runtime._borrowed_buffer_leases.len(), 1);
                assert_eq!(runtime._borrowed_buffer_leases[0].id(), id);
                let binding = runtime
                    .graph
                    .declaration()
                    .resources()
                    .iter()
                    .find_map(|resource| resource.external)
                    .unwrap();
                assert_eq!(binding.frame, 4);
                assert_eq!(binding.version, 7);
                assert_eq!(binding.physical, PhysicalResource::Buffer { id, offset });
                let probe = rhi.import_buffer(&buffer);
                assert_eq!(probe.id().index(), before.index());
                assert_eq!(
                    probe.id().generation(),
                    before.generation() + 1,
                    "runtime must not reimport buffers"
                );
                drop(probe);
                let (native, recorded) = runtime.encode(&rhi);
                assert_eq!(rhi.device().wgpu_buffer(id), *buffer.deref());
                // Caller-owned registrations can end after native encoding;
                // the encoded commands, not the resource table, retain handles.
                drop((
                    owner,
                    alias_lease,
                    foreign_lease,
                    seed,
                    seed_group,
                    update,
                    update_group,
                ));
                assert!(catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_buffer(id))).is_err());
                queue.submit([native]);
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fabric-existing-buffer-full-readback"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, size);
                queue.submit([encoder.finish()]);
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| result.unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let mapped = staging.slice(..).get_mapped_range().unwrap();
                let words: Vec<u32> = mapped
                    .chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                let start = offset as usize / 4;
                assert!(words[..start].iter().all(|word| *word == sentinel));
                assert_eq!(&words[start..start + 4], &[24, 4, 5, sentinel]);
                assert!(words[start + 4..].iter().all(|word| *word == sentinel));
                drop(mapped);
                staging.unmap();
                assert!(recorded.allocations().is_empty());
            }
        }

        #[test]
        fn existing_texture_leases_preserve_identity_validate_and_encode_without_reimport() {
            use prism_render_architecture::render_fabric::{
                TextureDimension, TextureFormat, TextureUsages,
            };
            use std::panic::{catch_unwind, AssertUnwindSafe};

            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let descriptor = wgpu::TextureDescriptor {
                label: Some("fabric-existing-texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };
            let native_texture = Texture::from(device.create_texture(&descriptor));
            let native_view = native_texture.create_view(&Default::default());
            let texture = rhi.import_texture(&native_texture);
            let view = rhi.import_texture_view(&native_view);
            let texture_id = texture.id();
            let view_id = view.id();
            let other = Texture::from(device.create_texture(&descriptor));
            let wrong_view = rhi.import_texture_view(&other.create_view(&Default::default()));
            let foreign =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let foreign_texture = foreign.import_texture(&native_texture);
            let foreign_view = foreign.import_texture_view(&native_view);
            // Same native device and colliding IDs are not proof of same table.
            assert_eq!(texture_id, foreign_texture.id());
            assert_eq!(view_id, foreign_view.id());

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-existing-texture-kernels"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var output: texture_storage_2d<r32uint, write>;
                     @compute @workgroup_size(1) fn seed() {
                         textureStore(output, vec2<i32>(0), vec4<u32>(7u));
                     }
                     @compute @workgroup_size(1) fn update() {
                         textureStore(output, vec2<i32>(0), vec4<u32>(24u));
                     }"
                    .into(),
                ),
            });
            let kernel = |entry_point| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(entry_point),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&native_view),
                    }],
                });
                (ComputePipeline::from(pipeline), BindGroup::from(group))
            };
            let (seed, seed_group) = kernel("seed");
            let (update, update_group) = kernel("update");
            let (scene, buffer_template) = scene(QueueClass::Compute);
            let template = ResourceTemplate {
                layout: ResourceLayout::Texture {
                    extent: [1, 1, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Uint,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::STORAGE_BINDING,
                },
                ..buffer_template
            };
            let request = || SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &[],
                active_passes: ACTIVE,
                fulfilled_dependencies: &[],
                resources: vec![SceneFabricResource::ExistingTexture {
                    key: OUTPUT,
                    frame: 4,
                    version: 7,
                    texture: &texture,
                    default_view: &view,
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::copy_src(),
                    mip_levels: 1,
                    array_layers: 1,
                }],
                recorders: vec![
                    SceneFabricPass::Compute {
                        pass: UPDATE,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &update,
                            bind_group: &update_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    },
                    SceneFabricPass::Compute {
                        pass: SEED,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &seed,
                            bind_group: &seed_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    },
                ],
            };
            for (candidate_texture, candidate_view) in [
                (&foreign_texture, &view),
                (&texture, &foreign_view),
                (&foreign_texture, &foreign_view),
            ] {
                let mut invalid = request();
                if let SceneFabricResource::ExistingTexture {
                    texture,
                    default_view,
                    ..
                } = &mut invalid.resources[0]
                {
                    *texture = candidate_texture;
                    *default_view = candidate_view;
                }
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, invalid, &[template]),
                    Err(SceneFabricRuntimeError::TextureOrigin { key: OUTPUT })
                ));
            }
            let mut invalid = request();
            if let SceneFabricResource::ExistingTexture { default_view, .. } =
                &mut invalid.resources[0]
            {
                *default_view = &wrong_view;
            }
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, invalid, &[template]),
                Err(SceneFabricRuntimeError::TextureMetadata { key: OUTPUT })
            ));
            for wrong_mips in [false, true] {
                let mut invalid = request();
                if let SceneFabricResource::ExistingTexture {
                    mip_levels,
                    array_layers,
                    ..
                } = &mut invalid.resources[0]
                {
                    if wrong_mips {
                        *mip_levels = 2;
                    } else {
                        *array_layers = 2;
                    }
                }
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, invalid, &[template]),
                    Err(SceneFabricRuntimeError::TextureMetadata { key: OUTPUT })
                ));
            }
            let wrong_template = ResourceTemplate {
                layout: ResourceLayout::Texture {
                    extent: [2, 1, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R32Uint,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::STORAGE_BINDING,
                },
                ..template
            };
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, request(), &[wrong_template]),
                Err(SceneFabricRuntimeError::TextureMetadata { key: OUTPUT })
            ));
            for reverse in [false, true] {
                let mut invalid = request();
                invalid.resources.push(SceneFabricResource::Texture {
                    key: OUTPUT,
                    frame: 3,
                    version: 1,
                    texture: &native_texture,
                    default_view: &native_view,
                    entry_state: TextureState::initial(),
                    exit_state: TextureState::copy_src(),
                    mip_levels: 1,
                    array_layers: 1,
                });
                if reverse {
                    invalid.resources.reverse();
                }
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, invalid, &[template]),
                    Err(SceneFabricRuntimeError::Compile(
                        FabricCompileError::OverlappingBindings { .. }
                    ))
                ));
            }
            let mut invalid = request();
            invalid.active_passes = &[UPDATE];
            invalid.recorders.pop();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, invalid, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::MissingActiveDependency { .. }
                ))
            ));

            let probe = rhi.import_texture(&native_texture);
            let before = probe.id();
            drop(probe);
            let probe = rhi.import_texture_view(&native_view);
            let before_view = probe.id();
            drop(probe);
            let runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
            assert!(runtime._texture_leases.is_empty());
            assert!(runtime._view_leases.is_empty());
            assert_eq!(runtime._borrowed_texture_leases.len(), 1);
            assert_eq!(runtime._borrowed_texture_leases[0].0.id(), texture_id);
            assert_eq!(runtime._borrowed_texture_leases[0].1.id(), view_id);
            let binding = runtime
                .graph
                .declaration()
                .resources()
                .iter()
                .find_map(|resource| resource.external)
                .unwrap();
            assert_eq!(binding.key, OUTPUT);
            assert_eq!(binding.frame, 4);
            assert_eq!(binding.version, 7);
            assert_eq!(binding.physical, PhysicalResource::Texture(texture_id));
            let probe = rhi.import_texture(&native_texture);
            assert_eq!(probe.id().index(), before.index());
            assert_eq!(
                probe.id().generation(),
                before.generation() + 1,
                "runtime must not reimport texture"
            );
            let probe_view = rhi.import_texture_view(&native_view);
            assert_eq!(probe_view.id().index(), before_view.index());
            assert_eq!(
                probe_view.id().generation(),
                before_view.generation() + 1,
                "runtime must not reimport view"
            );
            drop((probe, probe_view));
            drop((seed, seed_group, update, update_group, shader));
            let readback = native_texture.deref().clone();
            drop((native_texture, native_view));
            let (native, recorded) = runtime.encode(&rhi);
            assert_eq!(
                rhi.device().wgpu_texture(texture_id),
                *texture.texture().deref()
            );
            assert_eq!(
                rhi.device().wgpu_texture_view(view_id),
                *view.view().deref()
            );
            // Borrowing ends at encode, and caller-owned table entries are not
            // retired by the runtime. Native commands retain the GPU handles.
            drop((texture, view));
            assert!(
                catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_texture(texture_id))).is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_texture_view(view_id))).is_err()
            );
            queue.submit([native]);
            assert_eq!(read_texture_output(&device, &queue, &readback), 24);
            assert!(recorded.allocations().is_empty());
        }

        #[test]
        fn real_twenty_byte_push_constants_reach_every_dispatch_in_compiler_order() {
            let Some((device, queue, adapter, render_device)) =
                hardware_context_with_immediates(20)
            else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let output = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-push-constant-output"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-twenty-byte-immediates"),
                source: wgpu::ShaderSource::Wgsl(
                    "struct Params { a: u32, b: u32, c: u32, d: u32, e: u32 }
                     var<immediate> params: Params;
                     @group(0) @binding(0) var<storage, read_write> output: array<u32>;
                     @compute @workgroup_size(1) fn seed() {
                         output[0] = params.a + params.b + params.c + params.d + params.e;
                     }
                     @compute @workgroup_size(1) fn update() {
                         output[0] = output[0] * params.a + params.b + params.c + params.d + params.e;
                     }".into(),
                ),
            });
            let kernel = |entry_point| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(entry_point),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    }],
                });
                (ComputePipeline::from(pipeline), BindGroup::from(group))
            };
            let (seed, seed_group) = kernel("seed");
            let (update, update_group) = kernel("update");
            let constants = |values: [u32; 5]| {
                Some(SceneFabricPushConstants {
                    offset: 0,
                    data: values.into_iter().flat_map(u32::to_ne_bytes).collect(),
                })
            };
            let (scene, template) = scene(QueueClass::Compute);
            let request = SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &[],
                active_passes: ACTIVE,
                fulfilled_dependencies: &[],
                resources: vec![SceneFabricResource::Buffer {
                    key: OUTPUT,
                    frame: 4,
                    version: 0,
                    buffer: &output,
                    offset: 0,
                }],
                recorders: vec![
                    SceneFabricPass::Compute {
                        pass: UPDATE,
                        dispatches: vec![
                            SceneFabricDispatch {
                                pipeline: &update,
                                bind_group: &update_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: constants([2, 3, 4, 5, 6]),
                            },
                            SceneFabricDispatch {
                                pipeline: &update,
                                bind_group: &update_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: constants([1, 2, 3, 4, 5]),
                            },
                        ],
                    },
                    SceneFabricPass::Compute {
                        pass: SEED,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &seed,
                            bind_group: &seed_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: constants([1, 2, 3, 4, 5]),
                        }],
                    },
                ],
            };
            let runtime = prepare_with_templates(&scene, &rhi, request, &[template]).unwrap();
            let (native, recorded) = runtime.encode(&rhi);
            queue.submit([native]);
            // All five words and both UPDATE payloads matter: 15 -> 48 -> 62.
            assert_eq!(read_output(&device, &queue, &output), 62);
            assert!(recorded.allocations().is_empty());
        }

        #[test]
        fn real_multipass_helper_rejects_before_recording_and_encodes_in_compiler_order() {
            use prism_render_architecture::render_fabric::FabricRuntimeError as E;
            use prism_render_architecture::{
                gpu_scene::SceneHandle,
                motion::MotionSource,
                render_fabric::{PayloadRole, RenderObjectId, ResourcePayload},
            };
            use std::panic::{catch_unwind, AssertUnwindSafe};
            let Some((device, queue, adapter, render_device)) = hardware_context() else {
                return;
            };
            let rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let output = render_device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fabric-multipass-output"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&output, 0, &999u32.to_le_bytes());
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fabric-multipass-kernels"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> output: array<u32>;
                     @compute @workgroup_size(1) fn seed() { output[0] = output[1]; }
                     @compute @workgroup_size(1) fn transform() {
                         output[0] = output[0] * output[2] + 2u;
                     }
                     @compute @workgroup_size(1) fn increment() { output[0] += output[3]; }"
                        .into(),
                ),
            });
            let kernel = |entry_point| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(entry_point),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    }],
                });
                (ComputePipeline::from(pipeline), BindGroup::from(group))
            };
            let (seed, seed_group) = kernel("seed");
            let (transform, transform_group) = kernel("transform");
            let (increment, increment_group) = kernel("increment");
            let (scene, template) = scene(QueueClass::Compute);
            let items = [FrameRenderItem {
                object_id: RenderObjectId::Surface(SceneHandle::new(12, 3)),
                domain: RenderDomain::Surface,
                history_key: 8,
                motion_source: MotionSource::Rigid,
                payload: DomainPayload::Resource(ResourcePayload {
                    resource: OUTPUT,
                    lifetime: template.lifetime,
                    size: 16,
                    alignment: template.alignment,
                    layout: template.layout,
                    role: PayloadRole(0),
                }),
            }];
            let request = || SceneFabricRequest {
                frame: 4,
                extent: Extent3d::new_2d(1, 1),
                items: &items,
                active_passes: ACTIVE,
                // Both the data predecessor and control prerequisite execute
                // in this slice. Nothing is declared fulfilled externally.
                fulfilled_dependencies: &[],
                resources: vec![SceneFabricResource::Buffer {
                    key: OUTPUT,
                    frame: 4,
                    version: 0,
                    buffer: &output,
                    offset: 0,
                }],
                recorders: vec![
                    SceneFabricPass::Compute {
                        pass: UPDATE,
                        dispatches: vec![
                            SceneFabricDispatch {
                                pipeline: &transform,
                                bind_group: &transform_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: None,
                            },
                            SceneFabricDispatch {
                                pipeline: &increment,
                                bind_group: &increment_group,
                                bind_group_index: 0,
                                workgroups: [1, 1, 1],
                                push_constants: None,
                            },
                        ],
                    },
                    SceneFabricPass::Compute {
                        pass: SEED,
                        dispatches: vec![SceneFabricDispatch {
                            pipeline: &seed,
                            bind_group: &seed_group,
                            bind_group_index: 0,
                            workgroups: [1, 1, 1],
                            push_constants: None,
                        }],
                    },
                ],
            };

            // Payload metadata must pass the authoritative compiler before a
            // runtime capable of uploading or recording is returned.
            let mut wrong_items = items;
            if let DomainPayload::Resource(payload) = &mut wrong_items[0].payload {
                payload.size = 12;
            }
            let mut invalid_payload = request();
            invalid_payload.items = &wrong_items;
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, invalid_payload, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::PayloadLayoutMismatch(OUTPUT)
                ))
            ));

            let mut missing = request();
            missing.recorders.pop();
            // Even a bad resource cannot bypass the command coverage gate.
            if let SceneFabricResource::Buffer { offset, .. } = &mut missing.resources[0] {
                *offset = u64::MAX;
            }
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, missing, &[template]),
                Err(SceneFabricRuntimeError::Runtime(E::MissingRecorder(SEED)))
            ));
            let mut duplicate = request();
            *duplicate.recorders[1].pass_mut() = UPDATE;
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, duplicate, &[template]),
                Err(SceneFabricRuntimeError::Runtime(E::DuplicateRecorder(
                    UPDATE
                )))
            ));
            let mut unexpected = request();
            *unexpected.recorders[1].pass_mut() = PassKey(u32::MAX);
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, unexpected, &[template]),
                Err(SceneFabricRuntimeError::Runtime(E::UnexpectedRecorder(_)))
            ));
            let mut empty = request();
            empty.recorders[1].dispatches_mut().clear();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, empty, &[template]),
                Err(SceneFabricRuntimeError::EmptyDispatches(SEED))
            ));
            let mut zero = request();
            zero.recorders[0].dispatches_mut()[1].workgroups[2] = 0;
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, zero, &[template]),
                Err(SceneFabricRuntimeError::ZeroWorkgroups {
                    pass: UPDATE,
                    dispatch: 1
                })
            ));
            let limits = device.limits();
            let mut bad_group = request();
            bad_group.recorders[0].dispatches_mut()[1].bind_group_index = limits.max_bind_groups;
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, bad_group, &[template]),
                Err(SceneFabricRuntimeError::BindGroupIndex {
                    pass: UPDATE,
                    dispatch: 1,
                    ..
                })
            ));
            let mut too_many = request();
            too_many.recorders[1].dispatches_mut()[0].workgroups[0] = limits
                .max_compute_workgroups_per_dimension
                .checked_add(1)
                .unwrap();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, too_many, &[template]),
                Err(SceneFabricRuntimeError::WorkgroupsLimit {
                    pass: SEED,
                    dispatch: 0,
                    ..
                })
            ));
            for axis in 0..3 {
                let mut empty_extent = request();
                match axis {
                    0 => empty_extent.extent.width = 0,
                    1 => empty_extent.extent.height = 0,
                    _ => empty_extent.extent.depth_or_array_layers = 0,
                }
                assert!(matches!(
                    prepare_with_templates(&scene, &rhi, empty_extent, &[template]),
                    Err(SceneFabricRuntimeError::EmptyExtent)
                ));
            }

            // The first resource is valid; an invalid later one must reject
            // the entire request before even a temporary table import occurs.
            // Slot generation also detects an import immediately released by
            // error unwinding, which an index-only assertion would miss.
            let probe = rhi.import_buffer(&output);
            let before_preflight = probe.id();
            drop(probe);
            let mut bad_later_resource = request();
            bad_later_resource
                .resources
                .push(SceneFabricResource::Buffer {
                    key: OUTPUT,
                    frame: 3,
                    version: 0,
                    buffer: &output,
                    offset: u64::MAX,
                });
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, bad_later_resource, &[template]),
                Err(SceneFabricRuntimeError::BufferRange { key: OUTPUT, .. })
            ));
            let probe = rhi.import_buffer(&output);
            assert_eq!(probe.id().index(), before_preflight.index());
            assert_eq!(probe.id().generation(), before_preflight.generation() + 1);
            let before_compile = probe.id();
            drop(probe);
            let mut missing_dependency = request();
            missing_dependency.active_passes = &[UPDATE];
            missing_dependency.recorders.pop();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, missing_dependency, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::MissingActiveDependency {
                        pass: UPDATE,
                        dependency: SEED,
                    }
                ))
            ));
            // This failure happens after a real import. Its lease must unwind
            // without leaking a slot or destroying the caller-owned buffer.
            let probe = rhi.import_buffer(&output);
            assert_eq!(probe.id().index(), before_compile.index());
            assert_eq!(probe.id().generation(), before_compile.generation() + 2);
            drop(probe);
            let mut foreign = request();
            foreign.active_passes = &[UPDATE, PassKey(u32::MAX)];
            *foreign.recorders[1].pass_mut() = PassKey(u32::MAX);
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, foreign, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::ForeignActivePass(_)
                ))
            ));
            let mut future = request();
            if let SceneFabricResource::Buffer { frame, .. } = &mut future.resources[0] {
                *frame = 5;
            }
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, future, &[template]),
                Err(SceneFabricRuntimeError::FutureFrame {
                    key: OUTPUT,
                    request: 4,
                    resource: 5,
                })
            ));
            let mut duplicate_resource = request();
            duplicate_resource
                .resources
                .push(SceneFabricResource::Buffer {
                    key: OUTPUT,
                    frame: 4,
                    version: 0,
                    buffer: &output,
                    offset: 0,
                });
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, duplicate_resource, &[template]),
                Err(SceneFabricRuntimeError::DuplicateResource(OUTPUT))
            ));
            let mut no_resource = request();
            no_resource.resources.clear();
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, no_resource, &[template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::MissingBinding {
                        resource: OUTPUT,
                        frame: 4,
                    }
                ))
            ));
            let mut bad_range = request();
            if let SceneFabricResource::Buffer { offset, .. } = &mut bad_range.resources[0] {
                *offset = u64::MAX;
            }
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, bad_range, &[template]),
                Err(SceneFabricRuntimeError::BufferRange { key: OUTPUT, .. })
            ));
            assert!(matches!(
                prepare_with_templates(&scene, &rhi, request(), &[template, template]),
                Err(SceneFabricRuntimeError::Compile(
                    FabricCompileError::DuplicateTemplate(OUTPUT)
                ))
            ));
            let (graphics_scene, _) = self::scene(QueueClass::Graphics);
            assert!(matches!(
                prepare_with_templates(&graphics_scene, &rhi, request(), &[template]),
                Err(SceneFabricRuntimeError::NonComputePass(UPDATE))
            ));
            // No invalid request above may record or submit even its valid
            // first pass. Preparation of a valid request is also side-effect
            // free until its runtime is explicitly encoded.
            assert_eq!(read_output(&device, &queue, &output), 999);

            let imported_ids = |runtime: &SceneFabricRuntime| {
                (
                    runtime._buffer_leases[0].id(),
                    runtime
                        ._pipeline_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect::<Vec<_>>(),
                    runtime
                        ._bind_group_leases
                        .iter()
                        .map(|lease| lease.id())
                        .collect::<Vec<_>>(),
                )
            };
            let assert_released = |(buffer, pipelines, groups): (
                prism_render_driver::BufferId,
                Vec<prism_render_driver::ComputePipelineId>,
                Vec<prism_render_driver::BindGroupId>,
            )| {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| rhi.device().wgpu_buffer(buffer))).is_err()
                );
                for pipeline in pipelines {
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        rhi.device().wgpu_compute_pipeline(pipeline)
                    }))
                    .is_err());
                }
                for group in groups {
                    assert!(catch_unwind(AssertUnwindSafe(|| {
                        rhi.device().wgpu_bind_group(group)
                    }))
                    .is_err());
                }
            };
            let abandoned = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
            let ids = imported_ids(&abandoned);
            drop(abandoned);
            assert_released(ids);

            // Same native device is not sufficient: every bridge has its own
            // ID table. Both encode and submit must fail before graph recording.
            let foreign_rhi =
                SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
            let foreign_buffer = foreign_rhi.import_buffer(&output);
            for submit in [false, true] {
                let runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
                let ids = imported_ids(&runtime);
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    if submit {
                        drop(runtime.submit_tracked(&foreign_rhi));
                    } else {
                        drop(runtime.encode(&foreign_rhi));
                    }
                }))
                .is_err());
                assert_released(ids);
                assert_eq!(
                    foreign_rhi.device().wgpu_buffer(foreign_buffer.id()),
                    *output.deref()
                );
                assert_eq!(read_output(&device, &queue, &output), 999);
            }
            let mut runtime = prepare_with_templates(&scene, &rhi, request(), &[template]).unwrap();
            let ids = imported_ids(&runtime);
            assert_eq!(runtime.graph.declaration().items(), &items);
            assert_eq!(
                runtime
                    .graph
                    .declaration()
                    .passes()
                    .iter()
                    .map(|pass| pass.key())
                    .collect::<Vec<_>>(),
                [SEED, UPDATE]
            );
            assert_eq!(runtime._pipeline_leases.len(), 3);
            assert_eq!(runtime._bind_group_leases.len(), 3);
            assert_eq!(runtime._buffer_leases.len(), 1);
            assert_eq!(read_output(&device, &queue, &output), 999);
            let bytes = |words: [u32; 4]| {
                words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>()
            };
            let authored = bytes([999, 7, 3, 1]);
            assert!(matches!(
                runtime.upload_payload(&rhi, &items[0], &authored[..12]),
                Err(SceneFabricRuntimeError::PayloadUpload(_))
            ));
            assert!(matches!(
                runtime.upload_payload(&rhi, &wrong_items[0], &authored),
                Err(SceneFabricRuntimeError::PayloadUpload(_))
            ));
            assert!(matches!(
                runtime.upload_payload(&foreign_rhi, &items[0], &authored),
                Err(SceneFabricRuntimeError::PayloadUpload(_))
            ));
            assert_eq!(read_output(&device, &queue, &output), 999);
            runtime.upload_payload(&rhi, &items[0], &authored).unwrap();
            let submission = runtime.submit_tracked(&rhi);
            // A same-native-device foreign bridge cannot retire these IDs.
            let mut submission = submission
                .wait_for_completion(&foreign_rhi)
                .expect_err("foreign completion bridge must preserve the owner");
            assert_eq!(rhi.device().wgpu_buffer(ids.0), *output.deref());
            // Unsupported completion evidence is not success. The retry owns
            // the same imports; this simulates token rejection, not device loss.
            let actual_token = std::mem::replace(&mut submission.completion, Box::new(0u32));
            let mut submission = submission
                .wait_for_completion(&rhi)
                .expect_err("invalid completion token must not retire resources");
            assert_eq!(rhi.device().wgpu_buffer(ids.0), *output.deref());
            for &pipeline in &ids.1 {
                let _ = rhi.device().wgpu_compute_pipeline(pipeline);
            }
            for &group in &ids.2 {
                let _ = rhi.device().wgpu_bind_group(group);
            }
            submission.completion = actual_token;
            assert_eq!(submission.plan().pass_count(), 2);
            assert!(submission.allocations().is_empty(), "imported-only graph");
            submission
                .wait_for_completion(&rhi)
                .expect("real completion");
            assert_released(ids);
            assert_eq!(read_output(&device, &queue, &output), 24);

            // Same semantic resource, different frame/version and authored
            // bytes. A constant-output kernel or a disconnected payload fails.
            let mut next = request();
            next.frame = 5;
            if let SceneFabricResource::Buffer { frame, version, .. } = &mut next.resources[0] {
                *frame = 5;
                *version = 9;
            }
            let mut runtime = prepare_with_templates(&scene, &rhi, next, &[template]).unwrap();
            let ids = imported_ids(&runtime);
            runtime
                .upload_payload(&rhi, &items[0], &bytes([999, 11, 5, 4]))
                .unwrap();
            // Only the unified runtime's import leases now retain these native
            // command handles. They must survive all passes through encoding.
            drop((
                seed,
                seed_group,
                transform,
                transform_group,
                increment,
                increment_group,
            ));
            drop(shader);
            let submission = runtime.submit_tracked(&rhi);
            // The pending owner, rather than a fire-and-forget queue call,
            // retains the imported table leases until completion is proven.
            assert_eq!(
                rhi.device().wgpu_buffer(ids.0),
                *output.deref(),
                "imported buffer must remain leased before completion"
            );
            assert_eq!(submission.frame(), 5);
            // Dropping a pending owner is also completion-gated on the queue
            // it retained. It must not merely unregister IDs after submit.
            drop(submission);
            assert_released(ids);
            assert_eq!(read_output(&device, &queue, &output), 61);
        }
    }
}
