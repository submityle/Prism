//! Typed lowering to the existing graph runtime, not another executor.
use super::*;
use crate::frame_graph::{Barrier, ResourceId};
use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::fmt;
use prism_render_driver::{Extent3d, RenderDevice, RenderQueue};
use prism_render_graph::{
    full_range, BufferDesc, BufferHandle, BufferUse, CompileError, ExecuteContext, ExecutionPlan,
    ImportedBuffer, ImportedTexture, LogicalResource, PassBuilder, PassFlags, RenderGraph,
    SizeClass, TextureDesc, TextureHandle, TextureUse,
};

/// Entry/exit state and default view supplied by the external resource owner.
/// The semantic binding remains authoritative for physical ID, frame and layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeImport {
    Buffer {
        resource: ResourceId,
        imported: ImportedBuffer,
    },
    Texture {
        resource: ResourceId,
        imported: ImportedTexture,
    },
}
impl RuntimeImport {
    pub const fn resource(self) -> ResourceId {
        match self {
            Self::Buffer { resource, .. } | Self::Texture { resource, .. } => resource,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FabricRuntimeError {
    DuplicateRecorder(PassKey),
    UnexpectedRecorder(PassKey),
    MissingRecorder(PassKey),
    DuplicateImport(ResourceId),
    UnexpectedImport(ResourceId),
    MissingImport(ResourceId),
    InvalidImport(ResourceId),
    UnsupportedAccess {
        pass: PassKey,
        resource: ResourceKey,
    },
    VersionMismatch {
        pass: PassKey,
        resource: ResourceKey,
    },
    PlanMismatch,
    Graph(CompileError),
}
impl fmt::Display for FabricRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "render fabric runtime error: {self:?}")
    }
}
impl std::error::Error for FabricRuntimeError {}
impl From<CompileError> for FabricRuntimeError {
    fn from(error: CompileError) -> Self {
        Self::Graph(error)
    }
}

/// Graph-owned typed handle; buffer offsets are retained, never rounded to zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeResourceHandle {
    Buffer {
        handle: BufferHandle,
        offset: u64,
        bytes: u64,
    },
    Texture(TextureHandle),
}
impl RuntimeResourceHandle {
    fn version(self) -> u32 {
        match self {
            Self::Buffer { handle, .. } => handle.version(),
            Self::Texture(handle) => handle.version(),
        }
    }
    fn namespace_index(self) -> (u8, u32) {
        match self {
            Self::Texture(handle) => (0, handle.resource().get()),
            Self::Buffer { handle, .. } => (1, handle.resource().get()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeResourceUse {
    pub bound: BoundResourceUse,
    pub handle: RuntimeResourceHandle,
}

/// A recorder gets only its selected pass and already-lowered accesses.
/// It cannot declare resources, replace accesses, or reselect variants.
pub struct RuntimePassContext<'a, 'graph> {
    pass: &'a PassSpec,
    items: &'a [FrameRenderItem],
    accesses: &'a [RuntimeResourceUse],
    commands: &'a mut ExecuteContext<'graph>,
}
impl<'a, 'graph> RuntimePassContext<'a, 'graph> {
    pub const fn pass(&self) -> &PassSpec {
        self.pass
    }
    pub const fn accesses(&self) -> &[RuntimeResourceUse] {
        self.accesses
    }

    /// Subsystems see only their declared domain. Domainless services/responses
    /// may consume all items; no implicit Surface domain is introduced.
    pub fn items(&self) -> impl Iterator<Item = &FrameRenderItem> {
        self.items.iter().filter(|item| {
            self.pass
                .domain()
                .is_none_or(|domain| item.domain == domain)
        })
    }

    /// Resolve the access ordinal in the original PassTemplate order.
    pub fn buffer(&self, access: usize) -> Option<(BufferId, u64, u64)> {
        match self.accesses.get(access)?.handle {
            RuntimeResourceHandle::Buffer {
                handle,
                offset,
                bytes,
            } => Some((self.commands.buffer(handle), offset, bytes)),
            RuntimeResourceHandle::Texture(_) => None,
        }
    }
    pub fn texture(&self, access: usize) -> Option<TextureId> {
        match self.accesses.get(access)?.handle {
            RuntimeResourceHandle::Texture(handle) => Some(self.commands.texture(handle)),
            RuntimeResourceHandle::Buffer { .. } => None,
        }
    }
    pub fn texture_view(&self, access: usize) -> Option<prism_render_driver::TextureViewId> {
        match self.accesses.get(access)?.handle {
            RuntimeResourceHandle::Texture(handle) => Some(self.commands.texture_view(handle)),
            RuntimeResourceHandle::Buffer { .. } => None,
        }
    }
    /// Existing driver-command recorder, not a second graph setup context.
    pub fn commands(&mut self) -> &mut ExecuteContext<'graph> {
        self.commands
    }
}

type Recorder = Box<dyn for<'a, 'graph> FnOnce(&mut RuntimePassContext<'a, 'graph>) + Send>;

/// Move-only implementation selected by stable PassKey, with no setup closure.
pub struct RuntimePassBinding {
    key: PassKey,
    record: Recorder,
}
impl RuntimePassBinding {
    pub fn new<F>(key: PassKey, record: F) -> Self
    where
        F: for<'a, 'graph> FnOnce(&mut RuntimePassContext<'a, 'graph>) + Send + 'static,
    {
        Self {
            key,
            record: Box::new(record),
        }
    }
}

/// Prepared existing RenderGraph plus its verified declaration provenance.
/// All selected passes remain resident, including empty-domain composition.
pub struct RuntimeFeatureGraph {
    graph: RenderGraph,
    plan: ExecutionPlan,
    declaration: CompiledFeatureGraph,
}
impl RuntimeFeatureGraph {
    pub const fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
    pub const fn declaration(&self) -> &CompiledFeatureGraph {
        &self.declaration
    }

    /// Delegates realization, command recording and submission to RenderGraph.
    /// Its auto-sync path submits one command buffer; logical queue batches do
    /// not promise native multi-queue fences or external history publication.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        self,
        device: &D,
        queue: &Q,
    ) -> Result<(), CompileError> {
        self.graph.execute(device, queue)
    }
}

impl CompiledFeatureGraph {
    /// The production lowering seam. Import and recorder coverage is exact;
    /// no GPU allocation, user recorder or queue submission occurs on error.
    pub fn prepare_runtime(
        self,
        extent: Extent3d,
        imports: &[RuntimeImport],
        recorders: Vec<RuntimePassBinding>,
    ) -> Result<RuntimeFeatureGraph, FabricRuntimeError> {
        let mut callbacks = BTreeMap::new();
        for binding in recorders {
            if !self.passes().iter().any(|pass| pass.key() == binding.key) {
                return Err(FabricRuntimeError::UnexpectedRecorder(binding.key));
            }
            if callbacks.insert(binding.key, binding.record).is_some() {
                return Err(FabricRuntimeError::DuplicateRecorder(binding.key));
            }
        }
        for pass in self.passes() {
            if !callbacks.contains_key(&pass.key()) {
                return Err(FabricRuntimeError::MissingRecorder(pass.key()));
            }
            validate_pass(pass, self.resources())?;
        }
        let mut imported = BTreeMap::new();
        for &entry in imports {
            let id = entry.resource();
            let resource = self
                .resources()
                .get(id.0 as usize)
                .filter(|resource| resource.id == id && resource.external.is_some())
                .ok_or(FabricRuntimeError::UnexpectedImport(id))?;
            if !valid_import(entry, resource) {
                return Err(FabricRuntimeError::InvalidImport(id));
            }
            if imported.insert(id.0, entry).is_some() {
                return Err(FabricRuntimeError::DuplicateImport(id));
            }
        }
        let mut graph = RenderGraph::with_limits(extent, self.limits());
        let mut handles = vec![None; self.resources().len()];
        for resource in self
            .resources()
            .iter()
            .filter(|resource| resource.external.is_some())
        {
            let entry = imported
                .get(&resource.id.0)
                .ok_or(FabricRuntimeError::MissingImport(resource.id))?;
            let name = resource.descriptor.name.to_string();
            let handle = match *entry {
                RuntimeImport::Buffer { imported, .. } => {
                    let ResourceLayout::Buffer { bytes, usage } = resource.template.layout else {
                        unreachable!("validated import")
                    };
                    let PhysicalResource::Buffer { offset, .. } =
                        resource.external.expect("external").physical
                    else {
                        unreachable!("validated import")
                    };
                    RuntimeResourceHandle::Buffer {
                        handle: graph.try_import_buffer(
                            name,
                            BufferDesc::new(bytes, usage),
                            imported,
                        )?,
                        offset,
                        bytes,
                    }
                }
                RuntimeImport::Texture { imported, .. } => {
                    RuntimeResourceHandle::Texture(graph.try_import_texture(
                        name,
                        texture_desc(resource.template.layout),
                        imported,
                    )?)
                }
            };
            handles[resource.id.0 as usize] = Some(handle);
        }
        let items: Arc<[FrameRenderItem]> = self.items().into();
        let mut lowered = Vec::new();
        for (index, pass) in self.passes().iter().enumerate() {
            let callback = callbacks
                .remove(&pass.key())
                .expect("validated recorder coverage");
            let mut setup_error = None;
            let pass_items = Arc::clone(&items);
            let pass_spec = pass.clone();
            let mut pass_accesses = Vec::new();
            let setup = |builder: &mut PassBuilder<'_>| {
                // Creation adds no access and no synthetic pass. All transient
                // allocation still belongs to the existing graph runtime.
                let result = (|| -> Result<(), FabricRuntimeError> {
                    if index == 0 {
                        create_transients(builder, self.resources(), &mut handles)?;
                    }
                    for &bound in pass.accesses() {
                        let handle = &mut handles[bound.resource.0 as usize];
                        let next = lower_access(
                            builder,
                            handle.expect("bound resource"),
                            bound.declaration.access,
                        )?;
                        if next.version() != bound.ssa_version {
                            return Err(FabricRuntimeError::VersionMismatch {
                                pass: pass.key(),
                                resource: bound.declaration.key,
                            });
                        }
                        *handle = Some(next);
                        pass_accesses.push(RuntimeResourceUse {
                            bound,
                            handle: next,
                        });
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    setup_error = Some(error);
                }
                lowered.push(pass_accesses.clone());
                move |commands: &mut ExecuteContext<'_>| {
                    callback(&mut RuntimePassContext {
                        pass: &pass_spec,
                        items: &pass_items,
                        accesses: &pass_accesses,
                        commands,
                    })
                }
            };
            let flags = PassFlags::NEVER_CULL;
            match pass.template().queue {
                QueueClass::Graphics => {
                    graph.try_add_raster_pass(pass.descriptor().name.to_string(), flags, setup)?
                }
                QueueClass::Compute => graph.try_add_compute_pass(
                    pass.descriptor().name.to_string(),
                    flags | PassFlags::ASYNC_COMPUTE,
                    setup,
                )?,
                QueueClass::Transfer => {
                    graph.try_add_transfer_pass(pass.descriptor().name.to_string(), flags, setup)?
                }
            }
            if let Some(error) = setup_error {
                return Err(error);
            }
        }
        let plan = graph.compile()?;
        verify_plan(&self, &plan, &handles, &lowered)?;
        Ok(RuntimeFeatureGraph {
            graph,
            plan,
            declaration: self,
        })
    }
}

fn valid_import(entry: RuntimeImport, resource: &BoundResource) -> bool {
    let binding = resource.external.expect("external import");
    match (entry, binding.physical, binding.layout) {
        (
            RuntimeImport::Buffer { imported, .. },
            PhysicalResource::Buffer { id, .. },
            ResourceLayout::Buffer { .. },
        ) => imported.buffer == id,
        (
            RuntimeImport::Texture { imported, .. },
            PhysicalResource::Texture(id),
            ResourceLayout::Texture {
                extent,
                dimension,
                mip_levels,
                ..
            },
        ) => {
            imported.texture == id
                && imported.mip_levels == mip_levels
                && imported.array_layers
                    == if dimension == TextureDimension::D3 {
                        1
                    } else {
                        extent[2]
                    }
        }
        _ => false,
    }
}

fn validate_pass(pass: &PassSpec, resources: &[BoundResource]) -> Result<(), FabricRuntimeError> {
    let mut depths = 0;
    for usage in pass.accesses() {
        let attachment = matches!(
            usage.declaration.access,
            AccessKind::ColorAttachment | AccessKind::DepthAttachment
        );
        depths += usize::from(usage.declaration.access == AccessKind::DepthAttachment);
        let invalid_attachment = attachment
            && !matches!(
                resources[usage.resource.0 as usize].template.layout,
                ResourceLayout::Texture {
                    extent: [_, _, 1],
                    dimension: TextureDimension::D2,
                    mip_levels: 1,
                    ..
                }
            );
        if attachment && pass.template().queue != QueueClass::Graphics
            || invalid_attachment
            || depths > 1
            || pass.template().queue == QueueClass::Transfer
                && !matches!(
                    usage.declaration.access,
                    AccessKind::TransferRead | AccessKind::TransferWrite
                )
        {
            return Err(FabricRuntimeError::UnsupportedAccess {
                pass: pass.key(),
                resource: usage.declaration.key,
            });
        }
    }
    Ok(())
}

fn texture_desc(layout: ResourceLayout) -> TextureDesc {
    let ResourceLayout::Texture {
        extent: [width, height, depth_or_array_layers],
        dimension,
        format,
        mip_levels,
        samples,
        usage,
    } = layout
    else {
        unreachable!("validated texture layout")
    };
    TextureDesc {
        size: SizeClass::Absolute {
            width,
            height,
            depth_or_array_layers,
        },
        dimension,
        format,
        mip_level_count: mip_levels,
        sample_count: samples,
        usage,
    }
}

fn create_transients(
    builder: &mut PassBuilder<'_>,
    resources: &[BoundResource],
    handles: &mut [Option<RuntimeResourceHandle>],
) -> Result<(), CompileError> {
    for resource in resources
        .iter()
        .filter(|resource| resource.external.is_none())
    {
        let name = resource.descriptor.name.to_string();
        handles[resource.id.0 as usize] = Some(match resource.template.layout {
            ResourceLayout::Buffer { bytes, usage } => RuntimeResourceHandle::Buffer {
                handle: builder.try_create_buffer(name, BufferDesc::new(bytes, usage))?,
                offset: 0,
                bytes,
            },
            layout @ ResourceLayout::Texture { .. } => RuntimeResourceHandle::Texture(
                builder.try_create_texture(name, texture_desc(layout))?,
            ),
        });
    }
    Ok(())
}

fn lower_access(
    builder: &mut PassBuilder<'_>,
    handle: RuntimeResourceHandle,
    access: AccessKind,
) -> Result<RuntimeResourceHandle, CompileError> {
    Ok(match handle {
        RuntimeResourceHandle::Buffer {
            handle,
            offset,
            bytes,
        } => {
            let usage = match access {
                AccessKind::StorageRead => BufferUse::StorageRead,
                AccessKind::StorageWrite => BufferUse::StorageWrite,
                AccessKind::IndirectRead => BufferUse::Indirect,
                AccessKind::TransferRead => BufferUse::CopySrc,
                AccessKind::TransferWrite => BufferUse::CopyDst,
                _ => unreachable!("all-variant layout validation"),
            };
            RuntimeResourceHandle::Buffer {
                handle: builder.try_use_buffer(handle, usage)?,
                offset,
                bytes,
            }
        }
        RuntimeResourceHandle::Texture(handle) => {
            let usage = match access {
                AccessKind::SampledRead => TextureUse::Sampled,
                AccessKind::StorageRead => TextureUse::StorageRead,
                AccessKind::StorageWrite => TextureUse::StorageWrite,
                AccessKind::ColorAttachment => TextureUse::ColorAttachment,
                AccessKind::DepthAttachment => TextureUse::DepthAttachment,
                AccessKind::TransferRead => TextureUse::CopySrc,
                AccessKind::TransferWrite => TextureUse::CopyDst,
                AccessKind::IndirectRead => unreachable!("all-variant layout validation"),
            };
            RuntimeResourceHandle::Texture(builder.try_access_texture(
                handle,
                usage,
                full_range(),
            )?)
        }
    })
}

// Projection only: the authority solves both graphs. Compare every logical
// hazard after undoing the typed runtime's texture-first access namespace.
fn verify_plan(
    declaration: &CompiledFeatureGraph,
    plan: &ExecutionPlan,
    handles: &[Option<RuntimeResourceHandle>],
    lowered: &[Vec<RuntimeResourceUse>],
) -> Result<(), FabricRuntimeError> {
    if plan.order
        != declaration
            .plan()
            .execution_order
            .iter()
            .map(|id| id.0 as usize)
            .collect::<Vec<_>>()
        || plan.alive.iter().any(|alive| !alive)
        || plan.queue_batches != declaration.plan().queue_batches
    {
        return Err(FabricRuntimeError::PlanMismatch);
    }
    let resources: BTreeMap<_, _> = handles
        .iter()
        .enumerate()
        .filter_map(|(id, handle)| {
            handle.map(|handle| (handle.namespace_index(), ResourceId(id as u32)))
        })
        .collect();
    let ordinals: Vec<Vec<_>> = lowered
        .iter()
        .map(|accesses| {
            accesses
                .iter()
                .filter(|a| matches!(a.handle, RuntimeResourceHandle::Texture(_)))
                .chain(
                    accesses
                        .iter()
                        .filter(|a| matches!(a.handle, RuntimeResourceHandle::Buffer { .. })),
                )
                .collect()
        })
        .collect();
    let mut projected = Vec::new();
    for barrier in &plan.logical_barriers {
        let namespace_index = match barrier.resource {
            LogicalResource::Texture(id) => (0, id.get()),
            LogicalResource::Buffer(id) => (1, id.get()),
        };
        projected.push(Barrier {
            resource: resources[&namespace_index],
            source: crate::frame_graph::PassId(barrier.source as u32),
            destination: crate::frame_graph::PassId(barrier.destination as u32),
            before: ordinals[barrier.source][barrier.before]
                .bound
                .declaration
                .access,
            after: ordinals[barrier.destination][barrier.after]
                .bound
                .declaration
                .access,
            queue_transfer: barrier.queue_transfer,
        });
    }
    let key = |barrier: &Barrier| {
        (
            barrier.resource.0,
            barrier.source.0,
            barrier.destination.0,
            barrier.before as u8,
            barrier.after as u8,
            barrier.queue_transfer,
        )
    };
    projected.sort_by_key(key);
    let mut expected = declaration.plan().barriers.clone();
    expected.sort_by_key(key);
    if projected != expected {
        return Err(FabricRuntimeError::PlanMismatch);
    }
    Ok(())
}
