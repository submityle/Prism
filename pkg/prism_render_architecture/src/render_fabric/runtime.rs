//! Typed lowering to the existing graph runtime, not another executor.
use super::*;
use crate::frame_graph::ResourceId;
use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::fmt;
use prism_render_driver::{Extent3d, RenderDevice, RenderQueue};
use prism_render_graph::{
    full_range, BufferDesc, BufferHandle, CompileError, ExecuteContext, ExecutionPlan,
    ImportedBuffer, ImportedTexture, LogicalResource, PassBuilder, PassFlags, PassKind,
    PreparedGraph, RecordedGraph, RenderGraph, SizeClass, TextureDesc, TextureHandle,
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

/// Object imports authenticate the complete compiled semantic address and
/// storage description, including buffer range and external content version.
/// A raw ResourceId/physical ID alone cannot distinguish suballocated objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectRuntimeImport {
    pub address: ResourceAddress,
    pub binding: FrameResourceBinding,
    pub imported: RuntimeImport,
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
    InvalidObjectImport(ResourceAddress),
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeResourceUse {
    pub bound: BoundResourceUse,
    pub handle: RuntimeResourceHandle,
}

/// One immutable frame allocation is shared with the compiled declaration.
/// Stable domain indices cost one usize per item, rather than another payload
/// copy or a full-frame scan every time a subsystem recorder visits its items.
pub(super) struct RuntimeFrameItems {
    items: Arc<[FrameRenderItem]>,
    indices: Box<[usize]>,
    offsets: [usize; 6],
}

impl RuntimeFrameItems {
    pub(super) fn new(items: Arc<[FrameRenderItem]>) -> Self {
        let mut offsets = [0; 6];
        for item in items.iter() {
            offsets[Self::domain_index(item.domain) + 1] += 1;
        }
        for domain in 1..offsets.len() {
            offsets[domain] += offsets[domain - 1];
        }
        let mut next = offsets;
        let mut indices = vec![0; items.len()].into_boxed_slice();
        for (index, item) in items.iter().enumerate() {
            let domain = Self::domain_index(item.domain);
            indices[next[domain]] = index;
            next[domain] += 1;
        }
        Self {
            items,
            indices,
            offsets,
        }
    }

    const fn domain_index(domain: RenderDomain) -> usize {
        // Exhaustive on purpose: a new domain must update the index contract,
        // not silently rely on enum discriminants or alias another domain.
        match domain {
            RenderDomain::Surface => 0,
            RenderDomain::Particle => 1,
            RenderDomain::Volume => 2,
            RenderDomain::Hair => 3,
            RenderDomain::Decal => 4,
        }
    }

    pub(super) fn iter(
        &self,
        domain: Option<RenderDomain>,
    ) -> impl ExactSizeIterator<Item = &FrameRenderItem> + DoubleEndedIterator + core::iter::FusedIterator
    {
        let indices = domain.map(|domain| {
            let domain = Self::domain_index(domain);
            &self.indices[self.offsets[domain]..self.offsets[domain + 1]]
        });
        let len = indices.map_or(self.items.len(), <[usize]>::len);
        (0..len).map(move |index| &self.items[indices.map_or(index, |indices| indices[index])])
    }
}

/// A recorder gets only its selected pass and already-lowered accesses.
/// It cannot declare resources, replace accesses, or reselect variants.
pub struct RuntimePassContext<'a, 'graph> {
    pass: &'a PassSpec,
    items: &'a RuntimeFrameItems,
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
    /// The index is built once per prepared frame; repeated visits allocate
    /// nothing and touch only this domain's items, in their original order.
    pub fn items(
        &self,
    ) -> impl ExactSizeIterator<Item = &FrameRenderItem> + DoubleEndedIterator + core::iter::FusedIterator
    {
        self.items.iter(self.pass.domain())
    }

    /// Resolve the original template ordinal with explicit shared/object scope.
    /// An absent instance never falls back to a shared or first-object access.
    pub fn access(
        &self,
        identity: Option<ObjectResourceIdentity>,
        ordinal: usize,
    ) -> Option<&RuntimeResourceUse> {
        self.accesses
            .get(self.pass.access_index(identity, ordinal)?)
    }
    pub fn shared_access(&self, ordinal: usize) -> Option<&RuntimeResourceUse> {
        self.access(None, ordinal)
    }
    pub fn object_access(
        &self,
        identity: ObjectResourceIdentity,
        ordinal: usize,
    ) -> Option<&RuntimeResourceUse> {
        self.access(Some(identity), ordinal)
    }

    /// Shared-only access in the original PassTemplate order.
    pub fn buffer(&self, access: usize) -> Option<(BufferId, u64, u64)> {
        self.resolve_buffer(self.shared_access(access)?)
    }
    pub fn object_buffer(
        &self,
        identity: ObjectResourceIdentity,
        ordinal: usize,
    ) -> Option<(BufferId, u64, u64)> {
        self.resolve_buffer(self.object_access(identity, ordinal)?)
    }
    fn resolve_buffer(&self, access: &RuntimeResourceUse) -> Option<(BufferId, u64, u64)> {
        match access.handle {
            RuntimeResourceHandle::Buffer {
                handle,
                offset,
                bytes,
            } => Some((self.commands.buffer(handle), offset, bytes)),
            RuntimeResourceHandle::Texture(_) => None,
        }
    }
    pub fn texture(&self, access: usize) -> Option<TextureId> {
        self.resolve_texture(self.shared_access(access)?)
    }
    pub fn object_texture(
        &self,
        identity: ObjectResourceIdentity,
        ordinal: usize,
    ) -> Option<TextureId> {
        self.resolve_texture(self.object_access(identity, ordinal)?)
    }
    fn resolve_texture(&self, access: &RuntimeResourceUse) -> Option<TextureId> {
        match access.handle {
            RuntimeResourceHandle::Texture(handle) => Some(self.commands.texture(handle)),
            RuntimeResourceHandle::Buffer { .. } => None,
        }
    }
    pub fn texture_view(&self, access: usize) -> Option<prism_render_driver::TextureViewId> {
        self.resolve_texture_view(self.shared_access(access)?)
    }
    pub fn object_texture_view(
        &self,
        identity: ObjectResourceIdentity,
        ordinal: usize,
    ) -> Option<prism_render_driver::TextureViewId> {
        self.resolve_texture_view(self.object_access(identity, ordinal)?)
    }
    fn resolve_texture_view(
        &self,
        access: &RuntimeResourceUse,
    ) -> Option<prism_render_driver::TextureViewId> {
        match access.handle {
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
    graph: PreparedGraph,
    declaration: CompiledFeatureGraph,
}
impl RuntimeFeatureGraph {
    pub fn plan(&self) -> &ExecutionPlan {
        self.graph.plan()
    }
    pub const fn declaration(&self) -> &CompiledFeatureGraph {
        &self.declaration
    }

    /// Transfers the exact verified plan to the existing graph submitter.
    /// This consumes the Fabric wrapper without recompiling, realizing resources,
    /// or recording commands. Completion and imported-resource ownership remain
    /// the submitter's responsibility.
    #[must_use]
    pub fn into_prepared(self) -> PreparedGraph {
        self.graph
    }

    /// Realizes and records with the exact plan verified during preparation.
    /// No recompilation or submission occurs. The external frame owner submits
    /// the commands on the creating device and retains the returned allocation
    /// owner until real completion; imports remain externally owned.
    #[must_use]
    pub fn record<D: RenderDevice>(self, device: &D) -> RecordedGraph {
        self.graph.record(device)
    }

    /// Records and submits one ordered command buffer on an auto-sync queue.
    /// Submission is not completion: retain the returned owner until the real
    /// queue confirms completion, then release it on the creating device.
    /// Logical queue batches do not imply native multi-queue fences or history
    /// publication. Use `record` when an external renderer owns submission.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(self, device: &D, queue: &Q) -> RecordedGraph {
        let recorded = self.record(device);
        queue.submit(core::slice::from_ref(recorded.commands()));
        recorded
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
        self.prepare_runtime_with_object_imports(extent, imports, &[], recorders)
    }

    /// Explicit object-import entrypoint. Recorder coverage remains per PassKey,
    /// not per instance; existing graph origin/SSA/plan checks stay authoritative.
    pub fn prepare_runtime_with_object_imports(
        self,
        extent: Extent3d,
        shared_imports: &[RuntimeImport],
        object_imports: &[ObjectRuntimeImport],
        recorders: Vec<RuntimePassBinding>,
    ) -> Result<RuntimeFeatureGraph, FabricRuntimeError> {
        let mut imports = Vec::with_capacity(shared_imports.len() + object_imports.len());
        for &entry in shared_imports {
            if self
                .resources()
                .get(entry.resource().0 as usize)
                .is_some_and(|resource| resource.identity.is_some())
            {
                return Err(FabricRuntimeError::UnexpectedImport(entry.resource()));
            }
            imports.push(entry);
        }
        for entry in object_imports {
            let id = entry.imported.resource();
            if entry.address.identity.is_none()
                || self.resource_id_at(entry.address) != Some(id)
                || self
                    .resources()
                    .get(id.0 as usize)
                    .is_none_or(|resource| resource.external != Some(entry.binding))
            {
                return Err(FabricRuntimeError::InvalidObjectImport(entry.address));
            }
            imports.push(entry.imported);
        }
        // Resolve stable keys once, then keep move-only recorders in pass order.
        // Coverage is O((P + R) log P), not a full pass scan per recorder.
        let pass_indices: BTreeMap<_, _> = self
            .passes()
            .iter()
            .enumerate()
            .map(|(index, pass)| (pass.key(), index))
            .collect();
        let mut callbacks: Vec<Option<Recorder>> = (0..self.passes().len()).map(|_| None).collect();
        for binding in recorders {
            let &index = pass_indices
                .get(&binding.key)
                .ok_or(FabricRuntimeError::UnexpectedRecorder(binding.key))?;
            if callbacks[index].is_some() {
                return Err(FabricRuntimeError::DuplicateRecorder(binding.key));
            }
            callbacks[index] = Some(binding.record);
        }
        for (pass, callback) in self.passes().iter().zip(&callbacks) {
            if callback.is_none() {
                return Err(FabricRuntimeError::MissingRecorder(pass.key()));
            }
            validate_pass(pass, self.resources())?;
        }
        let mut imported = BTreeMap::new();
        for &entry in &imports {
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
        let mut versions = vec![BTreeMap::new(); self.resources().len()];
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
            versions[resource.id.0 as usize].insert(0, handle);
        }
        let items = Arc::new(RuntimeFrameItems::new(self.shared_items()));
        for (index, (pass, callback)) in self.passes().iter().zip(callbacks).enumerate() {
            let callback = callback.expect("validated recorder coverage");
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
                        for (resource, handle) in handles.iter().enumerate() {
                            versions[resource].insert(0, handle.expect("created resource"));
                        }
                    }
                    for &bound in pass.accesses() {
                        let resource = bound.resource.0 as usize;
                        let input = versions[resource]
                            .get(&bound.input_ssa_version)
                            .copied()
                            .ok_or(FabricRuntimeError::VersionMismatch {
                                pass: pass.key(),
                                resource: bound.declaration.key,
                            })?;
                        let next = lower_access(builder, input, bound.declaration.access)?;
                        if next.version() != bound.ssa_version {
                            return Err(FabricRuntimeError::VersionMismatch {
                                pass: pass.key(),
                                resource: bound.declaration.key,
                            });
                        }
                        if bound.declaration.access.writes() {
                            versions[resource].insert(next.version(), next);
                            handles[resource] = Some(next);
                        }
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
            let (kind, flags) = match pass.template().queue {
                QueueClass::Graphics => (PassKind::Raster, flags),
                QueueClass::Compute => (PassKind::Compute, flags | PassFlags::ASYNC_COMPUTE),
                QueueClass::Transfer => (PassKind::Transfer, flags),
            };
            graph.try_add_pass_with_dependencies(
                pass.descriptor().name.to_string(),
                kind,
                flags,
                pass.descriptor()
                    .depends_on
                    .iter()
                    .map(|id| id.0 as usize)
                    .collect(),
                setup,
            )?;
            if let Some(error) = setup_error {
                return Err(error);
            }
        }
        let graph = graph.prepare()?;
        verify_plan(&self, graph.plan(), &handles)?;
        Ok(RuntimeFeatureGraph {
            graph,
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
        let access = usage.declaration.access;
        let attachment = access.is_attachment();
        depths += usize::from(matches!(
            access,
            AccessKind::DepthAttachment | AccessKind::DepthAttachmentRead
        ));
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
        if (attachment || access == AccessKind::IndexRead)
            && pass.template().queue != QueueClass::Graphics
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
            let usage = access.buffer_use().expect("all-variant layout validation");
            RuntimeResourceHandle::Buffer {
                handle: builder.try_use_buffer(handle, usage)?,
                offset,
                bytes,
            }
        }
        RuntimeResourceHandle::Texture(handle) => {
            let usage = access.texture_use().expect("all-variant layout validation");
            RuntimeResourceHandle::Texture(builder.try_access_texture(
                handle,
                usage,
                full_range(),
            )?)
        }
    })
}

// Projection only: the authority solves both graphs. Namespace-local resource
// indices and texture-first access ordinals are not semantic identities: the
// runtime registers imports first and uses actual layouts, while declaration
// lowering infers texture kinds from sampling/attachment uses.
pub(super) fn verify_plan(
    declaration: &CompiledFeatureGraph,
    plan: &ExecutionPlan,
    handles: &[Option<RuntimeResourceHandle>],
) -> Result<(), FabricRuntimeError> {
    let expected = declaration.plan();
    if plan.declared_pass_count != expected.declared_pass_count
        || plan.order != expected.order
        || plan.alive != expected.alive
        || plan.position_by_pass != expected.position_by_pass
        || plan.queue_batches != expected.queue_batches
    {
        return Err(FabricRuntimeError::PlanMismatch);
    }

    let mut declared_textures = vec![false; declaration.resources().len()];
    for usage in declaration.passes().iter().flat_map(PassSpec::accesses) {
        if usage.declaration.access.requires_texture() {
            declared_textures[usage.resource.0 as usize] = true;
        }
    }
    // (is_texture, namespace-local index), indexed by semantic ResourceId.
    let mut next_texture = 0;
    let mut next_buffer = 0;
    let declared_resources: Vec<_> = declared_textures
        .into_iter()
        .map(|is_texture| {
            let next = if is_texture {
                &mut next_texture
            } else {
                &mut next_buffer
            };
            let identity = (is_texture, *next);
            *next += 1;
            identity
        })
        .collect();
    let runtime_resources = handles
        .iter()
        .map(|handle| match handle {
            Some(RuntimeResourceHandle::Buffer { handle, .. }) => {
                Ok((false, handle.resource().get()))
            }
            Some(RuntimeResourceHandle::Texture(handle)) => Ok((true, handle.resource().get())),
            None => Err(FabricRuntimeError::PlanMismatch),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if semantic_accesses(declaration.passes(), plan, &runtime_resources)?
        != semantic_accesses(declaration.passes(), expected, &declared_resources)?
    {
        return Err(FabricRuntimeError::PlanMismatch);
    }
    if semantic_hazards(declaration.passes(), plan, &runtime_resources)?
        != semantic_hazards(declaration.passes(), expected, &declared_resources)?
    {
        return Err(FabricRuntimeError::PlanMismatch);
    }
    Ok(())
}

type SemanticAccess = (u32, u32, Option<u32>);

fn semantic_accesses(
    passes: &[PassSpec],
    plan: &ExecutionPlan,
    resources: &[(bool, u32)],
) -> Result<Vec<Vec<SemanticAccess>>, FabricRuntimeError> {
    if plan.accesses.len() != passes.len() {
        return Err(FabricRuntimeError::PlanMismatch);
    }
    passes
        .iter()
        .zip(&plan.accesses)
        .map(|(pass, compiled)| {
            let mut order: Vec<_> = pass.accesses().iter().enumerate().collect();
            order.sort_by_key(|(_, usage)| !resources[usage.resource.0 as usize].0);
            if order.len() != compiled.len() {
                return Err(FabricRuntimeError::PlanMismatch);
            }
            let mut result = vec![(0, 0, None); order.len()];
            for ((ordinal, usage), access) in order.into_iter().zip(compiled) {
                let identity = match access.resource {
                    LogicalResource::Texture(index) => (true, index.get()),
                    LogicalResource::Buffer(index) => (false, index.get()),
                };
                if resources[usage.resource.0 as usize] != identity {
                    return Err(FabricRuntimeError::PlanMismatch);
                }
                result[ordinal] = (
                    usage.resource.0,
                    access.input_version,
                    access.output_version,
                );
            }
            Ok(result)
        })
        .collect()
}

// Original pass/access ordinals, semantic resource ID, and queue transfer.
// Sorting removes only namespace-dependent iteration order, not multiplicity.
type SemanticHazard = (usize, usize, usize, usize, u32, bool);

fn semantic_hazards(
    passes: &[PassSpec],
    plan: &ExecutionPlan,
    resources: &[(bool, u32)],
) -> Result<Vec<SemanticHazard>, FabricRuntimeError> {
    let normalized: Vec<Vec<_>> = passes
        .iter()
        .map(|pass| {
            let accesses = pass.accesses();
            let is_texture = |usage: &BoundResourceUse| resources[usage.resource.0 as usize].0;
            // Keep each occurrence: repeated reads carry distinct destination
            // ordinals for RAW hazards and distinct source ordinals for WAR.
            accesses
                .iter()
                .enumerate()
                .filter(|(_, usage)| is_texture(usage))
                .chain(
                    accesses
                        .iter()
                        .enumerate()
                        .filter(|(_, usage)| !is_texture(usage)),
                )
                .collect()
        })
        .collect();
    let mut hazards = Vec::with_capacity(plan.logical_barriers.len());
    for barrier in &plan.logical_barriers {
        let &(before, source) = normalized
            .get(barrier.source)
            .and_then(|accesses| accesses.get(barrier.before))
            .ok_or(FabricRuntimeError::PlanMismatch)?;
        let &(after, destination) = normalized
            .get(barrier.destination)
            .and_then(|accesses| accesses.get(barrier.after))
            .ok_or(FabricRuntimeError::PlanMismatch)?;
        let identity = match barrier.resource {
            LogicalResource::Texture(index) => (true, index.get()),
            LogicalResource::Buffer(index) => (false, index.get()),
        };
        if source.resource != destination.resource
            || resources[destination.resource.0 as usize] != identity
        {
            return Err(FabricRuntimeError::PlanMismatch);
        }
        hazards.push((
            barrier.source,
            before,
            barrier.destination,
            after,
            destination.resource.0,
            barrier.queue_transfer,
        ));
    }
    hazards.sort_unstable();
    Ok(hazards)
}
