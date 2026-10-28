//! SceneFabric Decal producer/recorder.

use bevy_app::SubApp;
use bevy_ecs::{prelude::*, schedule::IntoScheduleConfigs};
use bevy_render::{
    render_resource::{BindGroup, ComputePipeline, PipelineCache},
    renderer::{RenderContext, RenderDevice},
};
use bytemuck::bytes_of;
use prism_render_architecture::render_fabric::{
    DecalAbiBuffer, FrameRenderItem, RenderDomain, RenderObjectId, DECAL_ABI_BUFFERS,
    DECAL_ABI_BUFFER_COUNT,
};
use prism_render_driver::{
    BufferDescriptor, BufferState, BufferUsages, Extent3d, RenderQueue as _,
};
use prism_render_graph::RecordedGraph;
use prism_render_shading::gi::decal::DecalProjector;
use std::sync::Arc;
use thiserror::Error;

use crate::{
    completion::GpuCompletionTracker,
    fabric_domains::{
        build_decal_domain_items,
        consumers::prepare_decal_visibility,
        gpu::{DomainRecordReadyResource, DomainRecordReadyView},
        owners::DecalRenderWorldOwner,
        DecalDomainInput, DomainFrameIdentity,
    },
    fabric_runtime::{
        self, SceneFabricCoordinatorInput, SceneFabricCoordinatorInputError, SceneFabricDispatch,
        SceneFabricObjectResource, SceneFabricPass, SceneFabricRequest, SceneFabricRuntime,
        SceneFabricRuntimeError,
    },
    gpu::{fabric::SceneFabric, rhi::SceneRhi},
};

use super::{
    bind_groups::create_decal_bind_group,
    fabric::{DECAL_OBJECT_ACCESSES, DECAL_PASS},
    pipeline::DecalResolvePipeline,
};

#[derive(Debug, Error)]
pub(crate) enum DecalResolveError {
    #[error("invalid Decal semantic frame: {0}")]
    Semantic(#[from] crate::fabric_domains::consumers::DomainConsumerError),
    #[error("invalid Decal bind group: {0}")]
    BindGroup(&'static str),
    #[error("Decal resolve pipeline is not ready")]
    PipelineNotReady,
    #[error("SceneFabric Decal preparation failed: {0}")]
    Runtime(#[from] SceneFabricRuntimeError),
}

/// One renderer-owned Decal publication ready for the host Core3d recorder.
///
/// The publication owns the authenticated domain view and semantic items. It
/// therefore keeps every real RHI buffer lease alive from PrepareResources
/// through command recording; no recorder-side lookup can fall back to a
/// table-local ID or manufacture a buffer.
pub(crate) struct DecalResolvePublication {
    pub(crate) frame: u64,
    pub(crate) extent: Extent3d,
    pub(crate) identity: DomainFrameIdentity,
    pub(crate) items: Vec<FrameRenderItem>,
    pub(crate) domain_view: DomainRecordReadyView,
    pub(crate) workgroups: [u32; 3],
}

/// Error raised while admitting a real Decal publication fan-out to the
/// shared SceneFabric coordinator.
#[derive(Debug, Error)]
pub(crate) enum DecalResolveBatchError {
    #[error("Decal batch must contain at least one publication")]
    Empty,
    #[error("Decal batch publications must share frame and extent")]
    FrameOrExtentMismatch,
    #[error("Decal batch contains duplicate object/history identity")]
    DuplicateIdentity,
    #[error("Decal batch publication is not a complete six-buffer record-ready view")]
    IncompleteResources,
    #[error("Decal batch publication has an invalid workgroup dimension")]
    InvalidWorkgroups,
    #[error("Decal batch publication failed semantic validation: {0}")]
    Semantic(String),
    #[error("Decal batch resource is not a buffer: {0}")]
    Resource(String),
    #[error("Decal batch coordinator input rejected: {0}")]
    Input(#[from] SceneFabricCoordinatorInputError),
    #[error("Decal batch completion token must be non-zero")]
    InvalidCompletionToken,
}

/// One real object-scoped Decal fan-out prepared for the shared coordinator.
///
/// The adapter owns the publications and the bind groups created from their
/// authenticated [`DomainRecordReadyView`] values. `append_to` only borrows
/// this object while the coordinator compiles and records the shared batch;
/// the adapter must then be moved into [`DecalResolveBatchSubmission`] so the
/// six-buffer leases cannot retire before the host queue completion.
pub(crate) struct DecalResolveBatchAdapter<'pipeline> {
    frame: u64,
    extent: Extent3d,
    items: Vec<FrameRenderItem>,
    publications: Vec<DecalResolvePublication>,
    bind_groups: Vec<BindGroup>,
    native_pipeline: &'pipeline ComputePipeline,
}

impl<'pipeline> DecalResolveBatchAdapter<'pipeline> {
    /// Builds one real fan-out from producer publications and the resolved
    /// native compute pipeline. No publication is accepted unless its six
    /// physical resources are record-ready and its semantic identity matches.
    pub(crate) fn from_publications(
        rhi: &SceneRhi,
        device: &RenderDevice,
        pipeline: &DecalResolvePipeline,
        native_pipeline: &'pipeline ComputePipeline,
        publications: Vec<DecalResolvePublication>,
    ) -> Result<Self, DecalResolveBatchError> {
        let first = publications.first().ok_or(DecalResolveBatchError::Empty)?;
        let frame = first.frame;
        let extent = first.extent;
        let mut items = Vec::new();
        let mut bind_groups = Vec::with_capacity(publications.len());

        for (index, publication) in publications.iter().enumerate() {
            if publication.frame != frame || publication.extent != extent {
                return Err(DecalResolveBatchError::FrameOrExtentMismatch);
            }
            if publications[..index]
                .iter()
                .any(|candidate| candidate.identity == publication.identity)
            {
                return Err(DecalResolveBatchError::DuplicateIdentity);
            }
            if publication.workgroups.iter().any(|&group| group == 0) {
                return Err(DecalResolveBatchError::InvalidWorkgroups);
            }
            let consumer = prepare_decal_visibility(&publication.items, publication.identity)
                .map_err(|error| DecalResolveBatchError::Semantic(error.to_string()))?;
            if consumer.object() != RenderObjectId::Decal(publication.identity.object)
                || publication.domain_view.frame() != frame
                || publication.domain_view.resources().len() != DECAL_ABI_BUFFER_COUNT
                || publication.domain_view.resources().iter().any(|resource| {
                    !matches!(
                        resource.record_ready(),
                        Ok(DomainRecordReadyResource::Buffer { .. })
                    )
                })
            {
                return Err(DecalResolveBatchError::IncompleteResources);
            }
            let bind_group =
                create_decal_bind_group(rhi, device, pipeline, &publication.domain_view)
                    .map_err(|error| DecalResolveBatchError::Resource(error.into()))?;
            items.extend_from_slice(&publication.items);
            bind_groups.push(bind_group);
        }

        Ok(Self {
            frame,
            extent,
            items,
            publications,
            bind_groups,
            native_pipeline,
        })
    }

    /// Appends this fan-out to the shared coordinator boundary.
    ///
    /// `SceneFabricCoordinatorInput` intentionally has no object-resource
    /// field. Putting multiple Decal publications into its shared resource
    /// vector would collapse distinct object/history addresses to the same
    /// `(ResourceKey, frame)` and be rejected by the authoritative runtime.
    /// The six real buffers therefore move into `object_resources`, while the
    /// shared input receives the complete semantic item set, one Decal pass,
    /// and one real dispatch per publication. The caller must pass the
    /// returned object resources to `fabric_runtime::prepare_with_objects`
    /// together with [`Self::object_accesses`].
    pub(crate) fn append_to<'a, 'lease>(
        &'a self,
        batch: &mut SceneFabricCoordinatorInput<'a, 'lease>,
        object_resources: &mut Vec<SceneFabricObjectResource<'a, 'a>>,
    ) -> Result<(), DecalResolveBatchError> {
        let mut object_bindings =
            Vec::with_capacity(self.publications.len() * DECAL_ABI_BUFFER_COUNT);
        for publication in &self.publications {
            for resource in publication.domain_view.resources() {
                let ready = resource
                    .record_ready()
                    .map_err(|error| DecalResolveBatchError::Resource(error.to_string()))?;
                if !matches!(ready, DomainRecordReadyResource::Buffer { .. }) {
                    return Err(DecalResolveBatchError::IncompleteResources);
                }
                object_bindings.push(
                    SceneFabricObjectResource::from_domain(resource)
                        .map_err(|error| DecalResolveBatchError::Resource(error.to_string()))?,
                );
            }
        }

        let dispatches = self
            .publications
            .iter()
            .zip(&self.bind_groups)
            .map(|(publication, bind_group)| SceneFabricDispatch {
                pipeline: self.native_pipeline,
                bind_group,
                bind_group_index: 0,
                workgroups: publication.workgroups,
                push_constants: None,
            })
            .collect::<Vec<_>>();
        if dispatches.is_empty()
            || object_bindings.len() != self.publications.len() * DECAL_ABI_BUFFER_COUNT
        {
            return Err(DecalResolveBatchError::IncompleteResources);
        }

        batch.append_parts(
            self.frame,
            self.extent,
            &self.items,
            &[DECAL_PASS],
            &[],
            Vec::new(),
            vec![SceneFabricPass::Compute {
                pass: DECAL_PASS,
                dispatches,
            }],
        )?;
        object_resources.extend(object_bindings);
        Ok(())
    }

    /// Returns the exact object-access declarations required when the shared
    /// coordinator input is prepared through the object-aware runtime path.
    #[must_use]
    pub(crate) const fn object_accesses(
        &self,
    ) -> &'static [prism_render_architecture::render_fabric::ObjectResourceAccess] {
        DECAL_OBJECT_ACCESSES
    }

    #[must_use]
    pub(crate) const fn frame(&self) -> u64 {
        self.frame
    }

    #[must_use]
    pub(crate) const fn extent(&self) -> Extent3d {
        self.extent
    }

    #[must_use]
    pub(crate) fn publication_count(&self) -> usize {
        self.publications.len()
    }

    #[must_use]
    pub(crate) fn resource_count(&self) -> usize {
        self.publications.len() * DECAL_ABI_BUFFER_COUNT
    }

    /// Transfers the publication and bind-group ownership to a completion
    /// ticket after the shared host has submitted the recorded batch.
    pub(crate) fn submit(
        self,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<DecalResolveBatchSubmission<'pipeline>, DecalResolveBatchError> {
        if completion.0 == 0 {
            return Err(DecalResolveBatchError::InvalidCompletionToken);
        }
        Ok(DecalResolveBatchSubmission {
            completion,
            adapter: Some(self),
        })
    }
}

/// Completion-gated ownership for one recorded Decal fan-out.
pub(crate) struct DecalResolveBatchSubmission<'pipeline> {
    completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    adapter: Option<DecalResolveBatchAdapter<'pipeline>>,
}

/// Completion-gated ownership retained by the unified SceneFabric host.
///
/// The native coordinator imports the bind groups into its recorded graph, but
/// the publication views and Bevy bind-group wrappers still need an explicit
/// owner until the host queue has completed the command buffer. This lease is
/// deliberately independent of the pipeline borrow so it can live in an ECS
/// resource without manufacturing a `'static` adapter reference.
pub(crate) struct DecalResolveBatchLease {
    _publications: Vec<DecalResolvePublication>,
    _bind_groups: Vec<BindGroup>,
}

impl DecalResolveBatchLease {
    pub(crate) fn from_adapter<'pipeline>(adapter: DecalResolveBatchAdapter<'pipeline>) -> Self {
        let DecalResolveBatchAdapter {
            publications,
            bind_groups,
            ..
        } = adapter;
        Self {
            _publications: publications,
            _bind_groups: bind_groups,
        }
    }
}

impl<'pipeline> DecalResolveBatchSubmission<'pipeline> {
    #[must_use]
    pub(crate) const fn completion(
        &self,
    ) -> prism_render_architecture::gpu_scene::GpuCompletionValue {
        self.completion
    }

    /// Retires all six-buffer publications only after the host completion
    /// tracker has observed this exact submission value.
    pub(crate) fn complete(
        mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) -> Result<(), Self> {
        if completed < self.completion {
            return Err(self);
        }
        self.adapter.take();
        Ok(())
    }
}

impl Drop for DecalResolveBatchSubmission<'_> {
    fn drop(&mut self) {
        if let Some(adapter) = self.adapter.take() {
            // Without a confirmed completion value, releasing these leases
            // could race the host queue. Preserve the same safety rule as the
            // shared coordinator and the Hair batch adapter.
            std::mem::forget(adapter);
        }
    }
}

impl DecalResolvePublication {
    pub(crate) fn new(
        frame: u64,
        extent: Extent3d,
        identity: DomainFrameIdentity,
        items: Vec<FrameRenderItem>,
        domain_view: DomainRecordReadyView,
        workgroups: [u32; 3],
    ) -> Self {
        Self {
            frame,
            extent,
            identity,
            items,
            domain_view,
            workgroups,
        }
    }
}

/// A producer-owned admission ticket for the six physical Decal resources.
///
/// The owner of these resources must construct the [`DomainRecordReadyView`]
/// during `PrepareResources` from live, authenticated RHI leases. This type
/// deliberately contains no allocation helper and has no default value for a
/// missing resource set: until a producer publishes a ticket, the Decal
/// resolve path has no work to record.
pub(crate) struct DecalPreparedResourcePublication {
    pub(crate) frame: u64,
    pub(crate) identity: DomainFrameIdentity,
    pub(crate) extent: Extent3d,
    pub(crate) domain_view: DomainRecordReadyView,
    pub(crate) workgroups: [u32; 3],
}

/// Builds the production six-buffer Decal publication from the semantic
/// projector/material/receiver records. The same function is used by the
/// strict GPU harness so the test cannot pass with a test-only allocator or a
/// generic placeholder buffer.
pub(crate) fn build_decal_resource_publication(
    rhi: &SceneRhi,
    frame: u64,
    identity: DomainFrameIdentity,
    items: &[FrameRenderItem],
    projector: &DecalProjector,
    material: &prism_render_architecture::render_fabric::DecalMaterial,
    receiver: &prism_render_architecture::render_fabric::DecalReceiver,
) -> Result<DecalPreparedResourcePublication, String> {
    build_decal_resource_publication_with_extent(
        rhi,
        frame,
        identity,
        items,
        projector,
        material,
        receiver,
        Extent3d::new_2d(1, 1),
        [1, 1, 1],
    )
}

/// Builds one object-scoped Decal publication with explicit graph extent and
/// dispatch dimensions. The six buffers remain independently owned by this
/// identity; changing the extent never aliases or mutates an in-flight
/// publication from an earlier frame.
pub(crate) fn build_decal_resource_publication_with_extent(
    rhi: &SceneRhi,
    frame: u64,
    identity: DomainFrameIdentity,
    items: &[FrameRenderItem],
    projector: &DecalProjector,
    material: &prism_render_architecture::render_fabric::DecalMaterial,
    receiver: &prism_render_architecture::render_fabric::DecalReceiver,
    extent: Extent3d,
    workgroups: [u32; 3],
) -> Result<DecalPreparedResourcePublication, String> {
    if extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0 {
        return Err("Decal publication extent must be non-zero".into());
    }
    if workgroups.iter().any(|&group| group == 0) {
        return Err("Decal publication workgroups must be non-zero".into());
    }
    let consumer = prepare_decal_visibility(items, identity).map_err(|error| error.to_string())?;
    if consumer.object() != RenderObjectId::Decal(identity.object) {
        return Err("Decal publication semantic identity does not match its object".into());
    }
    if consumer.resource_declarations().len() != DECAL_ABI_BUFFER_COUNT {
        return Err("Decal publication must cover the complete six-buffer ABI".into());
    }
    let instance = prism_render_architecture::render_fabric::DecalInstance::new(
        projector.world_to_decal.to_cols_array(),
        projector.projection_dir.to_array(),
        projector.angle_fade_start,
        projector.angle_fade_end,
        projector.edge_fade,
        0,
    );
    let range = prism_render_architecture::render_fabric::DecalClusterRange {
        offset: 0,
        count: 1,
        reserved: [0; 2],
    };
    let index = 0_u32;
    let output = prism_render_architecture::render_fabric::DecalResolvedReceiver {
        albedo: receiver.albedo,
        normal: receiver.world_normal,
        scalar_coverage: [receiver.scalar[0], receiver.scalar[1], 0.0, 0.0],
    };
    let usage = BufferUsages::STORAGE
        .union(BufferUsages::COPY_SRC)
        .union(BufferUsages::COPY_DST);
    let buffers = [
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<prism_render_architecture::render_fabric::DecalInstance>() as u64,
            usage,
        ))),
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<prism_render_architecture::render_fabric::DecalMaterial>() as u64,
            usage,
        ))),
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<prism_render_architecture::render_fabric::DecalReceiver>() as u64,
            usage,
        ))),
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<prism_render_architecture::render_fabric::DecalClusterRange>()
                as u64,
            usage,
        ))),
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<u32>() as u64,
            usage,
        ))),
        Arc::new(rhi.create_buffer(&BufferDescriptor::new(
            core::mem::size_of::<prism_render_architecture::render_fabric::DecalResolvedReceiver>()
                as u64,
            usage,
        ))),
    ];
    debug_assert_eq!(DECAL_ABI_BUFFERS.len(), buffers.len());
    debug_assert_eq!(
        DECAL_ABI_BUFFERS.map(DecalAbiBuffer::stride),
        core::array::from_fn(|index| buffers[index].buffer().size())
    );
    let payloads: [&[u8]; 6] = [
        bytes_of(&instance),
        bytes_of(material),
        bytes_of(receiver),
        bytes_of(&range),
        bytes_of(&index),
        bytes_of(&output),
    ];
    for (buffer, payload) in buffers.iter().zip(payloads) {
        rhi.queue().write_buffer(buffer.id(), 0, payload);
    }

    let requirements = crate::fabric_domains::resources::contract(RenderDomain::Decal)
        .map_err(|error| error.to_string())?
        .requirements;
    let bindings = requirements
        .iter()
        .enumerate()
        .map(|(slot, requirement)| {
            let resource_identity = crate::fabric_domains::gpu::DomainResourceIdentity::new(
                RenderDomain::Decal,
                identity.object,
                identity.history_key,
                requirement.role,
            )
            .map_err(|error| error.to_string())?;
            crate::fabric_domains::gpu::DomainGpuBinding::new(
                rhi,
                resource_identity,
                frame,
                slot as u32 + 1,
                crate::fabric_domains::gpu::DomainGpuResourceOwner::Buffer {
                    owner: crate::fabric_domains::gpu::DomainBufferOwner::Allocated(Arc::clone(
                        &buffers[slot],
                    )),
                    offset: 0,
                    entry_state: BufferState::initial(),
                },
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain_view = DomainRecordReadyView::from_items(rhi, frame, items, bindings)
        .map_err(|error| error.to_string())?;
    Ok(DecalPreparedResourcePublication::new(
        frame,
        identity,
        extent,
        domain_view,
        workgroups,
    ))
}

/// Lowers every currently-owned Decal object into a deterministic fan-out of
/// independent six-buffer publications. No object shares a physical buffer,
/// resource version, or history identity with another object.
pub(crate) fn build_decal_resource_publications_from_owner(
    rhi: &SceneRhi,
    frame: u64,
    owner: &DecalRenderWorldOwner,
) -> Result<Vec<DecalPreparedResourcePublication>, String> {
    owner
        .iter()
        .map(|entry| {
            let identity = entry.identity();
            let projector = entry.projector();
            let material = entry.material();
            let receiver = entry.receiver();
            let semantic_input = [DecalDomainInput {
                identity,
                projector: &projector,
                material: &material,
                receiver: &receiver,
            }];
            let items =
                build_decal_domain_items(&semantic_input).map_err(|error| error.to_string())?;
            build_decal_resource_publication(
                rhi, frame, identity, &items, &projector, &material, &receiver,
            )
        })
        .collect()
}

impl DecalPreparedResourcePublication {
    pub(crate) fn new(
        frame: u64,
        identity: DomainFrameIdentity,
        extent: Extent3d,
        domain_view: DomainRecordReadyView,
        workgroups: [u32; 3],
    ) -> Self {
        Self {
            frame,
            identity,
            extent,
            domain_view,
            workgroups,
        }
    }
}

/// Render-world mailbox for the six-resource Decal PrepareResources publisher.
///
/// A producer publishes one entry per owner after creating the exact semantic
/// buffers, writing their ABI records, and authenticating their leases. The
/// mailbox itself never allocates or substitutes resources.
#[derive(Resource, Default)]
pub(crate) struct DecalPreparedResources {
    pending: Vec<DecalPreparedResourcePublication>,
}

impl DecalPreparedResources {
    /// Publishes a complete, already-authenticated physical resource view.
    ///
    /// This is the only production admission point for physical Decal
    /// resources. It accepts ownership of the view; it never allocates,
    /// uploads, substitutes, or fills missing resources.
    pub(crate) fn publish(
        &mut self,
        frame: u64,
        identity: DomainFrameIdentity,
        extent: Extent3d,
        domain_view: DomainRecordReadyView,
        workgroups: [u32; 3],
    ) -> Result<(), &'static str> {
        if domain_view.frame() != frame {
            return Err("Decal prepared resource frame does not match publication frame");
        }
        if domain_view.resources().len() != DECAL_ABI_BUFFER_COUNT
            || domain_view.resources().iter().any(|resource| {
                !matches!(
                    resource.record_ready(),
                    Ok(DomainRecordReadyResource::Buffer { .. })
                )
            })
        {
            return Err("Decal PrepareResources must admit exactly six record-ready buffers");
        }
        if workgroups.iter().any(|&group| group == 0) {
            return Err("Decal prepared resource workgroups must be non-zero");
        }
        if self
            .pending
            .iter()
            .any(|publication| publication.frame == frame && publication.identity == identity)
        {
            return Err("Decal PrepareResources already has this owner publication");
        }
        self.pending.push(DecalPreparedResourcePublication {
            frame,
            identity,
            extent,
            domain_view,
            workgroups,
        });
        Ok(())
    }

    pub(crate) fn take_for(
        &mut self,
        frame: u64,
        identity: DomainFrameIdentity,
    ) -> Result<Option<DecalPreparedResourcePublication>, &'static str> {
        let Some(index) = self
            .pending
            .iter()
            .position(|publication| publication.identity == identity)
        else {
            return Ok(None);
        };
        if self.pending[index].frame != frame {
            return Err("Decal prepared resource frame does not match Render Fabric frame");
        }
        Ok(Some(self.pending.swap_remove(index)))
    }

    fn clear(&mut self) {
        self.pending.clear();
    }
}

/// One-shot RenderApp mailbox for the real Decal Core3d consumer.
///
/// Allocation and semantic upload remain producer-owned: a publication is
/// accepted only after the producer has built a complete authenticated
/// [`DomainRecordReadyView`]. This resource only transfers that ownership to
/// the host recorder and records errors instead of replaying stale work.
#[derive(Resource, Default)]
pub(crate) struct DecalResolveHost {
    pending: Vec<DecalResolvePublication>,
    in_flight: Vec<DecalInFlightPublication>,
    last_error: Option<String>,
    last_recorded_frame: Option<u64>,
}

/// Keeps both graph allocations and producer-owned domain leases alive until
/// the actual Bevy host submission containing the Decal commands completes.
struct DecalInFlightPublication {
    completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    // The leading underscores are intentional: these fields are the lifetime
    // anchors for graph allocations and producer-owned buffer leases.
    _recorded: Option<RecordedGraph>,
    _publications: Vec<DecalResolvePublication>,
}

impl DecalResolveHost {
    pub(crate) fn publish(
        &mut self,
        publication: DecalResolvePublication,
    ) -> Result<(), &'static str> {
        if self
            .pending
            .first()
            .is_some_and(|pending| pending.frame != publication.frame)
        {
            return Err("Decal host cannot mix publications from different frames");
        }
        if self.pending.iter().any(|pending| {
            pending.frame == publication.frame && pending.identity == publication.identity
        }) {
            return Err("Decal host already has this owner publication");
        }
        self.last_error = None;
        self.pending.push(publication);
        Ok(())
    }

    pub(crate) fn take_for_frame(
        &mut self,
        frame: u64,
    ) -> Result<Vec<DecalResolvePublication>, String> {
        if let Some(publication) = self
            .pending
            .iter()
            .find(|publication| publication.frame != frame)
        {
            let error = format!(
                "Decal publication frame {} does not match Render Fabric frame {frame}",
                publication.frame
            );
            self.last_error = Some(error.clone());
            self.pending.clear();
            return Err(error);
        }
        Ok(core::mem::take(&mut self.pending))
    }

    fn record_success(&mut self, frame: u64) {
        self.last_error = None;
        self.last_recorded_frame = Some(frame);
    }

    fn record_error(&mut self, error: impl Into<String>) {
        self.last_error = Some(error.into());
        self.last_recorded_frame = None;
    }

    fn clear_without_work(&mut self) {
        self.pending.clear();
        self.last_error = None;
    }

    fn retain_in_flight(
        &mut self,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
        recorded: RecordedGraph,
        publications: Vec<DecalResolvePublication>,
    ) {
        self.in_flight.push(DecalInFlightPublication {
            completion,
            _recorded: Some(recorded),
            _publications: publications,
        });
    }

    fn retire_completed(
        &mut self,
        completed: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) {
        self.in_flight.retain(|entry| entry.completion > completed);
    }

    #[cfg(test)]
    fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    #[cfg(test)]
    fn last_recorded_frame(&self) -> Option<u64> {
        self.last_recorded_frame
    }

    #[cfg(test)]
    fn arm_test_completion(
        &mut self,
        completion: prism_render_architecture::gpu_scene::GpuCompletionValue,
    ) {
        self.in_flight.push(DecalInFlightPublication {
            completion,
            _recorded: None,
            _publications: Vec::new(),
        });
    }

    #[cfg(test)]
    fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }
}

/// PrepareResources scheduling seam for a real Decal resource publisher.
///
/// Physical resources are producer-owned. This system deliberately does not
/// allocate buffers, upload semantic values, or publish synthetic data. A
/// producer with real allocations must call [`DecalPreparedResources::publish`]
/// after constructing an authenticated [`DomainRecordReadyView`].
pub(crate) fn prepare_decal_resources_from_owner(
    owner: Res<DecalRenderWorldOwner>,
    frame: Res<crate::fabric_frame::RenderFabricFrame>,
    rhi: Res<SceneRhi>,
    mut prepared: ResMut<DecalPreparedResources>,
) {
    prepared.clear();
    if owner.is_empty() {
        return;
    }
    let Ok(publications) =
        build_decal_resource_publications_from_owner(rhi.as_ref(), frame.frame(), owner.as_ref())
    else {
        prepared.clear();
        return;
    };
    for publication in publications {
        let DecalPreparedResourcePublication {
            frame,
            identity,
            extent,
            domain_view,
            workgroups,
        } = publication;
        if prepared
            .publish(frame, identity, extent, domain_view, workgroups)
            .is_err()
        {
            prepared.clear();
            return;
        }
    }
}

/// Admits the current render-world Decal owner to the real six-resource
/// publication supplied by `PrepareResources`.
///
/// The owner provides semantic data and identity; the preceding
/// PrepareResources producer provides the authenticated physical leases. If
/// either side is missing or mismatched, this system records an error and
/// emits no Core3d work.
pub(crate) fn publish_decal_resolve_from_owner(
    owner: Res<DecalRenderWorldOwner>,
    frame: Res<crate::fabric_frame::RenderFabricFrame>,
    mut prepared: ResMut<DecalPreparedResources>,
    mut host: ResMut<DecalResolveHost>,
) {
    if owner.is_empty() {
        prepared.clear();
        host.clear_without_work();
        return;
    }
    for entry in owner.iter() {
        let identity = entry.identity();
        if !frame.items().iter().any(|item| {
            item.object_id == RenderObjectId::Decal(identity.object)
                && item.domain == RenderDomain::Decal
                && item.history_key == identity.history_key
        }) {
            host.record_error("Decal owner has no matching shared-frame semantic items");
            continue;
        }
        let projector = entry.projector();
        let material = entry.material();
        let receiver = entry.receiver();
        let semantic_input = [DecalDomainInput {
            identity,
            projector: &projector,
            material: &material,
            receiver: &receiver,
        }];
        let Ok(items) = build_decal_domain_items(&semantic_input) else {
            host.record_error("Decal owner failed semantic item construction");
            continue;
        };
        let Some(prepared_publication) = (match prepared.take_for(frame.frame(), identity) {
            Ok(publication) => publication,
            Err(error) => {
                host.record_error(error);
                continue;
            }
        }) else {
            host.record_error(
                "Decal six-resource PrepareResources publication is unavailable; resolve skipped",
            );
            continue;
        };
        let admitted_items: Vec<_> = prepared_publication
            .domain_view
            .resources()
            .iter()
            .map(|resource| resource.declaration())
            .collect();
        let expected_resources: Vec<_> = items
            .iter()
            .copied()
            .filter(|item| {
                matches!(
                    item.payload,
                    prism_render_architecture::render_fabric::DomainPayload::Resource(_)
                )
            })
            .collect();
        if admitted_items != expected_resources {
            host.record_error(
                "Decal prepared resources do not exactly cover the owner's semantic resource set",
            );
            continue;
        }
        let publication = DecalResolvePublication::new(
            prepared_publication.frame,
            prepared_publication.extent,
            prepared_publication.identity,
            items,
            prepared_publication.domain_view,
            prepared_publication.workgroups,
        );
        if let Err(error) = host.publish(publication) {
            host.record_error(error);
        }
    }
}

/// Prepares one real Decal object for the frozen SceneFabric resolve pass.
/// The six resources are authenticated domain leases; no placeholder buffer or
/// table-local ID is accepted by this path.
pub(crate) fn prepare_decal_resolve<'a>(
    scene: &SceneFabric,
    rhi: &SceneRhi,
    device: &RenderDevice,
    pipeline: &DecalResolvePipeline,
    native_pipeline: &'a ComputePipeline,
    frame: u64,
    extent: Extent3d,
    identity: DomainFrameIdentity,
    items: &'a [FrameRenderItem],
    domain_view: &'a DomainRecordReadyView,
    workgroups: [u32; 3],
) -> Result<SceneFabricRuntime<'a>, DecalResolveError> {
    let bind_group = create_decal_bind_group(rhi, device, pipeline, domain_view)
        .map_err(DecalResolveError::BindGroup)?;
    let expected = RenderObjectId::Decal(identity.object);
    let consumer = prepare_decal_visibility(items, identity)?;
    if consumer.object() != expected || domain_view.frame() != frame {
        return Err(DecalResolveError::BindGroup(
            "Decal identity/frame mismatch",
        ));
    }
    let runtime = fabric_runtime::prepare_with_domains(
        scene,
        rhi,
        SceneFabricRequest {
            frame,
            extent,
            items,
            active_passes: &[DECAL_PASS],
            fulfilled_dependencies: &[],
            resources: Vec::new(),
            recorders: vec![SceneFabricPass::Compute {
                pass: DECAL_PASS,
                dispatches: vec![SceneFabricDispatch {
                    pipeline: native_pipeline,
                    bind_group: &bind_group,
                    bind_group_index: 0,
                    workgroups,
                    push_constants: None,
                }],
            }],
        },
        &[],
        DECAL_OBJECT_ACCESSES,
        &[domain_view],
    )?;
    Ok(runtime)
}

/// Prepares all Decal objects for one frozen SceneFabric pass.
///
/// The six resources remain object-scoped, but the compiler sees one request
/// containing every object declaration, every authenticated view, and one
/// dispatch per object. This is the fan-out boundary: no object is flattened
/// into a shared ledger and no publication is recorded in an isolated graph.
pub(crate) fn prepare_decal_resolves<'a>(
    scene: &SceneFabric,
    rhi: &SceneRhi,
    native_pipeline: &'a ComputePipeline,
    publications: &'a [DecalResolvePublication],
    bind_groups: &'a [BindGroup],
) -> Result<SceneFabricRuntime<'a>, DecalResolveError> {
    if publications.is_empty() {
        return Err(DecalResolveError::BindGroup(
            "Decal fan-out cannot prepare an empty publication set",
        ));
    }
    if publications.len() != bind_groups.len() {
        return Err(DecalResolveError::BindGroup(
            "Decal fan-out publication/bind-group count mismatch",
        ));
    }
    let frame = publications[0].frame;
    let extent = publications[0].extent;
    let mut items = Vec::new();
    let mut views = Vec::with_capacity(publications.len());
    for (publication_index, publication) in publications.iter().enumerate() {
        if publication.frame != frame || publication.extent != extent {
            return Err(DecalResolveError::BindGroup(
                "Decal fan-out publications must share frame and extent",
            ));
        }
        if publications[..publication_index]
            .iter()
            .any(|candidate| candidate.identity == publication.identity)
        {
            return Err(DecalResolveError::BindGroup(
                "Decal fan-out contains duplicate object/history identity",
            ));
        }
        let expected = RenderObjectId::Decal(publication.identity.object);
        let consumer = prepare_decal_visibility(&publication.items, publication.identity)?;
        if consumer.object() != expected || publication.domain_view.frame() != frame {
            return Err(DecalResolveError::BindGroup(
                "Decal identity/frame mismatch",
            ));
        }
        items.extend_from_slice(&publication.items);
        views.push(&publication.domain_view);
    }
    let compute_dispatches = publications
        .iter()
        .zip(bind_groups)
        .map(|(publication, bind_group)| SceneFabricDispatch {
            pipeline: native_pipeline,
            bind_group,
            bind_group_index: 0,
            workgroups: publication.workgroups,
            push_constants: None,
        })
        .collect::<Vec<_>>();
    let runtime = fabric_runtime::prepare_with_domains(
        scene,
        rhi,
        SceneFabricRequest {
            frame,
            extent,
            items: &items,
            active_passes: &[DECAL_PASS],
            fulfilled_dependencies: &[],
            resources: Vec::new(),
            recorders: vec![SceneFabricPass::Compute {
                pass: DECAL_PASS,
                dispatches: compute_dispatches,
            }],
        },
        &[],
        DECAL_OBJECT_ACCESSES,
        &views,
    )?;
    Ok(runtime)
}

/// Records all object-scoped Decal publications into one SceneFabric graph.
///
/// One host command buffer owns the whole fan-out. The returned publication
/// vector is retained as one completion-gated unit so removal or resize of a
/// later frame cannot recycle any object buffer still referenced by this
/// submission.
pub(crate) fn record_decal_resolves(
    scene: &SceneFabric,
    rhi: &SceneRhi,
    device: &RenderDevice,
    cache: &PipelineCache,
    pipeline: &DecalResolvePipeline,
    publications: Vec<DecalResolvePublication>,
    context: &mut RenderContext,
) -> Result<(RecordedGraph, Vec<DecalResolvePublication>), DecalResolveError> {
    let native_pipeline = cache
        .get_compute_pipeline(pipeline.resolve())
        .ok_or(DecalResolveError::PipelineNotReady)?;
    let bind_groups = publications
        .iter()
        .map(|publication| {
            create_decal_bind_group(rhi, device, pipeline, &publication.domain_view)
                .map_err(DecalResolveError::BindGroup)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let runtime = prepare_decal_resolves(scene, rhi, native_pipeline, &publications, &bind_groups)?;
    Ok((runtime.record_into(rhi, context), publications))
}

/// Real RenderApp/Core3d consumer for the Decal mailbox.
///
/// The function is intentionally exposed as an installer instead of being
/// silently registered from a pipeline-init system. That keeps schedule
/// ownership explicit and lets the top-level plugin install it at the same
/// point as the other Fabric domains.
pub(crate) fn install_decal_resolve(render_app: &mut SubApp) {
    render_app.init_resource::<DecalResolveHost>();
    render_app.init_resource::<DecalPreparedResources>();
    render_app.add_systems(
        bevy_core_pipeline::Core3d,
        record_decal_resolve_system.before(bevy_core_pipeline::Core3dSystems::MainPass),
    );
}

fn record_decal_resolve_system(
    mut host: ResMut<DecalResolveHost>,
    completion: Res<GpuCompletionTracker>,
    frame: Res<crate::fabric_frame::RenderFabricFrame>,
    scene: Res<SceneFabric>,
    rhi: Res<SceneRhi>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipeline: Res<DecalResolvePipeline>,
    mut context: RenderContext,
) {
    let publications = match host.take_for_frame(frame.frame()) {
        Ok(publication) => publication,
        Err(error) => {
            host.record_error(error);
            return;
        }
    };
    if publications.is_empty() {
        return;
    }
    let frame_number = publications[0].frame;
    match record_decal_resolves(
        scene.as_ref(),
        rhi.as_ref(),
        device.as_ref(),
        cache.as_ref(),
        pipeline.as_ref(),
        publications,
        &mut context,
    ) {
        Ok((recorded, publications)) => {
            host.record_success(frame_number);
            host.retain_in_flight(completion.next_submission_value(), recorded, publications);
        }
        Err(error) => host.record_error(error.to_string()),
    }
}

/// Releases Decal graph allocations and producer-owned resource leases only
/// after the host queue confirms completion of the submission containing them.
pub(crate) fn retire_decal_resolve_resources(
    mut host: ResMut<DecalResolveHost>,
    completion: Res<GpuCompletionTracker>,
) {
    host.retire_completed(completion.completed_value());
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::gpu_scene::GpuCompletionValue;

    #[test]
    fn completion_gated_retirement_keeps_decal_leases_until_and_only_until_completion() {
        let mut host = DecalResolveHost::default();
        host.arm_test_completion(GpuCompletionValue(7));
        assert_eq!(host.in_flight_len(), 1);

        host.retire_completed(GpuCompletionValue(6));
        assert_eq!(host.in_flight_len(), 1);

        host.retire_completed(GpuCompletionValue(7));
        assert_eq!(host.in_flight_len(), 0);

        // Retirement is idempotent; a later poll cannot retire the same batch twice.
        host.retire_completed(GpuCompletionValue(8));
        assert_eq!(host.in_flight_len(), 0);
    }

    #[test]
    fn decal_batch_contract_keeps_each_abi_slot_object_scoped() {
        assert_eq!(DECAL_OBJECT_ACCESSES.len(), DECAL_ABI_BUFFER_COUNT);
        for (ordinal, access) in DECAL_OBJECT_ACCESSES.iter().enumerate() {
            assert_eq!(access.pass, DECAL_PASS);
            assert_eq!(access.ordinal, ordinal);
        }
    }
}
