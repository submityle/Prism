//! Same-device placement registration, graph submission, and completion retirement.
//!
//! This optional adapter owns a single RHI table adopted from the original
//! `GpuContext` tuple. Producers are moved into registered wrappers; a frame
//! borrows their current requests until it is submitted or cancelled. Native
//! registrations are cached across requests and retained alongside graph
//! allocations through completion. No CPU candidate readback is used for rendering.
//!
//! The draw shader is supplied by the caller, not a product material renderer.
//! Group 0 has four vertex-stage, read-only storage buffers: records, terrain
//! samples, attributes, and compact source slots at bindings 0..=3. A deletion
//! stream binds its filtered compact output, never the upstream compact output.
//! Shaders and any IDs in pipeline descriptors must come from `device()`.
//! Do not manually destroy resources obtained through native inspection APIs.

use std::{
    collections::BTreeSet,
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard,
    },
    time::Duration,
};

use prism_render_driver::{self as rhi, RenderDevice};
use prism_render_driver_wgpu::{WgpuDevice, WgpuQueue};
use prism_render_graph::{
    GraphAllocations, ImportedBuffer, ImportedTexture, RecordedGraph, TextureDesc,
};
use prism_world_residency::ScatterDensityField;
use prism_world_stream::SectorId;
use prism_world_terrain::{
    HeightfieldPlacementConfig, HeightfieldPlacementDeletionSet, HeightfieldPlacementFilters,
    HeightfieldPlacementRegion, PlanarHeightfieldSurface,
};

use crate::placement_render_bridge::build_placement_render_graph_with_resident_upstreams;
use crate::{
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementBufferSnapshot,
    GpuHeightfieldPlacementDeletionError, GpuHeightfieldPlacementDeletionWorkBudget,
    GpuHeightfieldPlacementDeletions, GpuHeightfieldPlacementError,
    GpuHeightfieldPlacementRegistrationSnapshot, GpuHeightfieldPlacementStageSnapshot,
    GpuHeightfieldPlacementWorkBudget, PlacementComputeStage, PlacementDeletionWork,
    PlacementDrawStage, PlacementEmptyDeletionWork, PlacementImportedBuffer,
    PlacementRenderBridgeError, PlacementRenderStream, PlacementRenderWork, PlacementSceneTarget,
    PlacementStream,
};

/// Failures before submission, or while waiting for an exact submission.
#[derive(Debug)]
pub enum PlacementRhiError {
    /// The resource belongs to a different logical device or RHI table.
    ForeignResource,
    /// All admitted in-flight slots are occupied. Poll/wait before retrying.
    InFlightLimit,
    /// This adapter accepts single-layer, single-sample 2D color targets only.
    InvalidScene,
    /// The draw must use group 0's fixed ABI and one color attachment, without
    /// vertex buffers, depth, or multisampling.
    InvalidDrawPipeline,
    Placement(GpuHeightfieldPlacementError),
    Deletion(GpuHeightfieldPlacementDeletionError),
    Graph(Box<PlacementRenderBridgeError>),
    /// Native encode/submit failed validation, allocation, or internal checks.
    /// No upstream residency is published. Drain the conservatively retained
    /// pending packet before recovery; resource retirement is not execution.
    Native(Box<wgpu::Error>),
    Poll(wgpu::PollError),
}

impl std::fmt::Display for PlacementRhiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForeignResource => {
                f.write_str("placement resource belongs to another device/table")
            }
            Self::InFlightLimit => f.write_str("placement in-flight frame limit reached"),
            Self::InvalidScene => f.write_str("placement requires a single-sample 2D color target"),
            Self::InvalidDrawPipeline => {
                f.write_str("placement draw pipeline violates the group/attachment ABI")
            }
            Self::Placement(e) => e.fmt(f),
            Self::Deletion(e) => e.fmt(f),
            Self::Graph(e) => e.fmt(f),
            Self::Native(e) => e.fmt(f),
            Self::Poll(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PlacementRhiError {}

#[derive(Default)]
struct Completion {
    gpu_done: AtomicBool,
    retired: AtomicBool,
}

/// A same-bridge receipt; completion means its allocations have been retired,
/// not merely that queue submission returned. Dropping it does not block.
pub struct PlacementRhiSubmission {
    table: Arc<()>,
    index: wgpu::SubmissionIndex,
    completion: Arc<Completion>,
}

impl PlacementRhiSubmission {
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completion.retired.load(Ordering::Acquire)
    }
}

struct Pending {
    // None means native submit unwound after it may have enqueued work. Only
    // waiting for the entire queue can prove this packet safe to retire.
    index: Option<wgpu::SubmissionIndex>,
    completion: Arc<Completion>,
    allocations: Option<GraphAllocations>,
    resources: Option<FrameResources>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        // Only explicit, verified retirement may release native registrations.
        // Failed shutdown must not destroy resources still used by the GPU.
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
    }
}

/// One adopted RHI table and a bounded retirement queue. Call `poll` each frame
/// or `wait`/`drain` at explicit synchronization points. Normal frame/token Drop
/// never waits; bridge Drop drains at shutdown (30-second timeout per wait).
/// A failed wait never authorizes retirement. Explicit `drain` lets the host
/// handle such errors; failed shutdown intentionally preserves graph pins.
pub struct PlacementRhiBridge {
    context: GpuContext,
    device: WgpuDevice,
    queue: WgpuQueue,
    table: Arc<()>,
    limit: NonZeroUsize,
    pending: Vec<Pending>,
}

impl PlacementRhiBridge {
    #[must_use]
    pub fn new(context: &GpuContext, max_in_flight: NonZeroUsize) -> Self {
        let (device, queue) = context.adopt_placement_rhi();
        Self {
            context: context.clone(),
            device,
            queue,
            table: Arc::new(()),
            limit: max_in_flight,
            pending: Vec::with_capacity(max_in_flight.get()),
        }
    }

    /// Creates caller shaders on this exact table. Raw resource destruction or
    /// foreign IDs remain a caller-certified RHI escape hatch, not authenticated
    /// imports. Producer, scene, and draw wrappers do not expose their RHI IDs.
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub fn register_heightfield(
        &self,
        producer: GpuHeightfieldPlacement,
    ) -> Result<RegisteredHeightfieldPlacement, PlacementRhiError> {
        self.check_device(producer.device_identity())?;
        Ok(RegisteredHeightfieldPlacement {
            producer,
            registrations: self.registrations(),
        })
    }

    pub fn register_deletions(
        &self,
        producer: GpuHeightfieldPlacementDeletions,
    ) -> Result<RegisteredPlacementDeletions, PlacementRhiError> {
        self.check_device(producer.device_identity())?;
        Ok(RegisteredPlacementDeletions {
            producer,
            registrations: self.registrations(),
        })
    }

    fn check_device(&self, identity: &Arc<()>) -> Result<(), PlacementRhiError> {
        if Arc::ptr_eq(self.context.device_identity(), identity) {
            Ok(())
        } else {
            Err(PlacementRhiError::ForeignResource)
        }
    }

    fn registrations(&self) -> Arc<Mutex<Registrations>> {
        Arc::new(Mutex::new(Registrations {
            device: self.device.clone(),
            table: self.table.clone(),
            buffers: Vec::new(),
            stages: Vec::new(),
            resident_upstream: ResidentUpstream::default(),
        }))
    }

    pub fn create_scene(
        &self,
        desc: &rhi::TextureDescriptor,
    ) -> Result<PlacementRhiScene, PlacementRhiError> {
        if desc.size.width == 0
            || desc.size.height == 0
            || desc.size.depth_or_array_layers != 1
            || desc.dimension != rhi::TextureDimension::D2
            || desc.mip_level_count != 1
            || desc.sample_count != 1
            || !desc.usage.contains(rhi::TextureUsages::RENDER_ATTACHMENT)
            || desc.format.is_depth_stencil()
        {
            return Err(PlacementRhiError::InvalidScene);
        }
        let mut owner = SceneOwner {
            device: self.device.clone(),
            table: self.table.clone(),
            texture: None,
            view: None,
            extent: desc.size,
            format: desc.format,
        };
        let texture = self.device.create_texture(desc);
        owner.texture = Some(texture);
        owner.view = Some(
            self.device
                .create_texture_view(texture, &Default::default()),
        );
        Ok(PlacementRhiScene(Arc::new(owner)))
    }

    /// Replaces `layout` with the adapter-owned group-0 storage ABI. Shader
    /// module IDs must be created using `device()`; modules can be released
    /// after pipeline creation. The adapter owns pipeline/layout lifetimes.
    pub fn create_draw_pipeline(
        &self,
        desc: &rhi::RenderPipelineDescriptor,
    ) -> Result<PlacementRhiDrawPipeline, PlacementRhiError> {
        let target = desc
            .fragment
            .as_ref()
            .and_then(|f| {
                if f.targets.len() == 1 {
                    f.targets[0].as_ref()
                } else {
                    None
                }
            })
            .ok_or(PlacementRhiError::InvalidDrawPipeline)?;
        if !desc.is_valid()
            || !desc.vertex.buffers.is_empty()
            || desc.depth_stencil.is_some()
            || desc.multisample.count != 1
            || target.format.is_depth_stencil()
        {
            return Err(PlacementRhiError::InvalidDrawPipeline);
        }
        let mut owner = DrawOwner {
            device: self.device.clone(),
            table: self.table.clone(),
            format: target.format,
            bind_layout: None,
            layout: None,
            pipeline: None,
        };
        let bind_layout = self
            .device
            .create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
                label: Some("world placement draw ABI".into()),
                entries: [64, 64, 16, 4]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, bytes)| rhi::BindGroupLayoutEntry {
                        binding: binding as u32,
                        visibility: rhi::ShaderStages::VERTEX,
                        ty: rhi::BindingType::Buffer {
                            ty: rhi::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(bytes),
                        },
                        count: None,
                    })
                    .collect(),
            });
        owner.bind_layout = Some(bind_layout);
        let layout = self
            .device
            .create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
                label: Some("world placement draw layout".into()),
                bind_group_layouts: vec![bind_layout],
                push_constant_ranges: vec![],
            });
        owner.layout = Some(layout);
        let mut desc = desc.clone();
        desc.layout = Some(layout);
        owner.pipeline = Some(self.device.create_render_pipeline(&desc));
        Ok(PlacementRhiDrawPipeline(Arc::new(owner)))
    }

    pub fn begin_frame<'bridge, 'request>(
        &'bridge mut self,
        scene: &PlacementRhiScene,
    ) -> Result<PlacementRhiFrame<'bridge, 'request>, PlacementRhiError> {
        if !Arc::ptr_eq(&self.table, &scene.0.table) {
            return Err(PlacementRhiError::ForeignResource);
        }
        self.poll()?;
        if self.pending.len() >= self.limit.get() {
            return Err(PlacementRhiError::InFlightLimit);
        }
        Ok(PlacementRhiFrame {
            resources: FrameResources {
                device: self.device.clone(),
                scene: scene.0.clone(),
                registrations: Vec::new(),
                pipelines: Vec::new(),
                draw_groups: Vec::new(),
            },
            bridge: self,
            streams: Vec::new(),
            resident_upstreams: BTreeSet::new(),
            request: PhantomData,
        })
    }

    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Nonblocking device progress followed by completion-signal retirement.
    pub fn poll(&mut self) -> Result<usize, PlacementRhiError> {
        self.context
            .device()
            .poll(wgpu::PollType::Poll)
            .map_err(PlacementRhiError::Poll)?;
        Ok(self.retire())
    }

    fn retire(&mut self) -> usize {
        let before = self.pending.len();
        // Compact in place in O(in-flight frames), preserving queue order for
        // drain's latest-index wait and retaining the preallocated capacity.
        self.pending.retain_mut(|item| {
            if item.completion.gpu_done.load(Ordering::Acquire) {
                item.allocations
                    .take()
                    .expect("pending graph allocations")
                    .release(&self.device);
                drop(item.resources.take());
                item.completion.retired.store(true, Ordering::Release);
                false
            } else {
                true
            }
        });
        before - self.pending.len()
    }

    pub fn wait(
        &mut self,
        submission: &PlacementRhiSubmission,
    ) -> Result<usize, PlacementRhiError> {
        if !Arc::ptr_eq(&self.table, &submission.table) {
            return Err(PlacementRhiError::ForeignResource);
        }
        if submission.is_complete() {
            return Ok(0);
        }
        self.wait_index(Some(submission.index.clone()))?;
        // Queue callbacks can include later submissions from other threads.
        // The exact native wait itself proves this ordered prefix complete.
        self.confirm_through(&submission.completion);
        Ok(self.retire())
    }

    fn confirm_through(&self, completion: &Arc<Completion>) {
        confirm_completed_prefix(&self.pending, completion);
    }

    fn wait_index(&self, index: Option<wgpu::SubmissionIndex>) -> Result<(), PlacementRhiError> {
        self.context
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: index,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(PlacementRhiError::Poll)?;
        Ok(())
    }

    /// Waits through this bridge's latest submission, or the entire queue if a
    /// panicking submit could not return its index. A successful wait retires
    /// every pending packet, independently of delayed completion callbacks.
    pub fn drain(&mut self) -> Result<usize, PlacementRhiError> {
        if let Some(last) = self.pending.last() {
            self.wait_index(last.index.clone())?;
            for item in &self.pending {
                item.completion.gpu_done.store(true, Ordering::Release);
            }
        }
        Ok(self.retire())
    }
}

impl Drop for PlacementRhiBridge {
    fn drop(&mut self) {
        // On failed shutdown, GraphAllocations deliberately keeps its pins.
        // A timeout or rejected index is not permission to destroy GPU users.
        // A host callback can panic inside native polling. Do not double-panic
        // during shutdown; unretired packets retain their native owners.
        contain_shutdown_panic(|| {
            let _ = self.drain();
        });
    }
}

fn contain_shutdown_panic(shutdown: impl FnOnce()) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(shutdown)) {
        // An arbitrary host panic payload can itself panic in Drop. Shutdown
        // must contain both panics, including during an existing unwind.
        std::mem::forget(payload);
    }
}

/// Owns a stable producer and its cached, same-table registration lease.
/// A frame's shared borrow prevents mutation until submit or cancellation.
pub struct RegisteredHeightfieldPlacement {
    producer: GpuHeightfieldPlacement,
    registrations: Arc<Mutex<Registrations>>,
}

impl RegisteredHeightfieldPlacement {
    #[must_use]
    pub fn producer(&self) -> &GpuHeightfieldPlacement {
        &self.producer
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_request(
        &mut self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        budget: GpuHeightfieldPlacementWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementError> {
        self.producer
            .write_request(config, region, field, surface, filters, budget)
    }
}

/// Full-key deletion producer with the same borrow/ordered-queue contract.
/// Successfully submitted upstream outputs are reused while the complete
/// placement recipe remains unchanged; deletion-only edits rerun the filter.
pub struct RegisteredPlacementDeletions {
    producer: GpuHeightfieldPlacementDeletions,
    registrations: Arc<Mutex<Registrations>>,
}

impl RegisteredPlacementDeletions {
    /// Read-only inspection of the owned producer. Before modifying its native
    /// output buffers through an escape hatch (including native `encode`), call
    /// [`Self::invalidate_resident_upstream`]. External writes must obey the
    /// same queue ordering and request-borrow contract as adapter submissions:
    /// enqueue the external writes before the next rebuilding submission, not
    /// after that submission has reestablished residency.
    #[must_use]
    pub fn producer(&self) -> &GpuHeightfieldPlacementDeletions {
        &self.producer
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_request(
        &mut self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        placement_budget: GpuHeightfieldPlacementWorkBudget,
        deletions: &HeightfieldPlacementDeletionSet,
        budget: GpuHeightfieldPlacementDeletionWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementDeletionError> {
        let previous_generation = self.producer.upstream_generation();
        let previous_key = self.producer.cache_key().map(|key| key.upstream.clone());
        self.producer.write_request(
            config,
            region,
            field,
            surface,
            filters,
            placement_budget,
            deletions,
            budget,
        )?;
        let unchanged = previous_key.as_ref() == self.producer.cache_key().map(|key| &key.upstream);
        lock(&self.registrations).resident_upstream.rewrite(
            previous_generation,
            self.producer.upstream_generation(),
            unchanged,
        );
        Ok(())
    }

    /// Forces the next nonempty frame to regenerate all placement outputs.
    /// Use this before any external native mutation of upstream output buffers.
    /// The mutable borrow prevents invalidation while a frame using the current
    /// cache proof is declared or recorded. No GPU work or wait is performed.
    pub fn invalidate_resident_upstream(&mut self) {
        lock(&self.registrations).resident_upstream = ResidentUpstream::default();
    }
}

// A generation is published only after native submit returns. Output validity
// needs queue ordering, not CPU completion. Successful recipe-equivalent writes
// can advance the proof; any intervening recipe change invalidates it (including
// A -> B -> A without submitting B). No equality-only cache resurrection.
#[derive(Default)]
struct ResidentUpstream(Option<u64>);

impl ResidentUpstream {
    fn contains(&self, generation: u64) -> bool {
        self.0 == Some(generation)
    }

    fn rewrite(&mut self, previous: u64, next: u64, unchanged: bool) {
        self.0 = (unchanged && self.contains(previous)).then_some(next);
    }

    fn submitted(&mut self, generation: u64) {
        self.0 = Some(generation);
    }
}

struct Registrations {
    device: WgpuDevice,
    table: Arc<()>,
    buffers: Vec<(wgpu::Buffer, PlacementImportedBuffer)>,
    stages: Vec<RegisteredStage>,
    resident_upstream: ResidentUpstream,
}

struct RegisteredStage {
    device: WgpuDevice,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    registered_pipeline: Option<rhi::ComputePipelineId>,
    registered_group: Option<rhi::BindGroupId>,
}

impl Drop for RegisteredStage {
    fn drop(&mut self) {
        if let Some(group) = self.registered_group {
            self.device.release_bind_group(group);
        }
        if let Some(pipeline) = self.registered_pipeline {
            self.device.release_compute_pipeline(pipeline);
        }
    }
}

impl RegisteredStage {
    fn work(&self, dispatch: [u32; 3]) -> PlacementComputeStage {
        PlacementComputeStage {
            pipeline: self
                .registered_pipeline
                .expect("registered compute pipeline"),
            bind_group: self.registered_group.expect("registered compute group"),
            dispatch,
        }
    }
}

fn lock(registrations: &Mutex<Registrations>) -> MutexGuard<'_, Registrations> {
    registrations
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Registrations {
    fn buffer(
        &mut self,
        native: &GpuHeightfieldPlacementBufferSnapshot,
    ) -> PlacementImportedBuffer {
        if let Some((_, imported)) = self
            .buffers
            .iter()
            .find(|(buffer, _)| buffer == native.buffer())
        {
            return *imported;
        }
        // Reserve before installing a table entry so allocation panic cannot
        // strand a newly registered handle outside this RAII owner.
        self.buffers.reserve(1);
        let imported = PlacementImportedBuffer {
            bytes: native.bytes(),
            imported: ImportedBuffer {
                buffer: self.device.register_buffer(native.buffer()),
                entry_state: rhi::BufferState::initial(),
            },
        };
        self.buffers.push((native.buffer().clone(), imported));
        imported
    }

    fn stage(&mut self, native: &GpuHeightfieldPlacementStageSnapshot) -> PlacementComputeStage {
        if let Some(stage) = self.stages.iter().find(|stage| {
            stage.pipeline == *native.pipeline() && stage.group == *native.bind_group()
        }) {
            return stage.work(native.dispatch());
        }
        self.stages.reserve(1);
        let mut registration = RegisteredStage {
            device: self.device.clone(),
            pipeline: native.pipeline().clone(),
            group: native.bind_group().clone(),
            registered_pipeline: None,
            registered_group: None,
        };
        registration.registered_pipeline =
            Some(self.device.register_compute_pipeline(native.pipeline()));
        registration.registered_group = Some(self.device.register_bind_group(native.bind_group()));
        let stage = registration.work(native.dispatch());
        self.stages.push(registration);
        stage
    }

    fn stream(
        &mut self,
        snapshot: &GpuHeightfieldPlacementRegistrationSnapshot,
        cell: SectorId,
        revision: u64,
    ) -> PlacementRenderStream {
        PlacementRenderStream {
            stream: PlacementStream {
                cell,
                revision,
                candidate_count: snapshot.candidate_count,
            },
            source_params: self.buffer(&snapshot.source_params),
            project_params: self.buffer(&snapshot.project_params),
            field_params: self.buffer(&snapshot.field_params),
            records: self.buffer(&snapshot.records),
            samples: self.buffer(&snapshot.samples),
            attributes: self.buffer(&snapshot.attributes),
            compact: self.buffer(&snapshot.compact),
            indirect: self.buffer(&snapshot.indirect),
            status: self.buffer(&snapshot.status),
            heights: self.buffer(&snapshot.heights),
            brushes: self.buffer(&snapshot.brushes),
            reset: self.stage(&snapshot.reset),
            work: None,
            empty_deletion: None,
        }
    }
}

impl Drop for Registrations {
    fn drop(&mut self) {
        self.stages.clear();
        for (_, buffer) in &self.buffers {
            self.device.release_buffer(buffer.imported.buffer);
        }
    }
}

struct SceneOwner {
    device: WgpuDevice,
    table: Arc<()>,
    texture: Option<rhi::TextureId>,
    view: Option<rhi::TextureViewId>,
    extent: rhi::Extent3d,
    format: rhi::TextureFormat,
}

impl Drop for SceneOwner {
    fn drop(&mut self) {
        if let Some(view) = self.view {
            self.device.destroy_texture_view(view);
        }
        if let Some(texture) = self.texture {
            self.device.destroy_texture(texture);
        }
    }
}

/// Adapter-owned color texture/view, pinned by every submitted frame using it.
#[derive(Clone)]
pub struct PlacementRhiScene(Arc<SceneOwner>);

impl PlacementRhiScene {
    /// Native handle for explicit caller readback/composition after suitable
    /// synchronization; do not destroy it while the scene/frame can be used.
    #[must_use]
    pub fn native_texture(&self) -> wgpu::Texture {
        self.0
            .device
            .wgpu_texture(self.0.texture.expect("owned scene texture"))
    }
}

struct DrawOwner {
    device: WgpuDevice,
    table: Arc<()>,
    format: rhi::TextureFormat,
    bind_layout: Option<rhi::BindGroupLayoutId>,
    layout: Option<rhi::PipelineLayoutId>,
    pipeline: Option<rhi::RenderPipelineId>,
}

impl Drop for DrawOwner {
    fn drop(&mut self) {
        if let Some(pipeline) = self.pipeline {
            self.device.destroy_render_pipeline(pipeline);
        }
        if let Some(layout) = self.layout {
            self.device.destroy_pipeline_layout(layout);
        }
        if let Some(layout) = self.bind_layout {
            self.device.destroy_bind_group_layout(layout);
        }
    }
}

#[derive(Clone)]
pub struct PlacementRhiDrawPipeline(Arc<DrawOwner>);

struct FrameResources {
    device: WgpuDevice,
    scene: Arc<SceneOwner>,
    registrations: Vec<FrameRegistration>,
    pipelines: Vec<Arc<DrawOwner>>,
    draw_groups: Vec<rhi::BindGroupId>,
}

struct FrameRegistration {
    lease: Arc<Mutex<Registrations>>,
    // Only deletion producers currently participate in upstream reuse.
    upstream_generation: Option<u64>,
}

impl Drop for FrameResources {
    fn drop(&mut self) {
        for group in &self.draw_groups {
            self.device.destroy_bind_group(*group);
        }
    }
}

/// Move-only declaration. Its request borrow is retained by `record` and only
/// ends at cancellation/Drop or actual submission; commands are not exposed.
pub struct PlacementRhiFrame<'bridge, 'request> {
    bridge: &'bridge mut PlacementRhiBridge,
    streams: Vec<PlacementRenderStream>,
    resident_upstreams: BTreeSet<SectorId>,
    resources: FrameResources,
    request: PhantomData<&'request ()>,
}

impl<'bridge, 'request> PlacementRhiFrame<'bridge, 'request> {
    fn validate(
        &self,
        registrations: &Mutex<Registrations>,
        pipeline: &PlacementRhiDrawPipeline,
    ) -> Result<(), PlacementRhiError> {
        if !Arc::ptr_eq(&lock(registrations).table, &self.bridge.table)
            || !Arc::ptr_eq(&pipeline.0.table, &self.bridge.table)
        {
            return Err(PlacementRhiError::ForeignResource);
        }
        if pipeline.0.format != self.resources.scene.format {
            return Err(PlacementRhiError::InvalidDrawPipeline);
        }
        Ok(())
    }

    fn draw(
        &mut self,
        stream: &PlacementRenderStream,
        compact: PlacementImportedBuffer,
        pipeline: &PlacementRhiDrawPipeline,
    ) -> PlacementDrawStage {
        self.resources.draw_groups.reserve(1);
        self.resources.pipelines.reserve(1);
        let group = self
            .bridge
            .device
            .create_bind_group(&rhi::BindGroupDescriptor {
                label: Some("world placement draw bindings".into()),
                layout: pipeline.0.bind_layout,
                entries: [stream.records, stream.samples, stream.attributes, compact]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, buffer)| rhi::BindGroupEntry {
                        binding: binding as u32,
                        resource: rhi::BindingResource::Buffer {
                            buffer: buffer.imported.buffer,
                            offset: 0,
                            size: Some(buffer.bytes),
                        },
                    })
                    .collect(),
            });
        self.resources.draw_groups.push(group);
        self.resources.pipelines.push(pipeline.0.clone());
        PlacementDrawStage {
            pipeline: pipeline.0.pipeline.expect("owned draw pipeline"),
            bind_group: group,
            indirect_offset: 0,
        }
    }

    pub fn add_heightfield(
        &mut self,
        producer: &'request RegisteredHeightfieldPlacement,
        cell: SectorId,
        revision: u64,
        pipeline: &PlacementRhiDrawPipeline,
    ) -> Result<(), PlacementRhiError> {
        self.validate(&producer.registrations, pipeline)?;
        let snapshot = producer
            .producer
            .registration_snapshot()
            .map_err(PlacementRhiError::Placement)?;
        let mut registrations = lock(&producer.registrations);
        let mut stream = registrations.stream(&snapshot, cell, revision);
        if let Some(active) = &snapshot.active {
            stream.work = Some(PlacementRenderWork {
                source: registrations.stage(&active.source),
                project: registrations.stage(&active.project),
                field: registrations.stage(&active.field),
                finalize: registrations.stage(&active.finalize),
                deletion: None,
                draw: self.draw(&stream, stream.compact, pipeline),
            });
        }
        self.resources.registrations.push(FrameRegistration {
            lease: producer.registrations.clone(),
            upstream_generation: None,
        });
        self.streams.push(stream);
        Ok(())
    }

    pub fn add_deletions(
        &mut self,
        producer: &'request RegisteredPlacementDeletions,
        cell: SectorId,
        revision: u64,
        pipeline: &PlacementRhiDrawPipeline,
    ) -> Result<(), PlacementRhiError> {
        self.validate(&producer.registrations, pipeline)?;
        let snapshot = producer
            .producer
            .registration_snapshot()
            .map_err(PlacementRhiError::Deletion)?;
        let mut registrations = lock(&producer.registrations);
        let mut stream = registrations.stream(&snapshot.upstream, cell, revision);
        let indirect = registrations.buffer(&snapshot.indirect);
        let clear_instance_count = registrations.stage(&snapshot.clear_instance_count);
        if let (Some(active), Some(filter)) = (&snapshot.upstream.active, &snapshot.active) {
            if registrations
                .resident_upstream
                .contains(snapshot.upstream.request_generation)
            {
                self.resident_upstreams.insert(cell);
            }
            let deletion = PlacementDeletionWork {
                request: snapshot.request,
                params: registrations.buffer(&snapshot.params),
                keys: registrations.buffer(&snapshot.keys),
                compact: registrations.buffer(&snapshot.compact),
                indirect,
                clear_instance_count,
                stage: registrations.stage(filter),
            };
            stream.work = Some(PlacementRenderWork {
                source: registrations.stage(&active.source),
                project: registrations.stage(&active.project),
                field: registrations.stage(&active.field),
                finalize: registrations.stage(&active.finalize),
                draw: self.draw(&stream, deletion.compact, pipeline),
                deletion: Some(deletion),
            });
        } else {
            stream.empty_deletion = Some(PlacementEmptyDeletionWork {
                indirect,
                clear_instance_count,
            });
        }
        self.resources.registrations.push(FrameRegistration {
            lease: producer.registrations.clone(),
            upstream_generation: snapshot
                .upstream
                .active
                .as_ref()
                .map(|_| snapshot.upstream.request_generation),
        });
        self.streams.push(stream);
        Ok(())
    }

    pub fn record(self) -> Result<PlacementRhiRecorded<'bridge, 'request>, PlacementRhiError> {
        let scene = &self.resources.scene;
        let target = PlacementSceneTarget {
            descriptor: TextureDesc::color(scene.format),
            imported: ImportedTexture {
                texture: scene.texture.expect("owned scene texture"),
                default_view: scene.view.expect("owned scene view"),
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        };
        let graph = build_placement_render_graph_with_resident_upstreams(
            scene.extent,
            target,
            &self.streams,
            &self.resident_upstreams,
        )
        .map_err(|e| PlacementRhiError::Graph(Box::new(e)))?;
        let recorded = graph
            .record(&self.bridge.device)
            .map_err(|e| PlacementRhiError::Graph(Box::new(e.into())))?;
        Ok(PlacementRhiRecorded {
            bridge: self.bridge,
            recorded: Some(recorded),
            resources: Some(self.resources),
            reused_upstream_count: self.resident_upstreams.len(),
            request: PhantomData,
        })
    }
}

/// Owns the only replayable commands for an unsubmitted frame. Dropping it
/// discards commands before explicitly releasing graph allocations and pins.
///
/// A recorded request cannot be replaced before it is submitted or cancelled:
///
/// ```compile_fail,E0502
/// use prism_world_gpu::placement_rhi::{PlacementRhiBridge, PlacementRhiScene,
///     PlacementRhiDrawPipeline, RegisteredHeightfieldPlacement};
/// use prism_world_stream::SectorId;
/// fn overwrite(bridge: &mut PlacementRhiBridge, scene: &PlacementRhiScene,
///     draw: &PlacementRhiDrawPipeline, producer: &mut RegisteredHeightfieldPlacement,
///     replacement: RegisteredHeightfieldPlacement) {
///     let mut frame = bridge.begin_frame(scene).unwrap();
///     frame.add_heightfield(producer, SectorId::new(0, 0, 0, 0), 1, draw).unwrap();
///     let recorded = frame.record().unwrap();
///     let _old = std::mem::replace(producer, replacement);
///     recorded.submit();
/// }
/// ```
///
/// The same restriction applies to full-key deletion requests during declaration:
///
/// ```compile_fail,E0502
/// use prism_world_gpu::placement_rhi::{PlacementRhiBridge, PlacementRhiScene,
///     PlacementRhiDrawPipeline, RegisteredPlacementDeletions};
/// use prism_world_stream::SectorId;
/// fn overwrite(bridge: &mut PlacementRhiBridge, scene: &PlacementRhiScene,
///     draw: &PlacementRhiDrawPipeline, producer: &mut RegisteredPlacementDeletions,
///     replacement: RegisteredPlacementDeletions) {
///     let mut frame = bridge.begin_frame(scene).unwrap();
///     frame.add_deletions(producer, SectorId::new(0, 0, 0, 0), 1, draw).unwrap();
///     let _old = std::mem::replace(producer, replacement);
///     frame.record().unwrap().submit();
/// }
/// ```
///
/// Native output invalidation is also excluded while commands borrow a proof:
///
/// ```compile_fail,E0502
/// use prism_world_gpu::placement_rhi::{PlacementRhiBridge, PlacementRhiScene,
///     PlacementRhiDrawPipeline, RegisteredPlacementDeletions};
/// use prism_world_stream::SectorId;
/// fn invalidate(bridge: &mut PlacementRhiBridge, scene: &PlacementRhiScene,
///     draw: &PlacementRhiDrawPipeline, producer: &mut RegisteredPlacementDeletions) {
///     let mut frame = bridge.begin_frame(scene).unwrap();
///     frame.add_deletions(producer, SectorId::new(0, 0, 0, 0), 1, draw).unwrap();
///     let recorded = frame.record().unwrap();
///     producer.invalidate_resident_upstream();
///     recorded.submit();
/// }
/// ```
#[must_use = "submit or drop to cancel this recorded placement frame"]
pub struct PlacementRhiRecorded<'bridge, 'request> {
    bridge: &'bridge mut PlacementRhiBridge,
    recorded: Option<RecordedGraph>,
    resources: Option<FrameResources>,
    reused_upstream_count: usize,
    request: PhantomData<&'request ()>,
}

impl PlacementRhiRecorded<'_, '_> {
    /// Number of nonempty deletion streams that import previously submitted
    /// placement outputs instead of dispatching reset/source/project/field/
    /// finalize. Deletion filtering and drawing still execute on each stream.
    #[must_use]
    pub fn reused_upstream_count(&self) -> usize {
        self.reused_upstream_count
    }

    /// Consumes commands exactly once. Resource pins transfer to the bridge's
    /// bounded pending queue; callback captures only a completion signal.
    /// Panics on native encode/submit errors; use [`Self::try_submit`] to handle
    /// these errors explicitly. Host callbacks may still panic in either API.
    pub fn submit(self) -> PlacementRhiSubmission {
        self.try_submit()
            .expect("placement native submission failed")
    }

    /// Captures native validation/internal/OOM errors without replacing host
    /// callbacks or relying on an outer error scope. Only an error-free native
    /// encode/submit establishes reusable upstream outputs. Native scope pops
    /// resolve immediately on the supported wgpu-core backends; this does not
    /// poll or wait for GPU completion. An error retains the packet until
    /// whole-queue `drain` and invalidates its upstream proofs before returning.
    pub fn try_submit(mut self) -> Result<PlacementRhiSubmission, PlacementRhiError> {
        let completion = Arc::new(Completion::default());
        let native_device = self.bridge.context.device();
        let oom_scope = native_device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = native_device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = native_device.push_error_scope(wgpu::ErrorFilter::Validation);
        // Withdraw old proofs before crossing any native or host-callback
        // boundary. A cached stream can fail too; publishing only after success
        // is insufficient if an earlier proof survives an encode/submit panic.
        for registration in &self
            .resources
            .as_ref()
            .expect("unsubmitted resources")
            .registrations
        {
            lock(&registration.lease).resident_upstream = ResidentUpstream::default();
        }
        let recorded = self.recorded.as_ref().expect("unsubmitted frame");
        let native = self.bridge.queue.encode(recorded.commands());
        let (commands, plan, allocations) = self
            .recorded
            .take()
            .expect("unsubmitted frame")
            .into_parts();
        drop(commands);
        drop(plan);
        // Establish conservative ownership before entering native submit: it
        // may enqueue work and then run a panicking user completion callback.
        // Unwinding must not take the unsubmitted/cancellation release path.
        self.bridge.pending.push(Pending {
            index: None,
            completion: completion.clone(),
            allocations: Some(allocations),
            resources: self.resources.take(),
        });
        let index = self.bridge.context.queue().submit([native]);
        let pending = self.bridge.pending.last_mut().expect("submitted packet");
        pending.index = Some(index.clone());
        let errors = [
            futures_lite::future::block_on(validation_scope.pop()),
            futures_lite::future::block_on(internal_scope.pop()),
            futures_lite::future::block_on(oom_scope.pop()),
        ];
        if let Some(error) = errors.into_iter().flatten().next() {
            // Native failure may have a returned index without a successful
            // command submission. Never publish a proof or retire by that
            // index; whole-queue drain is the conservative recovery boundary.
            pending.index = None;
            for registration in &pending
                .resources
                .as_ref()
                .expect("submitted resources")
                .registrations
            {
                lock(&registration.lease).resident_upstream = ResidentUpstream::default();
            }
            return Err(PlacementRhiError::Native(Box::new(error)));
        }
        // This is deliberately after submit, not declaration/record/encode.
        // If submit unwinds after enqueueing, the next frame conservatively
        // rebuilds rather than relying on an unproven cache entry.
        for registration in &pending
            .resources
            .as_ref()
            .expect("submitted resources")
            .registrations
        {
            if let Some(generation) = registration.upstream_generation {
                lock(&registration.lease)
                    .resident_upstream
                    .submitted(generation);
            }
        }
        let signal = completion.clone();
        self.bridge.context.queue().on_submitted_work_done(move || {
            signal.gpu_done.store(true, Ordering::Release);
        });
        Ok(PlacementRhiSubmission {
            table: self.bridge.table.clone(),
            index,
            completion,
        })
    }
}

impl Drop for PlacementRhiRecorded<'_, '_> {
    fn drop(&mut self) {
        if let Some(recorded) = self.recorded.take() {
            let (commands, plan, allocations) = recorded.into_parts();
            drop(commands);
            drop(plan);
            allocations.release(&self.bridge.device);
        }
    }
}

fn confirm_completed_prefix(pending: &[Pending], completion: &Arc<Completion>) {
    if let Some(last) = pending
        .iter()
        .position(|item| Arc::ptr_eq(&item.completion, completion))
    {
        for item in &pending[..=last] {
            item.completion.gpu_done.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    #[test]
    fn resident_proof_requires_submission_and_exact_generation() {
        let mut proof = ResidentUpstream::default();
        assert!(!proof.contains(0));
        proof.rewrite(0, 1, true);
        assert!(!proof.contains(1));
        proof.submitted(1);
        assert!(proof.contains(1));
        assert!(!proof.contains(0));
        proof.rewrite(1, 2, true);
        assert!(proof.contains(2));
        assert!(!proof.contains(1));
        // A mismatched old generation cannot transfer a proof.
        proof.rewrite(1, 3, true);
        assert!(!proof.contains(3));
    }

    #[test]
    fn changed_recipe_cannot_resurrect_resident_outputs() {
        let mut proof = ResidentUpstream::default();
        proof.submitted(1);
        proof.rewrite(1, 2, false); // A -> B, not submitted
        proof.rewrite(2, 3, false); // B -> A, not submitted
        assert!(!proof.contains(3));
        proof.rewrite(3, 4, true);
        assert!(!proof.contains(4));
        proof.submitted(4);
        assert!(proof.contains(4));
        proof = ResidentUpstream::default(); // native external write
        assert!(!proof.contains(4));
    }

    #[test]
    fn shutdown_contains_panics_without_dropping_host_payload() {
        struct HostPayload(Arc<AtomicBool>);
        impl Drop for HostPayload {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
                panic!("host panic payload destructor");
            }
        }
        let normal = AtomicBool::new(false);
        contain_shutdown_panic(|| normal.store(true, Ordering::Release));
        assert!(normal.load(Ordering::Acquire));

        let dropped = Arc::new(AtomicBool::new(false));
        let payload = HostPayload(dropped.clone());
        contain_shutdown_panic(|| std::panic::panic_any(payload));
        assert!(!dropped.load(Ordering::Acquire));
    }

    fn pending() -> Pending {
        Pending {
            index: None,
            completion: Arc::new(Completion::default()),
            allocations: None,
            resources: None,
        }
    }

    #[test]
    fn exact_completion_confirms_ordered_prefix_without_callbacks() {
        let packets = [pending(), pending(), pending()];
        // Includes an earlier possibly-submitted packet without an index.
        confirm_completed_prefix(&packets, &packets[1].completion);
        for (packet, expected) in packets.iter().zip([true, true, false]) {
            assert_eq!(packet.completion.gpu_done.load(Ordering::Acquire), expected);
            // Completion confirmation alone does not release allocations.
            assert!(!packet.completion.retired.load(Ordering::Acquire));
        }
    }

    #[test]
    fn missing_or_already_retired_receipt_cannot_confirm_later_packets() {
        let packets = [pending(), pending()];
        let retired = Arc::new(Completion::default());
        retired.retired.store(true, Ordering::Release);
        confirm_completed_prefix(&packets, &retired);
        assert!(packets
            .iter()
            .all(|packet| !packet.completion.gpu_done.load(Ordering::Acquire)));
    }
}
