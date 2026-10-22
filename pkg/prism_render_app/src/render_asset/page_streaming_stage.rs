//! ECS wiring for the fixed-page CPU-to-GPU streaming slice.
//!
//! The main world owns requests, source reads, and CPU residency. Extraction
//! moves owned page payloads into the render world. The render prepare stage
//! performs the real queue upload, while cleanup only retires replacements after
//! the runtime has reported a real submission completion value.

use alloc::sync::{Arc, Weak};
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};
use std::sync::Mutex;

use prism_app::{App, PostUpdate};
use prism_ecs::prelude::{FromWorld, Res, ResMut, Resource};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules, SystemSet, SystemSetId};
use prism_ecs::world::World;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{FrameSync, Instance, RenderDevice};

use super::page_streaming::StreamGeneration;
use super::PagePoolError;
use crate::label::RenderApp;
use crate::render_asset::{
    FixedPagePoolDescriptor, FixedPageRenderQueue, FixedPageResidencyBridge,
    FixedPageStreamingSource, GpuPrepareBudget, PageResidencyBridgeError, PrepareCancellation,
};
use crate::runtime_resource::RenderRuntimeResource;
use crate::sets::RenderSystems;
use crate::startup::GpuResourceAppExt;

/// Main-world policy for the fixed-page streaming bridge.
#[derive(Clone, Debug, Resource)]
pub struct FixedPageStreamingConfig {
    /// GPU fixed-page backing-buffer descriptor.
    pub gpu_pool: FixedPagePoolDescriptor,
    /// Maximum number of CPU range reads performed by the automatic pump.
    pub max_io_pages_per_frame: usize,
    /// Maximum number of owned payloads uploaded in one render prepare pass.
    pub max_uploads_per_frame: usize,
    /// Maximum number of decoded feedback demands admitted to CPU residency
    /// in one main-world frame.
    pub max_feedback_admissions_per_frame: usize,
}

/// Last-frame diagnostics for the fixed-page streaming bridge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Resource)]
pub struct FixedPageStreamingStats {
    /// Number of decoded feedback demands admitted to CPU residency last frame.
    pub feedback_admitted: usize,
    /// Number of decoded feedback demands rejected by page admission last frame.
    pub feedback_request_errors: usize,
    /// Number of decoded feedback demands still queued after admission.
    pub feedback_queued: usize,
    /// Number of CPU page reads attempted by the last main-world pump.
    pub io_attempted: usize,
    /// Number of CPU page reads completed by the last main-world pump.
    pub io_loaded: usize,
    /// Number of CPU page read/admission errors observed by the last pump.
    pub io_errors: usize,
    /// Number of page uploads committed by the last render prepare pass.
    pub uploads: usize,
    /// Number of page uploads rejected by the last render prepare pass.
    pub upload_errors: usize,
    /// The last explicit GPU bridge error, if any.
    pub last_upload_error: Option<PageResidencyBridgeError>,
}

impl FixedPageStreamingConfig {
    /// Creates a config with one bounded I/O and upload operation per frame.
    #[must_use]
    pub const fn new(gpu_pool: FixedPagePoolDescriptor) -> Self {
        Self {
            gpu_pool,
            max_io_pages_per_frame: 1,
            max_uploads_per_frame: 1,
            max_feedback_admissions_per_frame: 1,
        }
    }
}

#[derive(Clone, Resource)]
pub(super) struct FixedPageStreamingRenderConfig {
    pub(super) gpu_pool: FixedPagePoolDescriptor,
    max_uploads_per_frame: usize,
}

/// Orders table publication after physical page uploads within resource preparation.
pub(super) struct FixedPageUploadSet;

impl SystemSet for FixedPageUploadSet {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::of::<Self>()
    }
}

// A callback must not keep the device containing that callback alive. The live
// resource owns the device; retired records use a weak creator handle and let
// device shutdown release its native resource tables if completion is lost.
const MAX_POOL_COMPLETION_CALLBACKS: usize = 3;

struct PoolRetirement<D> {
    device: Weak<D>,
    state: Mutex<PoolRetirementState>,
}

struct PoolRetirementState {
    latest: StreamGeneration,
    completed: bool,
    outstanding: usize,
    retired: Option<FixedPageResidencyBridge>,
}

impl<D: RenderDevice + Send + Sync + 'static> PoolRetirement<D> {
    fn protect(self: &Arc<Self>, queue: &impl FrameSync) -> StreamGeneration {
        let generation = StreamGeneration::default();
        let register = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.latest = generation.clone();
            state.completed = false;
            if state.outstanding < MAX_POOL_COMPLETION_CALLBACKS {
                state.outstanding += 1;
                true
            } else {
                false
            }
        };
        if !register {
            // Backends need not deliver callbacks without polling. Bound their
            // storage even then; only a successful real barrier certifies the
            // newer use. Failed barriers quarantine retirement until teardown.
            if queue.wait_idle() {
                self.complete(&generation);
            }
            return generation;
        }
        let owner = self.clone();
        let callback_generation = generation.clone();
        queue.on_submitted_work_done(Box::new(move || {
            let retired = {
                let mut state = owner
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.outstanding -= 1;
                if state.latest != callback_generation {
                    return;
                }
                state.completed = true;
                state.retired.take()
            };
            if let (Some(bridge), Some(device)) = (retired, owner.device.upgrade()) {
                bridge.destroy(device.as_ref());
            }
        }));
        generation
    }

    fn complete(&self, generation: &StreamGeneration) {
        let retired = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if &state.latest != generation {
                return;
            }
            state.completed = true;
            state.retired.take()
        };
        if let (Some(bridge), Some(device)) = (retired, self.device.upgrade()) {
            bridge.destroy(device.as_ref());
        }
    }

    fn retire(&self, bridge: FixedPageResidencyBridge) {
        let ready = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.completed {
                Some(bridge)
            } else {
                state.retired = Some(bridge);
                None
            }
        };
        if let (Some(bridge), Some(device)) = (ready, self.device.upgrade()) {
            bridge.destroy(device.as_ref());
        }
    }
}

struct OwnedPageBridge<D: RenderDevice + Send + Sync + 'static> {
    bridge: Option<FixedPageResidencyBridge>,
    owner: Arc<PoolRetirement<D>>,
    device: Arc<D>,
}

impl<D: RenderDevice + Send + Sync + 'static> OwnedPageBridge<D> {
    fn new(bridge: FixedPageResidencyBridge, device: Arc<D>) -> Self {
        Self {
            bridge: Some(bridge),
            owner: Arc::new(PoolRetirement {
                device: Arc::downgrade(&device),
                state: Mutex::new(PoolRetirementState {
                    latest: StreamGeneration::default(),
                    completed: true,
                    outstanding: 0,
                    retired: None,
                }),
            }),
            device,
        }
    }
}

impl<D: RenderDevice + Send + Sync + 'static> Deref for OwnedPageBridge<D> {
    type Target = FixedPageResidencyBridge;

    fn deref(&self) -> &Self::Target {
        self.bridge.as_ref().expect("live page bridge")
    }
}

impl<D: RenderDevice + Send + Sync + 'static> DerefMut for OwnedPageBridge<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bridge.as_mut().expect("live page bridge")
    }
}

impl<D: RenderDevice + Send + Sync + 'static> Drop for OwnedPageBridge<D> {
    fn drop(&mut self) {
        if let Some(bridge) = self.bridge.take() {
            self.owner.retire(bridge);
        }
    }
}

/// Render-world GPU state for one fixed-page stream.
pub struct FixedPageStreamingGpu<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    bridge: OwnedPageBridge<I::Device>,
    max_uploads_per_frame: usize,
    pending_completion: Option<(u64, u64)>,
    source_generation: Option<StreamGeneration>,
    marker: PhantomData<fn() -> I>,
}

impl<I> Resource for FixedPageStreamingGpu<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
}

impl<I> FixedPageStreamingGpu<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    /// Committed physical allocations; the bridge remains owned by prepare.
    pub fn bridge(&self) -> &FixedPageResidencyBridge {
        &self.bridge
    }

    pub(super) fn has_source(&self, generation: &StreamGeneration) -> bool {
        self.source_generation.as_ref() == Some(generation)
    }

    pub(super) fn owns_device(&self, device: &Arc<I::Device>) -> bool {
        Arc::ptr_eq(&self.bridge.device, device)
    }

    fn finish_submission(&mut self, runtime: &prism_render_runtime::RenderRuntime<I>) {
        if !self.owns_device(&runtime.device_handle()) {
            return;
        }
        // Cleanup follows all frame consumers, including native queue submits.
        // The record survives removal of this GPU resource on a later startup.
        self.bridge.owner.protect(runtime.queue());
        let Some(pending) = self.pending_completion else {
            return;
        };
        if pending.0 != runtime.device_generation() {
            // Idling a replacement device cannot prove completion on its predecessor.
            return;
        }
        if retirement_was_submitted(
            pending,
            runtime.device_generation(),
            runtime.next_completion_value(),
        ) {
            self.pending_completion = None;
        } else if runtime.wait_idle() {
            // No tracked submit covered this retirement. A queue-wide barrier
            // also covers native feedback/consumer work submitted directly.
            if self
                .bridge
                .reclaim_completed(GpuCompletionValue(u64::MAX))
                .is_ok()
            {
                self.pending_completion = None;
            }
        }
    }
}

impl<I> FromWorld for FixedPageStreamingGpu<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    fn from_world(world: &mut World) -> Self {
        let config = world.resource::<FixedPageStreamingRenderConfig>().clone();
        let runtime = world.resource::<RenderRuntimeResource<I>>();
        let device = runtime.runtime().device_handle();
        if let Some(previous) = world.get_resource::<Self>() {
            if previous.owns_device(&device) {
                // Also cover same-device startup requested between frames. A
                // different runtime's queue must never certify the old pool.
                let proof = previous.bridge.owner.protect(runtime.runtime().queue());
                if runtime.runtime().wait_idle() {
                    previous.bridge.owner.complete(&proof);
                }
            }
        }
        let mut budget = GpuPrepareBudget::unlimited();
        let cancellation = PrepareCancellation::new(false);
        let bridge = FixedPageResidencyBridge::new_with_budget(
            runtime.runtime().device(),
            runtime.runtime().queue(),
            &config.gpu_pool,
            &mut budget,
            &cancellation,
        )
        .expect("fixed page streaming pool descriptor must be valid");
        world
            .resource_mut::<FixedPageRenderQueue>()
            .initialize_gpu();
        Self {
            bridge: OwnedPageBridge::new(bridge, device),
            max_uploads_per_frame: config.max_uploads_per_frame,
            pending_completion: None,
            source_generation: None,
            marker: PhantomData,
        }
    }
}

/// Registers the complete fixed-page stream path on an app with a render app.
///
/// The caller supplies the CPU source so the path cannot silently fall back to
/// a fake reader. Requests made during `Update` are pumped in `PostUpdate`,
/// extracted as owned bytes, and uploaded during render preparation.
pub fn register_fixed_page_streaming<I>(
    app: &mut App,
    source: FixedPageStreamingSource,
    config: FixedPageStreamingConfig,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    app.insert_resource(source);
    app.insert_resource(config.clone());
    app.insert_resource(FixedPageStreamingStats::default());
    app.add_systems(PostUpdate, pump_fixed_page_streaming_io);
    app.add_extract(RenderApp, extract_fixed_page_streaming);

    let render = app
        .sub_app_mut(RenderApp)
        .expect("PrismRenderPlugin must be installed before fixed-page streaming");
    render.world.insert_resource(FixedPageRenderQueue::new());
    render
        .world
        .insert_resource(FixedPageStreamingStats::default());
    render
        .world
        .insert_resource(FixedPageStreamingRenderConfig {
            gpu_pool: config.gpu_pool,
            max_uploads_per_frame: config.max_uploads_per_frame,
        });
    render.init_gpu_resource::<FixedPageStreamingGpu<I>>();
    let schedules = render.world.resource_mut::<Schedules>();
    if !schedules.contains(crate::Render) {
        schedules.insert(crate::Render, Schedule::new());
    }
    let schedule = schedules
        .get_mut(crate::Render)
        .expect("render schedule exists");
    schedule.add_systems(
        prepare_fixed_page_streaming::<I>
            .in_set(RenderSystems::PrepareResources)
            .in_set(FixedPageUploadSet),
    );
    schedule
        .add_systems(finish_fixed_page_streaming_submission::<I>.in_set(RenderSystems::Cleanup));
}

fn pump_fixed_page_streaming_io(
    mut source: ResMut<FixedPageStreamingSource>,
    config: Res<FixedPageStreamingConfig>,
    mut stats: ResMut<FixedPageStreamingStats>,
    #[cfg(feature = "fixed-page-feedback")] feedback: Option<ResMut<super::FixedPageFeedback>>,
) {
    let frame = source.advance_feedback_frame();
    let admissions = source.admit_feedback(frame, config.max_feedback_admissions_per_frame);
    let mut admitted = admissions.iter().filter(|a| a.result.is_ok()).count();
    let mut errors = admissions.len() - admitted;
    #[cfg(feature = "fixed-page-feedback")]
    if let Some(mut feedback) = feedback {
        let remaining = config
            .max_feedback_admissions_per_frame
            .saturating_sub(admissions.len());
        // Native misses are admitted in this pump or dropped. They never lose
        // their epoch by waiting in the source queue across registry mutations.
        feedback.admit_completed(&mut source, frame, remaining);
        let native = source.admit_feedback(frame, remaining);
        admitted += native.iter().filter(|a| a.result.is_ok()).count();
        errors += native.iter().filter(|a| a.result.is_err()).count();
    }
    let report = source.pump_io(config.max_io_pages_per_frame);
    stats.feedback_admitted = admitted;
    stats.feedback_request_errors = errors;
    stats.feedback_queued = source.feedback_len();
    stats.io_attempted = report.attempted;
    stats.io_loaded = report.loaded;
    stats.io_errors = report.errors.len();
}

fn extract_fixed_page_streaming(main: &mut World, render: &mut World) {
    #[cfg(feature = "fixed-page-feedback")]
    super::page_feedback::extract_feedback(main, render);
    let Some(config) = main.get_resource::<FixedPageStreamingConfig>().cloned() else {
        return;
    };
    let Some(source) = main.get_resource_mut::<FixedPageStreamingSource>() else {
        return;
    };
    let mut queue = render
        .get_resource_mut::<FixedPageRenderQueue>()
        .map(core::mem::take)
        .unwrap_or_default();
    source.extract_to_render(&mut queue);
    if let Some(existing) = render.get_resource_mut::<FixedPageRenderQueue>() {
        *existing = queue;
    } else {
        render.insert_resource(queue);
    }
    if render
        .get_resource::<FixedPageStreamingRenderConfig>()
        .is_none()
    {
        render.insert_resource(FixedPageStreamingRenderConfig {
            gpu_pool: config.gpu_pool,
            max_uploads_per_frame: config.max_uploads_per_frame,
        });
    }
}

pub(super) fn prepare_fixed_page_streaming<I>(
    mut queue: ResMut<FixedPageRenderQueue>,
    mut gpu: ResMut<FixedPageStreamingGpu<I>>,
    mut stats: ResMut<FixedPageStreamingStats>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    stats.uploads = 0;
    stats.upload_errors = 0;
    stats.last_upload_error = None;
    if !gpu.owns_device(&runtime.runtime().device_handle()) {
        stats.upload_errors = 1;
        stats.last_upload_error = Some(PageResidencyBridgeError::DeviceMismatch);
        return;
    }
    let completed = GpuCompletionValue(reclaimable_value(
        gpu.pending_completion,
        runtime.runtime().device_generation(),
        runtime.runtime().next_completion_value(),
        runtime.runtime().completed_value(),
    ));
    let _ = gpu.bridge.reclaim_completed(completed);
    let reclaim_after = GpuCompletionValue(runtime.runtime().next_completion_value());
    let pending = (runtime.runtime().device_generation(), reclaim_after.0);
    if gpu.source_generation != queue.source_generation {
        if gpu.bridge.pool().resident_page_count() != 0 {
            gpu.pending_completion = Some(pending);
        }
        if let Err(error) = gpu.bridge.release_all_deferred(reclaim_after) {
            stats.upload_errors += 1;
            stats.last_upload_error = Some(error);
            return;
        }
        gpu.source_generation = queue.source_generation.clone();
    }
    while let Some(release) = queue.pop_release() {
        match gpu.bridge.release_page_for_slot_deferred(
            &release.slot,
            release.page,
            release.revision,
            reclaim_after,
        ) {
            Ok(()) => gpu.pending_completion = Some(pending),
            Err(PageResidencyBridgeError::UnknownPage(_)) => {}
            Err(error) => {
                stats.upload_errors += 1;
                stats.last_upload_error = Some(error);
                // An obsolete release must not block independent releases or
                // the current source's uploads. Its allocation was untouched.
            }
        }
    }
    for _ in 0..gpu.max_uploads_per_frame {
        let Some(payload) = queue.pop() else {
            break;
        };
        match gpu.bridge.upload_page_bytes_for_slot(
            &payload.slot,
            payload.page,
            payload.revision,
            &payload.bytes,
            reclaim_after,
            runtime.runtime().queue(),
        ) {
            Ok(event) => {
                stats.uploads += 1;
                if matches!(event, crate::FixedPagePoolEvent::Replaced { .. }) {
                    gpu.pending_completion = Some(pending);
                }
            }
            Err(error) => {
                stats.upload_errors += 1;
                stats.last_upload_error = Some(error);
                if matches!(
                    error,
                    PageResidencyBridgeError::Pool(PagePoolError::OutOfPages { .. })
                ) {
                    queue.push_front(payload);
                    break;
                }
                // Malformed or obsolete payloads must not starve later pages.
            }
        }
    }
}

fn finish_fixed_page_streaming_submission<I>(
    mut gpu: ResMut<FixedPageStreamingGpu<I>>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    gpu.finish_submission(runtime.runtime());
}

fn retirement_was_submitted(pending: (u64, u64), generation: u64, next: u64) -> bool {
    pending.0 == generation && next > pending.1
}

fn reclaimable_value(
    pending: Option<(u64, u64)>,
    generation: u64,
    next: u64,
    completed: u64,
) -> u64 {
    match pending {
        Some(proof) if proof.0 != generation => 0,
        Some(proof) if !retirement_was_submitted(proof, generation, next) => {
            // At saturation, an earlier completed MAX must not certify a later
            // retirement for which no new submission value can be assigned.
            completed.min(proof.1.saturating_sub(1))
        }
        _ => completed,
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use prism_app::App;
    use prism_asset::{
        AssetSources, FeedbackDemand, FeedbackDemandQueue, MemSource, PageBudget, PageKey,
        PageResidency, PageSize,
    };
    use prism_render_driver::{
        BufferUsages, CommandBuffer, CompletionToken, DeviceDescriptor, RequestAdapterOptions,
    };
    use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

    use super::*;
    use crate::{
        render_asset::{
            FixedPagePoolDescriptor, FixedPageRenderQueue, FixedPageStreamingSource,
            FixedPageStreamingStats,
        },
        PrismRenderPlugin, RenderApp,
    };

    #[test]
    fn registration_runs_cpu_io_extract_prepare_and_gpu_upload() {
        let mut memory = MemSource::new();
        memory.insert("page.bin", b"abcdefgh".to_vec());
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let residency =
            PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(4, 1).unwrap()).unwrap();
        let source = FixedPageStreamingSource::with_feedback_queue(
            sources,
            residency,
            FeedbackDemandQueue::new(4, 60),
        );
        let mut source = source;
        source.submit_feedback(FeedbackDemand::new(
            PageKey::from_sources(source.sources(), "memory://page.bin", 0),
            0,
            1,
            0,
            prism_tasks::Priority::High,
        ));

        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
        register_fixed_page_streaming::<NullInstance>(
            &mut app,
            source,
            FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
                4,
                1,
                BufferUsages::COPY_DST,
            )),
        );
        app.update();

        let main_stats = app.world().resource::<FixedPageStreamingStats>();
        assert_eq!(main_stats.io_attempted, 1);
        assert_eq!(main_stats.io_loaded, 1);
        assert_eq!(main_stats.io_errors, 0);
        assert_eq!(main_stats.feedback_admitted, 1);
        assert_eq!(main_stats.feedback_request_errors, 0);
        assert_eq!(main_stats.feedback_queued, 0);
        let render = app.sub_app_mut(RenderApp).unwrap();
        assert!(render.world.resource::<FixedPageRenderQueue>().is_empty());
        let render_stats = render.world.resource::<FixedPageStreamingStats>();
        assert_eq!(render_stats.uploads, 1);
        assert_eq!(render_stats.upload_errors, 0);
        let gpu = render
            .world
            .resource::<FixedPageStreamingGpu<NullInstance>>();
        assert_eq!(gpu.bridge.pool().resident_page_count(), 1);
    }

    /// A real-device branch for the same feedback-driven vertical slice. The
    /// test is intentionally explicit about skipping when no adapter exists;
    /// Null-RHI coverage above is the mandatory deterministic regression.
    #[test]
    fn gpu_feedback_drives_fixed_page_upload_and_readback_when_available() {
        use prism_render_driver_wgpu::WgpuInstance;

        let Some((_device, _queue)) = WgpuInstance::new().request_default_device().ok() else {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU is set but no adapter was found"
            );
            eprintln!("SKIPPED fixed-page feedback GPU test: no adapter");
            return;
        };
        let mut memory = MemSource::new();
        memory.insert("page.bin", b"abcdefgh".to_vec());
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let residency =
            PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(4, 1).unwrap()).unwrap();
        let mut source = FixedPageStreamingSource::with_feedback_queue(
            sources,
            residency,
            FeedbackDemandQueue::new(4, 60),
        );
        let key = PageKey::from_sources(source.sources(), "memory://page.bin", 1);
        source.submit_feedback(FeedbackDemand::new(
            key,
            0,
            10,
            0,
            prism_tasks::Priority::Critical,
        ));

        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(WgpuInstance::new()));
        register_fixed_page_streaming::<WgpuInstance>(
            &mut app,
            source,
            FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
                4,
                1,
                BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )),
        );
        app.update();

        let main = app.world();
        let page = main
            .resource::<FixedPageStreamingSource>()
            .residency()
            .eviction_candidates()[0];
        let render = app.sub_app_mut(RenderApp).unwrap();
        let gpu = render
            .world
            .resource::<FixedPageStreamingGpu<WgpuInstance>>();
        let allocation = gpu.bridge.allocation(page).expect("feedback page upload");
        let bytes = render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime()
            .queue()
            .read_buffer(
                gpu.bridge.pool().buffer(),
                allocation.offset_bytes(gpu.bridge.pool().page_size()) + 4,
            );
        let start = allocation.offset_bytes(gpu.bridge.pool().page_size()) as usize;
        assert_eq!(&bytes[start..start + 4], b"efgh");
        assert_eq!(
            render.world.resource::<FixedPageStreamingStats>().uploads,
            1
        );
    }

    #[test]
    fn cpu_eviction_retires_the_gpu_page_through_the_render_queue() {
        let mut memory = MemSource::new();
        memory.insert("page.bin", b"abcdefgh".to_vec());
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let residency =
            PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(4, 1).unwrap()).unwrap();
        let mut source = FixedPageStreamingSource::new(sources, residency);
        source
            .request_page("memory://page.bin", 0, prism_tasks::Priority::High, 1)
            .unwrap();

        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
        register_fixed_page_streaming::<NullInstance>(
            &mut app,
            source,
            FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
                4,
                1,
                BufferUsages::COPY_DST,
            )),
        );
        app.update();

        let page = app
            .world()
            .resource::<FixedPageStreamingSource>()
            .residency()
            .eviction_candidates()[0];
        app.world_mut()
            .resource_mut::<FixedPageStreamingSource>()
            .residency_mut()
            .evict(page)
            .unwrap();
        app.update();

        let render = app.sub_app_mut(RenderApp).unwrap();
        let gpu = render
            .world
            .resource::<FixedPageStreamingGpu<NullInstance>>();
        assert_eq!(gpu.bridge.pool().resident_page_count(), 0);
        assert_eq!(gpu.bridge.pool().pending_reclaim_page_count(), 1);
        assert_eq!(gpu.bridge.pool().free_page_count(), 0);
        assert_eq!(
            render
                .world
                .resource::<FixedPageStreamingStats>()
                .upload_errors,
            0
        );
    }

    #[test]
    fn zero_upload_budget_keeps_owned_payload_queued() {
        let mut memory = MemSource::new();
        memory.insert("page.bin", b"abcdefgh".to_vec());
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let residency =
            PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(4, 1).unwrap()).unwrap();
        let mut source = FixedPageStreamingSource::new(sources, residency);
        source
            .request_page("memory://page.bin", 0, prism_tasks::Priority::High, 1)
            .unwrap();

        let mut config = FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
            4,
            1,
            BufferUsages::COPY_DST,
        ));
        config.max_uploads_per_frame = 0;
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
        register_fixed_page_streaming::<NullInstance>(&mut app, source, config);
        app.update();

        let render = app.sub_app_mut(RenderApp).unwrap();
        assert_eq!(
            render.world.resource::<FixedPageStreamingStats>().uploads,
            0
        );
        assert_eq!(render.world.resource::<FixedPageRenderQueue>().len(), 1);
        assert_eq!(
            render
                .world
                .resource::<FixedPageStreamingGpu<NullInstance>>()
                .bridge
                .pool()
                .resident_page_count(),
            0
        );
    }

    fn requested_source(index: u64) -> FixedPageStreamingSource {
        let mut memory = MemSource::new();
        memory.insert("page.bin", b"abcdefgh".to_vec());
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let residency =
            PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(8, 2).unwrap()).unwrap();
        let mut source = FixedPageStreamingSource::new(sources, residency);
        source
            .request_page("memory://page.bin", index, prism_tasks::Priority::High, 1)
            .unwrap();
        source
    }

    fn streaming_app(pool_pages: u32) -> App {
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::new(NullInstance::new()));
        let mut config = FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
            4,
            pool_pages,
            BufferUsages::COPY_DST,
        ));
        config.max_uploads_per_frame = 2;
        register_fixed_page_streaming::<NullInstance>(&mut app, requested_source(0), config);
        app.update();
        app
    }

    #[test]
    fn source_replacement_retires_old_slot_before_reusing_cpu_identity() {
        let mut app = streaming_app(1);
        let old_source = app.world().resource::<FixedPageStreamingSource>();
        let old_page = old_source.residency().eviction_candidates()[0];
        let revision = old_source.sources().revision();
        let old = app
            .get_sub_app(RenderApp)
            .unwrap()
            .world
            .resource::<FixedPageStreamingGpu<NullInstance>>()
            .bridge
            .allocation(old_page)
            .unwrap();
        let replacement = requested_source(1);
        assert_eq!(replacement.sources().revision(), revision);
        app.world_mut().insert_resource(replacement);
        app.update();
        let render = app.sub_app_mut(RenderApp).unwrap();
        let gpu = render
            .world
            .resource::<FixedPageStreamingGpu<NullInstance>>();
        assert!(gpu.bridge.allocation(old_page).is_none());
        assert_eq!(gpu.bridge.pool().resident_page_count(), 0);
        assert_eq!(gpu.bridge.pool().pending_reclaim_page_count(), 1);
        assert_eq!(render.world.resource::<FixedPageRenderQueue>().len(), 1);
        app.update();
        let render = app.get_sub_app(RenderApp).unwrap();
        let gpu = render
            .world
            .resource::<FixedPageStreamingGpu<NullInstance>>();
        let new = gpu.bridge.allocation(old_page).unwrap();
        assert_eq!(new.first_page(), old.first_page());
        assert_ne!(new.token(), old.token());
        assert_eq!(gpu.bridge.pool().resident_page_count(), 1);
        assert_eq!(gpu.bridge.pool().pending_reclaim_page_count(), 0);
        assert_eq!(gpu.bridge.pool().free_page_count(), 0);
    }

    #[test]
    fn gpu_resource_rebuild_reuploads_resident_bytes_without_io() {
        let mut app = streaming_app(1);
        {
            let render = app.sub_app_mut(RenderApp).unwrap();
            let device = render
                .world
                .resource::<RenderRuntimeResource<NullInstance>>()
                .runtime()
                .device_handle();
            let baseline = device.live_resource_count();
            for _ in 0..4 {
                let rebuilt = FixedPageStreamingGpu::<NullInstance>::from_world(&mut render.world);
                assert_eq!(device.live_resource_count(), baseline + 1);
                render.world.insert_resource(rebuilt);
                assert_eq!(
                    device.live_resource_count(),
                    baseline,
                    "repeated replacement releases the predecessor pool"
                );
                assert!(!render
                    .world
                    .resource::<FixedPageStreamingGpu<NullInstance>>()
                    .bridge
                    .pool()
                    .is_destroyed());
            }
        }
        app.update();
        assert_eq!(
            app.world()
                .resource::<FixedPageStreamingStats>()
                .io_attempted,
            0
        );
        let render = app.get_sub_app(RenderApp).unwrap();
        assert_eq!(
            render.world.resource::<FixedPageStreamingStats>().uploads,
            1
        );
        assert_eq!(
            render
                .world
                .resource::<FixedPageStreamingGpu<NullInstance>>()
                .bridge
                .pool()
                .resident_page_count(),
            1
        );
    }

    #[derive(Default)]
    struct ControlledCompletionQueue {
        callbacks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
        idle: core::sync::atomic::AtomicBool,
    }

    impl ControlledCompletionQueue {
        fn complete(&self, index: usize) {
            let callback = self.callbacks.lock().unwrap().remove(index);
            callback();
        }

        fn pending(&self) -> usize {
            self.callbacks.lock().unwrap().len()
        }
    }

    impl FrameSync for ControlledCompletionQueue {
        fn submit_tracked(&self, _: &[CommandBuffer]) -> CompletionToken {
            Box::new(())
        }

        fn wait_for(&self, _: &(dyn core::any::Any + Send + Sync)) -> bool {
            false
        }

        fn wait_idle(&self) -> bool {
            self.idle.load(core::sync::atomic::Ordering::Acquire)
        }

        fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
            self.callbacks.lock().unwrap().push(callback);
        }
    }

    fn owned_null_pool() -> (OwnedPageBridge<NullDevice>, Arc<NullDevice>) {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap();
        let device = Arc::new(device);
        let bridge = FixedPageResidencyBridge::new_with_budget(
            device.as_ref(),
            &queue,
            &FixedPagePoolDescriptor::new(4, 1, BufferUsages::COPY_DST),
            &mut GpuPrepareBudget::unlimited(),
            &PrepareCancellation::new(false),
        )
        .unwrap();
        (OwnedPageBridge::new(bridge, device.clone()), device)
    }

    #[test]
    fn pending_pool_retirement_uses_latest_creator_callback_and_never_new_device_ids() {
        let (old, old_device) = owned_null_pool();
        let (new, new_device) = owned_null_pool();
        assert_eq!(
            old.pool().buffer(),
            new.pool().buffer(),
            "independent devices reuse numeric IDs"
        );
        assert!(!Arc::ptr_eq(&old_device, &new_device));
        let old_queue = ControlledCompletionQueue::default();
        let new_queue = ControlledCompletionQueue::default();
        old.owner.protect(&old_queue);
        old.owner.protect(&old_queue);
        new.owner.protect(&new_queue);
        drop(old);
        assert_eq!(old_device.live_resource_count(), 1);
        new_queue.complete(0);
        assert_eq!(
            old_device.live_resource_count(),
            1,
            "replacement completion is irrelevant"
        );
        old_queue.complete(0);
        assert_eq!(
            old_device.live_resource_count(),
            1,
            "older creator callback cannot certify newer uses"
        );
        old_queue.complete(0);
        assert_eq!(old_device.live_resource_count(), 0);
        assert_eq!(
            new_device.live_resource_count(),
            1,
            "the new buffer remains live"
        );
        assert!(!new.pool().is_destroyed());
        drop(new);
        assert_eq!(new_device.live_resource_count(), 0);
    }

    #[test]
    fn pool_completion_callbacks_are_bounded_without_polling_or_frame_count_proofs() {
        let (pool, device) = owned_null_pool();
        let queue = ControlledCompletionQueue::default();
        for _ in 0..100 {
            pool.owner.protect(&queue);
            assert!(queue.pending() <= MAX_POOL_COMPLETION_CALLBACKS);
        }
        assert_eq!(queue.pending(), MAX_POOL_COMPLETION_CALLBACKS);
        while queue.pending() != 0 {
            queue.complete(0);
        }
        assert!(
            !pool.owner.state.lock().unwrap().completed,
            "older callbacks do not cover later failed barriers"
        );
        // This is a real queue-idle proof, independent of callback delivery.
        for _ in 0..MAX_POOL_COMPLETION_CALLBACKS {
            pool.owner.protect(&queue);
        }
        queue
            .idle
            .store(true, core::sync::atomic::Ordering::Release);
        pool.owner.protect(&queue);
        drop(pool);
        assert_eq!(device.live_resource_count(), 0);
        while queue.pending() != 0 {
            queue.complete(0);
        }
        assert_eq!(
            device.live_resource_count(),
            0,
            "late callbacks cannot destroy twice"
        );
    }

    #[test]
    fn pending_or_failed_completion_cannot_keep_shutdown_device_alive() {
        for exceed_callback_budget in [false, true] {
            let (pool, device) = owned_null_pool();
            let device_lifetime = Arc::downgrade(&device);
            let record_lifetime = Arc::downgrade(&pool.owner);
            let queue = ControlledCompletionQueue::default();
            let count = if exceed_callback_budget { 100 } else { 1 };
            for _ in 0..count {
                pool.owner.protect(&queue);
            }
            drop(pool);
            assert_eq!(device.live_resource_count(), 1, "no completion proof yet");
            drop(device);
            assert!(
                device_lifetime.upgrade().is_none(),
                "callbacks must not retain the shutdown device"
            );
            // Both cancellation on queue teardown and late delivery are safe.
            if exceed_callback_budget {
                drop(queue);
            } else {
                queue.complete(0);
            }
            assert!(
                record_lifetime.upgrade().is_none(),
                "shutdown releases the retired host record too"
            );
        }
    }

    #[test]
    fn completion_before_retirement_releases_pool_on_drop() {
        let (pool, device) = owned_null_pool();
        pool.owner.protect(&NullQueue::new());
        assert_eq!(device.live_resource_count(), 1);
        drop(pool);
        assert_eq!(device.live_resource_count(), 0);
    }

    fn pool_startup_world() -> World {
        let mut world = World::new();
        world.insert_resource(RenderRuntimeResource::new(
            prism_render_runtime::RenderRuntime::headless(
                NullInstance::new(),
                prism_render_driver::Extent3d::new_2d(4, 4),
            )
            .unwrap(),
        ));
        world.insert_resource(FixedPageStreamingRenderConfig {
            gpu_pool: FixedPagePoolDescriptor::new(4, 1, BufferUsages::COPY_DST),
            max_uploads_per_frame: 1,
        });
        world.insert_resource(FixedPageRenderQueue::new());
        world.insert_resource(FixedPageStreamingStats::default());
        let mut startup = Schedule::new();
        startup
            .add_systems(crate::startup::init_gpu_resource::<FixedPageStreamingGpu<NullInstance>>);
        let mut schedules = Schedules::default();
        schedules.insert(crate::RenderStartup, startup);
        world.insert_resource(schedules);
        crate::startup::ensure_render_startup(&mut world);
        world
    }

    #[test]
    fn render_startup_repeatedly_releases_only_predecessor_pool() {
        let mut world = pool_startup_world();
        let device = world
            .resource::<RenderRuntimeResource<NullInstance>>()
            .runtime()
            .device_handle();
        assert_eq!(device.live_resource_count(), 1);
        for _ in 0..8 {
            world
                .resource_mut::<crate::RenderStartupState>()
                .invalidate();
            crate::startup::ensure_render_startup(&mut world);
            assert_eq!(device.live_resource_count(), 1);
            let gpu = world.resource::<FixedPageStreamingGpu<NullInstance>>();
            assert!(gpu.owns_device(&device));
            assert!(!gpu.bridge.pool().is_destroyed());
        }
        world.remove_resource::<FixedPageStreamingGpu<NullInstance>>();
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn runtime_replacement_rejects_old_pool_then_startup_destroys_on_creator_only() {
        let mut world = pool_startup_world();
        let old_device = world
            .resource::<RenderRuntimeResource<NullInstance>>()
            .runtime()
            .device_handle();
        let old_buffer = world
            .resource::<FixedPageStreamingGpu<NullInstance>>()
            .bridge
            .pool()
            .buffer();
        let replacement = prism_render_runtime::RenderRuntime::headless(
            NullInstance::new(),
            prism_render_driver::Extent3d::new_2d(4, 4),
        )
        .unwrap();
        let new_device = replacement.device_handle();
        // Both runtimes start at generation zero; this must use actual identity.
        assert_eq!(
            replacement.device_generation(),
            world
                .resource::<RenderRuntimeResource<NullInstance>>()
                .runtime()
                .device_generation()
        );
        world.insert_resource(RenderRuntimeResource::new(replacement));
        let mut prepare = Schedule::new();
        prepare.add_systems(prepare_fixed_page_streaming::<NullInstance>);
        prepare.run(&mut world);
        assert_eq!(
            world
                .resource::<FixedPageStreamingStats>()
                .last_upload_error,
            Some(PageResidencyBridgeError::DeviceMismatch)
        );
        assert_eq!(old_device.live_resource_count(), 1);
        assert_eq!(new_device.live_resource_count(), 0);
        world
            .resource_mut::<crate::RenderStartupState>()
            .invalidate();
        crate::startup::ensure_render_startup(&mut world);
        let gpu = world.resource::<FixedPageStreamingGpu<NullInstance>>();
        assert_eq!(
            old_buffer,
            gpu.bridge.pool().buffer(),
            "numeric IDs collide across devices"
        );
        assert!(gpu.owns_device(&new_device));
        assert_eq!(old_device.live_resource_count(), 0);
        assert_eq!(new_device.live_resource_count(), 1);
        assert!(!gpu.bridge.pool().is_destroyed());
        world.remove_resource::<FixedPageStreamingGpu<NullInstance>>();
        assert_eq!(new_device.live_resource_count(), 0);
    }

    #[test]
    fn permanent_upload_error_does_not_block_later_valid_payload() {
        for stale in [false, true] {
            let mut app = streaming_app(2);
            let source = app.world_mut().resource_mut::<FixedPageStreamingSource>();
            let revision = source.sources().revision();
            let old_page = source.residency().eviction_candidates()[0];
            let page = source
                .request_page("memory://page.bin", 1, prism_tasks::Priority::High, 1)
                .unwrap();
            let invalid = super::super::FixedPageRenderPayload {
                page: if stale { old_page } else { page },
                slot: super::super::FixedPageSlotKey {
                    path: "memory://page.bin".into(),
                    page_index: if stale { 0 } else { 1 },
                },
                revision: if stale { revision - 1 } else { revision },
                bytes: if stale { b"abcd".to_vec() } else { vec![0] },
            };
            app.sub_app_mut(RenderApp)
                .unwrap()
                .world
                .resource_mut::<FixedPageRenderQueue>()
                .push_front(invalid);
            app.update();
            let render = app.get_sub_app(RenderApp).unwrap();
            let stats = render.world.resource::<FixedPageStreamingStats>();
            assert_eq!(stats.upload_errors, 1);
            assert_eq!(stats.uploads, 1);
            assert!(render.world.resource::<FixedPageRenderQueue>().is_empty());
            assert!(render
                .world
                .resource::<FixedPageStreamingGpu<NullInstance>>()
                .bridge
                .allocation(page)
                .is_some());
            app.update();
            assert_eq!(
                app.get_sub_app(RenderApp)
                    .unwrap()
                    .world
                    .resource::<FixedPageStreamingStats>()
                    .upload_errors,
                0
            );
        }
    }

    #[test]
    fn retirement_proof_uses_planned_value_and_device_generation() {
        // Registration after 400 earlier frames, with no new submission.
        assert!(!retirement_was_submitted((3, 401), 3, 401));
        assert!(retirement_was_submitted((3, 401), 3, 402));
        assert!(!retirement_was_submitted((3, 401), 4, 900));
        assert!(!retirement_was_submitted((3, u64::MAX), 3, u64::MAX));
        assert_eq!(
            reclaimable_value(Some((3, u64::MAX)), 3, u64::MAX, u64::MAX),
            u64::MAX - 1
        );
        assert_eq!(reclaimable_value(Some((3, 401)), 3, 402, 401), 401);
        assert_eq!(reclaimable_value(Some((3, 401)), 4, 900, 899), 0);
    }

    #[test]
    fn retirement_without_new_submission_uses_queue_barrier_despite_old_frame_count() {
        let mut app = streaming_app(1);
        let page = app
            .world()
            .resource::<FixedPageStreamingSource>()
            .residency()
            .eviction_candidates()[0];
        let revision = app
            .world()
            .resource::<FixedPageStreamingSource>()
            .sources()
            .revision();
        let render = app.sub_app_mut(RenderApp).unwrap();
        render
            .world
            .resource_mut::<crate::RenderFrameStats>()
            .frames_submitted = 10_000;
        let mut gpu = render
            .world
            .remove_resource::<FixedPageStreamingGpu<NullInstance>>()
            .unwrap();
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<NullInstance>>()
            .runtime();
        let next = runtime.next_completion_value();
        gpu.bridge
            .release_page_deferred(page, revision, GpuCompletionValue(next))
            .unwrap();
        gpu.pending_completion = Some((runtime.device_generation(), next));
        assert_eq!(gpu.bridge.pool().pending_reclaim_page_count(), 1);
        gpu.finish_submission(runtime);
        assert_eq!(
            runtime.next_completion_value(),
            next,
            "no new tracked frame"
        );
        assert!(gpu.pending_completion.is_none());
        assert_eq!(gpu.bridge.pool().pending_reclaim_page_count(), 0);
        assert_eq!(gpu.bridge.pool().free_page_count(), 1);
        render.world.insert_resource(gpu);
    }
}
