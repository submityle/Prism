//! ECS wiring for the fixed-page CPU-to-GPU streaming slice.
//!
//! The main world owns requests, source reads, and CPU residency. Extraction
//! moves owned page payloads into the render world. The render prepare stage
//! performs the real queue upload, while cleanup only retires replacements after
//! the runtime has reported a real submission completion value.

use core::marker::PhantomData;

use prism_app::{App, PostUpdate};
use prism_ecs::prelude::{FromWorld, Res, ResMut, Resource};
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, Schedules};
use prism_ecs::world::World;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{FrameSync, Instance};

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
}

/// Last-frame diagnostics for the fixed-page streaming bridge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Resource)]
pub struct FixedPageStreamingStats {
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
        }
    }
}

#[derive(Clone, Resource)]
struct FixedPageStreamingRenderConfig {
    gpu_pool: FixedPagePoolDescriptor,
    max_uploads_per_frame: usize,
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
    bridge: FixedPageResidencyBridge,
    max_uploads_per_frame: usize,
    /// A replacement was prepared before the current frame submission.
    pending_submission: bool,
    observed_submissions: u64,
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
        Self {
            bridge,
            max_uploads_per_frame: config.max_uploads_per_frame,
            pending_submission: false,
            observed_submissions: 0,
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
    schedule.add_systems(prepare_fixed_page_streaming::<I>.in_set(RenderSystems::PrepareResources));
    schedule
        .add_systems(finish_fixed_page_streaming_submission::<I>.in_set(RenderSystems::Cleanup));
}

fn pump_fixed_page_streaming_io(
    mut source: ResMut<FixedPageStreamingSource>,
    config: Res<FixedPageStreamingConfig>,
    mut stats: ResMut<FixedPageStreamingStats>,
) {
    let report = source.pump_io(config.max_io_pages_per_frame);
    stats.io_attempted = report.attempted;
    stats.io_loaded = report.loaded;
    stats.io_errors = report.errors.len();
}

fn extract_fixed_page_streaming(main: &mut World, render: &mut World) {
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

fn prepare_fixed_page_streaming<I>(
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
    let completed = GpuCompletionValue(runtime.runtime().completed_value());
    let _ = gpu.bridge.reclaim_completed(completed);
    let reclaim_after = GpuCompletionValue(runtime.runtime().next_completion_value());
    while let Some(release) = queue.pop_release() {
        match gpu
            .bridge
            .release_page_deferred(release.page, release.revision, reclaim_after)
        {
            Ok(()) => gpu.pending_submission = true,
            Err(PageResidencyBridgeError::UnknownPage(_)) => {}
            Err(error) => {
                stats.upload_errors += 1;
                stats.last_upload_error = Some(error);
                break;
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
                    gpu.pending_submission = true;
                }
            }
            Err(error) => {
                stats.upload_errors += 1;
                stats.last_upload_error = Some(error);
                queue.push_front(payload);
                break;
            }
        }
    }
}

fn finish_fixed_page_streaming_submission<I>(
    mut gpu: ResMut<FixedPageStreamingGpu<I>>,
    stats: Res<crate::RenderFrameStats>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if stats.frames_submitted > gpu.observed_submissions {
        gpu.observed_submissions = stats.frames_submitted;
        gpu.pending_submission = false;
    } else if gpu.pending_submission {
        // A zero-sized frame or graph compile failure did not produce the
        // completion value used by a replacement. Drain the queue before
        // releasing that unsubmitted replacement's retired range.
        if runtime.runtime().wait_idle() {
            let _ = gpu.bridge.reclaim_completed(GpuCompletionValue(u64::MAX));
            gpu.pending_submission = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use prism_app::App;
    use prism_asset::{AssetSources, MemSource, PageBudget, PageResidency, PageSize};
    use prism_render_driver::BufferUsages;
    use prism_render_driver_null::NullInstance;

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

        let main_stats = app.world().resource::<FixedPageStreamingStats>();
        assert_eq!(main_stats.io_attempted, 1);
        assert_eq!(main_stats.io_loaded, 1);
        assert_eq!(main_stats.io_errors, 0);
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
}
