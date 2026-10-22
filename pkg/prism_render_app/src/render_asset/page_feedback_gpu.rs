use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Mutex},
};

use prism_app::App;
use prism_ecs::{
    prelude::{FromWorld, Res, ResMut, Resource},
    schedule::{IntoSystemConfigs, Schedules},
    world::World,
};
use prism_render_architecture::texture_streaming::GpuPageTable;
use prism_render_driver_wgpu::WgpuInstance;

use super::super::page_streaming::StreamGeneration;
use super::super::{FixedPagePoolDescriptor, FixedPageStreamingGpu};
use super::{
    copy_vec, Completed, FixedPageFeedbackConfig, FixedPageFeedbackError as Error,
    FixedPageFeedbackSample, Input,
};
use crate::{GpuResourceAppExt, RenderApp, RenderRuntimeResource, RenderSystems};

const SHADER: &str = include_str!("page_feedback.wgsl");

/// Persistent table and bounded feedback storage on the RenderApp's own device.
/// Consumers must submit on that runtime's queue before its completion signal.
/// This is buffer-page residency, not a material texture atlas.
#[derive(Resource)]
pub struct FixedPageFeedbackGpu {
    state: Result<Buffers, Error>,
    /// Last device/admission failure. Initialization is retried on device startup.
    pub last_error: Option<Error>,
}

impl FixedPageFeedbackGpu {
    /// Shader-visible persistent table in the canonical four-word VT ABI.
    /// The matching physical buffer is `FixedPageStreamingGpu::bridge().pool()`.
    pub fn table_buffer(&self) -> Option<&wgpu::Buffer> {
        if self.last_error.is_some() {
            return None;
        }
        self.state.as_ref().ok().map(|state| &state.table)
    }
    /// Number of valid entries for the current publication.
    pub fn entry_count(&self) -> u32 {
        if self.last_error.is_some() {
            return 0;
        }
        self.state
            .as_ref()
            .map_or(0, |state| (state.table_words.len() / 4) as u32)
    }
}

struct Buffers {
    table: wgpu::Buffer,
    queries: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    table_words: Vec<u32>,
    pending: Option<Pending>,
}

struct Pending {
    source_generation: StreamGeneration,
    target_generation: StreamGeneration,
    ready: Arc<Mutex<Option<bool>>>,
    queries: Vec<super::FixedPageQuery>,
    epoch: u64,
    revision: u64,
    frame: u64,
}

pub(super) fn validate_sizes(
    config: FixedPageFeedbackConfig,
    pool: &FixedPagePoolDescriptor,
) -> Result<[u64; 5], Error> {
    let size = |count: usize, stride: u64| {
        u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(stride))
            .ok_or(Error::Capacity)
    };
    let sizes = [
        size(config.max_bindings, 16)?,
        size(config.max_queries, 16)?,
        size(config.max_queries, 8)?,
        size(config.max_queries, 8)?,
        16,
    ];
    let pool_bytes = pool
        .page_size
        .checked_mul(u64::from(pool.page_count))
        .ok_or(Error::Capacity)?;
    if pool_bytes / 4 > u64::from(u32::MAX) {
        return Err(Error::Capacity);
    }
    let total = sizes
        .iter()
        .try_fold(pool_bytes, |sum, size| sum.checked_add(*size))
        .ok_or(Error::Capacity)?;
    if total > config.max_device_bytes {
        return Err(Error::InvalidConfig("device byte budget exceeded"));
    }
    Ok(sizes)
}

pub(super) fn validate_device(
    device: &wgpu::Device,
    config: FixedPageFeedbackConfig,
    pool: &FixedPagePoolDescriptor,
) -> Result<[u64; 5], Error> {
    let sizes = validate_sizes(config, pool)?;
    let limits = device.limits();
    let pool_bytes = pool.page_size * u64::from(pool.page_count);
    if sizes
        .iter()
        .chain(core::iter::once(&pool_bytes))
        .any(|&size| size > limits.max_buffer_size)
        || [sizes[0], sizes[1], sizes[2], pool_bytes]
            .iter()
            .any(|&size| size > u64::from(limits.max_storage_buffer_binding_size))
        || config.max_queries.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_bindings_per_bind_group < 5
        || limits.max_uniform_buffer_binding_size < 16
    {
        return Err(Error::InvalidConfig("device limits exceeded"));
    }
    Ok(sizes)
}

fn scoped<T>(device: &wgpu::Device, op: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = catch_unwind(AssertUnwindSafe(op))
        .unwrap_or_else(|_| Err(Error::Gpu("device operation panicked".into())));
    let errors = [
        futures_lite::future::block_on(validation.pop()),
        futures_lite::future::block_on(internal.pop()),
        futures_lite::future::block_on(oom.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(Error::Gpu(error.to_string()));
    }
    result
}

impl Buffers {
    fn new(
        device: &wgpu::Device,
        config: FixedPageFeedbackConfig,
        pool: &FixedPagePoolDescriptor,
    ) -> Result<Self, Error> {
        let sizes = validate_device(device, config, pool)?;
        scoped(device, || {
            let buffer = |label, size, usage| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let table = buffer(
                "fixed-page-table",
                sizes[0],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let queries = buffer(
                "fixed-page-queries",
                sizes[1],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let output = buffer(
                "fixed-page-feedback",
                sizes[2],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let staging = buffer(
                "fixed-page-feedback-readback",
                sizes[3],
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let params = buffer(
                "fixed-page-params",
                sizes[4],
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let entries: [_; 5] = core::array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fixed-page-feedback"),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fixed-page-feedback"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fixed-page-feedback"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fixed-page-feedback"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(Self {
                table,
                queries,
                output,
                staging,
                params,
                pipeline,
                layout,
                table_words: Vec::new(),
                pending: None,
            })
        })
    }

    fn collect(&mut self, device: &wgpu::Device) -> Result<Option<Completed>, Error> {
        if self.pending.is_none() {
            return Ok(None);
        }
        let polled = scoped(device, || {
            device
                .poll(wgpu::PollType::Poll)
                .map(|_| ())
                .map_err(|e| Error::Gpu(e.to_string()))
        });
        if let Err(error) = polled {
            self.pending = None;
            let _ = scoped(device, || {
                self.staging.unmap();
                Ok(())
            });
            return Err(error);
        }
        let pending = self.pending.as_ref().expect("pending feedback");
        let ready = pending.ready.lock().expect("feedback map callback").take();
        let Some(ready) = ready else {
            return Ok(None);
        };
        let pending = self.pending.take().expect("pending feedback");
        let result = scoped(device, || {
            if !ready {
                return Err(Error::Readback);
            }
            let view = self
                .staging
                .slice(..pending.queries.len() as u64 * 8)
                .get_mapped_range()
                .map_err(|_| Error::Readback)?;
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(pending.queries.len())
                .map_err(|_| Error::HostAllocation)?;
            for (query, bytes) in pending.queries.iter().zip(view.chunks_exact(8)) {
                samples.push(FixedPageFeedbackSample {
                    query: *query,
                    slot: u32::from_le_bytes(bytes[0..4].try_into().expect("word")),
                    value: u32::from_le_bytes(bytes[4..8].try_into().expect("word")),
                });
            }
            Ok(Completed {
                source_generation: pending.source_generation,
                target_generation: pending.target_generation,
                epoch: pending.epoch,
                revision: pending.revision,
                frame: pending.frame,
                samples,
            })
        });
        let unmap = scoped(device, || {
            self.staging.unmap();
            Ok(())
        });
        unmap?;
        result.map(Some)
    }

    fn publish(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &Input,
        pool: &FixedPageStreamingGpu<WgpuInstance>,
    ) -> Result<(), Error> {
        if let Some(error) = &input.publication_error {
            return Err(error.clone());
        }
        if !pool.has_source(&input.source_generation) {
            return Err(Error::StaleCapture);
        }
        let bridge = pool.bridge();
        let mut words = Vec::new();
        words
            .try_reserve_exact(input.bindings.len().checked_mul(4).ok_or(Error::Capacity)?)
            .map_err(|_| Error::HostAllocation)?;
        for binding in &input.bindings {
            if binding.page.source_revision() != input.revision
                || binding.source_generation.as_ref() != Some(&input.source_generation)
            {
                continue;
            }
            let Some(allocation) = bridge.allocation_for_slot(&binding.slot, input.revision) else {
                continue;
            };
            if allocation.buffer() != bridge.pool().buffer()
                || allocation.page_count() != 1
                || allocation.byte_len() != bridge.pool().page_size()
            {
                return Err(Error::InvalidConfig(
                    "invalid committed physical allocation",
                ));
            }
            words.extend_from_slice(&GpuPageTable::compare_words(binding.key));
            words.push(allocation.first_page());
        }
        if words != self.table_words {
            // Queue writes precede all later consumers on this queue. Older
            // allocations remain quarantined by the bridge's runtime fence.
            scoped(device, || {
                if !words.is_empty() {
                    queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&words));
                }
                Ok(())
            })?;
            self.table_words = words;
        }
        Ok(())
    }

    fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &Input,
        physical: &wgpu::Buffer,
        page_size: u64,
    ) -> Result<(), Error> {
        if self.pending.is_some()
            || input.queries.is_empty()
            || input
                .handoff
                .lock()
                .expect("feedback handoff")
                .completed
                .is_some()
        {
            return Ok(());
        }
        if input.queries.len() > input.config.max_queries
            || input
                .queries
                .iter()
                .any(|query| u64::from(query.word_in_page) >= page_size / 4)
        {
            return Err(Error::InvalidConfig("query outside fixed page"));
        }
        let mut queries = Vec::new();
        queries
            .try_reserve_exact(input.queries.len().checked_mul(4).ok_or(Error::Capacity)?)
            .map_err(|_| Error::HostAllocation)?;
        for query in &input.queries {
            queries.extend_from_slice(&GpuPageTable::compare_words(query.key));
            queries.push(query.word_in_page);
        }
        let captured = copy_vec(&input.queries)?;
        let ready = Arc::new(Mutex::new(None));
        let result = scoped(device, || {
            let params = [
                input.queries.len() as u32,
                (self.table_words.len() / 4) as u32,
                (page_size / 4) as u32,
                (physical.size() / 4) as u32,
            ];
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
            queue.write_buffer(&self.queries, 0, bytemuck::cast_slice(&queries));
            let entries = [
                &self.params,
                &self.table,
                &self.queries,
                physical,
                &self.output,
            ]
            .map(|buffer| buffer.as_entire_binding());
            let entries: [_; 5] = core::array::from_fn(|binding| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: entries[binding].clone(),
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fixed-page-feedback"),
                layout: &self.layout,
                entries: &entries,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fixed-page-feedback"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fixed-page-feedback"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((input.queries.len() as u32).div_ceil(64), 1, 1);
            }
            let bytes = input.queries.len() as u64 * 8;
            encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, bytes);
            queue.submit([encoder.finish()]);
            let callback = ready.clone();
            self.staging
                .slice(..bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    *callback.lock().expect("feedback map callback") = Some(result.is_ok());
                });
            Ok(())
        });
        if let Err(error) = result {
            // Cancel a map started before a late scoped error, so the reusable
            // staging buffer remains available to the next valid batch.
            let _ = scoped(device, || {
                self.staging.unmap();
                Ok(())
            });
            return Err(error);
        }
        self.pending = Some(Pending {
            source_generation: input.source_generation.clone(),
            target_generation: input.target_generation.clone(),
            ready,
            queries: captured,
            epoch: input.epoch,
            revision: input.revision,
            frame: input.frame,
        });
        Ok(())
    }
}

impl FromWorld for FixedPageFeedbackGpu {
    fn from_world(world: &mut World) -> Self {
        let input = world.resource::<Input>();
        let config =
            world.resource::<super::super::page_streaming_stage::FixedPageStreamingRenderConfig>();
        let runtime = world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime();
        let state = Buffers::new(
            runtime.device().wgpu_device(),
            input.config,
            &config.gpu_pool,
        );
        let last_error = state.as_ref().err().cloned();
        {
            // Old mapped captures belong to the old allocation target, even
            // when startup reinitializes resources on the same runtime device.
            let mut handoff = input.handoff.lock().expect("feedback handoff");
            if handoff.initialized {
                handoff.generation = StreamGeneration::default();
            }
            handoff.initialized = true;
            handoff.completed = None;
        }
        Self { state, last_error }
    }
}

fn prepare_feedback(
    mut feedback: ResMut<FixedPageFeedbackGpu>,
    pool: Res<FixedPageStreamingGpu<WgpuInstance>>,
    input: Res<Input>,
    runtime: Res<RenderRuntimeResource<WgpuInstance>>,
) {
    let runtime = runtime.runtime();
    let device = runtime.device().wgpu_device();
    let result = (|| {
        if !pool.owns_device(&runtime.device_handle()) {
            return Err(Error::StaleCapture);
        }
        let buffers = feedback.state.as_mut().map_err(|error| error.clone())?;
        if let Some(error) = &input.publication_error {
            return Err(error.clone());
        }
        if input.target_generation != input.handoff.lock().expect("feedback handoff").generation {
            return Err(Error::StaleCapture);
        }
        if input
            .handoff
            .lock()
            .expect("feedback handoff")
            .completed
            .is_none()
        {
            if let Some(capture) = buffers.collect(device)? {
                let mut handoff = input.handoff.lock().expect("feedback handoff");
                // At most one capture in flight and one completed capture.
                if handoff.completed.is_none() {
                    handoff.completed = Some(Ok(capture));
                }
            }
        }
        buffers.publish(device, runtime.queue().wgpu_queue(), &input, &pool)?;
        let physical = scoped(device, || {
            Ok(runtime.device().wgpu_buffer(pool.bridge().pool().buffer()))
        })?;
        buffers.dispatch(
            device,
            runtime.queue().wgpu_queue(),
            &input,
            &physical,
            pool.bridge().pool().page_size(),
        )
    })();
    feedback.last_error = result.as_ref().err().cloned();
    if let Err(error) = result {
        let mut handoff = input.handoff.lock().expect("feedback handoff");
        if handoff.completed.is_none() {
            handoff.completed = Some(Err(error));
        }
    }
}

pub(super) fn install(app: &mut App) {
    let source = app
        .world()
        .resource::<super::super::FixedPageStreamingSource>();
    let feedback = app.world().resource::<super::FixedPageFeedback>();
    let input = Input {
        source_generation: source.generation(),
        target_generation: feedback
            .handoff
            .lock()
            .expect("feedback handoff")
            .generation
            .clone(),
        publication_error: None,
        config: feedback.config,
        bindings: Vec::new(),
        queries: Vec::new(),
        epoch: feedback.epoch,
        revision: source.sources().revision(),
        frame: 0,
        handoff: feedback.handoff.clone(),
    };
    let render = app.sub_app_mut(RenderApp).expect("render app validated");
    render.world.insert_resource(input);
    render.init_gpu_resource::<FixedPageFeedbackGpu>();
    render
        .world
        .resource_mut::<Schedules>()
        .get_mut(crate::Render)
        .expect("render schedule")
        .add_systems(
            prepare_feedback
                .in_set(RenderSystems::PrepareResources)
                .after(super::super::page_streaming_stage::FixedPageUploadSet),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FixedPageFeedback, FixedPageStreamingConfig, FixedPageStreamingSource, PrismRenderPlugin,
        RenderAppConfig,
    };
    use prism_asset::{AssetSources, MemSource, PageBudget, PageKey, PageResidency, PageSize};
    use prism_ecs::schedule::Schedule;
    use prism_render_architecture::texture_streaming::TexturePageKey;
    use prism_render_driver::{BufferUsages, Extent3d};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError};

    #[test]
    fn failed_extract_withholds_previously_populated_gpu_table_and_recovers() {
        match WgpuInstance::new().request_default_hardware_device() {
            Ok(_) => {}
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(!require_hardware_gpu(), "physical GPU required");
                eprintln!("SKIPPED failed-extract publication GPU test: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(!require_hardware_gpu(), "physical GPU required: {evidence}");
                eprintln!("SKIPPED failed-extract publication GPU test: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => panic!("device creation: {error}"),
        }
        let mut memory = MemSource::new();
        memory.insert("page.bin", vec![1; 8]);
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(memory), 0);
        let mut source = FixedPageStreamingSource::new(
            sources,
            PageResidency::new(PageSize::new(8).unwrap(), PageBudget::new(8, 1).unwrap()).unwrap(),
        );
        source
            .request_page("memory://page.bin", 0, prism_tasks::Priority::High, 1)
            .unwrap();
        let page = PageKey::from_sources(source.sources(), "memory://page.bin", 0);
        let key = TexturePageKey {
            texture: 1,
            mip: 0,
            layer: 0,
            x: 0,
            y: 0,
        };
        let mut app = App::new();
        app.add_plugins(PrismRenderPlugin::with_config(
            WgpuInstance::new(),
            RenderAppConfig::new(Extent3d::new_2d(8, 8)).with_particle_capacity(0),
        ));
        super::super::register_fixed_page_feedback(
            &mut app,
            source,
            FixedPageStreamingConfig::new(FixedPagePoolDescriptor::new(
                8,
                1,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )),
            FixedPageFeedbackConfig {
                max_bindings: 1,
                max_queries: 1,
                max_device_bytes: 72,
            },
        )
        .unwrap();
        app.world_mut()
            .resource_mut::<FixedPageFeedback>()
            .bind(key, page)
            .unwrap();
        app.update();
        let initial = {
            let render = app.get_sub_app(RenderApp).unwrap();
            let gpu = render.world.resource::<FixedPageFeedbackGpu>();
            assert_eq!(gpu.entry_count(), 1);
            gpu.table_buffer().unwrap().clone()
        };
        let mut feedback = app
            .world_mut()
            .remove_resource::<FixedPageFeedback>()
            .unwrap();
        {
            let render = app.sub_app_mut(RenderApp).unwrap();
            super::super::apply_extracted_feedback(
                &mut feedback,
                &mut render.world,
                Err(Error::HostAllocation),
            );
            let mut prepare = Schedule::new();
            prepare.add_systems(prepare_feedback);
            prepare.run(&mut render.world);
            let gpu = render.world.resource::<FixedPageFeedbackGpu>();
            assert_eq!(gpu.last_error, Some(Error::HostAllocation));
            assert_eq!(gpu.entry_count(), 0);
            assert!(gpu.table_buffer().is_none());
        }
        app.world_mut().insert_resource(feedback);
        app.update();
        let render = app.get_sub_app(RenderApp).unwrap();
        let gpu = render.world.resource::<FixedPageFeedbackGpu>();
        assert_eq!(gpu.last_error, None);
        assert_eq!(gpu.entry_count(), 1);
        assert_eq!(gpu.table_buffer().unwrap(), &initial);
        assert!(render
            .world
            .resource::<RenderRuntimeResource<WgpuInstance>>()
            .runtime()
            .wait_idle());
    }
}
