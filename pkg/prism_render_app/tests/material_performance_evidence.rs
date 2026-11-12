//! Strict admission for native material performance evidence.
//!
//! The gate accepts only raw per-sample records. Percentiles are recomputed
//! here, so a producer cannot provide a hand-written summary and call it GPU
//! evidence. The producer identity is also part of the contract: the generic
//! RHI timestamp probe is useful capability evidence, but it is not a native
//! material frame and must not pass this gate.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process::Command as ProcessCommand;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use prism_app::App;
use prism_asset::{AssetId, Assets};
use prism_math::{Aabb3, Frustum, Mat4, Plane, Vec3};
use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
use prism_render_app::prelude::{PrismRenderPlugin, RenderApp, RenderAppConfig};
use prism_render_app::visibility::{
    sync_scene, GeometryHandle, LocalBounds, PrismCamera, PrismFrustum, PrismRenderScene,
    PrismRenderable, PrismVisibilityPlugin,
};
use prism_render_app::{RenderRuntimeResource, SceneRegistry};
use prism_render_architecture::geometry::{
    GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
};
use prism_render_architecture::gpu_scene::SceneHandle;
use prism_render_driver as rhi;
use prism_render_driver::{
    Adapter, BufferReadbackRange, BufferUsages, CommandEncoder, ComputeCommand,
    ComputePipelineDescriptor, CopyCommand, DeviceDescriptor, Features, FrameSync, Instance, Pass,
    RenderCommand, RenderDevice, RenderQueue, RequestAdapterOptions, ShaderModuleDescriptor,
    TimestampQuerySetDescriptor, TimestampWrites,
};
use prism_render_driver_wgpu::{AdapterEvidence, WgpuAdapter, WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_material::{fallback_material_record, BuiltinProgram, ShadingPlan};
use prism_render_visibility::{GeometryLod, GeometryLodChain};
use prism_transform::GlobalTransform;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const SCHEMA: &str = "prism-material-perf-v2";
const CONTENT: &str = "native_unlit_quad_v1";
const FRAME_WIDTH: u32 = 64;
const FRAME_HEIGHT: u32 = 64;
const REQUIRED_FIELDS: [&str; 11] = [
    "schema",
    "commit",
    "scene",
    "content",
    "capture_scope",
    "producer",
    "backend",
    "adapter",
    "resolution",
    "timestamp_source",
    "timestamp_unit",
];
const PRODUCTION_SCOPE: &str = "production_native_material_frame";
const PRODUCTION_PRODUCER: &str = "prism_native_material_frame_capture_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Upload {
    buffer: rhi::BufferId,
    bytes: u64,
}

#[derive(Debug)]
struct PendingGpuFrame {
    query_set: rhi::TimestampQuerySetId,
    resolve_buffer: rhi::BufferId,
    uploads: Vec<Upload>,
}

#[derive(Default, Debug)]
struct QueueCaptureState {
    uploads: Vec<Upload>,
    frames: Vec<PendingGpuFrame>,
}

/// A queue wrapper that instruments the real production submission.
///
/// The Native frame command buffers are not replaced or replayed. They are
/// submitted in their original order between two timestamped RHI passes, and
/// the query resolve is appended to that same queue submission.
#[derive(Clone)]
struct MaterialPerfQueue {
    device: WgpuDevice,
    queue: WgpuQueue,
    captures: Arc<Mutex<QueueCaptureState>>,
    timestamp_pipeline: Arc<rhi::ComputePipelineId>,
}

impl MaterialPerfQueue {
    fn new(device: WgpuDevice, queue: WgpuQueue) -> Self {
        let module = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            "@compute @workgroup_size(1)\nfn main() {}",
        ));
        let timestamp_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism-material-perf-timestamp-pipeline".into()),
            layout: None,
            module,
            entry_point: "main".into(),
        });
        device.destroy_shader_module(module);
        Self {
            device,
            queue,
            captures: Arc::new(Mutex::new(QueueCaptureState::default())),
            timestamp_pipeline: Arc::new(timestamp_pipeline),
        }
    }

    fn capture_count(&self) -> usize {
        self.captures
            .lock()
            .expect("material performance capture lock")
            .frames
            .len()
    }

    fn submit_instrumented(&self, command_buffers: &[rhi::CommandBuffer]) -> bool {
        command_buffers.iter().any(|command_buffer| {
            command_buffer.passes.iter().any(|pass| match pass {
                Pass::Render { commands, .. } => commands.iter().any(|command| {
                    matches!(
                        command,
                        RenderCommand::Draw { .. }
                            | RenderCommand::DrawIndexed { .. }
                            | RenderCommand::DrawIndirect { .. }
                            | RenderCommand::DrawIndexedIndirect { .. }
                            | RenderCommand::MultiDrawIndirect { .. }
                    )
                }),
                Pass::Compute { .. } | Pass::Transfer { .. } => false,
            })
        })
    }

    fn timestamp_boundary(
        &self,
        query_set: rhi::TimestampQuerySetId,
        index: u32,
        label: &'static str,
        beginning: bool,
    ) -> rhi::CommandBuffer {
        let mut encoder = CommandEncoder::labeled(label);
        encoder.push_timestamped_compute_pass(
            Some(label.into()),
            TimestampWrites {
                query_set,
                beginning_of_pass_write_index: beginning.then_some(index),
                end_of_pass_write_index: (!beginning).then_some(index),
            },
            vec![
                ComputeCommand::SetPipeline(*self.timestamp_pipeline),
                ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
            ],
        );
        encoder.finish()
    }

    fn resolve_timestamp(
        query_set: rhi::TimestampQuerySetId,
        resolve_buffer: rhi::BufferId,
    ) -> rhi::CommandBuffer {
        let mut encoder = CommandEncoder::labeled("prism-material-perf-resolve");
        encoder.push_transfer_pass(
            Some("prism-material-perf-resolve".into()),
            vec![CopyCommand::ResolveTimestampQueries {
                query_set,
                first_query: 0,
                query_count: 2,
                destination: resolve_buffer,
                destination_offset: 0,
            }],
        );
        encoder.finish()
    }

    fn instrumented_submission(
        &self,
        command_buffers: &[rhi::CommandBuffer],
    ) -> Option<(
        rhi::TimestampQuerySetId,
        rhi::BufferId,
        Vec<rhi::CommandBuffer>,
    )> {
        if !self.submit_instrumented(command_buffers) {
            return None;
        }

        let query_set = self
            .device
            .create_timestamp_query_set(&TimestampQuerySetDescriptor::new(2));
        let resolve_buffer = self.device.create_buffer(&rhi::BufferDescriptor {
            label: Some("prism-material-perf-timestamp-resolve".into()),
            size: 16,
            usage: BufferUsages::QUERY_RESOLVE.union(BufferUsages::COPY_SRC),
            mapped_at_creation: false,
        });
        let mut instrumented = Vec::with_capacity(command_buffers.len() + 3);
        instrumented.push(Self::timestamp_boundary(
            self,
            query_set,
            0,
            "prism-material-perf-begin",
            true,
        ));
        instrumented.extend_from_slice(command_buffers);
        instrumented.push(Self::timestamp_boundary(
            self,
            query_set,
            1,
            "prism-material-perf-end",
            false,
        ));
        instrumented.push(Self::resolve_timestamp(query_set, resolve_buffer));
        Some((query_set, resolve_buffer, instrumented))
    }

    fn record_frame(&self, query_set: rhi::TimestampQuerySetId, resolve_buffer: rhi::BufferId) {
        let uploads = std::mem::take(
            &mut self
                .captures
                .lock()
                .expect("material performance capture lock")
                .uploads,
        );
        self.captures
            .lock()
            .expect("material performance capture lock")
            .frames
            .push(PendingGpuFrame {
                query_set,
                resolve_buffer,
                uploads,
            });
    }

    fn drain_samples(&self, material_buffers: &[rhi::BufferId]) -> Result<Vec<Sample>, String> {
        if !self.queue.wait_idle() {
            return Err("GPU queue did not reach idle while collecting material samples".into());
        }
        let frames = std::mem::take(
            &mut self
                .captures
                .lock()
                .expect("material performance capture lock")
                .frames,
        );
        let period_ns = self
            .queue
            .timestamp_period_ns()
            .filter(|period| period.is_finite() && *period > 0.0)
            .ok_or_else(|| "production queue did not expose a valid timestamp period".to_owned())?;
        let mut samples = Vec::with_capacity(frames.len());
        for (index, frame) in frames.into_iter().enumerate() {
            let mut readback = self
                .queue
                .request_buffer_readback(
                    &[BufferReadbackRange {
                        buffer: frame.resolve_buffer,
                        offset: 0,
                        size: 16,
                    }],
                    16,
                )
                .map_err(|error| format!("timestamp resolve readback failed: {error:?}"))?;
            let ranges = loop {
                if let Some(ranges) = readback
                    .try_complete()
                    .map_err(|error| format!("timestamp readback failed: {error:?}"))?
                {
                    break ranges;
                }
                if !self.device.poll_wait() {
                    return Err("GPU device was lost while mapping timestamp readback".into());
                }
            };
            let bytes = ranges
                .into_iter()
                .next()
                .ok_or_else(|| "timestamp readback returned no range".to_owned())?;
            let begin = u64::from_ne_bytes(
                bytes
                    .get(0..8)
                    .ok_or_else(|| "timestamp readback is shorter than two queries".to_owned())?
                    .try_into()
                    .expect("eight-byte timestamp chunk"),
            );
            let end = u64::from_ne_bytes(
                bytes
                    .get(8..16)
                    .ok_or_else(|| "timestamp readback is shorter than two queries".to_owned())?
                    .try_into()
                    .expect("eight-byte timestamp chunk"),
            );
            if begin == 0 || end <= begin {
                return Err(format!(
                    "production frame {index} returned invalid GPU timestamps: begin={begin} end={end}"
                ));
            }
            let gpu_timestamp_ns = ((end - begin) as f64 * f64::from(period_ns)).round();
            if !gpu_timestamp_ns.is_finite()
                || gpu_timestamp_ns <= 0.0
                || gpu_timestamp_ns > u64::MAX as f64
            {
                return Err(format!(
                    "production frame {index} returned invalid GPU duration: {gpu_timestamp_ns} ns"
                ));
            }
            let material_upload_bytes = frame
                .uploads
                .iter()
                .filter(|upload| material_buffers.contains(&upload.buffer))
                .map(|upload| upload.bytes)
                .sum();
            samples.push(Sample {
                index: index as u64,
                cpu_wall_ns: 0,
                gpu_timestamp_ns: gpu_timestamp_ns as u64,
                material_upload_bytes,
            });
            self.device.destroy_buffer(frame.resolve_buffer);
            self.device.destroy_timestamp_query_set(frame.query_set);
        }
        Ok(samples)
    }
}

impl Drop for MaterialPerfQueue {
    fn drop(&mut self) {
        if Arc::strong_count(&self.timestamp_pipeline) == 1 {
            self.device
                .destroy_compute_pipeline(*self.timestamp_pipeline);
        }
    }
}

impl RenderQueue for MaterialPerfQueue {
    fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
        self.captures
            .lock()
            .expect("material performance capture lock")
            .uploads
            .push(Upload {
                buffer,
                bytes: data.len() as u64,
            });
        self.queue.write_buffer(buffer, offset, data);
    }

    fn write_texture(
        &self,
        destination: rhi::TextureWrite,
        data: &[u8],
        layout: rhi::ImageDataLayout,
    ) {
        self.queue.write_texture(destination, data, layout);
    }

    fn submit(&self, command_buffers: &[rhi::CommandBuffer]) {
        if let Some((query_set, resolve_buffer, instrumented)) =
            self.instrumented_submission(command_buffers)
        {
            self.queue.submit(&instrumented);
            self.record_frame(query_set, resolve_buffer);
        } else {
            self.queue.submit(command_buffers);
        }
    }

    fn timestamp_period_ns(&self) -> Option<f32> {
        self.queue.timestamp_period_ns()
    }

    fn request_buffer_readback(
        &self,
        ranges: &[BufferReadbackRange],
        max_bytes: u64,
    ) -> Result<Box<dyn rhi::BufferReadback>, rhi::BufferReadbackError> {
        self.queue.request_buffer_readback(ranges, max_bytes)
    }
}

impl FrameSync for MaterialPerfQueue {
    fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
        let Some((query_set, resolve_buffer, instrumented)) =
            self.instrumented_submission(command_buffers)
        else {
            return Box::new(self.queue.submit_tracked(command_buffers));
        };

        let token = Box::new(self.queue.submit_tracked(&instrumented));
        self.record_frame(query_set, resolve_buffer);
        token
    }

    fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
        self.queue.wait_for_submission_erased(token)
    }

    fn wait_idle(&self) -> bool {
        self.queue.wait_idle()
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        self.queue.on_submitted_work_done(callback);
    }
}

struct MaterialPerfInstance {
    inner: WgpuInstance,
}

impl MaterialPerfInstance {
    fn new() -> Self {
        Self {
            inner: WgpuInstance::new(),
        }
    }

    fn timestamp_capable_hardware(&self) -> Option<AdapterEvidence> {
        self.enumerate_adapters().into_iter().find_map(|adapter| {
            let evidence = AdapterEvidence::from_info(adapter.info());
            (evidence.is_hardware()
                && adapter
                    .capabilities()
                    .features
                    .contains(Features::TIMESTAMP_QUERY))
            .then_some(evidence)
        })
    }
}

impl Instance for MaterialPerfInstance {
    type Adapter = WgpuAdapter;
    type Device = WgpuDevice;
    type Queue = MaterialPerfQueue;
    type Surface = prism_render_driver_wgpu::WgpuSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        <WgpuInstance as Instance>::enumerate_adapters(&self.inner)
    }

    fn request_adapter(&self, _options: &RequestAdapterOptions) -> Option<Self::Adapter> {
        self.enumerate_adapters().into_iter().find(|adapter| {
            matches!(
                adapter.info().device_type,
                rhi::DeviceType::IntegratedGpu | rhi::DeviceType::DiscreteGpu
            ) && adapter
                .capabilities()
                .features
                .contains(Features::TIMESTAMP_QUERY)
        })
    }

    fn request_adapter_for_surface(
        &self,
        options: &RequestAdapterOptions,
        _surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        self.request_adapter(options)
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        if !adapter
            .capabilities()
            .features
            .contains(Features::TIMESTAMP_QUERY)
        {
            return Err(rhi::DeviceError::Request(
                "production material sampler requires TIMESTAMP_QUERY".into(),
            ));
        }
        let mut descriptor = descriptor.clone();
        descriptor.required_features |= Features::TIMESTAMP_QUERY;
        let (device, queue) =
            <WgpuInstance as Instance>::create_device(&self.inner, adapter, &descriptor)?;
        let queue = MaterialPerfQueue::new(device.clone(), queue);
        Ok((device, queue))
    }

    fn create_surface<W>(
        &self,
        _window: Arc<W>,
        _device: &Self::Device,
        _queue: &Self::Queue,
    ) -> Result<Self::Surface, rhi::SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        Err(rhi::SurfaceError::Unsupported)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Evidence {
    fields: BTreeMap<String, String>,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sample {
    index: u64,
    cpu_wall_ns: u64,
    gpu_timestamp_ns: u64,
    material_upload_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Percentiles {
    p50: u64,
    p95: u64,
    p99: u64,
}

fn parse(input: &str) -> Result<Evidence, String> {
    let mut fields = BTreeMap::new();
    let mut samples = Vec::new();

    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split(',').map(str::trim).collect();
        match parts.first().copied() {
            Some("sample") => {
                if parts.len() != 5 {
                    return Err(format!(
                        "line {}: sample requires index,cpu_wall_ns,gpu_timestamp_ns,material_upload_bytes",
                        line_index + 1
                    ));
                }
                let values = [parts[1], parts[2], parts[3], parts[4]]
                    .into_iter()
                    .map(|value| {
                        value.parse::<u64>().map_err(|_| {
                            format!("line {}: invalid unsigned sample value", line_index + 1)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                samples.push(Sample {
                    index: values[0],
                    cpu_wall_ns: values[1],
                    gpu_timestamp_ns: values[2],
                    material_upload_bytes: values[3],
                });
            }
            Some("metric") => {
                return Err(format!(
                    "line {}: percentile summaries are not accepted; provide raw sample lines",
                    line_index + 1
                ));
            }
            Some(kind) if parts.len() == 2 && !kind.is_empty() && !parts[1].is_empty() => {
                if fields
                    .insert(kind.to_owned(), parts[1].to_owned())
                    .is_some()
                {
                    return Err(format!("line {}: field is duplicated", line_index + 1));
                }
            }
            _ => {
                return Err(format!(
                    "line {}: expected key,value or sample,index,cpu_wall_ns,gpu_timestamp_ns,material_upload_bytes",
                    line_index + 1
                ));
            }
        }
    }

    let evidence = Evidence { fields, samples };
    validate(&evidence)?;
    Ok(evidence)
}

fn validate(evidence: &Evidence) -> Result<(), String> {
    let field = |name: &str| {
        evidence
            .fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required field {name}"))
    };

    for name in REQUIRED_FIELDS {
        field(name)?;
    }
    if field("schema")? != SCHEMA {
        return Err(format!("schema must be {SCHEMA}"));
    }
    let commit = field("commit")?;
    if !(7..=64).contains(&commit.len()) || !commit.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("commit must be a hexadecimal revision id".into());
    }
    if field("scene")? != "default_native_material_frame" {
        return Err("scene must identify the default native material frame".into());
    }
    if field("content")? != CONTENT {
        return Err(format!(
            "content must identify the fixed production content: {CONTENT}"
        ));
    }
    if field("capture_scope")? != PRODUCTION_SCOPE {
        return Err("capture_scope must identify a production native material frame".into());
    }
    if field("producer")? != PRODUCTION_PRODUCER {
        return Err("producer is not the production native material frame sampler".into());
    }
    for name in ["backend", "adapter"] {
        if field(name)?.trim().is_empty() || field(name)?.contains(',') {
            return Err(format!("{name} must be nonempty and comma-free"));
        }
    }
    if field("timestamp_source")? != "gpu_timestamp_query" {
        return Err("timestamp_source must be gpu_timestamp_query".into());
    }
    if field("timestamp_unit")? != "ns" {
        return Err("timestamp_unit must be ns".into());
    }

    let resolution = field("resolution")?;
    let (width, height) = resolution
        .split_once('x')
        .ok_or_else(|| "resolution must be WIDTHxHEIGHT".to_owned())?;
    if width
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .is_none()
        || height
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .is_none()
    {
        return Err("resolution must contain positive dimensions".into());
    }

    let warmup = field("warmup")?
        .parse::<u64>()
        .map_err(|_| "warmup must be an unsigned integer".to_owned())?;
    let sample_count = field("samples")?
        .parse::<u64>()
        .map_err(|_| "samples must be an unsigned integer".to_owned())?;
    if warmup == 0 || sample_count < 30 {
        return Err("warmup must be positive and samples must be at least 30".into());
    }
    if evidence.samples.len() as u64 != sample_count {
        return Err(format!(
            "raw sample count is {}, expected {sample_count}",
            evidence.samples.len()
        ));
    }
    for (expected, sample) in evidence.samples.iter().enumerate() {
        if sample.index != expected as u64 {
            return Err(format!(
                "sample indices must be contiguous from zero; expected {}, got {}",
                expected, sample.index
            ));
        }
        if sample.cpu_wall_ns == 0 || sample.gpu_timestamp_ns == 0 {
            return Err(format!(
                "sample {} must contain positive CPU and GPU durations",
                sample.index
            ));
        }
    }

    Ok(())
}

fn percentiles(values: impl IntoIterator<Item = u64>) -> Percentiles {
    let mut values = values.into_iter().collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "percentiles require at least one sample"
    );
    values.sort_unstable();
    let nearest_rank = |percent: usize| {
        let rank = (values.len() * percent).div_ceil(100);
        values[rank.saturating_sub(1)]
    };
    Percentiles {
        p50: nearest_rank(50),
        p95: nearest_rank(95),
        p99: nearest_rank(99),
    }
}

fn metric_percentiles(evidence: &Evidence, metric: &str) -> Percentiles {
    match metric {
        "frame_cpu_wall_ns" => {
            percentiles(evidence.samples.iter().map(|sample| sample.cpu_wall_ns))
        }
        "frame_gpu_timestamp_ns" => percentiles(
            evidence
                .samples
                .iter()
                .map(|sample| sample.gpu_timestamp_ns),
        ),
        "material_upload_bytes" => percentiles(
            evidence
                .samples
                .iter()
                .map(|sample| sample.material_upload_bytes),
        ),
        _ => unreachable!("the material gate only names known metrics"),
    }
}

fn assert_budgets(evidence: &Evidence) {
    for (metric_name, env_name, percentile_name) in [
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P50_MAX_NS", "p50"),
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P95_MAX_NS", "p95"),
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P99_MAX_NS", "p99"),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P50_MAX_NS",
            "p50",
        ),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P95_MAX_NS",
            "p95",
        ),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P99_MAX_NS",
            "p99",
        ),
    ] {
        let budget = std::env::var(env_name)
            .unwrap_or_else(|_| panic!("set {env_name} to an agreed nanosecond budget"))
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("{env_name} must be an unsigned nanosecond budget"));
        let percentiles = metric_percentiles(evidence, metric_name);
        let actual = match percentile_name {
            "p50" => percentiles.p50,
            "p95" => percentiles.p95,
            "p99" => percentiles.p99,
            _ => unreachable!("the gate only names supported percentiles"),
        };
        assert!(
            actual <= budget,
            "{metric_name} {percentile_name} is {actual} ns, above {env_name}={budget} ns"
        );
    }
}

fn complete_artifact() -> String {
    let mut artifact = String::from(
        "schema,prism-material-perf-v2\n\
         commit,0123456789abcdef0123456789abcdef01234567\n\
         scene,default_native_material_frame\n\
         content,native_unlit_quad_v1\n\
         capture_scope,production_native_material_frame\n\
         producer,prism_native_material_frame_capture_v1\n\
         backend,metal\n\
         adapter,physical-gpu\n\
         resolution,1280x720\n\
         warmup,30\n\
         samples,60\n\
         timestamp_source,gpu_timestamp_query\n\
         timestamp_unit,ns\n",
    );
    for index in 0..60_u64 {
        let cpu = 100_000 + index * 1_000;
        let gpu = 80_000 + index * 500;
        let upload = 1_024 + index * 16;
        artifact.push_str(&format!("sample,{index},{cpu},{gpu},{upload}\n"));
    }
    artifact
}

#[test]
fn raw_material_artifact_recomputes_real_percentiles() {
    let evidence = parse(&complete_artifact()).expect("complete artifact must be admissible");
    assert_eq!(evidence.fields["timestamp_source"], "gpu_timestamp_query");
    assert_eq!(evidence.samples.len(), 60);
    assert_eq!(
        metric_percentiles(&evidence, "frame_gpu_timestamp_ns").p95,
        108_000
    );
}

#[test]
fn cpu_wall_time_cannot_be_declared_gpu_timestamp_evidence() {
    let artifact = complete_artifact().replace(
        "timestamp_source,gpu_timestamp_query",
        "timestamp_source,cpu_wall_clock",
    );
    let error = parse(&artifact).expect_err("CPU wall time must not pass the GPU gate");
    assert!(
        error.contains("timestamp_source"),
        "unexpected error: {error}"
    );
}

#[test]
fn summary_only_or_incomplete_raw_samples_are_rejected() {
    let mut summary = complete_artifact();
    let sample_start = summary.find("sample,").expect("fixture has samples");
    summary.truncate(sample_start);
    summary.push_str("metric,frame_gpu_timestamp_ns,60,80000,120000,160000\n");
    let error = parse(&summary).expect_err("summary-only evidence must fail the gate");
    assert!(error.contains("raw sample"), "unexpected error: {error}");

    let incomplete = complete_artifact().replace("sample,29,", "sample,60,");
    let error = parse(&incomplete).expect_err("non-contiguous samples must fail the gate");
    assert!(error.contains("contiguous"), "unexpected error: {error}");
}

#[test]
fn generic_rhi_timestamp_probe_cannot_pass_as_material_frame_evidence() {
    let probe = complete_artifact()
        .replace(
            "capture_scope,production_native_material_frame",
            "capture_scope,rhi_timestamp_probe",
        )
        .replace(
            "producer,prism_native_material_frame_capture_v1",
            "producer,prism_native_rhi_timestamp_probe_v1",
        );
    let error = parse(&probe).expect_err("generic RHI work must not pass material gate");
    assert!(error.contains("capture_scope"), "unexpected error: {error}");
}

fn broad_frustum() -> PrismFrustum {
    PrismFrustum(Frustum::from_planes([
        Plane::new(Vec3::X, 10.0),
        Plane::new(-Vec3::X, 10.0),
        Plane::new(Vec3::Y, 10.0),
        Plane::new(-Vec3::Y, 10.0),
        Plane::new(Vec3::Z, 10.0),
        Plane::new(-Vec3::Z, 10.0),
    ]))
}

fn production_material_record(revision: u64, phase: u64) -> prism_render_material::MaterialRecord {
    let mut record = fallback_material_record(GeometryHandle::new(2, 1), revision);
    record.response_plan = ShadingPlan::single_builtin(BuiltinProgram::Unlit);
    let authored = if phase % 2 == 0 { 1.0 } else { 0.999 };
    record.surface.base_color = [authored; 4];
    record
}

fn production_scene_fixture(app: &mut App) {
    if app.world().get_resource::<PrismRenderScene>().is_none() {
        app.add_plugins(PrismVisibilityPlugin);
    }
    let mesh = Mesh::new(rhi::PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Position", rhi::VertexFormat::Float32x3, 0),
            VertexAttributeValues::Float32x3(vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]),
        )
        .expect("positions fit")
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Normal", rhi::VertexFormat::Float32x3, 1),
            VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 1.0]; 4]),
        )
        .expect("normals fit")
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Uv", rhi::VertexFormat::Float32x2, 2),
            VertexAttributeValues::Float32x2(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )
        .expect("UVs fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]));
    let mut meshes = Assets::new();
    let mesh_id: AssetId<Mesh> = meshes.insert(mesh).id();
    app.world_mut().insert_resource(meshes);

    // Keep this fixture on the same production seam as the native acceptance
    // tests: the main-world scene is authoritative and PrepareAssets performs
    // the render-world material reconcile.
    let geometry = GeometryHandle::new(1, 1);
    let material = GeometryHandle::new(2, 1);
    let material_record = production_material_record(2, 0);

    {
        let scene = app.world_mut().resource_mut::<PrismRenderScene>();
        scene.insert_geometry(GeometryLodChain {
            geometry,
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.0,
                resident: true,
                fallback: true,
            }],
        });
        scene.insert_material(material_record.clone());
    }

    app.world_mut().spawn((
        PrismCamera::new(Mat4::IDENTITY, [0, 0, FRAME_WIDTH, FRAME_HEIGHT]),
        broad_frustum(),
        GlobalTransform::IDENTITY,
    ));
    let instance = app.world_mut().spawn((
        PrismRenderable::new(geometry, material),
        LocalBounds(Aabb3::from_center_half_extents(
            Vec3::ZERO,
            Vec3::splat(0.5),
        )),
        GlobalTransform::IDENTITY,
        prism_render_app::visibility::PrismViewVisibility::default(),
    ));
    sync_scene(app.world_mut());
    let scene_handle = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(instance)
        .expect("the production renderable receives a stable scene identity");
    assert_eq!(scene_handle, SceneHandle::new(1, 1));

    let render = app.sub_app_mut(RenderApp).expect("native render sub-app");
    render
        .world
        .resource_mut::<SceneRegistry>()
        .insert_geometry(GeometryRecord {
            handle: geometry,
            revision: 1,
            vertex_buffer_class: 5,
            index_buffer_class: 0,
            primitive_start: 0,
            primitive_count: 1,
            lods: vec![GeometryLodRecord {
                level: 0,
                primitive_kind: GeometryPrimitiveKind::Indexed,
                element_count: 6,
                first_element: 0,
                base_vertex: 0,
                vertex_count: 4,
                screen_error: 0.0,
                resident: true,
                fallback: false,
            }],
        });
    render
        .world
        .resource_mut::<SceneRegistry>()
        .bind_geometry_mesh(geometry, mesh_id);
}

fn current_commit() -> String {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("git must be available to stamp a production artifact");
    assert!(
        output.status.success(),
        "git rev-parse HEAD failed while stamping the production artifact"
    );
    let commit = String::from_utf8(output.stdout)
        .expect("git revision must be UTF-8")
        .trim()
        .to_owned();
    assert!(
        (7..=64).contains(&commit.len()) && commit.chars().all(|c| c.is_ascii_hexdigit()),
        "git returned an invalid HEAD revision: {commit:?}"
    );
    commit
}

fn csv_field(value: &str) -> String {
    value.replace(',', "%2C").replace('\n', "%0A")
}

fn production_material_buffer_ids(app: &App) -> Vec<rhi::BufferId> {
    let render = app.get_sub_app(RenderApp).expect("native render sub-app");
    let buffers = render
        .world
        .resource::<prism_render_app::prelude::MaterialParameterBuffers>();
    [buffers.headers(), buffers.parameters(), buffers.textures()]
        .into_iter()
        .flatten()
        .collect()
}

fn write_production_artifact(
    path: &Path,
    commit: &str,
    adapter: &AdapterEvidence,
    samples: &[Sample],
    warmup: u64,
) -> Result<(), String> {
    let mut output = String::new();
    writeln!(output, "schema,{SCHEMA}").unwrap();
    writeln!(output, "commit,{commit}").unwrap();
    writeln!(output, "scene,default_native_material_frame").unwrap();
    writeln!(output, "content,{CONTENT}").unwrap();
    writeln!(output, "capture_scope,{PRODUCTION_SCOPE}").unwrap();
    writeln!(output, "producer,{PRODUCTION_PRODUCER}").unwrap();
    writeln!(output, "backend,{:?}", adapter.info.backend).unwrap();
    writeln!(output, "adapter,{}", csv_field(&adapter.info.name)).unwrap();
    writeln!(output, "resolution,{FRAME_WIDTH}x{FRAME_HEIGHT}").unwrap();
    writeln!(output, "warmup,{warmup}").unwrap();
    writeln!(output, "samples,{}", samples.len()).unwrap();
    writeln!(output, "timestamp_source,gpu_timestamp_query").unwrap();
    writeln!(output, "timestamp_unit,ns").unwrap();
    for sample in samples {
        writeln!(
            output,
            "sample,{},{},{},{}",
            sample.index, sample.cpu_wall_ns, sample.gpu_timestamp_ns, sample.material_upload_bytes
        )
        .unwrap();
    }
    fs::write(path, output).map_err(|error| format!("write {}: {error}", path.display()))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} must be an unsigned integer"))
        })
        .unwrap_or(default)
}

/// Runs the actual production app graph and writes the raw material evidence
/// consumed by `material_gpu_performance_evidence_gate`.
///
/// This test is intentionally separate from the generic RHI timestamp probe:
/// every retained sample must come from a submitted command buffer containing
/// a production Native draw, and the timestamps bracket that same submission.
#[test]
#[ignore = "requires PRISM_REQUIRE_GPU=1 and a physical timestamp-capable GPU"]
fn write_real_production_native_material_artifact() {
    assert_eq!(
        std::env::var("PRISM_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "the production material sampler requires PRISM_REQUIRE_GPU=1"
    );
    let path = std::env::var_os("PRISM_MATERIAL_PERF_EVIDENCE")
        .expect("set PRISM_MATERIAL_PERF_EVIDENCE to an output artifact path");
    let path = Path::new(&path);
    let expected_commit = std::env::var("PRISM_MATERIAL_EXPECTED_COMMIT")
        .expect("set PRISM_MATERIAL_EXPECTED_COMMIT to the tested HEAD revision");
    let commit = current_commit();
    assert_eq!(
        commit, expected_commit,
        "the sampler refuses to write an artifact for a different HEAD"
    );
    // The setup frame already warms material publication and pipeline creation;
    // keep one additional warmup frame so the fixed 32-target native fixture
    // leaves room for the required 30 retained samples.
    let warmup = env_u64("PRISM_MATERIAL_PERF_WARMUP", 1);
    let sample_count = env_u64("PRISM_MATERIAL_PERF_SAMPLES", 30);
    assert!(warmup > 0, "warmup must be positive");
    assert!(sample_count >= 30, "samples must be at least 30");

    let instance = MaterialPerfInstance::new();
    let expected_adapter = instance
        .timestamp_capable_hardware()
        .expect("no physical timestamp-capable adapter is available");
    let mut app = App::new();
    let config = RenderAppConfig::new(rhi::Extent3d::new_2d(FRAME_WIDTH, FRAME_HEIGHT))
        .with_max_frames_in_flight(1)
        .with_material_capacity(8)
        .with_material_parameter_words(256)
        .with_material_texture_rows(16);
    app.add_plugins(PrismRenderPlugin::with_config(instance, config));
    production_scene_fixture(&mut app);

    // The first frame publishes the extracted material/execution snapshot. It
    // is a setup frame, not a timed sample; later frames must exercise the
    // same production draw after that publication completes.
    app.update();
    {
        let render = app.get_sub_app(RenderApp).expect("native render sub-app");
        let runtime = render
            .world
            .resource::<RenderRuntimeResource<MaterialPerfInstance>>();
        assert!(
            runtime.runtime().wait_idle(),
            "material setup frame completes"
        );
    }
    app.sub_app_mut(RenderApp)
        .expect("native render sub-app")
        .world
        .insert_resource(prism_render_app::prelude::RenderFrameStats::default());

    let render = app.get_sub_app(RenderApp).expect("native render sub-app");
    let runtime = render
        .world
        .resource::<RenderRuntimeResource<MaterialPerfInstance>>();
    assert_eq!(runtime.runtime().adapter_info(), &expected_adapter.info);
    let adapter = AdapterEvidence::from_info(runtime.runtime().adapter_info());
    assert!(
        adapter.is_hardware(),
        "the production sampler selected non-GPU hardware"
    );
    let queue = runtime.runtime().queue().clone();
    let target_samples = warmup + sample_count;
    let max_attempts = target_samples.saturating_mul(4).max(target_samples + 1);
    let mut cpu_samples = Vec::with_capacity(target_samples as usize);
    let mut captured_samples = 0u64;
    let mut attempts = 0u64;
    while captured_samples < target_samples {
        attempts += 1;
        assert!(
            attempts <= max_attempts,
            "production sampler could not collect {target_samples} submitted draw frames within {max_attempts} attempts"
        );
        let captures_before = queue.capture_count();
        app.world_mut()
            .resource_mut::<PrismRenderScene>()
            .insert_material(production_material_record(3 + attempts, attempts));
        sync_scene(app.world_mut());
        let (draws_queued_before, draws_recorded_before, frames_failed_before) = {
            let stats_before = app
                .get_sub_app(RenderApp)
                .expect("native render sub-app")
                .world
                .resource::<prism_render_app::prelude::RenderFrameStats>();
            (
                stats_before.draws_queued,
                stats_before.draws_recorded,
                stats_before.frames_failed,
            )
        };
        let cpu_started = Instant::now();
        app.update();
        let cpu_wall_ns = u64::try_from(cpu_started.elapsed().as_nanos())
            .expect("CPU wall duration must fit in u64 nanoseconds");
        {
            let render = app.get_sub_app(RenderApp).expect("native render sub-app");
            let runtime = render
                .world
                .resource::<RenderRuntimeResource<MaterialPerfInstance>>();
            assert!(
                runtime.runtime().wait_idle(),
                "production runtime did not complete while collecting a material sample"
            );
        }
        let stats = app
            .get_sub_app(RenderApp)
            .expect("native render sub-app")
            .world
            .resource::<prism_render_app::prelude::RenderFrameStats>();
        let capture_delta = queue.capture_count() - captures_before;
        if capture_delta == 0 {
            assert_eq!(
                stats.frames_failed, frames_failed_before,
                "production frame failed while waiting for a submitted draw"
            );
            continue;
        }
        assert_eq!(
            capture_delta, 1,
            "one retained sample must contain one Native draw workload"
        );
        assert_eq!(
            stats.draws_queued - draws_queued_before,
            1,
            "the retained production frame did not queue one draw"
        );
        assert_eq!(
            stats.draws_recorded - draws_recorded_before,
            1,
            "the retained production frame did not record one draw"
        );
        cpu_samples.push(cpu_wall_ns);
        captured_samples += 1;
    }

    let material_buffers = production_material_buffer_ids(&app);
    assert_eq!(
        material_buffers.len(),
        3,
        "canonical material buffers are incomplete"
    );
    let mut samples = queue
        .drain_samples(&material_buffers)
        .unwrap_or_else(|error| panic!("collect production material samples: {error}"));
    assert_eq!(samples.len(), (warmup + sample_count) as usize);
    for (sample, cpu_wall_ns) in samples.iter_mut().zip(cpu_samples).skip(warmup as usize) {
        sample.cpu_wall_ns = cpu_wall_ns;
    }
    samples.drain(..warmup as usize);
    for (index, sample) in samples.iter_mut().enumerate() {
        sample.index = index as u64;
        assert!(sample.cpu_wall_ns > 0);
        assert!(sample.gpu_timestamp_ns > 0);
    }
    write_production_artifact(path, &commit, &adapter, &samples, warmup)
        .unwrap_or_else(|error| panic!("write production material artifact: {error}"));
    eprintln!(
        "wrote REAL production native-material evidence: adapter={}, backend={:?}, samples={}, path={}",
        adapter.info.name,
        adapter.info.backend,
        samples.len(),
        path.display()
    );
}

/// Run with a real production-native-material artifact, for example:
///
/// `PRISM_REQUIRE_GPU=1 PRISM_MATERIAL_EXPECTED_COMMIT=$(git rev-parse HEAD)
/// PRISM_MATERIAL_PERF_EVIDENCE=/abs/path/evidence.csv
/// PRISM_MATERIAL_CPU_P50_MAX_NS=... PRISM_MATERIAL_CPU_P95_MAX_NS=...
/// PRISM_MATERIAL_CPU_P99_MAX_NS=... PRISM_MATERIAL_GPU_P50_MAX_NS=...
/// PRISM_MATERIAL_GPU_P95_MAX_NS=... PRISM_MATERIAL_GPU_P99_MAX_NS=...
/// cargo test -p prism_render_app --test material_performance_evidence
/// material_gpu_performance_evidence_gate -- --ignored --nocapture`
#[test]
#[ignore = "requires a timestamped production native-material artifact"]
fn material_gpu_performance_evidence_gate() {
    assert_eq!(
        std::env::var("PRISM_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "the performance gate requires PRISM_REQUIRE_GPU=1"
    );
    let expected_commit = std::env::var("PRISM_MATERIAL_EXPECTED_COMMIT")
        .expect("set PRISM_MATERIAL_EXPECTED_COMMIT to the tested HEAD revision");
    let path = std::env::var_os("PRISM_MATERIAL_PERF_EVIDENCE")
        .expect("set PRISM_MATERIAL_PERF_EVIDENCE to a collected artifact path");
    let path = Path::new(&path);
    let input =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let evidence = parse(&input)
        .unwrap_or_else(|error| panic!("invalid material performance evidence: {error}"));
    assert_eq!(
        evidence.fields["commit"], expected_commit,
        "the artifact must be captured from the requested HEAD"
    );
    assert_budgets(&evidence);
    eprintln!(
        "accepted material performance evidence: adapter={}, backend={}, samples={}, frame_gpu_p95_ns={}",
        evidence.fields["adapter"],
        evidence.fields["backend"],
        evidence.samples.len(),
        metric_percentiles(&evidence, "frame_gpu_timestamp_ns").p95,
    );
}
