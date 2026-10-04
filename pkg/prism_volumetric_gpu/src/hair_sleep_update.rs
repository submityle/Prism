//! `wgpu` compute twin of the guide-strand groom sleep gate from the `CPU`
//! golden `prism_render_architecture::hair::sleep::update_sleep`.
//!
//! A crowd scene has far more grooms than the per-frame deformation budget can
//! simulate, so a groom that has come to rest is put to *sleep* and stops
//! consuming the shared budget until something disturbs it. The gate is
//! hysteretic: a groom only sleeps after a run of quiet frames, but it wakes
//! the instant its motion energy crosses a higher threshold, and the two
//! thresholds straddle a dead band to avoid chattering on the edge.
//!
//! This module ports that stateless, closed-form frame advance onto the device:
//! one thread advances one groom's sleep state. [`GpuHairSleepUpdate`] is the
//! on-device twin; a passing real-device parity test is direct evidence the
//! kernel takes the same ordered branches and the same integer counter
//! arithmetic the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! * `update_sleep(state, motion_energy, thresholds)` — the whole closed form:
//!   the wake-takes-priority test `!is_finite(e) || e >= wake_above`, the
//!   already-asleep hold, the quiet accumulation `e.is_finite() && e <=
//!   sleep_below`, the `saturating_add(1)` counter, the
//!   `quiet_frames >= max(frames_to_sleep, 1)` sleep latch, and the dead-band
//!   reset to the awake state.
//!
//! Unlike the cloth sleep gate, the reference performs **no sanitization** of
//! the thresholds: negative, zero or `NaN` thresholds flow through verbatim, so
//! the twin uses them verbatim too. Note the comparisons are inclusive
//! (`>=` for wake, `<=` for quiet), where the cloth gate uses strict ones.
//!
//! # Result encoding
//!
//! The reference returns a `GroomSleepState { asleep: bool, quiet_frames: u32
//! }`. The twin flattens `asleep` into a `u32` (`1`/`0`) and carries
//! `quiet_frames` directly. `valid` is always `1`: this gate has no degenerate
//! rejection path, since every finite or non-finite input maps to a defined
//! next state. The field exists so the layout matches the crate's other
//! classifier twins.
//!
//! # Correctness model
//!
//! Every output channel here is a discrete `u32`, so parity is checked with
//! exact equality. There is no continuous output and no rounding tie to guard,
//! but the random sweep still keeps samples clear of the exact
//! `motion_energy == wake_above` and `motion_energy == sleep_below` ties so a
//! host/device ordering difference on the inclusive compares cannot flip a
//! discrete channel; dedicated named fixtures pin those exact-boundary cases.
//!
//! # Degenerate inputs
//!
//! Non-finite motion energy (`NaN`, `+/-inf`) counts as motion and keeps the
//! groom awake, so a `NaN` can never latch a groom asleep. `frames_to_sleep` of
//! `0` is lifted to `1` by `max(frames_to_sleep, 1)`, so one quiet frame still
//! suffices. A saturating counter never wraps past `u32::MAX`. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `abs`, `max` and `u32` arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no float modulo and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only
//! float equality is the mandated `x == x` self-compare that detects `NaN`
//! inside `is_finite`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::sleep`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` groom sleep-gate kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `update_sleep` branch for branch; see the module
/// documentation for the algorithm.
const HAIR_SLEEP_UPDATE_WGSL: &str = r#"
// Hair groom sleep-gate twin: one thread per groom advances its hysteretic
// sleep state by one frame, mirroring hair::sleep::update_sleep. The reference
// does no threshold sanitization, so thresholds are consumed verbatim.
// Provenance: 孪生自本仓 prism_render_architecture::hair::sleep；无第三方引擎源码或衍生代码。

// Largest magnitude accepted as finite; matches f32::is_finite semantics.
const F32_MAX: f32 = 3.4e38;
// u32::MAX, the saturation ceiling of the quiet-frame counter.
const U32_MAX: u32 = 4294967295u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Current asleep flag: 1 asleep, 0 awake.
    asleep: u32,
    // Consecutive quiet frames accumulated so far.
    quiet_frames: u32,
    // This frame's groom motion energy.
    motion_energy: f32,
    // Energy at or below which a frame counts as quiet.
    sleep_below: f32,
    // Energy at or above which an asleep groom wakes immediately.
    wake_above: f32,
    // Consecutive quiet frames required before sleeping.
    frames_to_sleep: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Next asleep flag: 1 asleep, 0 awake.
    asleep: u32,
    // Next consecutive-quiet-frame counter.
    quiet_frames: u32,
    // Always 1; this gate has no degenerate rejection path.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Mirrors f32::is_finite: the self-compare rejects NaN and the magnitude bound
// rejects the infinities. The x == x self-compare is the mandated NaN probe.
fn is_finite(x: f32) -> bool {
    return (x == x) && (abs(x) < F32_MAX);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let energy = q.motion_energy;
    let finite = is_finite(energy);

    // Wake takes priority: non-finite energy or energy at/above wake_above.
    let energetic = (!finite) || (energy >= q.wake_above);

    let was_asleep = q.asleep == 1u;

    // Awake path, quiet frame: finite and at/below sleep_below.
    let quiet_cond = finite && (energy <= q.sleep_below);
    // Saturating increment of the quiet-frame counter.
    let next_quiet = select(q.quiet_frames + 1u, U32_MAX, q.quiet_frames == U32_MAX);
    let needed = max(q.frames_to_sleep, 1u);
    let latches = next_quiet >= needed;
    let accum_asleep = select(0u, 1u, latches);

    // Awake path composition: quiet -> (accum, next_quiet); dead band -> (0, 0).
    let awake_asleep = select(0u, accum_asleep, quiet_cond);
    let awake_quiet = select(0u, next_quiet, quiet_cond);

    // Already-asleep, not energetic: hold the incoming state unchanged.
    let hold_asleep = select(awake_asleep, q.asleep, was_asleep);
    let hold_quiet = select(awake_quiet, q.quiet_frames, was_asleep);

    // Energetic overrides everything with the AWAKE state (0, 0).
    var out: Result;
    out.asleep = select(hold_asleep, 0u, energetic);
    out.quiet_frames = select(hold_quiet, 0u, energetic);
    out.valid = 1u;
    out.pad0 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Six payload words plus two padding words keep the stride a flat `32` bytes,
/// a multiple of `16` with no vector-alignment surprises.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    asleep: u32,
    quiet_frames: u32,
    motion_energy: f32,
    sleep_below: f32,
    wake_above: f32,
    frames_to_sleep: u32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three payload words plus one padding word keep the stride a flat
/// `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    asleep: u32,
    quiet_frames: u32,
    valid: u32,
    pad0: u32,
}

/// One groom sleep-gate query: the current sleep state plus this frame's motion
/// energy and the hysteresis thresholds, flattened to scalars so the `std430`
/// stride stays an unambiguous flat layout. The thresholds are consumed
/// verbatim, exactly as the reference does (no sanitization).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairSleepUpdateQuery {
    /// Current asleep flag: `1` asleep, `0` awake.
    pub asleep: u32,
    /// Consecutive quiet frames accumulated so far.
    pub quiet_frames: u32,
    /// This frame's groom motion energy.
    pub motion_energy: f32,
    /// Energy at or below which a frame counts as quiet.
    pub sleep_below: f32,
    /// Energy at or above which an asleep groom wakes immediately.
    pub wake_above: f32,
    /// Consecutive quiet frames required before sleeping.
    pub frames_to_sleep: u32,
}

impl HairSleepUpdateQuery {
    /// Builds a sleep-gate query from the current state, this frame's motion
    /// energy and the hysteresis thresholds.
    #[must_use]
    pub fn new(
        asleep: u32,
        quiet_frames: u32,
        motion_energy: f32,
        sleep_below: f32,
        wake_above: f32,
        frames_to_sleep: u32,
    ) -> HairSleepUpdateQuery {
        HairSleepUpdateQuery {
            asleep,
            quiet_frames,
            motion_energy,
            sleep_below,
            wake_above,
            frames_to_sleep,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GroomSleepState` with the `asleep` flag flattened to a `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HairSleepUpdateResult {
    /// Next asleep flag: `1` asleep, `0` awake.
    pub asleep: u32,
    /// Next consecutive-quiet-frame counter.
    pub quiet_frames: u32,
    /// Always `1`; this gate has no degenerate rejection path.
    pub valid: u32,
}

/// Encodes one [`HairSleepUpdateQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HairSleepUpdateQuery) -> GpuQuery {
    GpuQuery {
        asleep: q.asleep,
        quiet_frames: q.quiet_frames,
        motion_energy: q.motion_energy,
        sleep_below: q.sleep_below,
        wake_above: q.wake_above,
        frames_to_sleep: q.frames_to_sleep,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairSleepUpdateResult`].
fn decode_result(raw: &GpuResult) -> HairSleepUpdateResult {
    HairSleepUpdateResult {
        asleep: raw.asleep,
        quiet_frames: raw.quiet_frames,
        valid: raw.valid,
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable groom sleep-gate compute pipeline, twinning the `CPU`
/// golden `update_sleep`.
pub struct GpuHairSleepUpdate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairSleepUpdate {
    /// Compiles the groom sleep-gate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairSleepUpdate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_sleep_update"),
            source: ShaderSource::Wgsl(HAIR_SLEEP_UPDATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairSleepUpdate {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairSleepUpdateResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairSleepUpdateQuery],
    ) -> Vec<HairSleepUpdateResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_sleep_update_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_sleep_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
