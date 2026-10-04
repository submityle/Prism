//! `wgpu` compute twin of the cloth sleep/activation gate from the `CPU` golden
//! `prism_render_architecture::cloth::sleep::SleepTracker::update` (together
//! with `SleepParams::sanitized`).
//!
//! A populated scene has far more garments than the per-frame deformation
//! budget can simulate, so a garment that has come to rest is put to *sleep* and
//! stops consuming budget until something disturbs it. The reference advances a
//! hysteretic state machine once per frame: a garment only sleeps after a run of
//! quiet frames, but wakes the instant its motion crosses a higher threshold,
//! with a dead band between the two to stop the gate chattering. This module
//! ports that stateless, no-`RNG` transition onto the device: one thread
//! advances one tracker, so a passing real-device parity test is direct
//! evidence the kernel takes the same wake/quiet/dead-band branch the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `SleepParams::sanitized` (clamp the two
//! thresholds to `[0, ∞)`, mapping negatives and non-finite values to `0`, then
//! raise the wake threshold to at least the sleep threshold) and
//! `SleepTracker::update`: wake takes priority (a non-finite indicator or one
//! strictly above the wake threshold wakes a sleeper and clears the quiet
//! counter); an awake garment accumulates quiet frames while its indicator is
//! strictly below the sleep threshold and sleeps once the saturating run reaches
//! `max(frames_to_sleep, 1)`; any other frame resets the counter. The per-scene
//! `max_kinetic_indicator` reduction and the `splitmix64`-free rest of the
//! module are not twinned here; this kernel is the pure per-garment transition.
//!
//! # State encoding
//!
//! State is the `u32` rank of the reference `SleepState`: `Awake = 0`,
//! `Sleeping = 1`. The kernel treats `state == 1` as sleeping and every other
//! value as awake, exactly matching the two-variant golden the host drives with
//! `0`/`1` only.
//!
//! # Correctness model
//!
//! Every output is discrete (`state`, `quiet_frames`, `valid`), so the parity
//! test compares them exactly; the few `f32` thresholds only steer branches.
//! `is_finite(x)` is replicated identically on host and device as
//! `(x == x) && (abs(x) < 3.4e38)`: the `x == x` self-compare is the portable
//! `NaN` test (a `NaN` is the only value unequal to itself), and the magnitude
//! guard rejects the infinities, so a `NaN` indicator reads as energetic and can
//! never latch a garment asleep. `saturating_add(q, 1)` is
//! `select(q + 1, 0xFFFFFFFF, q == 0xFFFFFFFF)`.
//!
//! # Degenerate inputs
//!
//! The golden has no rejected input: every query produces a defined next state,
//! so `valid` is always `1`; the field exists only to match the shared
//! three-piece shape across this crate's twins. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `max`, ordered
//! compares, integer `==`, `select` and `+ - *` on `u32`/`f32` — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no
//! `u64`/`i64`/`f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! one `f32` `==` is the documented `NaN` self-compare; every other equality is
//! on `u32` state/counter words.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::sleep`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cloth sleep-gate kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `SleepParams::sanitized` + `SleepTracker::update` branch for
/// branch; see the module documentation for the algorithm.
const CLOTH_SLEEP_UPDATE_WGSL: &str = r#"
// Cloth sleep-gate twin: one thread per query reproduces
// SleepParams::sanitized + SleepTracker::update. It clamps both thresholds to
// [0, inf) (negatives and non-finite map to 0) and lifts the wake threshold to
// at least the sleep threshold, then advances the hysteretic state machine:
// wake takes priority, an awake garment accumulates quiet frames strictly below
// the sleep threshold and sleeps at max(frames_to_sleep, 1), and any other
// frame resets the counter. It uses only the portable core-WGSL subset
// (abs/max, ordered compares, integer ==, select, + - * on u32/f32), takes no
// optional feature and has no unbounded loop.

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Current sleep state rank: Awake = 0, Sleeping = 1.
    state: u32,
    // Consecutive quiet frames observed while awake.
    quiet_frames: u32,
    // This frame's motion indicator (a squared speed).
    indicator: f32,
    // Indicator strictly below which a frame counts as quiet.
    linear_threshold: f32,
    // Consecutive quiet frames required before sleeping.
    frames_to_sleep: u32,
    // Indicator strictly above which a sleeping garment wakes at once.
    wake_threshold: f32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Next sleep state rank: Awake = 0, Sleeping = 1.
    state: u32,
    // Next consecutive-quiet-frame counter.
    quiet_frames: u32,
    // Always 1: the golden has no rejected input.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Largest magnitude accepted as finite; the two f32 infinities exceed it.
const F32_MAX_FINITE: f32 = 3.4e38;
// Largest u32, the saturation ceiling of saturating_add.
const U32_MAX: u32 = 4294967295u;

// Portable finiteness test: x == x is the NaN self-compare (a NaN is the only
// value unequal to itself) and the magnitude guard rejects the infinities.
fn is_finite(x: f32) -> bool {
    return (x == x) && (abs(x) < F32_MAX_FINITE);
}

// Clamp a scalar to [0, inf), mapping negatives and NaN/inf to 0, mirroring
// clamp_non_negative.
fn clamp_non_negative(v: f32) -> f32 {
    if (is_finite(v) && v > 0.0) {
        return v;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // SleepParams::sanitized.
    let lin = clamp_non_negative(q.linear_threshold);
    let wake_raw = clamp_non_negative(q.wake_threshold);
    var wake: f32 = lin;
    if (wake_raw > lin) {
        wake = wake_raw;
    }

    var st: u32 = q.state;
    var quiet: u32 = q.quiet_frames;

    // Wake takes priority: a non-finite indicator or one strictly above the
    // wake threshold is energetic, so a NaN can never latch a garment asleep.
    let energetic = (!is_finite(q.indicator)) || (q.indicator > wake);

    if (st == 1u) {
        // Sleeping: only an energetic frame wakes it; otherwise hold.
        if (energetic) {
            st = 0u;
            quiet = 0u;
        }
    } else {
        // Awake.
        if (energetic) {
            quiet = 0u;
        } else if (is_finite(q.indicator) && q.indicator < lin) {
            // Quiet frame: saturating increment, then sleep at the dwell.
            let bumped = select(quiet + 1u, U32_MAX, quiet == U32_MAX);
            quiet = bumped;
            let dwell = max(q.frames_to_sleep, 1u);
            if (quiet >= dwell) {
                st = 1u;
            }
        } else {
            // Dead band: awake, but making no progress toward sleep.
            quiet = 0u;
        }
    }

    var res: Result;
    res.state = st;
    res.quiet_frames = quiet;
    res.valid = 1u;
    res.pad0 = 0u;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_SLEEP_UPDATE_WGSL`].
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
/// All scalar lanes, so the layout is a flat `32`-byte stride with no vector
/// alignment rule to trip; the two trailing pad words round the tail up to a
/// `16`-byte multiple so a batch of two or more packs contiguously.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    state: u32,
    quiet_frames: u32,
    indicator: f32,
    linear_threshold: f32,
    frames_to_sleep: u32,
    wake_threshold: f32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three live words plus one pad word give a flat `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    state: u32,
    quiet_frames: u32,
    valid: u32,
    pad0: u32,
}

/// One query for the cloth sleep-gate twin: the garment's current `state` and
/// `quiet_frames`, this frame's motion `indicator`, and the three
/// [`SleepParams`](prism_render_architecture) fields (`linear_threshold`,
/// `frames_to_sleep`, `wake_threshold`).
///
/// The whole twinned transition is driven by this one tuple, so a single query
/// exercises the sanitize, the wake priority, the quiet accumulation and the
/// dead-band reset at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothSleepUpdateQuery {
    /// Current sleep state rank: `Awake = 0`, `Sleeping = 1`.
    pub state: u32,
    /// Consecutive quiet frames observed while awake.
    pub quiet_frames: u32,
    /// This frame's motion indicator (a squared speed).
    pub indicator: f32,
    /// Motion indicator strictly below which a frame counts as quiet.
    pub linear_threshold: f32,
    /// Consecutive quiet frames required before sleeping (treated as at least
    /// one).
    pub frames_to_sleep: u32,
    /// Motion indicator strictly above which a sleeping garment wakes at once.
    pub wake_threshold: f32,
}

impl ClothSleepUpdateQuery {
    /// Builds a query from the current state, quiet counter, indicator and the
    /// three sleep parameters.
    #[must_use]
    pub fn new(
        state: u32,
        quiet_frames: u32,
        indicator: f32,
        linear_threshold: f32,
        frames_to_sleep: u32,
        wake_threshold: f32,
    ) -> ClothSleepUpdateQuery {
        ClothSleepUpdateQuery {
            state,
            quiet_frames,
            indicator,
            linear_threshold,
            frames_to_sleep,
            wake_threshold,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `SleepTracker::update` output.
///
/// `state` is the next sleep-state rank (`Awake = 0`, `Sleeping = 1`) and
/// `quiet_frames` the next quiet counter. `valid` is always `1`: the golden has
/// no rejected input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothSleepUpdateResult {
    /// Next sleep state rank: `Awake = 0`, `Sleeping = 1`.
    pub state: u32,
    /// Next consecutive-quiet-frame counter.
    pub quiet_frames: u32,
    /// Always `1`: the golden produces a defined next state for every input.
    pub valid: u32,
}

/// Encodes one [`ClothSleepUpdateQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothSleepUpdateQuery) -> GpuQuery {
    GpuQuery {
        state: q.state,
        quiet_frames: q.quiet_frames,
        indicator: q.indicator,
        linear_threshold: q.linear_threshold,
        frames_to_sleep: q.frames_to_sleep,
        wake_threshold: q.wake_threshold,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothSleepUpdateResult`].
fn decode_result(raw: &GpuResult) -> ClothSleepUpdateResult {
    ClothSleepUpdateResult {
        state: raw.state,
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

/// A compiled, reusable cloth sleep-gate compute pipeline, twinning the `CPU`
/// golden `SleepTracker::update`.
pub struct GpuClothSleepUpdate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothSleepUpdate {
    /// Compiles the cloth sleep-gate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothSleepUpdate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update"),
            source: ShaderSource::Wgsl(CLOTH_SLEEP_UPDATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothSleepUpdate {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ClothSleepUpdateResult`]
    /// per input, in order.
    ///
    /// Every output is discrete and matches the reference exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothSleepUpdateQuery],
    ) -> Vec<ClothSleepUpdateResult> {
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
            label: Some("prism_volumetric_cloth_sleep_update_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_bind_group"),
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
            label: Some("prism_volumetric_cloth_sleep_update_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_sleep_update_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_sleep_update_pass"),
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
