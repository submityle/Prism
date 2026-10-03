//! `wgpu` compute twin of the per-frame thin-feature lock transition inside the
//! temporal-upscale reconstruction contract
//! ([`lock`](prism_render_architecture::temporal_upscale::lock)).
//!
//! The `CPU` golden
//! [`lock`](prism_render_architecture::temporal_upscale::lock) owns the
//! per-pixel thin-feature lifecycle that protects one-pixel luminance features
//! (a power line, a railing, a specular glint) from the neighborhood clamp. Its
//! sibling twin
//! [`taau_thin_feature_lock`](crate::taau_thin_feature_lock) ports the two
//! stateless closed forms (`thin_feature_strength` and `rejection_scale`) but
//! deliberately leaves the single-frame state transition
//! [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
//! on the host. That transition is itself a pure, branch-only, transcendental-
//! free function of one previous
//! [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
//! plus the frame's `current_luma`, `thin_strength`, and `disoccluded` verdict,
//! so one thread can reproduce one pixel's next lock state exactly. This module
//! is that twin.
//!
//! [`GpuTaauAdvanceLock`] evaluates the transition for a batch of pixels, so a
//! passing real-device parity test is direct evidence the ported kernel takes
//! the same discrete branch and emits the same next state the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one pixel the kernel reproduces
//! [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
//! rule-for-rule, in order:
//!
//! 1. a `disoccluded` pixel drops any lock and returns the unlocked default
//!    (`lifetime = 0`, `luma = 0`);
//! 2. an existing lock (`previous_lifetime > 0`) whose captured `luma` has
//!    drifted past the relative tolerance
//!    `LOCK_BREAK_LUMA_TOLERANCE * (|previous_luma| + 1e-3)` is dropped;
//! 3. a surviving lock loses one frame of lifetime, and only remains locked
//!    while `previous_lifetime - 1 > 0`, keeping its captured `luma`;
//! 4. independently, a `thin_strength` at or above `LOCK_CREATION_THRESHOLD`
//!    creates or refreshes the lock to the full `INITIAL_LOCK_LIFETIME`,
//!    capturing `current_luma`, superseding any decayed state.
//!
//! The emitted result carries the next lock's `lifetime`, its captured `luma`,
//! and the derived `locked` flag (the golden
//! [`LockState::is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked),
//! `lifetime > 0`).
//!
//! # What stays on the host
//!
//! The surrounding per-frame sequencing — carrying each pixel's
//! [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
//! across frames in the history buffer, and feeding this transition the
//! per-pixel `current_luma`, `thin_strength` (from
//! [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength))
//! and `disoccluded` verdict (from
//! [`reproject`](prism_render_architecture::temporal_upscale::reproject)) — is
//! orchestration the host owns; the kernel twins only the stateless single-step
//! transition. An empty batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The next `lifetime` is one of three exact values — `0`, the literal
//! `INITIAL_LOCK_LIFETIME`, or `previous_lifetime - 1` — and the next `luma` is
//! one of `0`, `previous_luma`, or `current_luma`, each a copy or a single
//! subtract of identical operands, so the two engines agree to within a legal
//! last-place difference; the parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on both scalars. The `locked` flag
//! is a discrete decision and is asserted exactly (`==`). The only comparison
//! whose two sides are *computed* (rather than copied inputs) is the luma-drift
//! break `drift <= tolerance`; fixtures keep that margin well clear of zero so
//! `CPU` and `GPU` cannot straddle it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /` and
//! unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `smoothstep`, no `round` and no `sqrt`.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of branches and arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::lock`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` thin-feature lock-transition kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
/// transition for one pixel; see the module documentation for the rules.
const TAAU_ADVANCE_LOCK_WGSL: &str = r#"
// Temporal-upscale lock-transition twin: one thread advances one pixel's
// thin-feature LockState by a single frame, mirroring the CPU golden
// `temporal_upscale::lock::advance_lock` rule-for-rule (disocclusion break,
// relative-luma staleness break, one-frame decay, and the create/refresh rule)
// with only abs, + - * / and unsigned index arithmetic. It owns no cross-frame
// history sequencing; the host carries each pixel's LockState between frames.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::lock；无第三方
// 引擎源码或衍生代码。

// Lifetime, in frames, granted to a freshly created lock. Mirrors the golden
// INITIAL_LOCK_LIFETIME.
const INITIAL_LOCK_LIFETIME: f32 = 4.0;
// Minimum thin-feature strength that creates a new lock. Mirrors the golden
// LOCK_CREATION_THRESHOLD.
const LOCK_CREATION_THRESHOLD: f32 = 0.25;
// Fractional luma change that breaks an existing lock. Mirrors the golden
// LOCK_BREAK_LUMA_TOLERANCE.
const LOCK_BREAK_LUMA_TOLERANCE: f32 = 0.25;

struct Params {
    // Number of pixels in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Remaining lock lifetime carried from the previous frame; <= 0 is unlocked.
    previous_lifetime: f32,
    // Luminance captured when the previous lock was created.
    previous_luma: f32,
    // This frame's luminance at the pixel.
    current_luma: f32,
    // This frame's thin-feature strength in [0, 1].
    thin_strength: f32,
    // 1 when the pixel was disoccluded this frame (golden `disoccluded`), else 0.
    disoccluded: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Next lock lifetime in frames; <= 0 is unlocked.
    lifetime: f32,
    // Next captured luminance.
    luma: f32,
    // 1 when the next state holds a live lock (golden is_locked), else 0.
    locked: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Default: the unlocked state. A disoccluded pixel invalidates its past, so
    // any lock is dropped and the defaults below are emitted unchanged.
    var out_lifetime: f32 = 0.0;
    var out_luma: f32 = 0.0;

    if (q.disoccluded == 0u) {
        // Decay / break an existing lock into `decay_*`.
        var decay_lifetime: f32 = 0.0;
        var decay_luma: f32 = 0.0;
        if (q.previous_lifetime > 0.0) {
            let drift = abs(q.current_luma - q.previous_luma);
            let tolerance = LOCK_BREAK_LUMA_TOLERANCE * (abs(q.previous_luma) + 1.0e-3);
            if (drift <= tolerance) {
                let life = q.previous_lifetime - 1.0;
                if (life > 0.0) {
                    decay_lifetime = life;
                    decay_luma = q.previous_luma;
                }
            }
        }

        // A detected thin feature creates / refreshes the lock, superseding the
        // decayed state; otherwise the decayed state is carried forward.
        if (q.thin_strength >= LOCK_CREATION_THRESHOLD) {
            out_lifetime = INITIAL_LOCK_LIFETIME;
            out_luma = q.current_luma;
        } else {
            out_lifetime = decay_lifetime;
            out_luma = decay_luma;
        }
    }

    var locked: u32 = 0u;
    if (out_lifetime > 0.0) {
        locked = 1u;
    }

    var out: Result;
    out.lifetime = out_lifetime;
    out.luma = out_luma;
    out.locked = locked;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pixel count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_ADVANCE_LOCK_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pixels in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pixel query, matching the `WGSL` `Query`
/// struct: the previous lock `lifetime` and `luma`, this frame's `current_luma`
/// and `thin_strength`, the `disoccluded` flag, and three pad words to a
/// `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Remaining lock lifetime carried from the previous frame.
    previous_lifetime: f32,
    /// Luminance captured when the previous lock was created.
    previous_luma: f32,
    /// This frame's luminance at the pixel.
    current_luma: f32,
    /// This frame's thin-feature strength in `[0, 1]`.
    thin_strength: f32,
    /// `1` when the pixel was disoccluded this frame, `0` otherwise.
    disoccluded: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pixel result, matching the `WGSL` `Result`
/// struct: the next lock `lifetime`, its captured `luma`, the derived `locked`
/// flag, and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Next lock lifetime in frames.
    lifetime: f32,
    /// Next captured luminance.
    luma: f32,
    /// `1` when the next state holds a live lock, `0` otherwise.
    locked: u32,
    /// Padding word.
    pad0: u32,
}

/// One per-pixel query for the lock-transition twin: the previous lock state
/// plus this frame's inputs to
/// [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock).
///
/// The host carries each pixel's
/// [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
/// across frames and enqueues one [`TaauAdvanceLockQuery`] per pixel, mirroring
/// the inputs the reference transition takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauAdvanceLockQuery {
    /// Remaining lock lifetime carried from the previous frame (the golden
    /// `LockState` `lifetime`); `<= 0` is unlocked.
    pub previous_lifetime: f32,
    /// Luminance captured when the previous lock was created (the golden
    /// `LockState` `luma`).
    pub previous_luma: f32,
    /// This frame's luminance at the pixel.
    pub current_luma: f32,
    /// This frame's thin-feature strength in `[0, 1]`.
    pub thin_strength: f32,
    /// Whether the pixel was disoccluded this frame (the golden `disoccluded`).
    pub disoccluded: bool,
}

impl TaauAdvanceLockQuery {
    /// Builds a query from the previous lock state and this frame's inputs.
    #[must_use]
    pub const fn new(
        previous_lifetime: f32,
        previous_luma: f32,
        current_luma: f32,
        thin_strength: f32,
        disoccluded: bool,
    ) -> TaauAdvanceLockQuery {
        TaauAdvanceLockQuery {
            previous_lifetime,
            previous_luma,
            current_luma,
            thin_strength,
            disoccluded,
        }
    }
}

/// One resolved pixel of the lock-transition twin, mirroring the next
/// [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
/// the reference
/// [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
/// produces.
///
/// `lifetime` and `luma` are the next lock state's fields; `locked` is the
/// derived
/// [`LockState::is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked)
/// flag (`lifetime > 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauAdvanceLockResult {
    /// Next lock lifetime in frames; `<= 0` is unlocked.
    pub lifetime: f32,
    /// Next captured luminance.
    pub luma: f32,
    /// Whether the next state holds a live lock.
    pub locked: bool,
}

/// Encodes one [`TaauAdvanceLockQuery`] into its `std430` [`GpuQuery`] slot,
/// turning the `disoccluded` [`bool`] into a `u32` word.
fn encode_query(q: &TaauAdvanceLockQuery) -> GpuQuery {
    GpuQuery {
        previous_lifetime: q.previous_lifetime,
        previous_luma: q.previous_luma,
        current_luma: q.current_luma,
        thin_strength: q.thin_strength,
        disoccluded: u32::from(q.disoccluded),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauAdvanceLockResult`],
/// turning the `locked` `u32` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> TaauAdvanceLockResult {
    TaauAdvanceLockResult {
        lifetime: raw.lifetime,
        luma: raw.luma,
        locked: raw.locked != 0,
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

/// A compiled, reusable lock-transition compute pipeline, twinning the single-
/// step state transition of the `CPU` golden
/// [`lock`](prism_render_architecture::temporal_upscale::lock).
pub struct GpuTaauAdvanceLock {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauAdvanceLock {
    /// Compiles the lock-transition kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauAdvanceLock {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_advance_lock"),
            source: ShaderSource::Wgsl(TAAU_ADVANCE_LOCK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauAdvanceLock {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances every pixel in `queries` by one frame and returns one
    /// [`TaauAdvanceLockResult`] per input, in order.
    ///
    /// Every output matches the reference transition exactly on the discrete
    /// `locked` flag and within the tolerance documented on this module for the
    /// continuous `lifetime` and `luma`. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauAdvanceLockQuery],
    ) -> Vec<TaauAdvanceLockResult> {
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
            label: Some("prism_volumetric_taau_advance_lock_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_bind_group"),
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
            label: Some("prism_volumetric_taau_advance_lock_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_advance_lock_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_advance_lock_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
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
