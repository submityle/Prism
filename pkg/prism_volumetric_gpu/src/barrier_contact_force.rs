//! `wgpu` compute twin of the rational soft-barrier contact scalars for hair
//! ([`barrier_contact`](prism_render_architecture::hair::barrier_contact), hair
//! design §8.5 item 4).
//!
//! The `CPU` golden pair
//! [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
//! turn a contact distance `d` and a
//! [`BarrierParams`](prism_render_architecture::hair::barrier_contact::BarrierParams)
//! into the soft-barrier energy `b(d)` and the repulsive force magnitude
//! `-b'(d)`. Both are closed form: evaluated on the clamped distance
//! `d_e = clamp(d, d_floor, dhat)` with `gap = dhat - d_e` and
//! `recip = 1/d_e - 1/dhat`,
//!
//! ```text
//!     energy = stiffness * gap^2 * recip
//!     force  = stiffness * gap * (2 * recip + gap / d_e^2)
//! ```
//!
//! and both return `0` in the free region `d >= dhat` and for a non-finite `d`.
//! The golden uses only `clamp`/`min`/`max`/`abs` and `+ - * /`, with no `ln`,
//! `exp` or `pow`, so the whole scalar core ports to the portable core-`WGSL`
//! subset.
//!
//! [`GpuBarrierContactForce`] is the on-device twin of that scalar core: one
//! thread solves one `(d, dhat, stiffness, d_floor)` query, first replicating
//! [`BarrierParams::sanitized`](prism_render_architecture::hair::barrier_contact::BarrierParams::sanitized)
//! (forcing `dhat` finite `> 0`, pulling `d_floor` into the open interval
//! `(0, dhat)` under the guard `hi = dhat * 0.5`, and clamping `stiffness` to a
//! finite `>= 0`) and then evaluating the energy and force. A passing
//! real-device parity test is therefore direct evidence the ported kernel
//! computes the same barrier response the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! The per-query sanitize-then-evaluate scalar pipeline:
//! [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
//! sharing one
//! [`BarrierParams::sanitized`](prism_render_architecture::hair::barrier_contact::BarrierParams::sanitized)
//! prelude. The `friction_mu` field is irrelevant to the energy and force, so
//! it is neither carried nor sanitized by the twin.
//!
//! # What stays on the host
//!
//! The non-finite-`d` short-circuit is pre-judged on the host: a non-finite `d`
//! is flagged in the encoded query so the device writes `0` for both outputs
//! without ever feeding a `NaN`/infinity into the barrier arithmetic, exactly as
//! the golden's `!d.is_finite()` early return does. The full contact-resolution
//! aggregate — [`resolve_contact`](prism_render_architecture::hair::barrier_contact)
//! and its batch form — is variable-length, mutable, inverse-mass-weighted state
//! that the host owns; the device never sees it.
//!
//! # Correctness model
//!
//! The sanitize prelude's decisions (`dhat` fallback, `d_floor` guard, the
//! `stiffness` floor) and the `d >= dhat` free-region test are magnitude
//! comparisons, so for fixtures chosen clear of a branch tie the `CPU` and
//! `GPU` take the same branch. The continuous `energy` and `force` thread
//! through divides, so `CPU` and `GPU` are not bit-exact: the parity test
//! asserts a tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`), tight enough
//! to catch a genuinely wrong port yet loose enough to admit a legal last-place
//! divide difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`,
//! `ceil` or `sqrt`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::barrier_contact`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` soft-barrier contact kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
/// and
/// [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
/// closed forms sharing their
/// [`BarrierParams::sanitized`](prism_render_architecture::hair::barrier_contact::BarrierParams::sanitized)
/// prelude; see the module documentation for the algorithm.
const BARRIER_CONTACT_FORCE_WGSL: &str = r#"
// Soft-barrier contact twin: one thread computes the barrier energy and the
// repulsive force magnitude for one contact distance, mirroring the CPU golden
// `hair::barrier_contact` closed forms with only min/max/clamp/abs and
// + - * /. It replicates `BarrierParams::sanitized` on-device; the non-finite
// distance short-circuit is pre-judged on the host via the `d_finite` flag.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::barrier_contact；无第三方
// 引擎源码或衍生代码。

// Golden `BarrierParams` defaults.
const DEFAULT_DHAT: f32 = 1.0e-2;
const DEFAULT_STIFFNESS: f32 = 1.0;
const DEFAULT_D_FLOOR: f32 = 1.0e-3;
// Smallest positive normal f32, the golden's `f32::MIN_POSITIVE` floor.
const MIN_POSITIVE: f32 = 1.17549435e-38;
// Largest finite f32; used to classify finiteness without an == comparison:
// a finite value lies in [-F32_MAX, F32_MAX]; +/-inf and NaN fall outside.
const F32_MAX: f32 = 3.40282347e38;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Contact distance d (finite; non-finite d is gated by d_finite below).
    d: f32,
    // Activation distance dhat before sanitize.
    dhat: f32,
    // Barrier stiffness k before sanitize.
    stiffness: f32,
    // Hard distance floor before sanitize.
    d_floor: f32,
    // 1 when the host saw a finite d, 0 when d was non-finite (-> 0, 0 output).
    d_finite: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Soft-barrier energy b(d).
    energy: f32,
    // Repulsive barrier force magnitude -b'(d).
    force: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A value is finite when it lies within the closed finite range; +/-inf and
// NaN fall outside (NaN fails both comparisons), so no == is needed.
fn is_finite(x: f32) -> bool {
    return (x <= F32_MAX) && (x >= -F32_MAX);
}

// Golden `finite_or`: keep x when finite, else the fallback.
fn finite_or(x: f32, fallback: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return fallback;
}

// Golden `sanitize_nonneg`: finite -> max(x, 0); non-finite -> 0.
fn sanitize_nonneg(x: f32) -> f32 {
    if (is_finite(x)) {
        return max(x, 0.0);
    }
    return 0.0;
}

// Golden `sanitize_pos`: finite and > 0 -> x; else the default.
fn sanitize_pos(x: f32, fallback: f32) -> f32 {
    if (is_finite(x) && x > 0.0) {
        return x;
    }
    return fallback;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.pad0 = 0u;
    out.pad1 = 0u;

    // Non-finite distance is treated as "far": no barrier (host pre-judged).
    if (q.d_finite == 0u) {
        out.energy = 0.0;
        out.force = 0.0;
        results[idx] = out;
        return;
    }

    // Replicate BarrierParams::sanitized.
    let dhat = sanitize_pos(q.dhat, DEFAULT_DHAT);
    let hi = dhat * 0.5;
    let floor_raw = sanitize_pos(q.d_floor, DEFAULT_D_FLOOR);
    let d_floor = max(min(floor_raw, hi), MIN_POSITIVE);
    let stiffness = sanitize_nonneg(q.stiffness);

    // Free region: no barrier.
    if (q.d >= dhat) {
        out.energy = 0.0;
        out.force = 0.0;
        results[idx] = out;
        return;
    }

    let de = clamp(q.d, d_floor, dhat);
    let gap = dhat - de;
    let recip = 1.0 / de - 1.0 / dhat;
    let energy = finite_or(max(stiffness * (gap * gap) * recip, 0.0), 0.0);

    let inv_de = 1.0 / de;
    let bracket = 2.0 * recip + gap * (inv_de * inv_de);
    let force = finite_or(max(stiffness * gap * bracket, 0.0), 0.0);

    out.energy = energy;
    out.force = force;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BARRIER_CONTACT_FORCE_WGSL`].
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

/// `repr(C)` `std430` layout of one barrier query: the four scalars plus a
/// finite flag for `d` and three pad words to a `32`-byte stride, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Contact distance `d` (replaced with `0` when non-finite).
    d: f32,
    /// Activation distance `dhat` before sanitize.
    dhat: f32,
    /// Barrier stiffness `k` before sanitize.
    stiffness: f32,
    /// Hard distance floor `d_floor` before sanitize.
    d_floor: f32,
    /// `1` when the host saw a finite `d`, `0` otherwise.
    d_finite: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one barrier result, matching the `WGSL`
/// `Result` struct: the energy, the force, and two pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Soft-barrier energy `b(d)`.
    energy: f32,
    /// Repulsive barrier force magnitude `-b'(d)`.
    force: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One soft-barrier contact query: the contact distance `d` and the three
/// [`BarrierParams`](prism_render_architecture::hair::barrier_contact::BarrierParams)
/// scalars the energy and force depend on.
///
/// The `friction_mu` field of the golden params does not affect the energy or
/// force, so it is intentionally absent here. The host owns the surrounding
/// contact-resolution aggregate and enqueues one [`BarrierContactForceQuery`]
/// per contact point, matching the reference
/// [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
/// and
/// [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
/// inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierContactForceQuery {
    /// Contact distance `d`; a non-finite value yields a zero energy and force.
    pub d: f32,
    /// Activation distance `dhat` before sanitize (forced finite `> 0`).
    pub dhat: f32,
    /// Barrier stiffness `k` before sanitize (clamped to finite `>= 0`).
    pub stiffness: f32,
    /// Hard distance floor `d_floor` before sanitize (pulled into `(0, dhat)`).
    pub d_floor: f32,
}

impl BarrierContactForceQuery {
    /// Builds a query for distance `d` with the given barrier scalars.
    #[must_use]
    pub const fn new(d: f32, dhat: f32, stiffness: f32, d_floor: f32) -> BarrierContactForceQuery {
        BarrierContactForceQuery {
            d,
            dhat,
            stiffness,
            d_floor,
        }
    }
}

/// One resolved soft-barrier contact response, mirroring the reference scalar
/// pair: the barrier `energy` and the repulsive `force` magnitude.
///
/// Both are `0` in the free region `d >= dhat` and for a non-finite `d`, and
/// never `NaN`/infinity, matching the golden's `finite_or` guards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarrierContactForceResult {
    /// Soft-barrier energy `b(d)` from
    /// [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy).
    pub energy: f32,
    /// Repulsive force magnitude `-b'(d)` from
    /// [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude).
    pub force: f32,
}

/// Encodes one [`BarrierContactForceQuery`] into its `std430` [`GpuQuery`] slot,
/// pre-judging the non-finite-`d` short-circuit on the host so a `NaN`/infinity
/// never reaches the device arithmetic.
fn encode_query(q: &BarrierContactForceQuery) -> GpuQuery {
    let finite = q.d.is_finite();
    GpuQuery {
        d: if finite { q.d } else { 0.0 },
        dhat: q.dhat,
        stiffness: q.stiffness,
        d_floor: q.d_floor,
        d_finite: u32::from(finite),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`BarrierContactForceResult`].
fn decode_result(raw: &GpuResult) -> BarrierContactForceResult {
    BarrierContactForceResult {
        energy: raw.energy,
        force: raw.force,
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

/// A compiled, reusable soft-barrier contact compute pipeline, twinning the
/// scalar core of the `CPU` golden
/// [`barrier_contact`](prism_render_architecture::hair::barrier_contact).
pub struct GpuBarrierContactForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBarrierContactForce {
    /// Compiles the soft-barrier contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBarrierContactForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_barrier_contact_force"),
            source: ShaderSource::Wgsl(BARRIER_CONTACT_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBarrierContactForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BarrierContactForceResult`] per input, in order.
    ///
    /// Each energy and force matches the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BarrierContactForceQuery],
    ) -> Vec<BarrierContactForceResult> {
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
            label: Some("prism_volumetric_barrier_contact_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_bind_group"),
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
            label: Some("prism_volumetric_barrier_contact_force_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_barrier_contact_force_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_barrier_contact_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per contact query, flattened to a 1-D dispatch.
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
