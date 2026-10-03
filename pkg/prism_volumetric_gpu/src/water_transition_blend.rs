//! `wgpu` compute twin of the solver transition blend
//! ([`transition`](prism_render_architecture::water::transition)).
//!
//! A single scene can run several water solvers at once — particle `FLIP`/`PBF`
//! up close, shallow-water (`SWE`) at mid range, and spectral ocean waves in the
//! distance. Where two regions meet, their surfaces must be cross-faded so no
//! seam appears. The `CPU` golden derives the three solver weights from the view
//! distance and two crossfade bands as a partition of unity, then blends three
//! solver normals by those weights and renormalizes. Both steps are built from
//! clamps, a guarded divide, signed comparisons and a single `sqrt`, so they
//! port to the device directly.
//!
//! [`GpuWaterTransitionBlend`] is the on-device twin: one thread resolves one
//! query, reproducing
//! [`solver_blend_weights`](prism_render_architecture::water::transition::solver_blend_weights)
//! and
//! [`blend_normal`](prism_render_architecture::water::transition::blend_normal). A
//! passing real-device parity test is direct evidence the ported kernel
//! reproduces the reference arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying a view `distance`, the two band midpoints
//! `particle_to_swe` and `swe_to_spectral`, the shared `half_width`, and the
//! three solver normals, the kernel reproduces:
//! - the near-band `to_swe` and far-band `to_spectral` linear ramps, each a
//!   `smooth_ramp` clamped to `0..=1` with a degenerate-band hard step when the
//!   band width is at or below `EPS`;
//! - the three weights `particle = 1 - to_swe`, `shallow_water = to_swe * (1 -
//!   to_spectral)` and `spectral = to_swe * to_spectral`, which sum to one; and
//! - the weighted normal sum passed through `normalize_or_zero`, which returns
//!   the zero vector when the composed length squared is at or below
//!   `EPS_LEN_SQ` (`1e-12`).
//!
//! # What stays on the host
//!
//! Nothing of the twinned pair stays on the host; the band tuning rides in each
//! query so the batch is self-contained. An empty batch short-circuits on the
//! host, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs clamps, a guarded divide by a non-degenerate band width,
//! signed comparisons and one `sqrt`, so a correct port reproduces the reference
//! to within floating-point rounding. The parity test asserts the shared
//! continuous tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! weight and normal component, checks the weights sum to one, and pins the
//! zero-length fallback exactly. Fixtures keep the band widths well above `EPS`
//! so the ramp denominators stay non-degenerate.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`, `+`,
//! `-`, `*`, `/`, a signed comparison and `sqrt` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, and no `64`-bit integers or
//! floats. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::transition`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` transition-blend kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `solver_blend_weights` and `blend_normal`; see the module
/// documentation for the algorithm.
const WATER_TRANSITION_BLEND_WGSL: &str = r#"
// Transition-blend twin: one thread resolves one query's three solver weights
// and the renormalized blended normal, mirroring the CPU golden
// `water::transition::{solver_blend_weights, blend_normal}` with only clamps, a
// guarded divide, signed comparisons and a single sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::transition；无第三方引擎
// 源码或衍生代码。

// Shared `water` module epsilons: the ramp-band guard EPS and the squared-length
// guard EPS_LEN_SQ, copied from the golden `water::{EPS, EPS_LEN_SQ}`.
const EPS: f32 = 1e-6;
const EPS_LEN_SQ: f32 = 1e-12;

// Linear ramp from 0 to 1 as x crosses [lo, hi], clamped at the ends. A
// degenerate band (hi - lo <= EPS) collapses to a hard step at lo.
fn smooth_ramp(x: f32, lo: f32, hi: f32) -> f32 {
    if (hi - lo <= EPS) {
        if (x < lo) {
            return 0.0;
        }
        return 1.0;
    }
    return clamp((x - lo) / (hi - lo), 0.0, 1.0);
}

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // View distance (clamped to >= 0 before use).
    distance: f32,
    // Midpoint distance of the particle-to-shallow-water crossfade.
    particle_to_swe: f32,
    // Midpoint distance of the shallow-water-to-spectral crossfade.
    swe_to_spectral: f32,
    // Half-width of each crossfade band (clamped to >= 0 before use).
    half_width: f32,
    // Particle-solver normal.
    np_x: f32,
    np_y: f32,
    np_z: f32,
    pad0: f32,
    // Shallow-water-solver normal.
    nsw_x: f32,
    nsw_y: f32,
    nsw_z: f32,
    pad1: f32,
    // Spectral-solver normal.
    nsp_x: f32,
    nsp_y: f32,
    nsp_z: f32,
    pad2: f32,
}

struct Result {
    // Particle solver weight.
    w_particle: f32,
    // Shallow-water solver weight.
    w_shallow_water: f32,
    // Spectral solver weight.
    w_spectral: f32,
    pad0: f32,
    // Renormalized blended normal.
    n_x: f32,
    n_y: f32,
    n_z: f32,
    pad1: f32,
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

    // Solver weights: near band hands particle to shallow-water, far band hands
    // shallow-water to spectral; the three always sum to one.
    let d = max(q.distance, 0.0);
    let hw = max(q.half_width, 0.0);
    let to_swe = smooth_ramp(d, q.particle_to_swe - hw, q.particle_to_swe + hw);
    let to_spectral = smooth_ramp(d, q.swe_to_spectral - hw, q.swe_to_spectral + hw);
    let w_particle = 1.0 - to_swe;
    let w_shallow_water = to_swe * (1.0 - to_spectral);
    let w_spectral = to_swe * to_spectral;

    // Weighted normal sum, then normalize-or-zero against EPS_LEN_SQ.
    let sx = q.np_x * w_particle + q.nsw_x * w_shallow_water + q.nsp_x * w_spectral;
    let sy = q.np_y * w_particle + q.nsw_y * w_shallow_water + q.nsp_y * w_spectral;
    let sz = q.np_z * w_particle + q.nsw_z * w_shallow_water + q.nsp_z * w_spectral;
    let lsq = sx * sx + sy * sy + sz * sz;
    var nx: f32 = 0.0;
    var ny: f32 = 0.0;
    var nz: f32 = 0.0;
    if (lsq > EPS_LEN_SQ) {
        let inv = 1.0 / sqrt(lsq);
        nx = sx * inv;
        ny = sy * inv;
        nz = sz * inv;
    }

    var out: Result;
    out.w_particle = w_particle;
    out.w_shallow_water = w_shallow_water;
    out.w_spectral = w_spectral;
    out.pad0 = 0.0;
    out.n_x = nx;
    out.n_y = ny;
    out.n_z = nz;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words,
/// filling a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_TRANSITION_BLEND_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the distance, two band midpoints, the
/// half-width and the three solver normals (each padded to a `vec4` lane), a
/// `64`-byte stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// View distance.
    distance: f32,
    /// Midpoint of the particle-to-shallow-water crossfade.
    particle_to_swe: f32,
    /// Midpoint of the shallow-water-to-spectral crossfade.
    swe_to_spectral: f32,
    /// Half-width of each crossfade band.
    half_width: f32,
    /// Particle normal x.
    np_x: f32,
    /// Particle normal y.
    np_y: f32,
    /// Particle normal z.
    np_z: f32,
    /// Padding word.
    pad0: f32,
    /// Shallow-water normal x.
    nsw_x: f32,
    /// Shallow-water normal y.
    nsw_y: f32,
    /// Shallow-water normal z.
    nsw_z: f32,
    /// Padding word.
    pad1: f32,
    /// Spectral normal x.
    nsp_x: f32,
    /// Spectral normal y.
    nsp_y: f32,
    /// Spectral normal z.
    nsp_z: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the three solver weights and the renormalized blended normal (each group
/// padded to a `vec4` lane), a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Particle solver weight.
    w_particle: f32,
    /// Shallow-water solver weight.
    w_shallow_water: f32,
    /// Spectral solver weight.
    w_spectral: f32,
    /// Padding word.
    pad0: f32,
    /// Blended normal x.
    n_x: f32,
    /// Blended normal y.
    n_y: f32,
    /// Blended normal z.
    n_z: f32,
    /// Padding word.
    pad1: f32,
}

/// One transition-blend query for the twin: a view distance, the two band
/// midpoints, the shared half-width, and the three solver normals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterTransitionBlendQuery {
    /// View distance.
    pub distance: f32,
    /// Midpoint of the particle-to-shallow-water crossfade.
    pub particle_to_swe: f32,
    /// Midpoint of the shallow-water-to-spectral crossfade.
    pub swe_to_spectral: f32,
    /// Half-width of each crossfade band.
    pub half_width: f32,
    /// Particle normal x.
    pub np_x: f32,
    /// Particle normal y.
    pub np_y: f32,
    /// Particle normal z.
    pub np_z: f32,
    /// Shallow-water normal x.
    pub nsw_x: f32,
    /// Shallow-water normal y.
    pub nsw_y: f32,
    /// Shallow-water normal z.
    pub nsw_z: f32,
    /// Spectral normal x.
    pub nsp_x: f32,
    /// Spectral normal y.
    pub nsp_y: f32,
    /// Spectral normal z.
    pub nsp_z: f32,
}

impl WaterTransitionBlendQuery {
    /// Builds a query from a distance, the two band midpoints, the half-width,
    /// and the three solver normals.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the golden's distance, three band scalars and three Vec3 normals as flat f32 lanes"
    )]
    pub const fn new(
        distance: f32,
        particle_to_swe: f32,
        swe_to_spectral: f32,
        half_width: f32,
        np_x: f32,
        np_y: f32,
        np_z: f32,
        nsw_x: f32,
        nsw_y: f32,
        nsw_z: f32,
        nsp_x: f32,
        nsp_y: f32,
        nsp_z: f32,
    ) -> WaterTransitionBlendQuery {
        WaterTransitionBlendQuery {
            distance,
            particle_to_swe,
            swe_to_spectral,
            half_width,
            np_x,
            np_y,
            np_z,
            nsw_x,
            nsw_y,
            nsw_z,
            nsp_x,
            nsp_y,
            nsp_z,
        }
    }
}

/// One resolved transition-blend query, mirroring the reference
/// [`solver_blend_weights`](prism_render_architecture::water::transition::solver_blend_weights)
/// and
/// [`blend_normal`](prism_render_architecture::water::transition::blend_normal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterTransitionBlendResult {
    /// Particle solver weight.
    pub w_particle: f32,
    /// Shallow-water solver weight.
    pub w_shallow_water: f32,
    /// Spectral solver weight.
    pub w_spectral: f32,
    /// Blended normal x.
    pub n_x: f32,
    /// Blended normal y.
    pub n_y: f32,
    /// Blended normal z.
    pub n_z: f32,
}

/// Encodes one [`WaterTransitionBlendQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterTransitionBlendQuery) -> GpuQuery {
    GpuQuery {
        distance: q.distance,
        particle_to_swe: q.particle_to_swe,
        swe_to_spectral: q.swe_to_spectral,
        half_width: q.half_width,
        np_x: q.np_x,
        np_y: q.np_y,
        np_z: q.np_z,
        pad0: 0.0,
        nsw_x: q.nsw_x,
        nsw_y: q.nsw_y,
        nsw_z: q.nsw_z,
        pad1: 0.0,
        nsp_x: q.nsp_x,
        nsp_y: q.nsp_y,
        nsp_z: q.nsp_z,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterTransitionBlendResult`].
fn decode_result(raw: &GpuResult) -> WaterTransitionBlendResult {
    WaterTransitionBlendResult {
        w_particle: raw.w_particle,
        w_shallow_water: raw.w_shallow_water,
        w_spectral: raw.w_spectral,
        n_x: raw.n_x,
        n_y: raw.n_y,
        n_z: raw.n_z,
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

/// A compiled, reusable transition-blend compute pipeline, twinning the `CPU`
/// golden solver transition blend from
/// [`transition`](prism_render_architecture::water::transition).
pub struct GpuWaterTransitionBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterTransitionBlend {
    /// Compiles the transition-blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterTransitionBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_transition_blend"),
            source: ShaderSource::Wgsl(WATER_TRANSITION_BLEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_transition_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_transition_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_transition_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterTransitionBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`WaterTransitionBlendResult`] per input, in order.
    ///
    /// Each output matches the reference: the three solver weights and the
    /// renormalized blended normal. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterTransitionBlendQuery],
    ) -> Vec<WaterTransitionBlendResult> {
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
            label: Some("prism_volumetric_water_transition_blend_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_transition_blend_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_transition_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_transition_blend_bind_group"),
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
            label: Some("prism_volumetric_water_transition_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_transition_blend_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_transition_blend_pass"),
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
