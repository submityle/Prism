//! `wgpu` compute twin of the `SPH` density estimate inside the Position-Based
//! Fluids solve ([`pbf`](prism_render_architecture::water::pbf)).
//!
//! The `CPU` golden [`pbf`](prism_render_architecture::water::pbf) binds an
//! incompressible liquid with one density constraint per particle. The density
//! itself is the `Poly6`-weighted neighbour sum
//! [`estimate_density`](prism_render_architecture::water::pbf::estimate_density)
//! `rho_i = m * sum_j W(r_ij, h)`, where `W` is the `Poly6` kernel
//! [`poly6`](prism_render_architecture::water::pbf::poly6). Both are pure
//! multiply/add polynomials with a single guarded divide, no `f32` equality
//! test and no transcendental.
//!
//! [`GpuWaterPbfDensity`] is the on-device twin of exactly that bounded density
//! sum. One thread estimates the density of one particle: it evaluates the
//! `Poly6` weight for each of that particle's squared neighbour distances,
//! sums them in the same order as the reference and scales by the clamped mass,
//! reproducing the reference's exact closed form. A passing real-device parity
//! test is direct evidence the ported kernel sums the same density the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one particle the kernel reproduces, in closed form:
//!
//! - [`poly6`](prism_render_architecture::water::pbf::poly6):
//!   `315 / (64 * pi * h^9) * (h^2 - r^2)^3` for `0 <= r^2 < h^2`, and `0`
//!   beyond the support radius or for a non-positive `h`. The squared distance
//!   is clamped up to `0` first, matching the reference `r_squared.max(0)`. The
//!   `pi` constant is the single-precision `3.1415927`, bit-identical to the
//!   golden `core::f32::consts::PI`.
//! - [`estimate_density`](prism_render_architecture::water::pbf::estimate_density):
//!   `mass.max(0) * sum_j poly6(r2_j, h)`, summed sequentially over the
//!   neighbour squared distances in input order, matching the reference's
//!   left-to-right accumulation.
//!
//! The emitted result carries the scalar `density`.
//!
//! # What stays on the host
//!
//! The neighbour-finding spatial hash, the density constraint and its scaling
//! factor, the artificial-pressure correction and the whole iterative
//! projection are variable-length aggregate and stateful work the host owns;
//! the device never sees them. The host finds each particle's neighbours and
//! down-feeds their squared distances (capped at [`MAX_NEIGHBORS`]) so the
//! device sees one bounded, fixed-stride particle record per thread. An empty
//! batch short-circuits with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! The density threads through multiplies, subtracts, a guarded divide and a
//! running sum, so the two engines are not bit-exact: a `GPU` divide may land a
//! few units in the last place from the scalar reference and the running sum
//! may reorder rounding. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, relative floor `1e-6`) on the
//! density, tight enough to catch a genuinely wrong port (a dropped clamp, a
//! wrong exponent on `h`, a wrong `pi`) yet loose enough to admit a legal
//! last-place difference. The support breakpoint `r^2` crossing `h^2` is a
//! discontinuity; fixtures and the randomized sweep keep every neighbour well
//! clear of it so `CPU` and `GPU` cannot straddle it, and the smoothing radius
//! `h` is kept far above the `1e-6` degeneracy floor except in the dedicated
//! degenerate fixture.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `+ - * /`,
//! unsigned index arithmetic and a bounded loop — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`
//! and no `sqrt`. No optional device feature is required, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The loop is bounded by the per-particle
//! neighbour count (at most [`MAX_NEIGHBORS`]), so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。
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

/// Upper bound on the number of squared neighbour distances the host may
/// down-feed per particle. The device record carries a fixed `MAX_NEIGHBORS`
/// slot array; the per-particle `count` selects how many are summed. The host
/// finds neighbours (a variable-length spatial-hash query) and caps the batch
/// at this bound before dispatch.
pub const MAX_NEIGHBORS: usize = 64;

/// The portable core-`WGSL` `SPH` density-estimate twin, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`poly6`](prism_render_architecture::water::pbf::poly6) and
/// [`estimate_density`](prism_render_architecture::water::pbf::estimate_density)
/// closed forms; see the module documentation for the algorithm.
const WATER_PBF_DENSITY_WGSL: &str = r#"
// SPH density-estimate twin: one thread estimates the density of one particle
// by summing the Poly6 kernel over its squared neighbour distances and scaling
// by the clamped mass. Mirrors the CPU golden `water::pbf` closed forms with
// only max and + - * / over a bounded loop. It owns no neighbour search, no
// density constraint and no iterative projection; those stay host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；
// 无第三方引擎源码或衍生代码。

const MAX_NEIGHBORS: u32 = 64u;
// Degeneracy floor on the smoothing radius, matching the golden EPS = 1e-6.
const EPS: f32 = 0.000001;
// Single-precision pi, bit-identical to core::f32::consts::PI.
const PI: f32 = 3.1415927;

struct Params {
    // Number of particles in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle mass (clamped up to 0 before scaling the sum).
    mass: f32,
    // Smoothing radius.
    h: f32,
    // Number of valid neighbour entries in r2 (at most MAX_NEIGHBORS).
    count: u32,
    pad0: u32,
    // Squared neighbour distances; only the first `count` are summed.
    r2: array<f32, 64>,
}

struct Result {
    // Estimated density rho = mass.max(0) * sum_j poly6(r2_j, h).
    density: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Poly6 smoothing kernel from a squared distance: 315/(64*pi*h^9)*(h^2-r^2)^3
// on the support, else 0. Mirrors the golden `poly6` exactly, including the
// h^9 = h2*h2*h2*h2*h multiply order and the r_squared.max(0) clamp.
fn poly6(r_squared: f32, h: f32) -> f32 {
    if (h <= EPS) {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = max(r_squared, 0.0);
    if (r2 >= h2) {
        return 0.0;
    }
    let h9 = h2 * h2 * h2 * h2 * h;
    let coeff = 315.0 / (64.0 * PI * h9);
    let d = h2 - r2;
    return coeff * d * d * d;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Clamp the neighbour count to the fixed slot bound, then sum the Poly6
    // weight over the valid neighbours in input order, matching the reference's
    // left-to-right accumulation.
    var n = q.count;
    if (n > MAX_NEIGHBORS) {
        n = MAX_NEIGHBORS;
    }
    var sum = 0.0;
    var i = 0u;
    loop {
        if (i >= n) {
            break;
        }
        sum = sum + poly6(q.r2[i], q.h);
        i = i + 1u;
    }

    let m = max(q.mass, 0.0);

    var out: Result;
    out.density = m * sum;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the particle count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_PBF_DENSITY_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid particles in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one particle query, matching the `WGSL` `Query`
/// struct: the mass, the smoothing radius, the neighbour count, one pad word,
/// then the fixed `MAX_NEIGHBORS` squared-distance slots, for a `272`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle mass.
    mass: f32,
    /// Smoothing radius.
    h: f32,
    /// Number of valid neighbour entries in `r2`.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Squared neighbour distances; only the first `count` are summed.
    r2: [f32; MAX_NEIGHBORS],
}

/// `repr(C)` `std430` layout of one particle result, matching the `WGSL`
/// `Result` struct: the estimated density plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Estimated density.
    density: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One per-particle query for the `SPH` density-estimate twin: the mass, the
/// smoothing radius and the squared neighbour distances.
///
/// [`estimate_density`](prism_render_architecture::water::pbf::estimate_density)
/// reads [`mass`](Self::mass), [`neighbor_r_squared`](Self::neighbor_r_squared)
/// and [`h`](Self::h). At most [`MAX_NEIGHBORS`] neighbours are summed on the
/// device; any beyond that are ignored, matching the host-side cap.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterPbfDensityQuery {
    /// Particle mass; clamped up to `0` before scaling the kernel sum.
    pub mass: f32,
    /// Smoothing radius `h`.
    pub h: f32,
    /// Squared neighbour distances summed through the `Poly6` kernel.
    pub neighbor_r_squared: Vec<f32>,
}

impl WaterPbfDensityQuery {
    /// Builds a query from the mass, smoothing radius and squared neighbour
    /// distances.
    #[must_use]
    pub fn new(mass: f32, h: f32, neighbor_r_squared: Vec<f32>) -> WaterPbfDensityQuery {
        WaterPbfDensityQuery {
            mass,
            h,
            neighbor_r_squared,
        }
    }
}

/// One estimated density of the `SPH` density-estimate twin, mirroring the
/// golden
/// [`estimate_density`](prism_render_architecture::water::pbf::estimate_density)
/// return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfDensityResult {
    /// Estimated density `rho = mass.max(0) * sum_j poly6(r2_j, h)`.
    pub density: f32,
}

/// Encodes one [`WaterPbfDensityQuery`] into its `std430` [`GpuQuery`] slot. The
/// neighbour count is capped at [`MAX_NEIGHBORS`]; unused slots are zeroed.
fn encode_query(q: &WaterPbfDensityQuery) -> GpuQuery {
    let mut r2 = [0.0_f32; MAX_NEIGHBORS];
    let count = q.neighbor_r_squared.len().min(MAX_NEIGHBORS);
    r2[..count].copy_from_slice(&q.neighbor_r_squared[..count]);
    GpuQuery {
        mass: q.mass,
        h: q.h,
        count: count as u32,
        pad0: 0,
        r2,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterPbfDensityResult`].
fn decode_result(raw: &GpuResult) -> WaterPbfDensityResult {
    WaterPbfDensityResult {
        density: raw.density,
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

/// A compiled, reusable `SPH` density-estimate compute pipeline, twinning the
/// numeric core of the `CPU` golden
/// [`pbf`](prism_render_architecture::water::pbf).
pub struct GpuWaterPbfDensity {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfDensity {
    /// Compiles the density-estimate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfDensity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_density"),
            source: ShaderSource::Wgsl(WATER_PBF_DENSITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_density_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_density_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_density_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfDensity {
            module,
            layout,
            pipeline,
        }
    }

    /// Estimates the density of every particle in `queries` and returns one
    /// [`WaterPbfDensityResult`] per input, in order.
    ///
    /// The density matches the reference within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfDensityQuery],
    ) -> Vec<WaterPbfDensityResult> {
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
            label: Some("prism_volumetric_water_pbf_density_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_density_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_density_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_density_bind_group"),
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
            label: Some("prism_volumetric_water_pbf_density_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_density_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_density_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per particle, flattened to a 1-D dispatch.
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
