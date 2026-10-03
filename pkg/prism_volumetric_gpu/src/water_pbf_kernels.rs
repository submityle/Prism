//! `wgpu` compute twin of the two `SPH` smoothing-kernel primitives inside the
//! Position-Based Fluids density solve
//! ([`pbf`](prism_render_architecture::water::pbf)).
//!
//! The `CPU` golden [`pbf`](prism_render_architecture::water::pbf) binds an
//! incompressible liquid with one density constraint per particle. Two pure,
//! classical polynomials feed that solve: the `Poly6` density kernel
//! [`poly6`](prism_render_architecture::water::pbf::poly6) and the `Spiky`
//! gradient kernel
//! [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient).
//! Both are evaluated with multiply/add and a single `sqrt`, with no `f32`
//! equality test and no transcendental.
//!
//! [`GpuWaterPbfKernels`] is the on-device twin of exactly those two numeric
//! cores. One thread evaluates one neighbour sample — the `Poly6` weight from a
//! squared distance and the `Spiky` gradient vector from a relative-position
//! vector, both at the same smoothing radius `h` — reproducing the reference's
//! exact closed form, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same weights and gradients the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one neighbour sample the kernel reproduces, in closed form:
//!
//! - [`poly6`](prism_render_architecture::water::pbf::poly6):
//!   `315 / (64 * pi * h^9) * (h^2 - r^2)^3` for `0 <= r^2 < h^2`, and `0`
//!   beyond the support radius or for a non-positive `h`. The squared distance
//!   is clamped up to `0` first, matching the reference `r_squared.max(0)`.
//! - [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient):
//!   `-45 / (pi * h^6) * (h - r)^2 * (r_vec / r)`, vanishing to the zero vector
//!   at or beyond the support radius, for a coincident pair
//!   (`r^2 <= EPS_LEN_SQ`, where the direction is undefined), and for a
//!   non-positive `h`.
//!
//! The emitted result carries the scalar `poly6` weight plus the three
//! components `spiky_x`, `spiky_y`, `spiky_z` of the gradient vector.
//!
//! # What stays on the host
//!
//! The neighbour-finding spatial hash, the `SPH` density sum
//! ([`estimate_density`](prism_render_architecture::water::pbf::estimate_density)),
//! the density constraint and its scaling factor, the artificial-pressure
//! correction and the whole iterative projection are variable-length aggregate
//! and stateful work the host owns; the device never sees them. The host
//! enqueues one neighbour sample per thread; an empty batch short-circuits with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Both kernels thread through only multiplies, subtracts, a guarded divide and
//! one `sqrt`, so the two engines are not bit-exact: a `GPU` `sqrt` or divide
//! may land a few units in the last place from the scalar reference. The parity
//! test asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, relative
//! floor `1e-6`) on every continuous field, tight enough to catch a genuinely
//! wrong port (a dropped clamp, a wrong exponent on `h`, a swapped sign) yet
//! loose enough to admit a legal last-place difference. The two support
//! breakpoints — `r^2` crossing `h^2`, and `r^2` crossing the coincident-pair
//! floor `EPS_LEN_SQ` — are discontinuities; fixtures and the randomized sweep
//! keep every sample well clear of both so `CPU` and `GPU` cannot straddle a
//! breakpoint.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`
//! and no `ceil`. No optional device feature is required, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
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

/// The portable core-`WGSL` `SPH` smoothing-kernel twin, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden [`poly6`](prism_render_architecture::water::pbf::poly6) and
/// [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient)
/// closed forms; see the module documentation for the algorithm.
const WATER_PBF_KERNELS_WGSL: &str = r#"
// SPH smoothing-kernel twin: one thread evaluates one neighbour sample's Poly6
// density weight (from a squared distance) and Spiky gradient vector (from a
// relative-position vector), both at the same smoothing radius h, mirroring the
// CPU golden `water::pbf` closed forms with only max, sqrt and + - * /. It owns
// no neighbour search, no density sum and no constraint projection; those
// variable-length and stateful parts stay host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；无第三方引擎
// 源码或衍生代码。

// Shared magnitude guards; mirror the water module `EPS` / `EPS_LEN_SQ`.
const EPS: f32 = 1.0e-6;
const EPS_LEN_SQ: f32 = 1.0e-12;
// Matches the golden `PI` (core::f32::consts::PI) to full f32 precision.
const PI: f32 = 3.14159265358979;

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Squared neighbour distance feeding the Poly6 kernel (clamped up to 0).
    r_squared: f32,
    // Smoothing radius h; the support radius of both kernels.
    h: f32,
    // Relative-position vector components feeding the Spiky gradient kernel.
    rx: f32,
    ry: f32,
    rz: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Poly6 density weight W(r, h).
    poly6: f32,
    // Spiky gradient vector components grad W(r_vec, h).
    spiky_x: f32,
    spiky_y: f32,
    spiky_z: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Poly6 density kernel: 315 / (64 * pi * h^9) * (h^2 - r^2)^3 on the support,
// 0 beyond it or for a non-positive h. Mirrors the golden `poly6`.
fn poly6_kernel(r_squared: f32, h: f32) -> f32 {
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
    let h = q.h;

    // Poly6 density weight from the squared distance and the radius.
    let poly = poly6_kernel(q.r_squared, h);

    // Spiky gradient: zero vector for a non-positive radius, a coincident pair
    // (direction undefined) or a sample at/beyond the support radius; otherwise
    // -45 / (pi * h^6) * (h - r)^2 * (r_vec / r).
    var sx: f32 = 0.0;
    var sy: f32 = 0.0;
    var sz: f32 = 0.0;
    if (h > EPS) {
        let r2 = q.rx * q.rx + q.ry * q.ry + q.rz * q.rz;
        if (r2 > EPS_LEN_SQ && r2 < h * h) {
            let r = sqrt(r2);
            let h6 = h * h * h * h * h * h;
            let coeff = -45.0 / (PI * h6);
            let scale = coeff * (h - r) * (h - r) / r;
            sx = q.rx * scale;
            sy = q.ry * scale;
            sz = q.rz * scale;
        }
    }

    var out: Result;
    out.poly6 = poly;
    out.spiky_x = sx;
    out.spiky_y = sy;
    out.spiky_z = sz;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_PBF_KERNELS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one sample query, matching the `WGSL` `Query`
/// struct: the squared distance and radius feeding `poly6`, the three
/// relative-position components feeding `spiky_gradient`, and three pad words to
/// a `32`-byte stride of eight `f32` words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Squared neighbour distance for `poly6`.
    r_squared: f32,
    /// Smoothing radius `h`.
    h: f32,
    /// Relative-position `x` for `spiky_gradient`.
    rx: f32,
    /// Relative-position `y` for `spiky_gradient`.
    ry: f32,
    /// Relative-position `z` for `spiky_gradient`.
    rz: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one sample result, matching the `WGSL` `Result`
/// struct: the scalar `poly6` weight and the three `spiky_gradient` components,
/// a `16`-byte stride of four `f32` words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `Poly6` density weight.
    poly6: f32,
    /// `Spiky` gradient `x` component.
    spiky_x: f32,
    /// `Spiky` gradient `y` component.
    spiky_y: f32,
    /// `Spiky` gradient `z` component.
    spiky_z: f32,
}

/// One per-sample query for the `SPH` smoothing-kernel twin: the squared
/// distance and radius `poly6` reads, plus the relative-position vector and the
/// same radius `spiky_gradient` reads.
///
/// `poly6` uses [`r_squared`](Self::r_squared) and [`h`](Self::h); the `Spiky`
/// gradient uses [`rx`](Self::rx)/[`ry`](Self::ry)/[`rz`](Self::rz) with the
/// same [`h`](Self::h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfKernelsQuery {
    /// Squared neighbour distance for the golden `poly6`.
    pub r_squared: f32,
    /// Smoothing radius `h` shared by both kernels.
    pub h: f32,
    /// Relative-position `x` for the golden `spiky_gradient`.
    pub rx: f32,
    /// Relative-position `y` for the golden `spiky_gradient`.
    pub ry: f32,
    /// Relative-position `z` for the golden `spiky_gradient`.
    pub rz: f32,
}

impl WaterPbfKernelsQuery {
    /// Builds a query from the `poly6` squared distance, the shared radius `h`,
    /// and the `spiky_gradient` relative-position components.
    #[must_use]
    pub const fn new(r_squared: f32, h: f32, rx: f32, ry: f32, rz: f32) -> WaterPbfKernelsQuery {
        WaterPbfKernelsQuery {
            r_squared,
            h,
            rx,
            ry,
            rz,
        }
    }
}

/// One resolved sample of the `SPH` smoothing-kernel twin, mirroring the golden
/// [`poly6`](prism_render_architecture::water::pbf::poly6) scalar weight and the
/// [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient)
/// vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfKernelsResult {
    /// `Poly6` density weight `W(r, h)`.
    pub poly6: f32,
    /// `Spiky` gradient `x` component.
    pub spiky_x: f32,
    /// `Spiky` gradient `y` component.
    pub spiky_y: f32,
    /// `Spiky` gradient `z` component.
    pub spiky_z: f32,
}

/// Encodes one [`WaterPbfKernelsQuery`] into its `std430` [`GpuQuery`] slot. The
/// real fields are a direct copy; the pad words are zeroed.
fn encode_query(q: &WaterPbfKernelsQuery) -> GpuQuery {
    GpuQuery {
        r_squared: q.r_squared,
        h: q.h,
        rx: q.rx,
        ry: q.ry,
        rz: q.rz,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterPbfKernelsResult`].
fn decode_result(raw: &GpuResult) -> WaterPbfKernelsResult {
    WaterPbfKernelsResult {
        poly6: raw.poly6,
        spiky_x: raw.spiky_x,
        spiky_y: raw.spiky_y,
        spiky_z: raw.spiky_z,
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

/// A compiled, reusable `SPH` smoothing-kernel compute pipeline, twinning the
/// numeric core of the `CPU` golden
/// [`pbf`](prism_render_architecture::water::pbf).
pub struct GpuWaterPbfKernels {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfKernels {
    /// Compiles the `SPH` smoothing-kernel kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfKernels {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels"),
            source: ShaderSource::Wgsl(WATER_PBF_KERNELS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfKernels {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every sample in `queries` and returns one
    /// [`WaterPbfKernelsResult`] per input, in order.
    ///
    /// Every output matches the reference within the tolerance documented on
    /// this module for both the `poly6` weight and the three `spiky_gradient`
    /// components. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfKernelsQuery],
    ) -> Vec<WaterPbfKernelsResult> {
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
            label: Some("prism_volumetric_water_pbf_kernels_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_bind_group"),
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
            label: Some("prism_volumetric_water_pbf_kernels_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_kernels_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_kernels_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
