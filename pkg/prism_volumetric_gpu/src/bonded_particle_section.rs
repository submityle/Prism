//! `wgpu` compute twin of the bonded-particle section-property closed form,
//! from the `CPU` golden `prism_physics_core::collider::bonded_particle`'s
//! `BondModel::new` disc-section derivation.
//!
//! A parallel bond cements two particles with a disc of radius `R`. Three
//! section properties follow from `R`: the cross-sectional area `A = π·R²`, the
//! second moment of area `I = π·R⁴/4`, and the polar moment `J = π·R⁴/2`. These
//! convert the bond's per-area stiffnesses into axial, bending and torsional
//! terms. This module ports that single stateless derivation onto the device:
//! one thread resolves one query, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same section properties the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the derived triple from one `radius`:
//!
//! * If `radius` is non-finite or `<= 0`, the bond is invalid
//!   (`valid = 0`, `area = inertia = polar = 0`).
//! * Otherwise, in the golden left-associative operator order with
//!   `r2 = radius·radius`: `area = π·r2`, `inertia = 0.25·π·r2·r2`, and
//!   `polar = 0.5·π·r2·r2`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through multiplies that a `GPU` may
//! contract, so `CPU` and `GPU` are not necessarily bit-exact; each valid scalar
//! is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity test keeps random radii strictly positive and finite so the validity
//! decision cannot be flipped by round-off. The kernel's `π` is the decimal
//! literal `3.1415927`, bit-identical to `std::f32::consts::PI`, which the host
//! oracle also uses so no relative drift creeps in from a differing constant.
//!
//! # Degenerate inputs
//!
//! A non-finite radius or a radius `<= 0` yields `valid = 0` with all three
//! outputs `0`. The kernel feeds the squared radius through a `select` guard so
//! the un-taken (invalid) branch never squares an out-of-range magnitude into
//! an overflow, and each output is `select`-gated to `0` when invalid. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - *`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and validity with ordered `> 0`; there is no
//! `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bonded_particle`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` section-property kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `BondModel::new` disc-section derivation; see the module
/// documentation for the closed form.
const BONDED_PARTICLE_SECTION_WGSL: &str = r#"
// Section-property twin: one thread per query reproduces the disc-section
// derivation of BondModel::new. It uses only the portable core-WGSL subset
// (abs, + - *, select plus unsigned index math), takes no optional feature, and
// has no loop and no branch, so it provably terminates. Finiteness is an
// ordered abs < 3.0e38 compare (rejecting infinities and NaN) and validity an
// ordered > 0 compare, both fed to select. There is no f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Cement disc radius R.
    radius: f32,
    // Padding words to a 16-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Cross-sectional area pi*R^2 when valid, else 0.
    area: f32,
    // Second moment of area 0.25*pi*R^4 when valid, else 0.
    inertia: f32,
    // Polar moment of area 0.5*pi*R^4 when valid, else 0.
    polar: f32,
    // 1 when radius is finite and strictly positive, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
// Bit-identical to std::f32::consts::PI, shared with the host oracle.
const PI: f32 = 3.1415927;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let radius = q.radius;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false), plus strict positivity. No bare f32
    // equality anywhere.
    let finite = abs(radius) < FINITE_LIMIT;
    let positive = radius > 0.0;
    let ok = finite && positive;

    // Guard the squared radius so the un-taken (invalid) branch never squares an
    // out-of-range magnitude into an overflow.
    let r = select(1.0, radius, ok);
    let r2 = r * r;
    // Golden left-associative operator order.
    let area = PI * r2;
    let inertia = 0.25 * PI * r2 * r2;
    let polar = 0.5 * PI * r2 * r2;

    var out: Result;
    out.area = select(0.0, area, ok);
    out.inertia = select(0.0, inertia, ok);
    out.polar = select(0.0, polar, ok);
    out.valid = select(0u, 1u, ok);
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// The radius is padded to `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    radius: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the three section properties and the validity flag — `4` words
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    area: f32,
    inertia: f32,
    polar: f32,
    valid: u32,
}

/// One section-property query: the cement disc radius `R`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondedParticleSectionQuery {
    /// Cement disc radius `R`.
    pub radius: f32,
}

impl BondedParticleSectionQuery {
    /// Builds a query from the cement disc radius `R`.
    #[must_use]
    pub fn new(radius: f32) -> BondedParticleSectionQuery {
        BondedParticleSectionQuery { radius }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `BondModel::new` disc-section outputs for that radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondedParticleSectionResult {
    /// Cross-sectional area `A = π·R²` when valid, else `0`.
    pub area: f32,
    /// Second moment of area `I = π·R⁴/4` when valid, else `0`.
    pub inertia: f32,
    /// Polar moment of area `J = π·R⁴/2` when valid, else `0`.
    pub polar: f32,
    /// `1` when the radius is finite and strictly positive, else `0`.
    pub valid: u32,
}

/// Encodes one [`BondedParticleSectionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &BondedParticleSectionQuery) -> GpuQuery {
    GpuQuery {
        radius: q.radius,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BondedParticleSectionResult`].
fn decode_result(raw: &GpuResult) -> BondedParticleSectionResult {
    BondedParticleSectionResult {
        area: raw.area,
        inertia: raw.inertia,
        polar: raw.polar,
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

/// A compiled, reusable section-property compute pipeline, twinning the `CPU`
/// golden `BondModel::new` disc-section derivation.
pub struct GpuBondedParticleSection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBondedParticleSection {
    /// Compiles the section-property kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBondedParticleSection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bonded_particle_section"),
            source: ShaderSource::Wgsl(BONDED_PARTICLE_SECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBondedParticleSection {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BondedParticleSectionResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the three section
    /// scalars to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BondedParticleSectionQuery],
    ) -> Vec<BondedParticleSectionResult> {
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
            label: Some("prism_volumetric_bonded_particle_section_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_bind_group"),
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
            label: Some("prism_volumetric_bonded_particle_section_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bonded_particle_section_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bonded_particle_section_pass"),
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
