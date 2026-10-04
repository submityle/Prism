//! `wgpu` compute twin of the Hertz elastic normal-force closed form, from the
//! `CPU` golden `prism_physics_core::collider::hertz_contact`'s private
//! `hertz_elastic_force`.
//!
//! Two elastic grains pressing into each other by a penetration `overlap`
//! develop a Hertzian normal force that grows as the `3/2` power of the
//! overlap. This module ports that scalar force magnitude onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same force the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `hertz_elastic_force` for one grain
//! pair with an effective modulus `E*` (`effective_modulus`), a reduced contact
//! radius `R*` (`effective_radius`) and a penetration `overlap`:
//!
//! * The elastic force is `F = (4 / 3) * E* * sqrt(R*) * overlap^(3/2)`.
//!
//! The validity gate follows the golden `evaluate_hertz_contact` degeneracy
//! rule: the result is valid only when `effective_modulus`, `effective_radius`
//! and `overlap` are all finite, `effective_radius > 0`, and `overlap > 0`. An
//! invalid query reports `valid = 0`, `force = 0`.
//!
//! # Correctness model
//!
//! The golden evaluates the square root and the `3/2` power in `f64`
//! (`r_eff.sqrt()`, `delta.powf(1.5)`) while the kernel uses `f32` `sqrt` and
//! `pow`, so `CPU` and `GPU` are not bit-exact; the valid `force` scalar is
//! compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity sweep keeps inputs well inside the valid region so the `f64`/`f32`
//! numerical gap cannot flip the validity decision.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a non-positive `effective_radius`, or a non-positive
//! `overlap` yields `valid = 0` with `force = 0`. The `sqrt` base and the `pow`
//! base are each fed through a `select` guard so an un-taken branch never
//! evaluates a square root or power of a non-positive argument. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset plus the `sqrt` and `pow`
//! builtins the closed form requires; it avoids `sin`, `cos`, `tan`, `exp`,
//! `log`, `round`, `f32` remainder and bare `f32` equality. Finiteness is
//! tested with the ordered compare `abs(x) < 3.0e38` (which rejects both
//! infinities and `NaN`) rather than a bare `x == x`, and validity with ordered
//! `> 0`; there is no `f32` equality anywhere, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。
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

/// The portable Hertz elastic-force kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `hertz_elastic_force`; see the module documentation for the closed
/// form.
const HERTZ_ELASTIC_FORCE_WGSL: &str = r#"
// Hertz elastic-force twin: one thread per query reproduces hertz_elastic_force.
// It uses the portable core-WGSL subset plus the sqrt and pow builtins the
// closed form requires. There is no loop and no branch, so it provably
// terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and validity ordered > 0 compares, all fed to select;
// there is no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Effective (reduced) elastic modulus E*.
    effective_modulus: f32,
    // Effective (reduced) contact radius R*.
    effective_radius: f32,
    // Penetration overlap delta.
    overlap: f32,
    // Padding word to a 16-byte stride.
    pad0: f32,
}

struct Result {
    // Hertz elastic normal-force magnitude when valid, else 0.
    force: f32,
    // 1 when all inputs finite, R* > 0 and overlap > 0, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const FOUR_THIRDS: f32 = 1.3333334;
const THREE_HALVES: f32 = 1.5;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let em = q.effective_modulus;
    let er = q.effective_radius;
    let overlap = q.overlap;

    // Validity gate matching the golden degeneracy rule: all inputs finite,
    // R* strictly positive and overlap strictly positive.
    let em_finite = abs(em) < FINITE_LIMIT;
    let er_finite = abs(er) < FINITE_LIMIT;
    let overlap_finite = abs(overlap) < FINITE_LIMIT;
    let er_ok = er > 0.0;
    let ov_ok = overlap > 0.0;
    let valid = em_finite && er_finite && overlap_finite && er_ok && ov_ok;

    // Guard the sqrt and pow bases so an un-taken (invalid) branch never
    // evaluates a root or power of a non-positive argument.
    let r_base = select(1.0, er, er_ok);
    let d_base = select(1.0, overlap, ov_ok);
    let computed = FOUR_THIRDS * em * sqrt(r_base) * pow(d_base, THREE_HALVES);

    var out: Result;
    out.force = select(0.0, computed, valid);
    out.valid = select(0u, 1u, valid);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the effective modulus, effective radius and overlap, padded to `4` `f32`
/// words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    effective_modulus: f32,
    effective_radius: f32,
    overlap: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the force magnitude and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    force: f32,
    valid: u32,
}

/// One Hertz elastic-force query: the effective modulus `E*`, the effective
/// contact radius `R*` and the penetration `overlap`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzElasticForceQuery {
    /// Effective (reduced) elastic modulus `E*`.
    pub effective_modulus: f32,
    /// Effective (reduced) contact radius `R*`.
    pub effective_radius: f32,
    /// Penetration overlap `delta`.
    pub overlap: f32,
}

impl HertzElasticForceQuery {
    /// Builds a query from the effective modulus, effective radius and overlap.
    #[must_use]
    pub fn new(
        effective_modulus: f32,
        effective_radius: f32,
        overlap: f32,
    ) -> HertzElasticForceQuery {
        HertzElasticForceQuery {
            effective_modulus,
            effective_radius,
            overlap,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `hertz_elastic_force` output for that configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HertzElasticForceResult {
    /// The Hertz elastic normal-force magnitude when valid, else `0`.
    pub force: f32,
    /// `1` when the query is valid (all inputs finite, `R* > 0`, `overlap > 0`),
    /// else `0`.
    pub valid: u32,
}

/// Encodes one [`HertzElasticForceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HertzElasticForceQuery) -> GpuQuery {
    GpuQuery {
        effective_modulus: q.effective_modulus,
        effective_radius: q.effective_radius,
        overlap: q.overlap,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HertzElasticForceResult`].
fn decode_result(raw: &GpuResult) -> HertzElasticForceResult {
    HertzElasticForceResult {
        force: raw.force,
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

/// A compiled, reusable Hertz elastic-force compute pipeline, twinning the
/// `CPU` golden `hertz_elastic_force`.
pub struct GpuHertzElasticForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHertzElasticForce {
    /// Compiles the Hertz elastic-force kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the `sqrt` and
    /// `pow` builtins the closed form requires, so no optional device feature is
    /// required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHertzElasticForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force"),
            source: ShaderSource::Wgsl(HERTZ_ELASTIC_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHertzElasticForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HertzElasticForceResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `force` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HertzElasticForceQuery],
    ) -> Vec<HertzElasticForceResult> {
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
            label: Some("prism_volumetric_hertz_elastic_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_bind_group"),
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
            label: Some("prism_volumetric_hertz_elastic_force_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hertz_elastic_force_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hertz_elastic_force_pass"),
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
