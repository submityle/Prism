//! `wgpu` compute twin of the Mohr-Coulomb angle trigonometry, from the `CPU`
//! golden
//! `prism_physics_core::collider::tet_fem_mohr_coulomb_plasticity`'s
//! `MohrCoulombModel::from_angles`.
//!
//! A Mohr-Coulomb plasticity model is parameterised by a friction angle `φ` and
//! a dilation angle `ψ`, both given in degrees. From those angles the reference
//! precomputes the three trigonometric terms that the multisurface return map
//! then reuses: `sin φ`, `cos φ` and `sin ψ`. This module ports that single
//! stateless derivation onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same trio the reference does, not merely that the shader
//! compiles.
//!
//! This twin reproduces only the angle-to-trig derivation; the cohesion- and
//! hardening-dependent fields of the full `MohrCoulombModel` are out of scope.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `from_angles`' trig trio for one angle
//! pair with explicit `friction_degrees` and `dilation_degrees`:
//!
//! * If `friction_degrees` is non-finite or outside `0 < φ < 90`, or
//!   `dilation_degrees` is non-finite or outside `0 <= ψ <= φ`, the pair is
//!   invalid (`valid = false`, all three outputs `0`).
//! * Otherwise `sin_phi = sin(radians(φ))`, `cos_phi = cos(radians(φ))` and
//!   `sin_psi = sin(radians(ψ))`, with `radians(d) = d * (PI / 180)`.
//!
//! # Correctness model
//!
//! The reference evaluates the trig in `f64` then narrows to `f32`; the kernel
//! evaluates `f32` `sin`/`cos` directly, so `CPU` and `GPU` are not bit-exact.
//! Each valid scalar is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity test keeps random angles well inside the valid band so the validity
//! decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite angle, a friction angle outside `(0, 90)`, or a dilation angle
//! outside `[0, φ]` yields `valid = false` with all outputs `0`. The invalid
//! branch is produced with `select`, so no out-of-range angle is ever fed to a
//! live trig evaluation path that matters. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `sin`, `cos`,
//! `+ - *`, `select` and unsigned index arithmetic — with no `tan`, `exp`,
//! `log`, `pow`, no `round` and no `f32` remainder, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the ordered compare
//! `abs(x) < 3.0e38` (which rejects both infinities and `NaN`) rather than a
//! bare `x == x`, and the angle-range gates use ordered compares; there is no
//! `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_mohr_coulomb_plasticity`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Mohr-Coulomb angle-trig kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `MohrCoulombModel::from_angles`; see the module
/// documentation for the closed form.
const MOHR_COULOMB_ANGLE_TRIG_WGSL: &str = r#"
// Mohr-Coulomb angle-trig twin: one thread per query reproduces from_angles'
// (sin_phi, cos_phi, sin_psi) trio. It uses only the portable core-WGSL subset
// (abs, sin, cos, + - *, select plus unsigned index math), takes no optional
// feature, and has no loop and no branch, so it provably terminates. Finiteness
// is an ordered abs < 3.0e38 compare (rejecting infinities and NaN) and the
// angle-range gates are ordered compares, all fed to select.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Friction angle in degrees.
    friction_degrees: f32,
    // Dilation angle in degrees.
    dilation_degrees: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // sin(radians(friction)) when valid, else 0.
    sin_phi: f32,
    // cos(radians(friction)) when valid, else 0.
    cos_phi: f32,
    // sin(radians(dilation)) when valid, else 0.
    sin_psi: f32,
    // 1 when both angles pass the validity gates, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const PI: f32 = 3.14159265358979;
const DEG_TO_RAD: f32 = 3.14159265358979 / 180.0;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let friction = q.friction_degrees;
    let dilation = q.dilation_degrees;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false), plus the angle-range gates with ordered
    // compares. No bare f32 equality anywhere.
    let friction_finite = abs(friction) < FINITE_LIMIT;
    let dilation_finite = abs(dilation) < FINITE_LIMIT;
    let friction_ok = friction_finite && (friction > 0.0) && (friction < 90.0);
    let dilation_ok = dilation_finite && (dilation >= 0.0) && (dilation <= friction);
    let ok = friction_ok && dilation_ok;

    // Feed only sanitised angles into the trig path so the invalid branch never
    // evaluates an out-of-range magnitude.
    let friction_safe = select(0.0, friction, ok);
    let dilation_safe = select(0.0, dilation, ok);
    let phi = friction_safe * DEG_TO_RAD;
    let psi = dilation_safe * DEG_TO_RAD;

    var out: Result;
    out.sin_phi = select(0.0, sin(phi), ok);
    out.cos_phi = select(0.0, cos(phi), ok);
    out.sin_psi = select(0.0, sin(psi), ok);
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
/// The two angles are padded to `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    friction_degrees: f32,
    dilation_degrees: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the three trig terms and the validity flag — `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    sin_phi: f32,
    cos_phi: f32,
    sin_psi: f32,
    valid: u32,
}

/// One Mohr-Coulomb angle-trig query: the friction and dilation angles, in
/// degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombAngleTrigQuery {
    /// Friction angle `φ`, in degrees.
    pub friction_degrees: f32,
    /// Dilation angle `ψ`, in degrees.
    pub dilation_degrees: f32,
}

impl MohrCoulombAngleTrigQuery {
    /// Builds a query from the friction and dilation angles, in degrees.
    #[must_use]
    pub fn new(friction_degrees: f32, dilation_degrees: f32) -> MohrCoulombAngleTrigQuery {
        MohrCoulombAngleTrigQuery {
            friction_degrees,
            dilation_degrees,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `MohrCoulombModel::from_angles` trig trio for that angle pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombAngleTrigResult {
    /// `sin φ` when valid, else `0`.
    pub sin_phi: f32,
    /// `cos φ` when valid, else `0`.
    pub cos_phi: f32,
    /// `sin ψ` when valid, else `0`.
    pub sin_psi: f32,
    /// `true` when both angles pass the validity gates, else `false`.
    pub valid: bool,
}

/// Encodes one [`MohrCoulombAngleTrigQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &MohrCoulombAngleTrigQuery) -> GpuQuery {
    GpuQuery {
        friction_degrees: q.friction_degrees,
        dilation_degrees: q.dilation_degrees,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MohrCoulombAngleTrigResult`].
fn decode_result(raw: &GpuResult) -> MohrCoulombAngleTrigResult {
    MohrCoulombAngleTrigResult {
        sin_phi: raw.sin_phi,
        cos_phi: raw.cos_phi,
        sin_psi: raw.sin_psi,
        valid: raw.valid != 0,
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

/// A compiled, reusable Mohr-Coulomb angle-trig compute pipeline, twinning the
/// `CPU` golden `MohrCoulombModel::from_angles`.
pub struct GpuMohrCoulombAngleTrig {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMohrCoulombAngleTrig {
    /// Compiles the angle-trig kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMohrCoulombAngleTrig {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig"),
            source: ShaderSource::Wgsl(MOHR_COULOMB_ANGLE_TRIG_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMohrCoulombAngleTrig {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`MohrCoulombAngleTrigResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the three trig
    /// scalars to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MohrCoulombAngleTrigQuery],
    ) -> Vec<MohrCoulombAngleTrigResult> {
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
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_bind_group"),
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
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_angle_trig_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mohr_coulomb_angle_trig_pass"),
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
