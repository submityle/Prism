//! `wgpu` compute twin of the hair rest-helix orthonormal-basis construction
//! from the `CPU` golden `prism_render_architecture::hair::rest_helix`.
//!
//! A rest-state hair strand is seeded from a single growth `axis`, and the
//! coiling math needs a stable pair of unit vectors spanning the plane
//! perpendicular to that axis. The reference builds `(normal, binormal)` with
//! a trig-free, panic-free recipe: normalise the axis (falling back to a fixed
//! cardinal direction when it is zero or non-finite), pick whichever cardinal
//! helper is least aligned with the axis so the seeding cross product is
//! well-conditioned, normalise that cross product (with a second cardinal
//! fallback for the parallel corner case), and complete the frame with
//! `binormal = axis x normal`. This module ports that stateless closed form
//! onto the device: one thread resolves one basis query.
//!
//! [`GpuHairOrthonormalBasis`] is the on-device twin, so a passing real-device
//! parity test is direct evidence the ported kernel takes the same
//! normalise / helper-selection / fallback branch the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `build_orthonormal_basis(axis)`. It
//! normalises `axis` with the crate's `normalize_or_zero` (a finite,
//! above-epsilon squared length divides by `sqrt(len_sq)`, otherwise it
//! collapses to zero), substitutes `FALLBACK_AXIS = (0, 1, 0)` when the
//! normalised axis is still degenerate, chooses the helper cardinal
//! (`(1, 0, 0)` when `|a.x| < 0.9`, else `(0, 1, 0)`), normalises
//! `a x helper` (falling back to `a x (0, 0, 1)` when the first cross is
//! degenerate), and returns `(normal, a x normal)`. There is no loop: each
//! thread performs a fixed, bounded sequence, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! The basis path threads through multiplies, adds and a guarded division by
//! `sqrt(len_sq)`, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact (a `GPU` may contract a multiply-add). The parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous channel and compares the discrete
//! `valid` flag exactly. Because the fallback guarantees a basis, `valid` is
//! always `1`.
//!
//! # Degenerate inputs
//!
//! A zero or non-finite axis normalises to zero and is replaced by
//! `FALLBACK_AXIS`, and the `0.9` helper guard plus the second-cardinal
//! fallback keep the seeding cross product well-conditioned, so every query
//! yields a valid orthonormal pair. Non-finite squared lengths are detected
//! with an exponent-bit test on the `u32` reinterpretation of the float,
//! matching the reference `is_finite` branch without any floating-point
//! equality. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, `abs`,
//! `sqrt`, `+ - * /`, `bitcast` and ordered `f32` comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, no `round`, no `f32` modulo, no `inverseSqrt`
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::rest_helix`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` hair orthonormal-basis kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `build_orthonormal_basis` branch for branch; see
/// the module documentation for the algorithm.
const HAIR_ORTHONORMAL_BASIS_WGSL: &str = r#"
// Hair orthonormal-basis twin: one thread per query reproduces
// build_orthonormal_basis(axis). It mirrors the CPU golden branch for branch,
// uses only the portable core-WGSL subset (select, abs, sqrt, bitcast and
// ordered f32 comparisons), takes no optional feature, and has no loop, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::rest_helix；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Growth axis whose perpendicular plane the basis spans.
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    pad0: u32,
}

struct Result {
    // The unit normal perpendicular to the (fallback-sanitised) axis.
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    // The unit binormal = axis x normal.
    binormal_x: f32,
    binormal_y: f32,
    binormal_z: f32,
    // 1 when the query produced a valid basis (always, thanks to the fallback).
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared epsilon below which a vector is treated as unnormalisable, matching
// the reference NORMALIZE_EPS_SQ.
const NORMALIZE_EPS_SQ: f32 = 1.0e-24;
// The cardinal axis substituted for a zero or non-finite input, matching the
// reference FALLBACK_AXIS = (0, 1, 0).
const FALLBACK_X: f32 = 0.0;
const FALLBACK_Y: f32 = 1.0;
const FALLBACK_Z: f32 = 0.0;
// IEEE-754 single-precision exponent mask; an all-ones exponent is inf/nan.
const EXP_MASK: u32 = 0x7F800000u;
// Threshold for the least-aligned helper selection, matching the reference 0.9.
const HELPER_GUARD: f32 = 0.9;

// True when x is finite: a non-all-ones exponent field, matching the reference
// is_finite branch without any floating-point equality test.
fn is_finite_f32(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & EXP_MASK) != EXP_MASK;
}

// Squared Euclidean length.
fn length_squared(v: vec3<f32>) -> f32 {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Cross product a x b, right-handed, written out to match the golden order.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// This vector scaled to unit length, or zero if it is too short to normalise or
// non-finite. Mirrors the reference normalize_or_zero: a finite, above-epsilon
// squared length divides by sqrt(len_sq); otherwise the result is zero. The
// unselected scaled branch is discarded by select, so a zero-length input never
// contaminates the output.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = length_squared(v);
    let ok = is_finite_f32(len_sq) && (len_sq > NORMALIZE_EPS_SQ);
    let scaled = v * (1.0 / sqrt(len_sq));
    return select(vec3<f32>(0.0, 0.0, 0.0), scaled, ok);
}

// Builds the orthonormal pair (normal, binormal) spanning the plane
// perpendicular to axis, replicating the golden step for step.
fn build_basis(axis: vec3<f32>) -> Result {
    let fallback = vec3<f32>(FALLBACK_X, FALLBACK_Y, FALLBACK_Z);
    let n = normalize_or_zero(axis);
    let a = select(fallback, n, length_squared(n) > NORMALIZE_EPS_SQ);

    let helper = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(a.x) < HELPER_GUARD,
    );

    let seeded = normalize_or_zero(cross3(a, helper));
    let fallback_normal = normalize_or_zero(cross3(a, vec3<f32>(0.0, 0.0, 1.0)));
    let normal = select(fallback_normal, seeded, length_squared(seeded) > NORMALIZE_EPS_SQ);

    let binormal = cross3(a, normal);

    var out: Result;
    out.normal_x = normal.x;
    out.normal_y = normal.y;
    out.normal_z = normal.z;
    out.binormal_x = binormal.x;
    out.binormal_y = binormal.y;
    out.binormal_z = binormal.z;
    out.valid = 1u;
    out.pad0 = 0u;
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = build_basis(vec3<f32>(q.axis_x, q.axis_y, q.axis_z));
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HAIR_ORTHONORMAL_BASIS_WGSL`].
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
/// Three `f32` plus a trailing pad word fill exactly one `16`-byte slot, so the
/// host stride matches the shader stride for batches of two or more elements.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Six `f32` plus `valid` and a trailing pad word fill exactly two
/// `16`-byte slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    binormal_x: f32,
    binormal_y: f32,
    binormal_z: f32,
    valid: u32,
    pad0: u32,
}

/// One query for the hair orthonormal-basis twin: the growth axis whose
/// perpendicular plane the returned basis spans.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairOrthonormalBasisQuery {
    /// `x` component of the growth axis.
    pub axis_x: f32,
    /// `y` component of the growth axis.
    pub axis_y: f32,
    /// `z` component of the growth axis.
    pub axis_z: f32,
}

impl HairOrthonormalBasisQuery {
    /// Builds a query from the three axis components.
    #[must_use]
    pub fn new(axis_x: f32, axis_y: f32, axis_z: f32) -> HairOrthonormalBasisQuery {
        HairOrthonormalBasisQuery {
            axis_x,
            axis_y,
            axis_z,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `build_orthonormal_basis` output `(normal, binormal)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairOrthonormalBasisResult {
    /// `x` component of the unit normal.
    pub normal_x: f32,
    /// `y` component of the unit normal.
    pub normal_y: f32,
    /// `z` component of the unit normal.
    pub normal_z: f32,
    /// `x` component of the unit binormal (`axis x normal`).
    pub binormal_x: f32,
    /// `y` component of the unit binormal (`axis x normal`).
    pub binormal_y: f32,
    /// `z` component of the unit binormal (`axis x normal`).
    pub binormal_z: f32,
    /// `1` when the query produced a valid basis (always, here).
    pub valid: u32,
}

/// Encodes one [`HairOrthonormalBasisQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &HairOrthonormalBasisQuery) -> GpuQuery {
    GpuQuery {
        axis_x: q.axis_x,
        axis_y: q.axis_y,
        axis_z: q.axis_z,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HairOrthonormalBasisResult`].
fn decode_result(raw: &GpuResult) -> HairOrthonormalBasisResult {
    HairOrthonormalBasisResult {
        normal_x: raw.normal_x,
        normal_y: raw.normal_y,
        normal_z: raw.normal_z,
        binormal_x: raw.binormal_x,
        binormal_y: raw.binormal_y,
        binormal_z: raw.binormal_z,
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

/// A compiled, reusable hair orthonormal-basis compute pipeline, twinning the
/// `CPU` golden `build_orthonormal_basis`.
pub struct GpuHairOrthonormalBasis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairOrthonormalBasis {
    /// Compiles the hair orthonormal-basis kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairOrthonormalBasis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis"),
            source: ShaderSource::Wgsl(HAIR_ORTHONORMAL_BASIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairOrthonormalBasis {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairOrthonormalBasisResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairOrthonormalBasisQuery],
    ) -> Vec<HairOrthonormalBasisResult> {
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
            label: Some("prism_volumetric_hair_orthonormal_basis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_bind_group"),
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
            label: Some("prism_volumetric_hair_orthonormal_basis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_orthonormal_basis_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_orthonormal_basis_pass"),
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
