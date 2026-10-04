//! `wgpu` compute twin of the quadric-error optimal-point solve from the `CPU`
//! golden `Quadric::optimal_point`.
//!
//! A quadric error metric accumulates a symmetric `3x3` system `A x = -b` whose
//! least-squares minimizer is the point that best satisfies a bundle of plane
//! constraints. The golden stores the upper triangle of `A` as `a2, ab, ac`,
//! `b2, bc`, `c2`, the linear term `b` as `ad, bd, cd`, and the constant `d2`,
//! and solves for the optimal point by an explicit cofactor (adjugate) inverse:
//! it forms the three first-row cofactors, takes the determinant, rejects a
//! (near-)singular system, and otherwise multiplies the full symmetric adjugate
//! against `-b` scaled by the reciprocal determinant. This module ports that
//! closed form onto the device: one thread resolves one quadric, so a passing
//! real-device parity test is direct evidence the kernel reproduces the same
//! optimal point and the same found/singular decision the reference does.
//!
//! # What is twinned
//!
//! `Quadric::optimal_point() -> Option<Vec3>`, operator for operator in the
//! golden's order: the first-row cofactors `c00, c01, c02`, the determinant
//! `det = m00*c00 + m01*c01 + m02*c02`, the singular guard `|det| <= 1e-12`, the
//! remaining symmetric cofactors `c11, c12, c22`, the right-hand side
//! `r = -(ad, bd, cd)`, and the three adjugate dot products scaled by
//! `1/det`. A non-finite result (which the det guard already precludes in
//! practice) also maps to `None`.
//!
//! # Correctness model
//!
//! The point components are continuous `f32`, compared with an
//! absolute-or-relative tolerance. The `found` word is discrete and compared
//! exactly: `1` when the system is solvable, `0` when the determinant is at or
//! below the `1e-12` singular threshold (or the solve would be non-finite). The
//! `valid` word is always `1`; it is carried only for layout parity with the
//! crate's other twins. Fixtures hold the singular/full-rank cases clear of the
//! `|det| = 1e-12` knee so the discrete decision never flips under `f32` noise.
//!
//! The singular test is an ordered compare (`|det| > 1e-12`) and the outputs are
//! chosen with `select`, so there is no bare `f32` equality anywhere in the
//! kernel. The divisor is guarded with `select(1.0, det, |det| > 1e-12)` so the
//! unselected arm computes `1/1` rather than forming `inf` or `nan`, and the
//! point is forced to the zero vector when the system is rejected. The finite
//! guard uses ordered magnitude compares (`|x| < 3e38`) rather than a `NaN`
//! self-comparison.
//!
//! # Degenerate inputs
//!
//! A rank-deficient or all-zero quadric has `|det| <= 1e-12` and reports
//! `found = 0` with a zeroed point. An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `abs`, and `+ - * /` on scalar `f32` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder, no bare `f32`
//! equality and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every coefficient and output lane is a scalar `f32` in
//! the storage buffers, so no vector alignment rule can perturb the `std430`
//! stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture` 的 `Quadric::optimal_point`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` quadric optimal-point kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// `Quadric::optimal_point`; see the module documentation for the algorithm.
const QUADRIC_OPTIMAL_POINT_WGSL: &str = r#"
// Quadric optimal-point twin: one thread per query reproduces
// Quadric::optimal_point. It builds the symmetric 3x3 matrix from the upper
// triangle, takes the explicit cofactor determinant, rejects |det| <= 1e-12 as
// singular, and otherwise multiplies the symmetric adjugate against the negated
// linear term scaled by 1/det. It uses only the portable core-WGSL subset
// (ordered compares, select, abs, + - * / on scalar f32).

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Upper triangle of the symmetric 3x3 matrix A and the linear/constant terms:
    // a2=A00, ab=A01, ac=A02, ad=b0, b2=A11, bc=A12, bd=b1, c2=A22, cd=b2, d2=const.
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
}

struct PointResult {
    // Optimal point, zero vector when the system is rejected.
    px: f32,
    py: f32,
    pz: f32,
    // 1 when the system is solvable, 0 when singular or non-finite.
    found: u32,
    // Always 1: carried for layout parity with the crate's other twins.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<PointResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Symmetric 3x3 matrix entries from the upper triangle.
    let m00 = q.a2;
    let m01 = q.ab;
    let m02 = q.ac;
    let m11 = q.b2;
    let m12 = q.bc;
    let m22 = q.c2;

    // First-row cofactors and the determinant, in the golden's exact order.
    let c00 = m11 * m22 - m12 * m12;
    let c01 = m02 * m12 - m01 * m22;
    let c02 = m01 * m12 - m02 * m11;
    let det = m00 * c00 + m01 * c01 + m02 * c02;

    // Ordered singular guard; the divisor is guarded so the unselected arm never
    // forms inf/nan.
    let det_ok = abs(det) > 1e-12;
    let inv_det = 1.0 / select(1.0, det, det_ok);

    // Remaining symmetric adjugate cofactors.
    let c11 = m00 * m22 - m02 * m02;
    let c12 = m02 * m01 - m00 * m12;
    let c22 = m00 * m11 - m01 * m01;

    // Right-hand side r = -(ad, bd, cd).
    let r0 = -q.ad;
    let r1 = -q.bd;
    let r2 = -q.cd;

    let x = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det;
    let y = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det;
    let z = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det;

    // Ordered finite guard (no NaN self-compare); det_ok already precludes a
    // non-finite solve in practice but the magnitude test mirrors the golden.
    let fin = abs(x) < 3.0e38 && abs(y) < 3.0e38 && abs(z) < 3.0e38;
    let found = det_ok && fin;

    var res: PointResult;
    res.px = select(0.0, x, found);
    res.py = select(0.0, y, found);
    res.pz = select(0.0, z, found);
    res.found = select(0u, 1u, found);
    res.valid = 1u;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUADRIC_OPTIMAL_POINT_WGSL`].
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
/// All ten lanes are scalar `f32`, so the layout is a flat `40`-byte stride with
/// alignment `4` and no internal padding, and a batch of two or more packs
/// contiguously with no vector alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `PointResult`
/// struct: the optimal point (three `f32`), a `found` word and a `valid` word.
/// Five scalar lanes give a flat `20`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    px: f32,
    py: f32,
    pz: f32,
    found: u32,
    valid: u32,
}

/// One query for the quadric optimal-point twin: the ten accumulated quadric
/// coefficients in the golden's storage order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricOptimalPointQuery {
    /// `A00` (squared `a` coefficient).
    pub a2: f32,
    /// `A01` (`a*b` coefficient).
    pub ab: f32,
    /// `A02` (`a*c` coefficient).
    pub ac: f32,
    /// Linear term `b0` (`a*d` coefficient).
    pub ad: f32,
    /// `A11` (squared `b` coefficient).
    pub b2: f32,
    /// `A12` (`b*c` coefficient).
    pub bc: f32,
    /// Linear term `b1` (`b*d` coefficient).
    pub bd: f32,
    /// `A22` (squared `c` coefficient).
    pub c2: f32,
    /// Linear term `b2` (`c*d` coefficient).
    pub cd: f32,
    /// Constant term (`d*d` coefficient); carried for layout parity, unused in
    /// the solve.
    pub d2: f32,
}

impl QuadricOptimalPointQuery {
    /// Builds a query from the ten quadric coefficients, in storage order.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the golden quadric is parameterized by exactly ten scalar coefficients"
    )]
    pub fn new(
        a2: f32,
        ab: f32,
        ac: f32,
        ad: f32,
        b2: f32,
        bc: f32,
        bd: f32,
        c2: f32,
        cd: f32,
        d2: f32,
    ) -> QuadricOptimalPointQuery {
        QuadricOptimalPointQuery {
            a2,
            ab,
            ac,
            ad,
            b2,
            bc,
            bd,
            c2,
            cd,
            d2,
        }
    }
}

/// One resolved answer for a single query, mirroring
/// `Quadric::optimal_point`.
///
/// (`point_x`, `point_y`, `point_z`) is the optimal point when `found` is `1`,
/// or the zero vector when the system is singular (`found` is `0`). `valid` is
/// always `1`: the layout carries it for parity with the crate's other twins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricOptimalPointResult {
    /// Optimal point, x component.
    pub point_x: f32,
    /// Optimal point, y component.
    pub point_y: f32,
    /// Optimal point, z component.
    pub point_z: f32,
    /// `1` when the system is solvable, `0` when singular or non-finite.
    pub found: u32,
    /// Always `1`: carried for layout parity with the crate's other twins.
    pub valid: u32,
}

/// Encodes one [`QuadricOptimalPointQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricOptimalPointQuery) -> GpuQuery {
    GpuQuery {
        a2: q.a2,
        ab: q.ab,
        ac: q.ac,
        ad: q.ad,
        b2: q.b2,
        bc: q.bc,
        bd: q.bd,
        c2: q.c2,
        cd: q.cd,
        d2: q.d2,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`QuadricOptimalPointResult`].
fn decode_result(raw: &GpuResult) -> QuadricOptimalPointResult {
    QuadricOptimalPointResult {
        point_x: raw.px,
        point_y: raw.py,
        point_z: raw.pz,
        found: raw.found,
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

/// A compiled, reusable quadric optimal-point compute pipeline, twinning the
/// `CPU` golden `Quadric::optimal_point`.
pub struct GpuQuadricOptimalPoint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricOptimalPoint {
    /// Compiles the quadric optimal-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricOptimalPoint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point"),
            source: ShaderSource::Wgsl(QUADRIC_OPTIMAL_POINT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricOptimalPoint {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`QuadricOptimalPointResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuadricOptimalPointQuery],
    ) -> Vec<QuadricOptimalPointResult> {
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
            label: Some("prism_volumetric_quadric_optimal_point_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_bind_group"),
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
            label: Some("prism_volumetric_quadric_optimal_point_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_optimal_point_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_optimal_point_pass"),
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
