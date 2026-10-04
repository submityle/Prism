//! `wgpu` compute twin of the `prism_math::projection` camera-projection
//! matrix family, selecting one of ten pure-scalar variants by an enum id and
//! returning a column-major `[f32; 16]`.
//!
//! `prism_math::projection` builds the perspective and orthographic matrices
//! that a renderer uploads every frame; the design doc's CPU/GPU-consistency
//! clause requires that a projection built on the host and one built in a shader
//! agree bit-close. This module ports the ten parameter-only constructors onto
//! the device so that mirror can be checked on real hardware. The two view
//! constructors (`look_at`/`look_to`) take `Vec3` eye/target/up triples and are
//! out of scope for this scalar-input twin.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, column-for-column, the golden
//! `Mat4::from_cols` of exactly one constructor, chosen by `variant_id`:
//!
//! * `0` `perspective_rh`, `1` `perspective_rh_gl`,
//!   `2` `perspective_reverse_z_rh`, `3` `perspective_infinite_rh`,
//!   `4` `perspective_infinite_reverse_z_rh`, `5` `perspective_lh`,
//!   `6` `perspective_lh_gl` — perspective family, reading
//!   `a0 = fovy_radians`, `a1 = aspect`, `a2 = z_near`, `a3 = z_far` (the two
//!   infinite variants ignore `z_far`).
//! * `7` `orthographic_rh`, `8` `orthographic_rh_gl`, `9` `orthographic_lh` —
//!   orthographic family, reading `a0 = left`, `a1 = right`, `a2 = bottom`,
//!   `a3 = top`, `a4 = z_near`, `a5 = z_far`.
//!
//! The focal term is `f = 1 / tan(fovy_radians * 0.5)`; the golden evaluates it
//! with the `f32` `tan`, and the kernel uses the native `WGSL` `tan(f32)`, which
//! differ by far less than the parity tolerance. The output `m` is laid out
//! column-major: `m[0..4]` is column `0` (`x, y, z, w`), `m[4..8]` column `1`,
//! `m[8..12]` column `2`, `m[12..16]` column `3`, matching `Mat4::from_cols`.
//!
//! # Correctness model
//!
//! Each of the sixteen matrix entries is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) against an
//! independent host oracle that re-derives every constructor's columns in the
//! golden operator order, written out directly so the test never imports
//! `prism_math`. The discrete `valid` flag is compared exactly: it is `1` for a
//! recognised `variant_id` (`<= 9`) and `0` otherwise, in which case `m` is all
//! zero.
//!
//! # Degenerate inputs
//!
//! Every divisor (`aspect`, the near/far differences, the extent widths) is
//! driven by the caller through the master-valid fixture contract, so no
//! division is guarded in the kernel; an out-of-range `variant_id` falls through
//! to the zeroed `else` arm with `valid = 0`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `tan`, `+ - * /`, and
//! unsigned index arithmetic — with no `exp`, `log`, `pow`, no `round`, no `f32`
//! remainder, and no `f64`/`u64`/`i64`. Variant dispatch is by unsigned integer
//! equality; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::projection`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` projection kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the ten
/// pure-scalar constructors of `prism_math::projection`; see the module
/// documentation for the column layout and input-slot mapping.
const PROJECTION_WGSL: &str = r#"
// Projection-matrix twin: one thread per query reproduces one of ten
// prism_math::projection constructors, chosen by variant_id. It uses only the
// portable core-WGSL subset (tan, + - * /, select plus unsigned index math),
// takes no optional feature, and has no loop, so it provably terminates.
// Variant dispatch is by unsigned integer equality; there is no f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Perspective: a0 = fovy_radians, a1 = aspect, a2 = z_near, a3 = z_far.
    // Orthographic: a0 = left, a1 = right, a2 = bottom, a3 = top,
    //               a4 = z_near, a5 = z_far.
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
    a5: f32,
    // Which constructor to evaluate (0..=9); anything else zeroes the matrix.
    variant_id: u32,
}

struct Result {
    // Column-major 4x4 matrix: m[0..4] = column 0, m[4..8] = column 1, etc.
    m: array<f32, 16>,
    // 1 for a recognised variant_id, 0 otherwise (then m is all zero).
    valid: u32,
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
    let vid = q.variant_id;

    var m: array<f32, 16> = array<f32, 16>(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );
    var valid: u32 = 1u;

    if (vid == 0u) {
        // perspective_rh: depth [0, 1], camera looks down -Z.
        let f = 1.0 / tan(q.a0 * 0.5);
        let r = q.a3 / (q.a2 - q.a3);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = r;
        m[11] = -1.0;
        m[14] = r * q.a2;
    } else if (vid == 1u) {
        // perspective_rh_gl: depth [-1, 1].
        let f = 1.0 / tan(q.a0 * 0.5);
        let inv = 1.0 / (q.a2 - q.a3);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = (q.a3 + q.a2) * inv;
        m[11] = -1.0;
        m[14] = 2.0 * q.a3 * q.a2 * inv;
    } else if (vid == 2u) {
        // perspective_reverse_z_rh: z_near -> 1, z_far -> 0.
        let f = 1.0 / tan(q.a0 * 0.5);
        let inv = 1.0 / (q.a3 - q.a2);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = q.a2 * inv;
        m[11] = -1.0;
        m[14] = q.a3 * q.a2 * inv;
    } else if (vid == 3u) {
        // perspective_infinite_rh: infinite far, depth [0, 1).
        let f = 1.0 / tan(q.a0 * 0.5);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = -1.0;
        m[11] = -1.0;
        m[14] = -q.a2;
    } else if (vid == 4u) {
        // perspective_infinite_reverse_z_rh: infinite far, z_near -> 1.
        let f = 1.0 / tan(q.a0 * 0.5);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = 0.0;
        m[11] = -1.0;
        m[14] = q.a2;
    } else if (vid == 5u) {
        // perspective_lh: depth [0, 1], camera looks down +Z.
        let f = 1.0 / tan(q.a0 * 0.5);
        let r = q.a3 / (q.a3 - q.a2);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = r;
        m[11] = 1.0;
        m[14] = -r * q.a2;
    } else if (vid == 6u) {
        // perspective_lh_gl: depth [-1, 1], camera looks down +Z.
        let f = 1.0 / tan(q.a0 * 0.5);
        let inv = 1.0 / (q.a3 - q.a2);
        m[0] = f / q.a1;
        m[5] = f;
        m[10] = (q.a3 + q.a2) * inv;
        m[11] = 1.0;
        m[14] = -2.0 * q.a3 * q.a2 * inv;
    } else if (vid == 7u) {
        // orthographic_rh: depth [0, 1].
        let rcp_w = 1.0 / (q.a1 - q.a0);
        let rcp_h = 1.0 / (q.a3 - q.a2);
        let rcp_d = 1.0 / (q.a4 - q.a5);
        m[0] = 2.0 * rcp_w;
        m[5] = 2.0 * rcp_h;
        m[10] = rcp_d;
        m[12] = -(q.a1 + q.a0) * rcp_w;
        m[13] = -(q.a3 + q.a2) * rcp_h;
        m[14] = q.a4 * rcp_d;
        m[15] = 1.0;
    } else if (vid == 8u) {
        // orthographic_rh_gl: depth [-1, 1].
        let rcp_w = 1.0 / (q.a1 - q.a0);
        let rcp_h = 1.0 / (q.a3 - q.a2);
        let rcp_d = 1.0 / (q.a5 - q.a4);
        m[0] = 2.0 * rcp_w;
        m[5] = 2.0 * rcp_h;
        m[10] = -2.0 * rcp_d;
        m[12] = -(q.a1 + q.a0) * rcp_w;
        m[13] = -(q.a3 + q.a2) * rcp_h;
        m[14] = -(q.a5 + q.a4) * rcp_d;
        m[15] = 1.0;
    } else if (vid == 9u) {
        // orthographic_lh: depth [0, 1].
        let rcp_w = 1.0 / (q.a1 - q.a0);
        let rcp_h = 1.0 / (q.a3 - q.a2);
        let rcp_d = 1.0 / (q.a5 - q.a4);
        m[0] = 2.0 * rcp_w;
        m[5] = 2.0 * rcp_h;
        m[10] = rcp_d;
        m[12] = -(q.a1 + q.a0) * rcp_w;
        m[13] = -(q.a3 + q.a2) * rcp_h;
        m[14] = -q.a4 * rcp_d;
        m[15] = 1.0;
    } else {
        valid = 0u;
    }

    var out: Result;
    out.m = m;
    out.valid = valid;
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
/// six scalar parameter slots and the variant selector — `7` words (`28`
/// bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
    a5: f32,
    variant_id: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the sixteen column-major matrix entries and the validity flag — `17`
/// words (`68` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    m: [f32; 16],
    valid: u32,
}

/// One projection query: six scalar parameter slots plus the `variant_id`
/// selecting which constructor to evaluate.
///
/// The slot meaning depends on the variant family. For the perspective variants
/// (`0..=6`) the slots are `a[0] = fovy_radians`, `a[1] = aspect`,
/// `a[2] = z_near`, `a[3] = z_far` (the infinite variants `3`/`4` ignore
/// `z_far`). For the orthographic variants (`7..=9`) the slots are
/// `a[0] = left`, `a[1] = right`, `a[2] = bottom`, `a[3] = top`,
/// `a[4] = z_near`, `a[5] = z_far`. Use [`ProjectionQuery::perspective`] and
/// [`ProjectionQuery::orthographic`] to fill the slots by name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionQuery {
    /// The six scalar parameter slots; interpretation depends on `variant_id`.
    pub a: [f32; 6],
    /// Which constructor to evaluate (`0..=9`); any other value zeroes the
    /// matrix and sets `valid = 0`.
    pub variant_id: u32,
}

impl ProjectionQuery {
    /// Builds a perspective query (`variant_id` in `0..=6`) from the field of
    /// view, aspect ratio, and near/far planes. The infinite variants ignore
    /// `z_far`; pass any finite value.
    #[must_use]
    pub fn perspective(
        variant_id: u32,
        fovy_radians: f32,
        aspect: f32,
        z_near: f32,
        z_far: f32,
    ) -> ProjectionQuery {
        ProjectionQuery {
            a: [fovy_radians, aspect, z_near, z_far, 0.0, 0.0],
            variant_id,
        }
    }

    /// Builds an orthographic query (`variant_id` in `7..=9`) from the view-box
    /// extents and near/far planes.
    #[must_use]
    pub fn orthographic(
        variant_id: u32,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> ProjectionQuery {
        ProjectionQuery {
            a: [left, right, bottom, top, z_near, z_far],
            variant_id,
        }
    }
}

/// One resolved answer for a single query: the column-major projection matrix
/// and the validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionResult {
    /// The column-major `4x4` matrix: `m[0..4]` is column `0`, `m[4..8]` column
    /// `1`, `m[8..12]` column `2`, `m[12..16]` column `3`.
    pub m: [f32; 16],
    /// `1` for a recognised `variant_id` (`<= 9`), `0` otherwise (then `m` is
    /// all zero).
    pub valid: u32,
}

/// Encodes one [`ProjectionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ProjectionQuery) -> GpuQuery {
    GpuQuery {
        a0: q.a[0],
        a1: q.a[1],
        a2: q.a[2],
        a3: q.a[3],
        a4: q.a[4],
        a5: q.a[5],
        variant_id: q.variant_id,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ProjectionResult`].
fn decode_result(raw: &GpuResult) -> ProjectionResult {
    ProjectionResult {
        m: raw.m,
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

/// A compiled, reusable projection compute pipeline, twinning the ten
/// pure-scalar constructors of `prism_math::projection`.
pub struct GpuProjection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuProjection {
    /// Compiles the projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuProjection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_projection"),
            source: ShaderSource::Wgsl(PROJECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_projection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_projection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_projection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProjection {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ProjectionResult`] per
    /// input, in order.
    ///
    /// Each matrix entry matches the reference to the module's tolerance and the
    /// `valid` flag exactly. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[ProjectionQuery]) -> Vec<ProjectionResult> {
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
            label: Some("prism_volumetric_projection_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_projection_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_projection_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_projection_bind_group"),
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
            label: Some("prism_volumetric_projection_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_projection_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_projection_pass"),
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
