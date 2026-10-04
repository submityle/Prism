//! `wgpu` compute twin of the `prism_math::projection` view-matrix family —
//! the `look_at`/`look_to` constructors — selecting one of four variants by an
//! enum id and returning a column-major `[f32; 16]`.
//!
//! `prism_math::projection` builds the view matrices a renderer uploads every
//! frame; the design doc's CPU/GPU-consistency clause requires that a view
//! matrix built on the host and one built in a shader agree bit-close. This
//! module ports the four orientation constructors onto the device so that
//! mirror can be checked on real hardware. The pure-scalar projection
//! constructors are twinned separately in the sibling `projection` module.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, column-for-column, the golden
//! `Mat4::from_cols` of exactly one constructor, chosen by `variant_id`:
//!
//! * `0` `look_at_rh`, `1` `look_to_rh`, `2` `look_at_lh`, `3` `look_to_lh`.
//!
//! The `look_at` variants first form the forward direction `dir = target - eye`
//! from the second input slot; the `look_to` variants read that slot as the
//! explicit forward `dir`. The third input slot is the `up` reference. The
//! right-handed basis is `f = normalize(dir)`, `s = normalize(cross(f, up))`,
//! `u = cross(s, f)` with columns `(s.x, u.x, -f.x, 0)`, …,
//! `(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)`. The left-handed basis is
//! `f = normalize(dir)`, `s = normalize(cross(up, f))`, `u = cross(f, s)` with
//! columns `(s.x, u.x, f.x, 0)`, …,
//! `(-dot(s, eye), -dot(u, eye), -dot(f, eye), 1)`.
//!
//! # Correctness model
//!
//! Each of the sixteen matrix entries is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) against an
//! independent host oracle that re-derives every constructor's columns in the
//! golden operator order, written out directly so the test never imports
//! `prism_math`. The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! The golden `normalize` is `v * (1 / length(v))`, which produces a
//! non-finite basis when `dir` or `cross(dir, up)` collapses to near-zero (a
//! zero-length direction or a direction parallel to `up`). The kernel guards
//! both normalizations with a `length > 1e-20` ordered test and reports
//! `valid = 0` with an all-zero matrix when either collapses; an out-of-range
//! `variant_id` falls through to the same zeroed arm. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `normalize`'s
//! building blocks `length`/`cross`/`dot`, `+ - * /`, and unsigned index
//! arithmetic — with no `exp`, `log`, `pow`, no `round`, no `f32` remainder,
//! and no `f64`/`u64`/`i64`. Variant dispatch is by unsigned integer equality;
//! there is no `f32` equality anywhere.
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

/// The portable core-`WGSL` view-matrix kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// four `look_at`/`look_to` constructors of `prism_math::projection`; see the
/// module documentation for the column layout and input-slot mapping.
const PROJECTION_VIEW_WGSL: &str = r#"
// View-matrix twin: one thread per query reproduces one of four
// prism_math::projection look_at/look_to constructors, chosen by variant_id. It
// uses only the portable core-WGSL subset (length/cross/dot, + - * /, plus
// unsigned index math), takes no optional feature, and has no loop, so it
// provably terminates. Variant dispatch is by unsigned integer equality; there
// is no f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // eye position; the second slot is target (look_at) or dir (look_to).
    eye: vec3<f32>,
    target_or_dir: vec3<f32>,
    up: vec3<f32>,
    // Which constructor to evaluate (0..=3); anything else zeroes the matrix.
    variant_id: u32,
}

struct ViewResult {
    // Column-major 4x4 matrix: m[0..4] = column 0, m[4..8] = column 1, etc.
    m: array<f32, 16>,
    // 1 for a recognised variant_id with a finite basis, 0 otherwise.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<ViewResult>;

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
    var valid: u32 = 0u;

    let eps = 1.0e-20;
    let eye = q.eye;
    let up = q.up;

    if (vid <= 3u) {
        // look_at (0, 2) forms dir = target - eye; look_to (1, 3) reads dir.
        var dir = q.target_or_dir;
        if (vid == 0u || vid == 2u) {
            dir = q.target_or_dir - eye;
        }
        let dir_len = length(dir);
        if (dir_len > eps) {
            let f = dir * (1.0 / dir_len);
            if (vid == 0u || vid == 1u) {
                // Right-handed basis.
                let sc = cross(f, up);
                let sc_len = length(sc);
                if (sc_len > eps) {
                    let s = sc * (1.0 / sc_len);
                    let u = cross(s, f);
                    m[0] = s.x;
                    m[1] = u.x;
                    m[2] = -f.x;
                    m[3] = 0.0;
                    m[4] = s.y;
                    m[5] = u.y;
                    m[6] = -f.y;
                    m[7] = 0.0;
                    m[8] = s.z;
                    m[9] = u.z;
                    m[10] = -f.z;
                    m[11] = 0.0;
                    m[12] = -dot(s, eye);
                    m[13] = -dot(u, eye);
                    m[14] = dot(f, eye);
                    m[15] = 1.0;
                    valid = 1u;
                }
            } else {
                // Left-handed basis (vid == 2u or 3u).
                let sc = cross(up, f);
                let sc_len = length(sc);
                if (sc_len > eps) {
                    let s = sc * (1.0 / sc_len);
                    let u = cross(f, s);
                    m[0] = s.x;
                    m[1] = u.x;
                    m[2] = f.x;
                    m[3] = 0.0;
                    m[4] = s.y;
                    m[5] = u.y;
                    m[6] = f.y;
                    m[7] = 0.0;
                    m[8] = s.z;
                    m[9] = u.z;
                    m[10] = f.z;
                    m[11] = 0.0;
                    m[12] = -dot(s, eye);
                    m[13] = -dot(u, eye);
                    m[14] = -dot(f, eye);
                    m[15] = 1.0;
                    valid = 1u;
                }
            }
        }
    }

    var out: ViewResult;
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
/// three `vec3<f32>` slots (each 16-byte aligned, so a trailing pad word) and
/// the variant selector — `48` bytes, aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    eye: [f32; 3],
    pad0: f32,
    target_or_dir: [f32; 3],
    pad1: f32,
    up: [f32; 3],
    variant_id: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `ViewResult`
/// struct: the sixteen column-major matrix entries and the validity flag — `17`
/// words (`68` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    m: [f32; 16],
    valid: u32,
}

/// One view-matrix query: the eye position, the second slot (target for
/// `look_at`, forward `dir` for `look_to`), the `up` reference, and the
/// `variant_id` selecting which constructor to evaluate.
///
/// For the `look_at` variants (`0`, `2`) the second slot is the target point;
/// for the `look_to` variants (`1`, `3`) it is the (unnormalized) forward
/// direction. Use [`ProjectionViewQuery::look_at`] and
/// [`ProjectionViewQuery::look_to`] to fill the slots by name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionViewQuery {
    /// The eye (camera) position.
    pub eye: [f32; 3],
    /// Target point (`look_at`) or forward direction (`look_to`).
    pub target_or_dir: [f32; 3],
    /// The `up` reference direction.
    pub up: [f32; 3],
    /// Which constructor to evaluate (`0..=3`); any other value zeroes the
    /// matrix and sets `valid = 0`.
    pub variant_id: u32,
}

impl ProjectionViewQuery {
    /// Builds a `look_at` query (`variant_id` `0` for right-handed, `2` for
    /// left-handed) from the eye, target point, and up reference.
    #[must_use]
    pub fn look_at(
        variant_id: u32,
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
    ) -> ProjectionViewQuery {
        ProjectionViewQuery {
            eye,
            target_or_dir: target,
            up,
            variant_id,
        }
    }

    /// Builds a `look_to` query (`variant_id` `1` for right-handed, `3` for
    /// left-handed) from the eye, forward direction, and up reference.
    #[must_use]
    pub fn look_to(
        variant_id: u32,
        eye: [f32; 3],
        dir: [f32; 3],
        up: [f32; 3],
    ) -> ProjectionViewQuery {
        ProjectionViewQuery {
            eye,
            target_or_dir: dir,
            up,
            variant_id,
        }
    }
}

/// One resolved answer for a single query: the column-major view matrix and the
/// validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionViewResult {
    /// The column-major `4x4` matrix: `m[0..4]` is column `0`, `m[4..8]` column
    /// `1`, `m[8..12]` column `2`, `m[12..16]` column `3`.
    pub m: [f32; 16],
    /// `1` for a recognised `variant_id` with a finite basis, `0` otherwise
    /// (then `m` is all zero).
    pub valid: u32,
}

/// Encodes one [`ProjectionViewQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ProjectionViewQuery) -> GpuQuery {
    GpuQuery {
        eye: q.eye,
        pad0: 0.0,
        target_or_dir: q.target_or_dir,
        pad1: 0.0,
        up: q.up,
        variant_id: q.variant_id,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ProjectionViewResult`].
fn decode_result(raw: &GpuResult) -> ProjectionViewResult {
    ProjectionViewResult {
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

/// A compiled, reusable view-matrix compute pipeline, twinning the four
/// `look_at`/`look_to` constructors of `prism_math::projection`.
pub struct GpuProjectionView {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuProjectionView {
    /// Compiles the view-matrix kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuProjectionView {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_projection_view"),
            source: ShaderSource::Wgsl(PROJECTION_VIEW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_projection_view_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_projection_view_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_projection_view_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProjectionView {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ProjectionViewResult`]
    /// per input, in order.
    ///
    /// Each matrix entry matches the reference to the module's tolerance and the
    /// `valid` flag exactly. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ProjectionViewQuery],
    ) -> Vec<ProjectionViewResult> {
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
            label: Some("prism_volumetric_projection_view_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_projection_view_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_projection_view_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_projection_view_bind_group"),
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
            label: Some("prism_volumetric_projection_view_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_projection_view_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_projection_view_pass"),
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
