//! `wgpu` compute twin of four signed-distance *domain transform* primitives of
//! the `CPU` golden path — `rotate_2d`, `rotate_axis`, `fold_plane` and
//! `fold_plane_offset` in
//! `prism_render_architecture::ray_scene::sdf_domain`.
//!
//! Procedural signed-distance modelling wraps a primitive in domain operators:
//! the query point is transformed before the field is sampled, so a single
//! modelled shape instances rotations and mirror folds without touching the
//! primitive. The reference bakes trigonometry on the host (passing a
//! `(sin, cos)` pair) so the runtime stays transcendental-free, and expresses
//! the folds as reflections built from a dot product and a `min`.
//! [`GpuSdfDomainTransform`] is the on-device twin: each thread reads one query
//! and writes the four transformed points, reproducing the reference closed
//! forms with only `abs`, `min`, `max`, products and quotients — never a
//! transcendental.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfDomainTransformQuery`] — a 2D `point_2d` with its
//! `rotate_2d` `(sin, cos)` pair, a 3D `point_3d` shared by the remaining three
//! transforms, the `rotate_axis` `axis` with its own `(sin, cos)` pair, and the
//! fold `normal` with the `fold_plane_offset` `offset` — and writes one
//! [`SdfDomainTransformResult`] holding the four transformed points
//! `rotated_2d`, `rotated_axis`, `folded_plane` and `folded_plane_offset`.
//!
//! The `rotate_2d` kernel applies the planar rotation matrix
//! `[[cos, -sin], [sin, cos]]`. The `rotate_axis` kernel applies Rodrigues'
//! rotation `v*cos + (axis x v)*sin + axis*(axis . v)*(1 - cos)` about the unit
//! `axis`. The `fold_plane` kernel reflects points on the plane's negative side
//! through the origin plane with unit `normal` via `p - 2*min(dot(p, n), 0)*n`,
//! and `fold_plane_offset` folds about the parallel plane
//! `dot(p, n) = offset` via `p - 2*min(dot(p, n) - offset, 0)*n`; points
//! already on the positive side pass through unchanged.
//!
//! # What stays on the host
//!
//! The `CSG` operators that compose these atoms, the ray-marcher that evaluates
//! the composed field along a ray, the host-side baking of each `(sin, cos)`
//! pair, and the normalisation of `axis`/`normal` to unit length all stay on
//! the host; the device sees only the four stateless, fixed-width point
//! transforms, one query at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The transforms thread through products, sums and (for the folds) a `min`
//! branch, so the `CPU` and `GPU` are not bit-exact: a fused multiply-add or a
//! differently ordered sum may land a few units in the last place from the
//! scalar reference. The parity test asserts each component within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough to catch a wrong port
//! yet loose enough to admit a legal last-place difference. Fixtures and the
//! random sweep keep the fold's signed distance clear of zero (the crease where
//! `min(dot, 0)` switches side) and keep `axis`/`normal` unit and the
//! `(sin, cos)` pairs on the unit circle, so the `CPU` and `GPU` never pick
//! different sides of the fold branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round`/`ceil`, and no
//! `f64`/`u64`/`u16`/`i64`/`i16`. The fold's degeneracy is expressed as an
//! ordered `min(..., 0.0)`, never a bare float equality. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_domain`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` domain-transform kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `rotate_2d`, `rotate_axis`, `fold_plane` and
/// `fold_plane_offset`, one query per thread.
const SDF_DOMAIN_TRANSFORM_WGSL: &str = r#"
// Signed-distance domain transforms twinned from the CPU golden path. Each
// thread reads one query and writes the four transformed points; the compose
// operators and the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_domain；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // 2D query point for rotate_2d.
    p2x: f32,
    p2y: f32,
    // Pre-baked (sin, cos) pair for rotate_2d.
    sin2: f32,
    cos2: f32,
    // 3D query point shared by rotate_axis / fold_plane / fold_plane_offset.
    p3x: f32,
    p3y: f32,
    p3z: f32,
    // Rotation axis for rotate_axis (already unit length).
    ax: f32,
    ay: f32,
    az: f32,
    // Pre-baked (sin, cos) pair for rotate_axis.
    sin3: f32,
    cos3: f32,
    // Fold plane normal (already unit length), shared by both folds.
    nx: f32,
    ny: f32,
    nz: f32,
    // Signed plane offset for fold_plane_offset.
    offset: f32,
}

struct Outputs {
    // rotate_2d result.
    r2x: f32,
    r2y: f32,
    // rotate_axis result.
    rax: f32,
    ray: f32,
    raz: f32,
    // fold_plane result.
    fpx: f32,
    fpy: f32,
    fpz: f32,
    // fold_plane_offset result.
    fox: f32,
    foy: f32,
    foz: f32,
    // Padding to a 48-byte stride.
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outputs>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Outputs;

    // --- rotate_2d: planar rotation [[cos, -sin], [sin, cos]]. ---
    out.r2x = q.cos2 * q.p2x - q.sin2 * q.p2y;
    out.r2y = q.sin2 * q.p2x + q.cos2 * q.p2y;

    // --- rotate_axis: Rodrigues' rotation about the unit axis. ---
    // cross = axis x point.
    let cx = q.ay * q.p3z - q.az * q.p3y;
    let cy = q.az * q.p3x - q.ax * q.p3z;
    let cz = q.ax * q.p3y - q.ay * q.p3x;
    let axis_dot = q.ax * q.p3x + q.ay * q.p3y + q.az * q.p3z;
    let w = axis_dot * (1.0 - q.cos3);
    out.rax = q.p3x * q.cos3 + cx * q.sin3 + q.ax * w;
    out.ray = q.p3y * q.cos3 + cy * q.sin3 + q.ay * w;
    out.raz = q.p3z * q.cos3 + cz * q.sin3 + q.az * w;

    // --- fold_plane: reflect the negative side through the origin plane. ---
    let dot_origin = q.p3x * q.nx + q.p3y * q.ny + q.p3z * q.nz;
    let k_origin = 2.0 * min(dot_origin, 0.0);
    out.fpx = q.p3x - k_origin * q.nx;
    out.fpy = q.p3y - k_origin * q.ny;
    out.fpz = q.p3z - k_origin * q.nz;

    // --- fold_plane_offset: reflect across the plane dot(p, n) = offset. ---
    let k_offset = 2.0 * min(dot_origin - q.offset, 0.0);
    out.fox = q.p3x - k_offset * q.nx;
    out.foy = q.p3y - k_offset * q.ny;
    out.foz = q.p3z - k_offset * q.nz;

    out.pad0 = 0.0;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_DOMAIN_TRANSFORM_WGSL`].
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
/// the 2D point and its `(sin, cos)` pair, the shared 3D point, the rotation
/// axis and its `(sin, cos)` pair, and the fold normal and offset — `16` `f32`
/// words at a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// 2D point `x`.
    p2x: f32,
    /// 2D point `y`.
    p2y: f32,
    /// `rotate_2d` sine.
    sin2: f32,
    /// `rotate_2d` cosine.
    cos2: f32,
    /// 3D point `x`.
    p3x: f32,
    /// 3D point `y`.
    p3y: f32,
    /// 3D point `z`.
    p3z: f32,
    /// Rotation axis `x`.
    ax: f32,
    /// Rotation axis `y`.
    ay: f32,
    /// Rotation axis `z`.
    az: f32,
    /// `rotate_axis` sine.
    sin3: f32,
    /// `rotate_axis` cosine.
    cos3: f32,
    /// Fold normal `x`.
    nx: f32,
    /// Fold normal `y`.
    ny: f32,
    /// Fold normal `z`.
    nz: f32,
    /// Signed plane offset.
    offset: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outputs`
/// struct: the four transformed points plus one pad word to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `rotate_2d` result `x`.
    r2x: f32,
    /// `rotate_2d` result `y`.
    r2y: f32,
    /// `rotate_axis` result `x`.
    rax: f32,
    /// `rotate_axis` result `y`.
    ray: f32,
    /// `rotate_axis` result `z`.
    raz: f32,
    /// `fold_plane` result `x`.
    fpx: f32,
    /// `fold_plane` result `y`.
    fpy: f32,
    /// `fold_plane` result `z`.
    fpz: f32,
    /// `fold_plane_offset` result `x`.
    fox: f32,
    /// `fold_plane_offset` result `y`.
    foy: f32,
    /// `fold_plane_offset` result `z`.
    foz: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the domain-transform twin: a 2D point for `rotate_2d`, a
/// shared 3D point for the three 3D transforms, and the per-transform
/// parameters.
///
/// `point_2d` is rotated by `rotate_2d_sin_cos` (a pre-baked `[sin, cos]`
/// pair). `point_3d` is the 3D position shared by `rotate_axis`, `fold_plane`
/// and `fold_plane_offset`. `axis` is the (unit) rotation axis and
/// `rotate_axis_sin_cos` its `[sin, cos]` pair. `normal` is the (unit) fold
/// plane normal shared by both folds, and `offset` is the signed plane distance
/// used only by `fold_plane_offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainTransformQuery {
    /// 2D query point `[x, y]`.
    pub point_2d: [f32; 2],
    /// Pre-baked `[sin, cos]` pair for `rotate_2d`.
    pub rotate_2d_sin_cos: [f32; 2],
    /// 3D query point `[x, y, z]` shared by the three 3D transforms.
    pub point_3d: [f32; 3],
    /// Rotation axis `[x, y, z]` (already unit length).
    pub axis: [f32; 3],
    /// Pre-baked `[sin, cos]` pair for `rotate_axis`.
    pub rotate_axis_sin_cos: [f32; 2],
    /// Fold plane normal `[x, y, z]` (already unit length).
    pub normal: [f32; 3],
    /// Signed plane offset for `fold_plane_offset`.
    pub offset: f32,
}

impl SdfDomainTransformQuery {
    /// Builds a query from the 2D point, the shared 3D point and the
    /// per-transform parameters.
    #[must_use]
    pub const fn new(
        point_2d: [f32; 2],
        rotate_2d_sin_cos: [f32; 2],
        point_3d: [f32; 3],
        axis: [f32; 3],
        rotate_axis_sin_cos: [f32; 2],
        normal: [f32; 3],
        offset: f32,
    ) -> SdfDomainTransformQuery {
        SdfDomainTransformQuery {
            point_2d,
            rotate_2d_sin_cos,
            point_3d,
            axis,
            rotate_axis_sin_cos,
            normal,
            offset,
        }
    }
}

/// One resolved query of the domain-transform twin: the four transformed
/// points.
///
/// `rotated_2d` is `point_2d` rotated in the plane; `rotated_axis` is
/// `point_3d` rotated about `axis`; `folded_plane` and `folded_plane_offset`
/// are `point_3d` mirror-folded across the origin plane and the offset plane
/// respectively (points on the positive side pass through unchanged).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDomainTransformResult {
    /// `rotate_2d` result `[x, y]`.
    pub rotated_2d: [f32; 2],
    /// `rotate_axis` result `[x, y, z]`.
    pub rotated_axis: [f32; 3],
    /// `fold_plane` result `[x, y, z]`.
    pub folded_plane: [f32; 3],
    /// `fold_plane_offset` result `[x, y, z]`.
    pub folded_plane_offset: [f32; 3],
}

/// Encodes one [`SdfDomainTransformQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfDomainTransformQuery) -> GpuQuery {
    GpuQuery {
        p2x: q.point_2d[0],
        p2y: q.point_2d[1],
        sin2: q.rotate_2d_sin_cos[0],
        cos2: q.rotate_2d_sin_cos[1],
        p3x: q.point_3d[0],
        p3y: q.point_3d[1],
        p3z: q.point_3d[2],
        ax: q.axis[0],
        ay: q.axis[1],
        az: q.axis[2],
        sin3: q.rotate_axis_sin_cos[0],
        cos3: q.rotate_axis_sin_cos[1],
        nx: q.normal[0],
        ny: q.normal[1],
        nz: q.normal[2],
        offset: q.offset,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfDomainTransformResult`].
fn decode_result(raw: &GpuResult) -> SdfDomainTransformResult {
    SdfDomainTransformResult {
        rotated_2d: [raw.r2x, raw.r2y],
        rotated_axis: [raw.rax, raw.ray, raw.raz],
        folded_plane: [raw.fpx, raw.fpy, raw.fpz],
        folded_plane_offset: [raw.fox, raw.foy, raw.foz],
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

/// A compiled, reusable domain-transform compute pipeline, twinning the `CPU`
/// golden `rotate_2d`, `rotate_axis`, `fold_plane` and `fold_plane_offset` of
/// `prism_render_architecture::ray_scene::sdf_domain`.
pub struct GpuSdfDomainTransform {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfDomainTransform {
    /// Compiles the domain-transform kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfDomainTransform {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform"),
            source: ShaderSource::Wgsl(SDF_DOMAIN_TRANSFORM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfDomainTransform {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfDomainTransformResult`] per input, in order.
    ///
    /// The transformed components match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfDomainTransformQuery],
    ) -> Vec<SdfDomainTransformResult> {
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
            label: Some("prism_volumetric_sdf_domain_transform_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_bind_group"),
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
            label: Some("prism_volumetric_sdf_domain_transform_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_domain_transform_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_domain_transform_pass"),
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
