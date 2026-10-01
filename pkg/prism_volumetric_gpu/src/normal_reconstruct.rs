//! `wgpu` compute twin of the view-space normal reconstruction golden
//! [`particle::normal_reconstruct`](prism_render_architecture::particle::normal_reconstruct)
//! (design §16-§21): rebuild a per-pixel view-space normal from a linear depth
//! buffer with the Valve/Unreal improved depth-to-normal technique.
//!
//! Deferred and screen-space effects (`SSAO`, contact shadows, screen-space
//! reflections and edge-aware blurs) want a per-pixel *view-space normal* but
//! only have a linear depth buffer: reconstructing the surface orientation from
//! depth avoids paying for a dedicated normal `G-buffer` target. The `CPU`
//! golden
//! [`particle::normal_reconstruct`](prism_render_architecture::particle::normal_reconstruct)
//! owns that reconstruction and documents that a future `GPU` kernel must
//! "match it bit for bit"; [`GpuNormalReconstruct`] is the on-device twin that
//! runs one thread per pixel query and returns, for every query, both the naive
//! two-tap normal and the improved four-tap normal.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly:
//!
//! - **Unproject.** [`view_pos_from_depth`](prism_render_architecture::particle::normal_reconstruct::view_pos_from_depth)
//!   maps a `UV` to normalized device coordinates (`ndc = uv * 2 - 1`) and
//!   scales by the precomputed `tan(FOV/2)` half-extents at the sampled linear
//!   depth, with the camera looking down `-Z`, so a pixel reprojects to
//!   `[ndc_x * half_tan_fov_x * d, ndc_y * half_tan_fov_y * d, -d]`. No tangent
//!   is evaluated here; the math stays linear/rational.
//! - **Naive.** The horizontal edge is `P(right) - P(center)` and the vertical
//!   edge is `P(up) - P(center)`, and the normal is
//!   `normalize(cross(ddx, ddy))`, matching
//!   [`reconstruct_normal_naive`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_naive).
//! - **Improved.** For each axis the neighbor whose linear depth is *closest*
//!   to the center depth is kept (the Valve "best four-tap" selection): if
//!   `|center - left| < |right - center|` the horizontal edge is
//!   `P(center) - P(left)`, else `P(right) - P(center)`; if
//!   `|center - down| < |up - center|` the vertical edge is
//!   `P(center) - P(down)`, else `P(up) - P(center)`. The normal is again
//!   `normalize(cross(ddx, ddy))`, matching
//!   [`reconstruct_normal_improved`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_improved).
//!
//! The absolute value used for the tap selection is computed as `max(x, -x)` so
//! the shader never calls a math builtin outside the portable subset, and the
//! normalization guards a squared length below [`CMP_EPS`](prism_render_architecture::particle::normal_reconstruct::CMP_EPS)
//! to the zero vector, exactly as the reference `normalize3` does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `min`, `max`
//! and `+ - * /` with unsigned integer compares — and takes no optional device
//! feature, so it runs unmodified on Metal, Vulkan and DX12. The only
//! non-arithmetic primitive is `sqrt` (vector normalization), exactly as in the
//! reference, and the one reciprocal is guarded by the squared-length floor so
//! no divide ever hits a vanishing denominator.
//!
//! # Correctness model
//!
//! The whole pipeline is affine reprojection plus a single cross product and a
//! guarded normalization, with no transcendental call and no reorderable
//! reduction, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The parity test asserts a tolerance
//! (`abs_diff <= 1e-5` or `rel_diff <= 1e-5`) tight enough to catch a genuinely
//! wrong port (a swapped tap, a flipped cross product, a dropped selection) yet
//! loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Valve/Unreal improved depth-to-normal reconstruction
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::normal_reconstruct::{DepthTaps, NormalReconstructParams};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Threads per workgroup; one thread reconstructs one pixel query.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` normal-reconstruction kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden exactly;
/// see the module documentation for the algorithm.
const NORMAL_RECONSTRUCT_WGSL: &str = r#"
// View-space normal reconstruction twin: one thread per pixel query reprojects
// the depth taps to view-space positions, then forms both the naive two-tap
// normal (right/up forward difference) and the improved four-tap normal (keep
// the depth-closest neighbor on each axis), writing both per query. It mirrors
// the CPU golden `particle::normal_reconstruct`, uses only the portable
// core-WGSL subset (sqrt/min/max and + - * /), and takes no optional feature,
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Valve/Unreal improved depth-to-normal reconstruction; no
// Unreal Engine source or derived code.

struct Params {
    // Precomputed tan(FOV_x / 2): the view-space half-width at unit depth.
    half_tan_fov_x: f32,
    // Precomputed tan(FOV_y / 2): the view-space half-height at unit depth.
    half_tan_fov_y: f32,
    // Number of queries (one thread each).
    count: u32,
    // Padding to a 16-byte std430 uniform struct.
    pad0: u32,
}

// One pixel query. 48-byte std430 stride (three vec4 slots): uv.xy and texel.xy,
// then the center/left/right/down taps, then the up tap plus three pad words.
struct Query {
    uv_texel: vec4<f32>,
    taps_a: vec4<f32>,
    taps_b: vec4<f32>,
}

// One query result. 32-byte std430 stride: the naive normal triple plus a pad
// word, then the improved normal triple plus a pad word.
struct Normal {
    naive_x: f32,
    naive_y: f32,
    naive_z: f32,
    pad0: f32,
    improved_x: f32,
    improved_y: f32,
    improved_z: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Normal>;

// Squared-length floor below which a vector is treated as degenerate and
// normalized to zero. Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

// Absolute value within the portable subset: max(x, -x) avoids calling abs so
// the kernel stays inside sqrt/min/max + arithmetic.
fn absf(x: f32) -> f32 {
    return max(x, -x);
}

// Reprojects a pixel to its view-space position from a linear depth, matching
// the reference `view_pos_from_depth`: ndc = uv * 2 - 1, scaled by the frustum
// half-extents at depth `d`, with the camera looking down -Z.
fn view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = uv.y * 2.0 - 1.0;
    let x = ndc_x * params.half_tan_fov_x * d;
    let y = ndc_y * params.half_tan_fov_y * d;
    let z = -d;
    return vec3<f32>(x, y, z);
}

// Right-handed cross product a x b, component order matching the reference
// `cross3`.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Normalizes a vector, guarding a squared length below `CMP_EPS` to the zero
// vector, matching the reference `normalize3`. The only place sqrt is used.
fn normalize_guarded(v: vec3<f32>) -> vec3<f32> {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq < CMP_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_len = 1.0 / sqrt(len_sq);
    return v * inv_len;
}

@compute @workgroup_size(64)
fn reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let uv = vec2<f32>(q.uv_texel.x, q.uv_texel.y);
    let texel = vec2<f32>(q.uv_texel.z, q.uv_texel.w);
    let center = q.taps_a.x;
    let left = q.taps_a.y;
    let right = q.taps_a.z;
    let down = q.taps_a.w;
    let up = q.taps_b.x;

    // Shared center and the right/up neighbors feed the naive reconstruction.
    let p_c = view_pos(uv, center);
    let p_r = view_pos(vec2<f32>(uv.x + texel.x, uv.y), right);
    let p_u = view_pos(vec2<f32>(uv.x, uv.y + texel.y), up);
    let naive = normalize_guarded(cross3(p_r - p_c, p_u - p_c));

    // The improved reconstruction additionally samples the left/down neighbors
    // and keeps, per axis, the side whose depth is closest to the center.
    let p_l = view_pos(vec2<f32>(uv.x - texel.x, uv.y), left);
    let p_d = view_pos(vec2<f32>(uv.x, uv.y - texel.y), down);

    let left_closer = absf(center - left) < absf(right - center);
    var ddx = p_r - p_c;
    if (left_closer) {
        ddx = p_c - p_l;
    }

    let down_closer = absf(center - down) < absf(up - center);
    var ddy = p_u - p_c;
    if (down_closer) {
        ddy = p_c - p_d;
    }
    let improved = normalize_guarded(cross3(ddx, ddy));

    var out: Normal;
    out.naive_x = naive.x;
    out.naive_y = naive.y;
    out.naive_z = naive.z;
    out.pad0 = 0.0;
    out.improved_x = improved.x;
    out.improved_y = improved.y;
    out.improved_z = improved.z;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One pixel's reconstruction query: the pixel `UV`, the `UV`-space step to a
/// neighbor `(du, dv)` and the five neighborhood depth taps.
///
/// The taps reuse the `CPU` golden
/// [`DepthTaps`](prism_render_architecture::particle::normal_reconstruct::DepthTaps)
/// type so the host and device share one depth-neighborhood contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalQuery {
    /// Pixel `UV` in `0..=1`.
    pub uv: [f32; 2],
    /// `UV`-space step to a neighbor `(du, dv)`.
    pub texel: [f32; 2],
    /// The five linear depth taps (center and four axis neighbors).
    pub taps: DepthTaps,
}

/// One query's reconstructed view-space normals: the naive two-tap normal and
/// the improved four-tap normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalResult {
    /// Naive forward-difference normal (right/up neighbors), matching
    /// [`reconstruct_normal_naive`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_naive).
    pub naive: [f32; 3],
    /// Improved four-tap normal (depth-closest neighbor per axis), matching
    /// [`reconstruct_normal_improved`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_improved).
    pub improved: [f32; 3],
}

/// Uniform parameters for one reconstruction dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`NORMAL_RECONSTRUCT_WGSL`]: the two frustum
/// half-extents, the query count and one pad word — `16` bytes with no interior
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Precomputed `tan(FOV_x / 2)`.
    half_tan_fov_x: f32,
    /// Precomputed `tan(FOV_y / 2)`.
    half_tan_fov_y: f32,
    /// Number of queries in this dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: `uv.xy`/`texel.xy`, then the center/left/right/down taps, then the
/// up tap plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Pixel `UV` `x`.
    uv_x: f32,
    /// Pixel `UV` `y`.
    uv_y: f32,
    /// Neighbor step `du`.
    texel_x: f32,
    /// Neighbor step `dv`.
    texel_y: f32,
    /// Center linear depth.
    center: f32,
    /// Left neighbor linear depth.
    left: f32,
    /// Right neighbor linear depth.
    right: f32,
    /// Down neighbor linear depth.
    down: f32,
    /// Up neighbor linear depth.
    up: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One query result as read back. `32`-byte `std430` stride matching `Normal`
/// in the shader: the naive normal triple plus a pad word, then the improved
/// normal triple plus a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNormal {
    /// Naive normal `x`.
    naive_x: f32,
    /// Naive normal `y`.
    naive_y: f32,
    /// Naive normal `z`.
    naive_z: f32,
    /// Padding word.
    pad0: f32,
    /// Improved normal `x`.
    improved_x: f32,
    /// Improved normal `y`.
    improved_y: f32,
    /// Improved normal `z`.
    improved_z: f32,
    /// Padding word.
    pad1: f32,
}

/// A compiled, reusable normal-reconstruction pipeline.
pub struct GpuNormalReconstruct {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuNormalReconstruct {
    /// Compiles the normal-reconstruction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNormalReconstruct {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_normal_reconstruct"),
            source: ShaderSource::Wgsl(NORMAL_RECONSTRUCT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reconstruct"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuNormalReconstruct {
            module,
            layout,
            pipeline,
        }
    }

    /// Reconstructs the view-space normals for every query in `queries` against
    /// the shared frustum `params`, returning one [`NormalResult`] per query in
    /// input order.
    ///
    /// The returned result for query `q` holds both the naive normal
    /// [`reconstruct_normal_naive`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_naive)`(q.uv, q.texel, q.taps.center, q.taps.right, q.taps.up, params)`
    /// and the improved normal
    /// [`reconstruct_normal_improved`](prism_render_architecture::particle::normal_reconstruct::reconstruct_normal_improved)`(q.uv, q.texel, &q.taps, params)`,
    /// each to within the tolerance documented on this module. An empty
    /// `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: NormalReconstructParams,
        queries: &[NormalQuery],
    ) -> Vec<NormalResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            half_tan_fov_x: params.half_tan_fov_x,
            half_tan_fov_y: params.half_tan_fov_y,
            count: queries.len() as u32,
            pad0: 0,
        };

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                uv_x: q.uv[0],
                uv_y: q.uv[1],
                texel_x: q.texel[0],
                texel_y: q.texel[1],
                center: q.taps.center,
                left: q.taps.left,
                right: q.taps.right,
                down: q.taps.down,
                up: q.taps.up,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<GpuNormal>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_normal_reconstruct_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_normal_reconstruct_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of `WORKGROUP_SIZE`.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuNormal>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(|r| NormalResult {
                naive: [r.naive_x, r.naive_y, r.naive_z],
                improved: [r.improved_x, r.improved_y, r.improved_z],
            })
            .collect()
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
