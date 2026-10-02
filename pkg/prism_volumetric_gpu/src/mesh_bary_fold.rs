//! `wgpu` compute twin of the `CPU` golden barycentric unit-square fold plus a
//! triangle barycentric interpolation (design §12, mesh-surface emission).
//!
//! Area-weighted mesh-surface emission draws two uniform numbers `(u, v)` on the
//! unit square and folds them onto the triangle with the standard reflection
//! `if u + v > 1 { u = 1 − u; v = 1 − v }`, yielding uniform barycentric weights
//! `(w0, w1, w2)` that sum to one. A particle spawned on the triangle then takes
//! position `w0·p0 + w1·p1 + w2·p2`. The `CPU` golden
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) module
//! owns the fold (its `fold_barycentric` helper); [`GpuMeshBaryFold`] is the
//! on-device twin of that fold plus the companion interpolation, validated
//! against a line-for-line mirror of the private golden helper so a passing
//! real-device parity test is direct evidence the ported kernel folds and
//! interpolates the same way, not merely that its shader compiles.
//!
//! # Private golden source
//!
//! The golden `fold_barycentric` is a *private* `fn` in
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission), so it
//! cannot be imported. The parity test carries a line-for-line mirror
//! (`golden_fold_barycentric`) transcribed from the reference and clearly
//! annotated as such; this twin is validated against that mirror.
//!
//! # Algorithm
//!
//! One thread owns one sample. For each `(u, v)` the kernel reproduces the fold
//! exactly:
//!
//! ```text
//! if u + v > 1.0 { su = 1 - u; sv = 1 - v } else { su = u; sv = v }
//! (w0, w1, w2) = (1 - su - sv, su, sv)
//! ```
//!
//! then interpolates the triangle position in the fixed accumulation order
//! `w0·p0 + w1·p1 + w2·p2` (left-to-right), matching the mirror used by the
//! parity test term for term. Comparison plus `+ − ×` only.
//!
//! # Degenerate inputs
//!
//! An empty query dispatches nothing and returns empty vectors, matching the
//! host guard. The fold branch `u + v > 1.0` depends only on the uploaded `f32`
//! inputs (never on a previously computed value), so the two sides take the same
//! branch for bit-identical inputs.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − ×` and comparison on `f32` — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and `DX12`.
//!
//! # Correctness model
//!
//! The fold and interpolation contain no transcendental call, so `CPU` and `GPU`
//! evaluate the same closed-form algebra. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar mirror leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a wrong
//! port (a dropped reflection, a swapped weight, a reordered interpolation) yet
//! loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`
//! 的私有重心折叠 `fold_barycentric` 加三角形重心插值与 `wgpu` 计算下发；无第三方
//! 引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` fold-plus-interpolate kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the private `CPU` golden
/// `fold_barycentric` plus the companion interpolation exactly; see the module
/// documentation for the algorithm.
const MESH_BARY_FOLD_WGSL: &str = r#"
// Barycentric fold + interpolation twin: one thread per sample folds the unit
// square draw (u, v) onto a triangle with the reflection
// `if u + v > 1 { su = 1 - u; sv = 1 - v }`, forms the weights
// (1 - su - sv, su, sv), then interpolates w0*p0 + w1*p1 + w2*p2 in that fixed
// order. It mirrors the private CPU golden
// `particle::mesh_emission::fold_barycentric` plus the companion interpolation,
// uses only the portable core-WGSL subset (integer index math plus + - * and
// comparison on f32), and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture 的私有重心折叠；无第三方引擎
// 源码或衍生代码。

struct Params {
    // Number of live samples; threads past this return early.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Per-sample input: the unit-square draw and the triangle's three positions.
struct Sample {
    u: f32,
    v: f32,
    pad_u0: f32,
    pad_u1: f32,
    p0x: f32,
    p0y: f32,
    p0z: f32,
    pad_p0: f32,
    p1x: f32,
    p1y: f32,
    p1z: f32,
    pad_p1: f32,
    p2x: f32,
    p2y: f32,
    p2z: f32,
    pad_p2: f32,
}

// Per-sample output: the folded barycentric weights and interpolated position.
struct OutSample {
    w0: f32,
    w1: f32,
    w2: f32,
    pad_w: f32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    pad_pos: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> out_samples: array<OutSample>;

@compute @workgroup_size(64)
fn fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let s = samples[idx];

    // Fold the unit-square draw onto the triangle (fold_barycentric).
    var su: f32;
    var sv: f32;
    if (s.u + s.v > 1.0) {
        su = 1.0 - s.u;
        sv = 1.0 - s.v;
    } else {
        su = s.u;
        sv = s.v;
    }
    let w0 = 1.0 - su - sv;
    let w1 = su;
    let w2 = sv;

    // Interpolate the triangle position in the fixed order w0*p0 + w1*p1 + w2*p2.
    let p0 = vec3<f32>(s.p0x, s.p0y, s.p0z);
    let p1 = vec3<f32>(s.p1x, s.p1y, s.p1z);
    let p2 = vec3<f32>(s.p2x, s.p2y, s.p2z);
    let pos = p0 * w0 + p1 * w1 + p2 * w2;

    var out: OutSample;
    out.w0 = w0;
    out.w1 = w1;
    out.w2 = w2;
    out.pad_w = 0.0;
    out.pos_x = pos.x;
    out.pos_y = pos.y;
    out.pos_z = pos.z;
    out.pad_pos = 0.0;
    out_samples[idx] = out;
}
"#;

/// Uniform parameters for one fold dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`MESH_BARY_FOLD_WGSL`]: the live `count` plus three pad words —
/// `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of live samples.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One fold sample as uploaded. `64`-byte `std430` stride matching `Sample` in
/// the shader: the unit-square draw `(u, v)` on its own `16`-byte lane, then the
/// three triangle positions each padded to a `16`-byte lane boundary.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// Unit-square draw `u`.
    u: f32,
    /// Unit-square draw `v`.
    v: f32,
    /// Padding lane.
    pad_u0: f32,
    /// Padding lane.
    pad_u1: f32,
    /// Vertex `p0` `x`.
    p0x: f32,
    /// Vertex `p0` `y`.
    p0y: f32,
    /// Vertex `p0` `z`.
    p0z: f32,
    /// Padding lane.
    pad_p0: f32,
    /// Vertex `p1` `x`.
    p1x: f32,
    /// Vertex `p1` `y`.
    p1y: f32,
    /// Vertex `p1` `z`.
    p1z: f32,
    /// Padding lane.
    pad_p1: f32,
    /// Vertex `p2` `x`.
    p2x: f32,
    /// Vertex `p2` `y`.
    p2y: f32,
    /// Vertex `p2` `z`.
    p2z: f32,
    /// Padding lane.
    pad_p2: f32,
}

/// One fold sample as read back. `32`-byte `std430` stride matching `OutSample`
/// in the shader: the folded barycentric weights on one `16`-byte lane, the
/// interpolated position on the next.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutSample {
    /// Barycentric weight `w0`.
    w0: f32,
    /// Barycentric weight `w1`.
    w1: f32,
    /// Barycentric weight `w2`.
    w2: f32,
    /// Padding lane.
    pad_w: f32,
    /// Interpolated position `x`.
    pos_x: f32,
    /// Interpolated position `y`.
    pos_y: f32,
    /// Interpolated position `z`.
    pos_z: f32,
    /// Padding lane.
    pad_pos: f32,
}

/// A batch of fold samples to resolve in one dispatch.
///
/// The two vectors are parallel and indexed by sample: `uvs[i]` is the
/// unit-square draw `(u, v)` and `triangles[i]` is the triangle `[p0, p1, p2]`
/// its folded weights interpolate across. The two vectors must share a length.
///
/// Provenance: 本模块新建的 GPU 批量查询类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshBaryFoldQuery {
    /// The per-sample unit-square draws `(u, v)`.
    pub uvs: Vec<[f32; 2]>,
    /// The per-sample triangles `[p0, p1, p2]`.
    pub triangles: Vec<[Vec3; 3]>,
}

/// The outcome of a `GPU` fold dispatch, one entry per input sample.
///
/// Provenance: 本模块新建的 GPU 批量结果类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshBaryFoldResult {
    /// The folded barycentric weights `(w0, w1, w2)` per sample.
    pub weights: Vec<Vec3>,
    /// The interpolated triangle positions `w0·p0 + w1·p1 + w2·p2` per sample.
    pub positions: Vec<Vec3>,
}

/// A compiled, reusable barycentric-fold pipeline.
///
/// Provenance: 本模块新建的 GPU 管线封装类型；无第三方引擎源码或衍生代码。
pub struct GpuMeshBaryFold {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshBaryFold {
    /// Compiles the fold kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 本模块新建的管线构造；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshBaryFold {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold"),
            source: ShaderSource::Wgsl(MESH_BARY_FOLD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("fold"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshBaryFold {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds every sample in `query` on the device and reads the results back.
    ///
    /// For sample `i` the returned `weights[i]` is the folded barycentric weight
    /// of `query.uvs[i]` (the twin of the private golden `fold_barycentric`), and
    /// `positions[i]` is `w0·p0 + w1·p1 + w2·p2` over `query.triangles[i]`, both
    /// to within the tolerance documented on this module. An empty batch
    /// dispatches nothing and returns empty vectors.
    ///
    /// # Panics
    ///
    /// Panics if `query.uvs` and `query.triangles` do not share a length, since
    /// the two are parallel per-sample inputs.
    ///
    /// Provenance: 本模块新建的下发/回读流程；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn fold(&self, ctx: &GpuContext, query: &GpuMeshBaryFoldQuery) -> GpuMeshBaryFoldResult {
        let count = query.uvs.len();
        assert_eq!(
            count,
            query.triangles.len(),
            "uvs and triangles must share a length"
        );
        if count == 0 {
            return GpuMeshBaryFoldResult {
                weights: Vec::new(),
                positions: Vec::new(),
            };
        }

        let device = ctx.device();
        let packed: Vec<GpuSample> = (0..count)
            .map(|i| {
                let [u, v] = query.uvs[i];
                let [p0, p1, p2] = query.triangles[i];
                GpuSample {
                    u,
                    v,
                    pad_u0: 0.0,
                    pad_u1: 0.0,
                    p0x: p0.x,
                    p0y: p0.y,
                    p0z: p0.z,
                    pad_p0: 0.0,
                    p1x: p1.x,
                    p1y: p1.y,
                    p1z: p1.z,
                    pad_p1: 0.0,
                    p2x: p2.x,
                    p2y: p2.y,
                    p2z: p2.z,
                    pad_p2: 0.0,
                }
            })
            .collect();
        let gpu_params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (count * size_of::<GpuOutSample>()) as u64;
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let sample_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_samples"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sample_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_bary_fold_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_bary_fold_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
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
        let gpu_out = bytemuck::cast_slice::<u8, GpuOutSample>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_out.len(), count);

        let mut weights = Vec::with_capacity(count);
        let mut positions = Vec::with_capacity(count);
        for o in gpu_out {
            weights.push(Vec3::new(o.w0, o.w1, o.w2));
            positions.push(Vec3::new(o.pos_x, o.pos_y, o.pos_z));
        }
        GpuMeshBaryFoldResult { weights, positions }
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
