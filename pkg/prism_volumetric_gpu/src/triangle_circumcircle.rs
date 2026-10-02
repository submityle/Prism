//! `wgpu` compute twin of the per-triangle circumcircle construction golden
//! ([`triangle_circumcircle`](prism_render_architecture::particle::triangle_circumcircle),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`triangle_circumcircle`](prism_render_architecture::particle::triangle_circumcircle)
//! owns the small, `CPU`-verifiable contract several particle meshing stages
//! share: turning three plane points into the unique circle that passes through
//! all of them. Its circumcenter
//! ([`Triangle::circumcenter`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumcenter))
//! is the intersection of two perpendicular bisectors, solved by Cramer's rule
//! as a pure determinant ratio over the three corners' squared magnitudes; its
//! circumradius
//! ([`Triangle::circumradius`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumradius))
//! is the single [`f32::sqrt`] of the distance from that center back to a
//! corner; and the whole circle
//! ([`Triangle::circumcircle`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumcircle))
//! bundles the two, returning [`None`] when the triangle is degenerate.
//!
//! [`GpuTriangleCircumcircle`] is the on-device twin: one thread builds the
//! circumcircle of one triangle, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same center and radius and
//! classifies the same degenerate (collinear or coincident) case the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent triangles the kernel reproduces exactly what
//! [`Triangle::circumcircle`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumcircle)
//! returns: the circumcenter, the circumradius, and a validity flag that mirrors
//! the reference [`Option`]. A valid result is decoded back into a reused golden
//! [`Circle`](prism_render_architecture::particle::triangle_circumcircle::Circle);
//! an invalid (degenerate) triangle decodes to [`None`]. The in-circle and
//! orientation predicates the golden also exposes are deliberately out of scope
//! here; this twin constructs the circle only.
//!
//! # Correctness model
//!
//! The validity flag is a discrete classification built from one `f32`
//! magnitude comparison against [`CMP_EPS`](prism_render_architecture::particle::triangle_circumcircle::CMP_EPS),
//! so for triangles clear of the degeneracy threshold the `CPU` and `GPU` agree
//! exactly and the parity test asserts an exact `==` on it. The center and
//! radius thread through multiplies, adds, one guarded division and one `sqrt`,
//! so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity,
//! tight enough to catch a genuinely wrong port (a dropped term, a swapped
//! corner, a wrong determinant sign) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A degenerate (zero-area) triangle has collinear or coincident corners, which
//! would divide the circumcenter by a near-zero determinant. The kernel checks
//! `2 * doubled-signed-area` against [`CMP_EPS`](prism_render_architecture::particle::triangle_circumcircle::CMP_EPS)
//! and reports an invalid flag (center and radius left at zero) instead,
//! matching the reference [`None`], so the result is never `NaN` or infinite. An
//! empty batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, the `dot`
//! builtin, `+ - * /`, one `sqrt` for the genuine Euclidean radius and unsigned
//! index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::triangle_circumcircle::{Circle, Triangle, Vec2};
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

/// The portable core-`WGSL` circumcircle kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`triangle_circumcircle`](prism_render_architecture::particle::triangle_circumcircle)
/// branch for branch; see the module documentation for the algorithm.
const TRIANGLE_CIRCUMCIRCLE_WGSL: &str = r#"
// Circumcircle twin: one thread per triangle reproduces the Cramer-rule
// circumcenter, the sqrt circumradius and the degenerate-triangle guard. It
// mirrors the CPU golden particle::triangle_circumcircle branch for branch, uses
// only the portable core-WGSL subset (abs and + - * / plus the dot builtin and
// one sqrt) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::triangle_circumcircle；
// 无第三方引擎源码或衍生代码。

// Magnitude below which the doubled determinant is treated as zero (collinear
// or coincident corners). Matches the reference CMP_EPS; the compare rule used
// instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of triangles in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Triangle corners a, b, c packed as three vec2, with a trailing vec2 pad so
    // the record fills two 16-byte std430 slots.
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
    pad: vec2<f32>,
}

struct Result {
    // Circumcenter (zero when degenerate), the circumradius and the validity
    // flag: four scalars filling one 16-byte slot.
    center: vec2<f32>,
    radius: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scalar 2D cross product (the z component of the 3D cross), i.e. twice the
// signed area spanned by `lhs` and `rhs`; mirrors the reference `cross`.
fn cross2(lhs: vec2<f32>, rhs: vec2<f32>) -> f32 {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a = q.a;
    let b = q.b;
    let c = q.c;

    // d = 2 * doubled-signed-area = 4 * area; when it vanishes the three points
    // are collinear/coincident and no finite circumcenter exists, so the twin
    // reports invalid rather than dividing by ~zero (mirrors the reference None).
    var center: vec2<f32> = vec2<f32>(0.0, 0.0);
    var radius: f32 = 0.0;
    var valid: u32 = 0u;
    let d = 2.0 * cross2(b - a, c - a);
    if (abs(d) > CMP_EPS) {
        let asq = dot(a, a);
        let bsq = dot(b, b);
        let csq = dot(c, c);
        let inv = 1.0 / d;
        let ux = (asq * (b.y - c.y) + bsq * (c.y - a.y) + csq * (a.y - b.y)) * inv;
        let uy = (asq * (c.x - b.x) + bsq * (a.x - c.x) + csq * (b.x - a.x)) * inv;
        center = vec2<f32>(ux, uy);
        let diff = center - a;
        radius = sqrt(dot(diff, diff));
        valid = 1u;
    }

    var out: Result;
    out.center = center;
    out.radius = radius;
    out.valid = valid;
    results[idx] = out;
}
"#;

/// The resolved circumcircle for one triangle, mirroring what the reference
/// [`Triangle::circumcircle`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumcircle)
/// returns: the circle through all three corners, or [`None`] for a degenerate
/// (collinear or coincident) triangle.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleCircumcircleResult {
    /// The circumscribed circle (reusing the golden
    /// [`Circle`](prism_render_architecture::particle::triangle_circumcircle::Circle)),
    /// or [`None`] when the triangle is degenerate.
    pub circle: Option<Circle>,
}

/// Evaluates the `CPU` golden for one triangle, delegating to the reference
/// [`Triangle::circumcircle`](prism_render_architecture::particle::triangle_circumcircle::Triangle::circumcircle)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn golden(triangle: &Triangle) -> TriangleCircumcircleResult {
    TriangleCircumcircleResult {
        circle: triangle.circumcircle(),
    }
}

/// `repr(C)` `std430` layout of one packed triangle: three `vec2` corners plus a
/// trailing `vec2` pad — `32` bytes filling two `16`-byte slots, each `vec2` on
/// its `8`-byte boundary exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First corner `a`.
    a: [f32; 2],
    /// Second corner `b`.
    b: [f32; 2],
    /// Third corner `c`.
    c: [f32; 2],
    /// Trailing padding lanes filling the second `vec4` slot.
    pad: [f32; 2],
}

impl GpuQuery {
    /// Packs one triangle into its `std430` image.
    fn new(triangle: &Triangle) -> GpuQuery {
        GpuQuery {
            a: [triangle.a.x, triangle.a.y],
            b: [triangle.b.x, triangle.b.y],
            c: [triangle.c.x, triangle.c.y],
            pad: [0.0, 0.0],
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(center.x, center.y, radius, valid)` — `16` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Circumcenter (zero when degenerate).
    center: [f32; 2],
    /// Circumradius (zero when degenerate).
    radius: f32,
    /// `1` when the triangle was non-degenerate, `0` otherwise.
    valid: u32,
}

/// Uniform parameters for one dispatch: the triangle count plus three pad words
/// to fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of triangles in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable circumcircle compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
pub struct GpuTriangleCircumcircle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriangleCircumcircle {
    /// Compiles the circumcircle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTriangleCircumcircle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle"),
            source: ShaderSource::Wgsl(TRIANGLE_CIRCUMCIRCLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriangleCircumcircle {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the circumcircle of every triangle on-device and returns one
    /// [`TriangleCircumcircleResult`] per input, in order.
    ///
    /// The validity flag equals the reference exactly for triangles clear of the
    /// degeneracy threshold; the center and radius match to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        triangles: &[Triangle],
    ) -> Vec<TriangleCircumcircleResult> {
        if triangles.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = triangles.len();

        let packed: Vec<GpuQuery> = triangles.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_triangle_circumcircle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_triangle_circumcircle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public
/// [`TriangleCircumcircleResult`], turning the validity flag back into an
/// [`Option`] over a reused golden
/// [`Circle`](prism_render_architecture::particle::triangle_circumcircle::Circle).
fn decode_result(raw: &GpuResult) -> TriangleCircumcircleResult {
    let circle = if raw.valid == 0 {
        None
    } else {
        Some(Circle::new(
            Vec2::new(raw.center[0], raw.center[1]),
            raw.radius,
        ))
    };
    TriangleCircumcircleResult { circle }
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
