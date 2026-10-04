//! `wgpu` compute twin of the slack-relaxed sphere containment test, from the
//! `CPU` golden `prism_physics_core::collider::bounding_sphere`'s
//! `BoundingSphere::contains_point`.
//!
//! A bounding sphere is a center and a non-negative radius. A point is
//! "contained" when its squared distance to the center is within the squared
//! radius grown by a small relative slack (`radius * CONTAIN_EPS + CONTAIN_EPS`,
//! `CONTAIN_EPS = 1e-5`), which keeps Welzl's minimal-enclosing-ball fit from
//! looping on round-off. This module ports that single stateless closed form
//! onto the device: one thread resolves one query, so a passing real-device
//! parity test is direct evidence the ported kernel takes the same containment
//! decision the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `contains_point` for one sphere/point
//! pair with explicit `center`, `radius` and `point`:
//!
//! * `slack = radius * 1e-5 + 1e-5`; `r = radius + slack`.
//! * `contains = dot(point - center, point - center) <= r * r`.
//!
//! The test uses the squared distance against the squared relaxed radius, so
//! there is no `sqrt` and no division anywhere; the comparison is a single
//! ordered `<=`. `valid` is always `1`.
//!
//! # Correctness model
//!
//! The continuous arithmetic (the difference, the dot product and the squared
//! radius) threads through multiplies and adds, so `CPU` and `GPU` are not
//! necessarily bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The discrete `contains` and `valid` flags are compared
//! exactly; the parity test keeps random points away from the boundary knee so
//! a last-bit difference cannot flip the decision.
//!
//! # Degenerate inputs
//!
//! There is no degenerate branch: a zero radius still admits points within the
//! absolute `CONTAIN_EPS` slack, and a large radius simply scales the slack —
//! both are the ordinary closed form, so `valid` is always `1`. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `+ - *`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder, no `sqrt` and no
//! division, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! boundary test is an ordered `<=` fed to `select`; there is no `f32`
//! equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` sphere-containment kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `BoundingSphere::contains_point`; see the module
/// documentation for the closed form.
const BOUNDING_SPHERE_CONTAINS_WGSL: &str = r#"
// Sphere-containment twin: one thread per query reproduces contains_point. It
// uses only the portable core-WGSL subset (dot, + - *, select plus unsigned
// index math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Every vec3 input is passed as scalar lanes and rebuilt
// inside the shader to avoid any std430 16-byte vector-alignment ambiguity.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sphere center, world space.
    cx: f32, cy: f32, cz: f32,
    // Sphere radius, non-negative.
    radius: f32,
    // Query point, world space.
    px: f32, py: f32, pz: f32,
    // Padding word to a 16-byte-friendly stride.
    pad0: f32,
}

struct Result {
    // 1 when the point lies within the slack-relaxed sphere, else 0.
    contains: u32,
    // Always 1; the closed form has no degenerate branch.
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
    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let point = vec3<f32>(q.px, q.py, q.pz);

    // slack = radius * 1e-5 + 1e-5, in the same order as the golden.
    let slack = q.radius * 1.0e-5 + 1.0e-5;
    let r = q.radius + slack;
    let d = point - center;
    let dist_sq = dot(d, d);
    let r_sq = r * r;

    // Ordered compare fed to select so Metal fast-math cannot fold it; no
    // bare f32 equality anywhere.
    let inside = select(0u, 1u, dist_sq <= r_sq);

    var out: Result;
    out.contains = inside;
    out.valid = 1u;
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
/// Every `vec3` input is flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`. The
/// struct is `8` `f32` words (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    px: f32,
    py: f32,
    pz: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the containment flag and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    contains: u32,
    valid: u32,
}

/// One sphere-containment query: the sphere center and radius, plus the point
/// to test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereContainsQuery {
    /// Sphere center, world space.
    pub center: [f32; 3],
    /// Sphere radius, non-negative.
    pub radius: f32,
    /// Query point, world space.
    pub point: [f32; 3],
}

impl BoundingSphereContainsQuery {
    /// Builds a query from the sphere center, radius and the test point.
    #[must_use]
    pub fn new(center: [f32; 3], radius: f32, point: [f32; 3]) -> BoundingSphereContainsQuery {
        BoundingSphereContainsQuery {
            center,
            radius,
            point,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `BoundingSphere::contains_point` output for that sphere/point pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereContainsResult {
    /// `1` when the point lies within the slack-relaxed sphere, else `0`.
    pub contains: u32,
    /// Always `1`; the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`BoundingSphereContainsQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &BoundingSphereContainsQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        radius: q.radius,
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingSphereContainsResult`].
fn decode_result(raw: &GpuResult) -> BoundingSphereContainsResult {
    BoundingSphereContainsResult {
        contains: raw.contains,
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

/// A compiled, reusable sphere-containment compute pipeline, twinning the `CPU`
/// golden `BoundingSphere::contains_point`.
pub struct GpuBoundingSphereContains {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingSphereContains {
    /// Compiles the sphere-containment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingSphereContains {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains"),
            source: ShaderSource::Wgsl(BOUNDING_SPHERE_CONTAINS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingSphereContains {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingSphereContainsResult`] per input, in order.
    ///
    /// The `contains` and `valid` flags match the reference exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingSphereContainsQuery],
    ) -> Vec<BoundingSphereContainsResult> {
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
            label: Some("prism_volumetric_bounding_sphere_contains_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_bind_group"),
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
            label: Some("prism_volumetric_bounding_sphere_contains_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_sphere_contains_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_sphere_contains_pass"),
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
