//! `wgpu` compute twin of three analytic signed-distance primitives of the
//! `CPU` golden path
//! ([`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box),
//! [`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere)
//! and
//! [`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere)).
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! three such distances: a filleted axis-aligned box
//! ([`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box),
//! the exact box field inflated outward by a corner radius), a sphere sliced by
//! a horizontal plane
//! ([`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere),
//! Inigo-Quilez `sdCutSphere`), and the hollow shell of that cut sphere
//! ([`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere),
//! Inigo-Quilez `sdCutHollowSphere`). [`GpuSdfSphereOps`] is the on-device twin:
//! each thread reads one point plus all three shapes' parameters and writes all
//! three signed distances, reproducing the reference closed forms with only
//! `sqrt`, `abs`, `min`, `max`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfSphereOpsQuery`] — a query `point` plus the box
//! `half_extent` and `round_radius`, the cut-sphere `cut_radius`/`cut_height`,
//! and the hollow-sphere `hollow_radius`/`hollow_cut_height`/`hollow_thickness`
//! — and writes one [`SdfSphereOpsResult`] holding the three signed distances
//! `round_box_sd`, `cut_sphere_sd` and `cut_hollow_sphere_sd`.
//!
//! The round-box kernel reduces the point to its per-axis overshoot
//! `q = |point| - half_extent`, forms the exact box distance (exterior
//! `length(max(q, 0))` plus interior `min(max(q.x, q.y, q.z), 0)`) and
//! subtracts the corner radius. The cut-sphere kernel works in the meridian
//! `(radial, y)` half-plane, where `radial = length(point.xz)`; a single
//! comparison `s` picks the spherical cap, the flat disc face, or the circular
//! rim. The hollow-sphere kernel reuses that meridian reduction: past the rim's
//! radial cone the rim circle governs, otherwise the sphere surface does, and
//! subtracting the thickness turns the zero-thickness cap into a shell.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every distance threads through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative error
//! floored at `1e-6`), tight enough to catch a genuinely wrong port yet loose
//! enough to admit a legal last-place difference. The cut-sphere selector `s`,
//! the `radial < w` disc test and the hollow-sphere cone test
//! `cut_height * radial < rim * y` are ordered comparisons; fixtures and the
//! randomized sweep stay a safe margin away from each boundary (and from a
//! near-zero cap radius `w`), so the `CPU` and `GPU` always pick the same
//! branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` sphere-ops signed-distance kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box),
/// [`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere)
/// and
/// [`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere)
/// closed forms; see the module documentation for the algorithm.
const SDF_SPHERE_OPS_WGSL: &str = r#"
// Sphere-ops signed-distance twin: one thread computes one query point's rounded
// box, cut-sphere and cut-hollow-sphere signed distances, mirroring the CPU
// golden `ray_scene::sdf_primitives::{round_box, cut_sphere, cut_hollow_sphere}`
// with only sqrt, abs, min, max, products and quotients. The domain/CSG
// operators and the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point components.
    px: f32,
    py: f32,
    pz: f32,
    // Rounded box: half extents and corner radius.
    hx: f32,
    hy: f32,
    hz: f32,
    round_radius: f32,
    // Cut sphere: sphere radius and slice height.
    cut_radius: f32,
    cut_height: f32,
    // Cut hollow sphere: sphere radius, slice height and shell thickness.
    hollow_radius: f32,
    hollow_cut_height: f32,
    hollow_thickness: f32,
}

struct Distances {
    // Rounded-box signed distance.
    round_box_sd: f32,
    // Cut-sphere signed distance.
    cut_sphere_sd: f32,
    // Cut-hollow-sphere signed distance.
    cut_hollow_sphere_sd: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector; the shared radial/meridian reduction.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // Rounded box = exact box distance - corner radius. The box distance is the
    // exterior overshoot length plus the interior least-negative face distance.
    let dx = abs(q.px) - q.hx;
    let dy = abs(q.py) - q.hy;
    let dz = abs(q.pz) - q.hz;
    let ox = max(dx, 0.0);
    let oy = max(dy, 0.0);
    let oz = max(dz, 0.0);
    let outside = sqrt(ox * ox + oy * oy + oz * oz);
    let inside = min(max(dx, max(dy, dz)), 0.0);
    out.round_box_sd = outside + inside - q.round_radius;

    // Cut sphere in the (radial, y) meridian half-plane. w is the cap radius.
    let r = q.cut_radius;
    let h = q.cut_height;
    let w = sqrt(max(r * r - h * h, 0.0));
    let cs_radial = length2(q.px, q.pz);
    let cs_y = q.py;
    let s = max(
        (h - r) * cs_radial * cs_radial + w * w * (h + r - 2.0 * cs_y),
        h * cs_radial - w * cs_y
    );
    if (s < 0.0) {
        // Spherical cap sector: distance to the sphere surface.
        out.cut_sphere_sd = length2(cs_radial, cs_y) - r;
    } else if (cs_radial < w) {
        // Directly under the cap: distance to the flat disc face.
        out.cut_sphere_sd = h - cs_y;
    } else {
        // Otherwise: distance to the circular rim where cap meets disc.
        out.cut_sphere_sd = length2(cs_radial - w, cs_y - h);
    }

    // Cut hollow sphere: the same meridian reduction, rim circle vs sphere,
    // offset inward by the shell thickness.
    let hr = q.hollow_radius;
    let hh = q.hollow_cut_height;
    let rim = sqrt(max(hr * hr - hh * hh, 0.0));
    let q0 = length2(q.px, q.pz);
    let q1 = q.py;
    var surface: f32;
    if (hh * q0 < rim * q1) {
        // Past the rim's radial cone: nearest feature is the rim circle.
        surface = length2(q0 - rim, q1 - hh);
    } else {
        // Otherwise: unsigned distance to the sphere surface.
        surface = abs(length2(q0, q1) - hr);
    }
    out.cut_hollow_sphere_sd = surface - q.hollow_thickness;

    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_SPHERE_OPS_WGSL`].
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
/// the point components plus all three shapes' parameters. Twelve `f32` fields
/// pack to a `48`-byte, `4`-byte-aligned stride with no trailing pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Rounded-box half extent `x`.
    hx: f32,
    /// Rounded-box half extent `y`.
    hy: f32,
    /// Rounded-box half extent `z`.
    hz: f32,
    /// Rounded-box corner radius.
    round_radius: f32,
    /// Cut-sphere radius.
    cut_radius: f32,
    /// Cut-sphere slice height.
    cut_height: f32,
    /// Cut-hollow-sphere radius.
    hollow_radius: f32,
    /// Cut-hollow-sphere slice height.
    hollow_cut_height: f32,
    /// Cut-hollow-sphere shell thickness.
    hollow_thickness: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rounded-box signed distance.
    round_box_sd: f32,
    /// Cut-sphere signed distance.
    cut_sphere_sd: f32,
    /// Cut-hollow-sphere signed distance.
    cut_hollow_sphere_sd: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the sphere-ops signed-distance twin: the query `point` plus
/// the rounded-box, cut-sphere and cut-hollow-sphere shape parameters.
///
/// `point` is the evaluation position; `half_extent`/`round_radius` are the
/// [`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box)
/// half extents and corner radius; `cut_radius`/`cut_height` are the
/// [`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere)
/// sphere radius and slice height; `hollow_radius`/`hollow_cut_height`/
/// `hollow_thickness` are the
/// [`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere)
/// sphere radius, slice height and shell thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSphereOpsQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Rounded-box half extents `[x, y, z]`.
    pub half_extent: [f32; 3],
    /// Rounded-box corner radius.
    pub round_radius: f32,
    /// Cut-sphere radius.
    pub cut_radius: f32,
    /// Cut-sphere slice height.
    pub cut_height: f32,
    /// Cut-hollow-sphere radius.
    pub hollow_radius: f32,
    /// Cut-hollow-sphere slice height.
    pub hollow_cut_height: f32,
    /// Cut-hollow-sphere shell thickness.
    pub hollow_thickness: f32,
}

impl SdfSphereOpsQuery {
    /// Builds a query from the point and all three shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        half_extent: [f32; 3],
        round_radius: f32,
        cut_radius: f32,
        cut_height: f32,
        hollow_radius: f32,
        hollow_cut_height: f32,
        hollow_thickness: f32,
    ) -> SdfSphereOpsQuery {
        SdfSphereOpsQuery {
            point,
            half_extent,
            round_radius,
            cut_radius,
            cut_height,
            hollow_radius,
            hollow_cut_height,
            hollow_thickness,
        }
    }
}

/// One resolved query of the sphere-ops signed-distance twin: the rounded-box,
/// cut-sphere and cut-hollow-sphere signed distances at the query point.
///
/// `round_box_sd` is
/// [`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box);
/// `cut_sphere_sd` is
/// [`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere);
/// `cut_hollow_sphere_sd` is
/// [`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere).
/// Each is negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSphereOpsResult {
    /// Rounded-box signed distance.
    pub round_box_sd: f32,
    /// Cut-sphere signed distance.
    pub cut_sphere_sd: f32,
    /// Cut-hollow-sphere signed distance.
    pub cut_hollow_sphere_sd: f32,
}

/// Encodes one [`SdfSphereOpsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfSphereOpsQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        hx: q.half_extent[0],
        hy: q.half_extent[1],
        hz: q.half_extent[2],
        round_radius: q.round_radius,
        cut_radius: q.cut_radius,
        cut_height: q.cut_height,
        hollow_radius: q.hollow_radius,
        hollow_cut_height: q.hollow_cut_height,
        hollow_thickness: q.hollow_thickness,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfSphereOpsResult`].
fn decode_result(raw: &GpuResult) -> SdfSphereOpsResult {
    SdfSphereOpsResult {
        round_box_sd: raw.round_box_sd,
        cut_sphere_sd: raw.cut_sphere_sd,
        cut_hollow_sphere_sd: raw.cut_hollow_sphere_sd,
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

/// A compiled, reusable sphere-ops signed-distance compute pipeline, twinning
/// the `CPU` golden
/// [`round_box`](prism_render_architecture::ray_scene::sdf_primitives::round_box),
/// [`cut_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_sphere)
/// and
/// [`cut_hollow_sphere`](prism_render_architecture::ray_scene::sdf_primitives::cut_hollow_sphere).
pub struct GpuSdfSphereOps {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSphereOps {
    /// Compiles the sphere-ops signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfSphereOps {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops"),
            source: ShaderSource::Wgsl(SDF_SPHERE_OPS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSphereOps {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfSphereOpsResult`]
    /// per input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfSphereOpsQuery],
    ) -> Vec<SdfSphereOpsResult> {
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
            label: Some("prism_volumetric_sdf_sphere_ops_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_bind_group"),
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
            label: Some("prism_volumetric_sdf_sphere_ops_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_sphere_ops_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_sphere_ops_pass"),
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
