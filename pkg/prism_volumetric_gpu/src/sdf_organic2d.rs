//! `wgpu` compute twin of four analytic two-dimensional signed-distance
//! primitives from the `CPU` golden path
//! (`prism_render_architecture::ray_scene::sdf_primitives`): the Inigo Quilez
//! `heart_2d`, `egg_2d`, `moon` and `cross_2d` closed forms.
//!
//! Vector graphics, decal masks and procedural profile extrusion evaluate these
//! analytic outlines whose exact Euclidean distance is known in closed form
//! rather than sampling a baked field. This module is the on-device twin of
//! four of those atoms, each a true distance built from folds, a branch select
//! and a single `sqrt`:
//!
//! - `heart_2d`: the unit heart with its tip at the origin and its cusp at
//!   `(0, 1)`; points above the `|x| + y = 1` diagonal measure against the lobe
//!   circle, the rest take the nearer of the top cusp and the diagonal flank,
//!   signed by `sign(|x| - y)`.
//! - `egg_2d`: Inigo Quilez's three-arc egg with its axis on `x = 0`; the fold
//!   to `x >= 0` routes the query to the bottom circle, the top cap or a cheek
//!   arc by the sign of `y` and the `sqrt 3 (x + r) < y` test.
//! - `moon`: the exact crescent, the constructive-solid difference of two disks
//!   with an explicit cusp-tip branch that keeps the field exact where a naive
//!   `max` would overshoot.
//! - `cross_2d`: the exact rounded plus sign, folded into the octant
//!   `x >= y >= 0` so the exterior is a plain box distance while the interior
//!   measures the reentrant corner, inset by the rounding radius.
//!
//! [`GpuSdfOrganic2d`] evaluates all four for one query per thread, reproducing
//! the reference closed forms with only `abs`, `min`, `max`, `clamp`, a sign
//! test and a final `sqrt` — no transcendental — so a passing real-device
//! parity test is direct evidence the ported kernel computes the same distances
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfOrganic2dQuery`] — the shared query point plus the
//! per-shape parameters (`egg` radii; `moon` offset and radii; `cross_2d` arm,
//! half-thickness and rounding) — and writes one [`SdfOrganic2dResult`] holding
//! the four signed distances. Every lane folds the query by `abs` to exploit
//! each shape's mirror symmetry, selects the governing feature branch, and
//! takes a single `sqrt`.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these outlines, the `extrude`
//! and `revolution` lift to three dimensions, and any acceleration structure
//! all stay on the host; the device sees only the stateless, fixed-width
//! distance evaluation, one query at a time, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! Every output threads through products, quotients and a `sqrt`, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` with a relative
//! floor of `1e-6`. Where a shape's internal `signum` flips sign the magnitude
//! simultaneously passes through zero (the sign flip sits on the surface), so
//! the field stays continuous and a disagreement there can only be a tiny
//! last-place difference; the randomized sweep still rejects samples near each
//! branch or fold boundary to keep the comparison far from any such edge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` kernel for the four two-dimensional organic
/// signed-distance outlines, embedded inline so the twin ships as a single
/// source file. The single entry point `solve` mirrors the `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives` closed forms
/// `heart_2d`, `egg_2d`, `moon` and `cross_2d`; see the module documentation
/// for each algorithm.
const SDF_ORGANIC2D_WGSL: &str = r#"
// 2D organic signed-distance twin: one thread computes one query's heart, egg,
// moon and cross distances, mirroring the CPU golden
// `ray_scene::sdf_primitives::{heart_2d, egg_2d, moon, cross_2d}` with only
// abs, min, max, clamp, select and a final sqrt. The CSG/extrude operators stay
// on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无
// 第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Shared query point.
    px: f32,
    py: f32,
    // Egg: bottom-circle radius and top-cap radius.
    egg_ra: f32,
    egg_rb: f32,
    // Moon: centre offset and the two disk radii.
    moon_d: f32,
    moon_ra: f32,
    moon_rb: f32,
    // Cross: arm half-length, half-thickness and rounding radius.
    cross_arm: f32,
    cross_thickness: f32,
    cross_r: f32,
    pad0: f32,
    pad1: f32,
}

struct Distances {
    // Signed distance to the unit heart.
    heart: f32,
    // Signed distance to the three-arc egg.
    egg: f32,
    // Signed distance to the crescent moon.
    moon: f32,
    // Signed distance to the rounded cross.
    cross: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector.
fn length2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Rust `f32::signum` reimplemented for the branch sign: +1 for a non-negative
// argument, -1 otherwise. Where each shape uses this the magnitude passes
// through zero at the same place, so the choice at an exact zero never governs.
fn sign_pos(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Exact signed distance to the unit heart (tip at the origin, cusp at (0, 1)).
fn heart_2d(point: vec2<f32>) -> f32 {
    let r = 0.35355338;
    let x = abs(point.x);
    let y = point.y;
    if (y + x > 1.0) {
        let dx = x - 0.25;
        let dy = y - 0.75;
        return sqrt(dx * dx + dy * dy) - r;
    }
    let cx = x;
    let cy = y - 1.0;
    let d_cusp = cx * cx + cy * cy;
    let s = 0.5 * max(x + y, 0.0);
    let fx = x - s;
    let fy = y - s;
    let d_flank = fx * fx + fy * fy;
    return sqrt(min(d_cusp, d_flank)) * sign_pos(x - y);
}

// Exact signed distance to the three-arc egg with its axis on x = 0.
fn egg_2d(point: vec2<f32>, ra: f32, rb: f32) -> f32 {
    let k = 1.7320508;
    let px = abs(point.x);
    let py = point.y;
    let r = ra - rb;
    var d: f32;
    if (py < 0.0) {
        d = length2(vec2<f32>(px, py)) - r;
    } else if (k * (px + r) < py) {
        d = length2(vec2<f32>(px, py - k * r));
    } else {
        d = length2(vec2<f32>(px + r, py)) - 2.0 * r;
    }
    return d - rb;
}

// Exact signed distance to the crescent moon: the difference of two disks with
// an explicit cusp-tip branch.
fn moon_sdf(point: vec2<f32>, d: f32, ra: f32, rb: f32) -> f32 {
    let p = vec2<f32>(point.x, abs(point.y));
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = sqrt(max(ra * ra - a * a, 0.0));
    if (d * (p.x * b - p.y * a) > d * d * max(b - p.y, 0.0)) {
        return length2(vec2<f32>(p.x - a, p.y - b));
    }
    let outer = length2(p) - ra;
    let inner = length2(vec2<f32>(p.x - d, p.y)) - rb;
    return max(outer, -inner);
}

// Exact signed distance to the rounded cross (plus sign), folded to the octant
// x >= y >= 0.
fn cross_2d(point: vec2<f32>, arm: f32, thickness: f32, r: f32) -> f32 {
    var p = vec2<f32>(abs(point.x), abs(point.y));
    if (p.y > p.x) {
        p = vec2<f32>(p.y, p.x);
    }
    let q = vec2<f32>(p.x - arm, p.y - thickness);
    let k = max(q.x, q.y);
    var w: vec2<f32>;
    if (k > 0.0) {
        w = q;
    } else {
        w = vec2<f32>(thickness - p.x, -k);
    }
    return sign_pos(k) * length2(vec2<f32>(max(w.x, 0.0), max(w.y, 0.0))) - r;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let point = vec2<f32>(q.px, q.py);

    var out: Distances;
    out.heart = heart_2d(point);
    out.egg = egg_2d(point, q.egg_ra, q.egg_rb);
    out.moon = moon_sdf(point, q.moon_d, q.moon_ra, q.moon_rb);
    out.cross = cross_2d(point, q.cross_arm, q.cross_thickness, q.cross_r);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_ORGANIC2D_WGSL`].
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
/// the shared query point plus the per-shape parameters flattened to scalar
/// `f32` lanes, with two pad words to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Egg bottom-circle radius `ra`.
    egg_ra: f32,
    /// Egg top-cap radius `rb`.
    egg_rb: f32,
    /// Moon centre offset `d`.
    moon_d: f32,
    /// Moon outer-disk radius `ra`.
    moon_ra: f32,
    /// Moon inner-disk radius `rb`.
    moon_rb: f32,
    /// Cross arm half-length.
    cross_arm: f32,
    /// Cross half-thickness.
    cross_thickness: f32,
    /// Cross rounding radius.
    cross_r: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the four signed distances to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the unit heart.
    heart: f32,
    /// Signed distance to the three-arc egg.
    egg: f32,
    /// Signed distance to the crescent moon.
    moon: f32,
    /// Signed distance to the rounded cross.
    cross: f32,
}

/// One query for the two-dimensional organic distance twin: the shared query
/// point and the per-shape parameters.
///
/// `point` is the position whose distances are sought. `egg_ra`/`egg_rb` are the
/// egg's bottom-circle and top-cap radii (`egg_ra > egg_rb > 0`). `moon_d`,
/// `moon_ra` and `moon_rb` are the crescent's centre offset and the two disk
/// radii (an overlapping configuration `|ra - rb| < d < ra + rb`).
/// `cross_arm`, `cross_thickness` and `cross_r` are the cross's arm half-length,
/// half-thickness and rounding radius (`arm >= thickness`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfOrganic2dQuery {
    /// Shared query point.
    pub point: [f32; 2],
    /// Egg bottom-circle radius.
    pub egg_ra: f32,
    /// Egg top-cap radius.
    pub egg_rb: f32,
    /// Moon centre offset.
    pub moon_d: f32,
    /// Moon outer-disk radius.
    pub moon_ra: f32,
    /// Moon inner-disk radius.
    pub moon_rb: f32,
    /// Cross arm half-length.
    pub cross_arm: f32,
    /// Cross half-thickness.
    pub cross_thickness: f32,
    /// Cross rounding radius.
    pub cross_r: f32,
}

impl SdfOrganic2dQuery {
    /// Builds a query from the shared point and the per-shape parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens four shapes' parameters into one record"
    )]
    pub const fn new(
        point: [f32; 2],
        egg_ra: f32,
        egg_rb: f32,
        moon_d: f32,
        moon_ra: f32,
        moon_rb: f32,
        cross_arm: f32,
        cross_thickness: f32,
        cross_r: f32,
    ) -> SdfOrganic2dQuery {
        SdfOrganic2dQuery {
            point,
            egg_ra,
            egg_rb,
            moon_d,
            moon_ra,
            moon_rb,
            cross_arm,
            cross_thickness,
            cross_r,
        }
    }
}

/// One resolved query of the two-dimensional organic distance twin: the four
/// signed distances.
///
/// `heart` is
/// `prism_render_architecture::ray_scene::sdf_primitives::heart_2d`, `egg` is
/// `egg_2d`, `moon` is `moon` and `cross` is `cross_2d`; each is negative inside
/// the respective shape and positive outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfOrganic2dResult {
    /// Signed distance to the unit heart.
    pub heart: f32,
    /// Signed distance to the three-arc egg.
    pub egg: f32,
    /// Signed distance to the crescent moon.
    pub moon: f32,
    /// Signed distance to the rounded cross.
    pub cross: f32,
}

/// Encodes one [`SdfOrganic2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfOrganic2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        egg_ra: q.egg_ra,
        egg_rb: q.egg_rb,
        moon_d: q.moon_d,
        moon_ra: q.moon_ra,
        moon_rb: q.moon_rb,
        cross_arm: q.cross_arm,
        cross_thickness: q.cross_thickness,
        cross_r: q.cross_r,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfOrganic2dResult`].
fn decode_result(raw: &GpuResult) -> SdfOrganic2dResult {
    SdfOrganic2dResult {
        heart: raw.heart,
        egg: raw.egg,
        moon: raw.moon,
        cross: raw.cross,
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

/// A compiled, reusable two-dimensional organic distance compute pipeline,
/// twinning the `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives` outlines `heart_2d`,
/// `egg_2d`, `moon` and `cross_2d`.
pub struct GpuSdfOrganic2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfOrganic2d {
    /// Compiles the two-dimensional organic distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfOrganic2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_organic2d"),
            source: ShaderSource::Wgsl(SDF_ORGANIC2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfOrganic2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfOrganic2dResult`]
    /// per input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfOrganic2dQuery],
    ) -> Vec<SdfOrganic2dResult> {
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
            label: Some("prism_volumetric_sdf_organic2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_organic2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_organic2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_organic2d_pass"),
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
