//! `wgpu` compute twin of the minimum-enclosing-ball geometry primitives
//! ([`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere),
//! particle design §12, §13).
//!
//! The `CPU` golden
//! [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere)
//! owns the exact smallest-enclosing circle (2D) and sphere (3D) of a particle
//! position cloud. Its public entry points
//! [`min_enclosing_circle`](prism_render_architecture::particle::welzl_min_sphere::min_enclosing_circle)
//! and
//! [`min_enclosing_sphere`](prism_render_architecture::particle::welzl_min_sphere::min_enclosing_sphere)
//! run Emo Welzl's expected-linear-time move-to-front recursion: a fixed-seed
//! shuffle feeds an incremental walk that, whenever a point escapes the current
//! ball, rebuilds the ball with that point pinned to the boundary and recurses
//! over the earlier points with a dynamically growing support set. That driver
//! is inherently stateful (random shuffle, data-dependent recursion depth, a
//! mutable support set), so it stays on the host; it is **not** twinned here.
//!
//! # What is twinned (strict boundary)
//!
//! This twin reproduces only the golden's two *stateless* building blocks, the
//! pieces Welzl's recursion calls at each leaf:
//!
//! * The containment predicates [`Sphere2::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere2::contains)
//!   and [`Sphere3::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere3::contains):
//!   given a circle or sphere (`center`, `radius`) and a query point, decide
//!   whether the point lies inside or on the ball within the golden's tolerant
//!   slack. The answer is a `bool`, so the twin matches the reference bit for
//!   bit (an exact `==` on the classification), not within an epsilon.
//! * The closed-form "trivial" balls resting on a small support set: the
//!   diameter circle / sphere of two points, the circumscribed circle of three
//!   points (planar in 2D, in-plane in 3D), and the circumscribed sphere of
//!   four points. Each is solved by the same Cramer / determinant form the
//!   golden uses, including the degeneracy guard that reports `None` (surfaced
//!   here as a `Degenerate` result) when the defining determinant collapses on
//!   collinear or coplanar inputs.
//!
//! A passing real-device parity test is therefore direct evidence the ported
//! kernels evaluate the same containment classification and the same
//! circumcenter algebra, and trip the same degeneracy branch, that the host
//! driver relies on; the host remains the sole owner of the Welzl recursion
//! that stitches these leaves into a global minimum-enclosing ball.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /` and the native `dot` on `vec2`/`vec3` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan` or any optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. `sqrt` appears only to turn a
//! squared distance into a radius or a containment distance, exactly as the
//! reference does.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact for the continuous fields: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! center and radius, while the containment `bool` and the degeneracy flag are
//! pinned by exact equality. Fixtures stay far from every branch tie, so the
//! two devices classify every degeneracy and containment identically.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere);
//! no third-party engine source or derived code.

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

/// Operation code: the [`Sphere2::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere2::contains)
/// planar containment predicate.
const OP_CONTAINS_2: u32 = 0;
/// Operation code: the [`Sphere3::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere3::contains)
/// spatial containment predicate.
const OP_CONTAINS_3: u32 = 1;
/// Operation code: the diameter circle of two planar points.
const OP_CIRCLE_DIAMETER_2: u32 = 2;
/// Operation code: the circumscribed circle of three planar points.
const OP_CIRCUMCIRCLE_2: u32 = 3;
/// Operation code: the diameter sphere of two spatial points.
const OP_SPHERE_DIAMETER_3: u32 = 4;
/// Operation code: the circumscribed circle of three spatial points.
const OP_CIRCUMCIRCLE_3: u32 = 5;
/// Operation code: the circumscribed sphere of four spatial points.
const OP_CIRCUMSPHERE_4: u32 = 6;

/// The portable core-`WGSL` minimum-enclosing-ball primitive kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere)
/// building blocks op by op; see the module documentation for the boundary.
const WELZL_MIN_SPHERE_WGSL: &str = r#"
// Minimum-enclosing-ball primitive twin: one thread per query reproduces either
// a containment predicate or a closed-form circum construction. The stateful
// Welzl recursion (shuffle + data-dependent recursion + growing support set)
// is not twinned; it stays on the host. This kernel only mirrors the stateless
// leaves the recursion invokes.
//
// Provenance: twinned from this repository's particle::welzl_min_sphere; no
// third-party engine source or derived code.

// Relative epsilon deciding that a defining determinant has collapsed
// (collinear in 2D, coplanar in 3D). Matches the reference `DEGEN_REL`.
const DEGEN_REL: f32 = 1.0e-7;
// Absolute floor on the degeneracy threshold so a fully coincident support set
// is classified as degenerate rather than dividing by a hard zero. Matches the
// reference `DEGEN_ABS`.
const DEGEN_ABS: f32 = 1.0e-30;
// Absolute containment slack so a boundary point is never rejected by rounding.
// Matches the reference `CONTAIN_ABS`.
const CONTAIN_ABS: f32 = 1.0e-5;
// Radius-relative containment slack. Matches the reference `CONTAIN_REL`.
const CONTAIN_REL: f32 = 1.0e-6;

const OP_CONTAINS_2: u32 = 0u;
const OP_CONTAINS_3: u32 = 1u;
const OP_CIRCLE_DIAMETER_2: u32 = 2u;
const OP_CIRCUMCIRCLE_2: u32 = 3u;
const OP_SPHERE_DIAMETER_3: u32 = 4u;
const OP_CIRCUMCIRCLE_3: u32 = 5u;
const OP_CIRCUMSPHERE_4: u32 = 6u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation selector; one of the OP_* classification codes.
    op: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Support point a (2D uses the x, y lanes); a pad lane follows.
    a: vec3<f32>,
    pad_a: f32,
    // Support point b; a pad lane follows.
    b: vec3<f32>,
    pad_b: f32,
    // Support point c; a pad lane follows.
    c: vec3<f32>,
    pad_c: f32,
    // Support point d; a pad lane follows.
    d: vec3<f32>,
    pad_d: f32,
    // Containment ball center (2D uses the x, y lanes); radius in the w lane.
    center: vec3<f32>,
    radius: f32,
    // Containment query point (2D uses the x, y lanes); a pad lane follows.
    point: vec3<f32>,
    pad_p: f32,
}

struct Result {
    // Constructed circle / sphere center (2D uses the x, y lanes); radius in w.
    center: vec3<f32>,
    radius: f32,
    // Containment answer: 1 inside/on, 0 outside. Unused by construction ops.
    contains: u32,
    // Construction validity: 1 valid, 0 degenerate (collinear/coplanar). The
    // containment ops leave it at 1.
    valid: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Determinant of a 3x3 matrix given as three row vectors, mirroring the
// reference `det3x3`.
fn det3(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>) -> f32 {
    return r0.x * (r1.y * r2.z - r1.z * r2.y)
         - r0.y * (r1.x * r2.z - r1.z * r2.x)
         + r0.z * (r1.x * r2.y - r1.y * r2.x);
}

// The tolerant containment test shared by Sphere2 and Sphere3: the point is in
// or on the ball when its distance does not exceed the radius plus the golden's
// absolute and radius-relative slack. Mirrors `Sphere2::contains` /
// `Sphere3::contains` exactly (same sqrt form, same constants).
fn ball_contains(dist_sq: f32, radius: f32) -> bool {
    let d = sqrt(dist_sq);
    return d <= radius + CONTAIN_ABS + CONTAIN_REL * radius;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.center = vec3<f32>(0.0, 0.0, 0.0);
    out.radius = 0.0;
    out.contains = 0u;
    out.valid = 1u;
    out.pad0 = 0u;
    out.pad1 = 0u;

    if (q.op == OP_CONTAINS_2) {
        // Sphere2::contains: planar point-in-circle predicate.
        let dxy = q.point.xy - q.center.xy;
        out.contains = select(0u, 1u, ball_contains(dot(dxy, dxy), q.radius));
    } else if (q.op == OP_CONTAINS_3) {
        // Sphere3::contains: spatial point-in-sphere predicate.
        let d = q.point - q.center;
        out.contains = select(0u, 1u, ball_contains(dot(d, d), q.radius));
    } else if (q.op == OP_CIRCLE_DIAMETER_2) {
        // circle_diameter_2: midpoint center, half-length radius (planar).
        let a = q.a.xy;
        let b = q.b.xy;
        let center = (a + b) * 0.5;
        let ab = b - a;
        out.center = vec3<f32>(center, 0.0);
        out.radius = sqrt(dot(ab, ab)) * 0.5;
    } else if (q.op == OP_CIRCUMCIRCLE_2) {
        // circumcircle_2: planar circumscribed circle via Cramer's rule.
        let a = q.a.xy;
        let ab = q.b.xy - a;
        let ac = q.c.xy - a;
        let d11 = dot(ab, ab);
        let d12 = dot(ab, ac);
        let d22 = dot(ac, ac);
        let det = d11 * d22 - d12 * d12;
        if (abs(det) <= DEGEN_REL * d11 * d22 + DEGEN_ABS) {
            out.valid = 0u;
        } else {
            let r1 = d11 * 0.5;
            let r2 = d22 * 0.5;
            let alpha = (r1 * d22 - r2 * d12) / det;
            let beta = (d11 * r2 - d12 * r1) / det;
            let center = a + alpha * ab + beta * ac;
            let rc = center - a;
            out.center = vec3<f32>(center, 0.0);
            out.radius = sqrt(dot(rc, rc));
        }
    } else if (q.op == OP_SPHERE_DIAMETER_3) {
        // sphere_diameter_3: midpoint center, half-length radius (spatial).
        let center = (q.a + q.b) * 0.5;
        let ab = q.b - q.a;
        out.center = center;
        out.radius = sqrt(dot(ab, ab)) * 0.5;
    } else if (q.op == OP_CIRCUMCIRCLE_3) {
        // circumcircle_3: spatial triangle circumscribed circle (in-plane).
        let a = q.a;
        let ab = q.b - a;
        let ac = q.c - a;
        let d11 = dot(ab, ab);
        let d12 = dot(ab, ac);
        let d22 = dot(ac, ac);
        let det = d11 * d22 - d12 * d12;
        if (abs(det) <= DEGEN_REL * d11 * d22 + DEGEN_ABS) {
            out.valid = 0u;
        } else {
            let r1 = d11 * 0.5;
            let r2 = d22 * 0.5;
            let alpha = (r1 * d22 - r2 * d12) / det;
            let beta = (d11 * r2 - d12 * r1) / det;
            let center = a + alpha * ab + beta * ac;
            let rc = center - a;
            out.center = center;
            out.radius = sqrt(dot(rc, rc));
        }
    } else {
        // OP_CIRCUMSPHERE_4: tetrahedron circumscribed sphere via Cramer's rule.
        let a = q.a;
        let ab = q.b - a;
        let ac = q.c - a;
        let ad = q.d - a;
        let rb = dot(ab, ab) * 0.5;
        let rc = dot(ac, ac) * 0.5;
        let rd = dot(ad, ad) * 0.5;
        let det = det3(ab, ac, ad);
        let na = sqrt(dot(ab, ab));
        let nb = sqrt(dot(ac, ac));
        let nc = sqrt(dot(ad, ad));
        if (abs(det) <= DEGEN_REL * na * nb * nc + DEGEN_ABS) {
            out.valid = 0u;
        } else {
            let ux = det3(
                vec3<f32>(rb, ab.y, ab.z),
                vec3<f32>(rc, ac.y, ac.z),
                vec3<f32>(rd, ad.y, ad.z),
            ) / det;
            let uy = det3(
                vec3<f32>(ab.x, rb, ab.z),
                vec3<f32>(ac.x, rc, ac.z),
                vec3<f32>(ad.x, rd, ad.z),
            ) / det;
            let uz = det3(
                vec3<f32>(ab.x, ab.y, rb),
                vec3<f32>(ac.x, ac.y, rc),
                vec3<f32>(ad.x, ad.y, rd),
            ) / det;
            let center = a + vec3<f32>(ux, uy, uz);
            let rr = center - a;
            out.center = center;
            out.radius = sqrt(dot(rr, rr));
        }
    }

    results[idx] = out;
}
"#;

/// One minimum-enclosing-ball primitive query: either a containment predicate
/// against a given ball, or a closed-form circum construction from a small
/// support set. The stateful Welzl recursion that chooses the support set stays
/// on the host; this twin only reproduces the individual leaves.
///
/// Provenance: twinned from this repository's
/// [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WelzlMinSphereQuery {
    /// Planar point-in-circle predicate mirroring
    /// [`Sphere2::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere2::contains):
    /// is `point` inside or on the circle `center`/`radius`?
    Contains2 {
        /// The circle center `[x, y]`.
        center: [f32; 2],
        /// The circle radius.
        radius: f32,
        /// The query point `[x, y]`.
        point: [f32; 2],
    },
    /// Spatial point-in-sphere predicate mirroring
    /// [`Sphere3::contains`](prism_render_architecture::particle::welzl_min_sphere::Sphere3::contains):
    /// is `point` inside or on the sphere `center`/`radius`?
    Contains3 {
        /// The sphere center `[x, y, z]`.
        center: [f32; 3],
        /// The sphere radius.
        radius: f32,
        /// The query point `[x, y, z]`.
        point: [f32; 3],
    },
    /// The circle whose diameter is the segment `a`-`b` (planar).
    CircleDiameter2 {
        /// First diameter endpoint `[x, y]`.
        a: [f32; 2],
        /// Second diameter endpoint `[x, y]`.
        b: [f32; 2],
    },
    /// The circumscribed circle of the planar triangle `a`, `b`, `c`.
    Circumcircle2 {
        /// First triangle vertex `[x, y]`.
        a: [f32; 2],
        /// Second triangle vertex `[x, y]`.
        b: [f32; 2],
        /// Third triangle vertex `[x, y]`.
        c: [f32; 2],
    },
    /// The sphere whose diameter is the segment `a`-`b` (spatial).
    SphereDiameter3 {
        /// First diameter endpoint `[x, y, z]`.
        a: [f32; 3],
        /// Second diameter endpoint `[x, y, z]`.
        b: [f32; 3],
    },
    /// The circumscribed circle of the spatial triangle `a`, `b`, `c` (its
    /// center lies in the triangle plane).
    Circumcircle3 {
        /// First triangle vertex `[x, y, z]`.
        a: [f32; 3],
        /// Second triangle vertex `[x, y, z]`.
        b: [f32; 3],
        /// Third triangle vertex `[x, y, z]`.
        c: [f32; 3],
    },
    /// The circumscribed sphere of the tetrahedron `a`, `b`, `c`, `d`.
    Circumsphere4 {
        /// First tetrahedron vertex `[x, y, z]`.
        a: [f32; 3],
        /// Second tetrahedron vertex `[x, y, z]`.
        b: [f32; 3],
        /// Third tetrahedron vertex `[x, y, z]`.
        c: [f32; 3],
        /// Fourth tetrahedron vertex `[x, y, z]`.
        d: [f32; 3],
    },
}

/// The resolved answer for one [`WelzlMinSphereQuery`].
///
/// Provenance: twinned from this repository's
/// [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WelzlMinSphereResult {
    /// Containment answer for a `Contains2` / `Contains3` query: `true` when the
    /// point lies inside or on the ball within the golden's tolerant slack.
    Contains(bool),
    /// A constructed planar circle: `center` `[x, y]` and non-negative
    /// `radius`.
    Circle {
        /// The circle center `[x, y]`.
        center: [f32; 2],
        /// The circle radius.
        radius: f32,
    },
    /// A constructed spatial ball: `center` `[x, y, z]` and non-negative
    /// `radius`.
    Ball {
        /// The ball center `[x, y, z]`.
        center: [f32; 3],
        /// The ball radius.
        radius: f32,
    },
    /// The construction's defining determinant collapsed (collinear triangle or
    /// coplanar tetrahedron), mirroring the golden returning `None`.
    Degenerate,
}

/// `repr(C)` `std430` image of one packed query: an op word plus three pads,
/// four `vec4` support-point slots `(a, pad)`, `(b, pad)`, `(c, pad)`,
/// `(d, pad)`, a `(center.xyz, radius)` slot and a `(point.xyz, pad)` slot —
/// `112` bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation selector; one of the `OP_*` codes.
    op: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Support point `a`.
    a: [f32; 3],
    /// Padding lane after `a`.
    pad_a: f32,
    /// Support point `b`.
    b: [f32; 3],
    /// Padding lane after `b`.
    pad_b: f32,
    /// Support point `c`.
    c: [f32; 3],
    /// Padding lane after `c`.
    pad_c: f32,
    /// Support point `d`.
    d: [f32; 3],
    /// Padding lane after `d`.
    pad_d: f32,
    /// Containment ball center.
    center: [f32; 3],
    /// Containment ball radius.
    radius: f32,
    /// Containment query point.
    point: [f32; 3],
    /// Padding lane after the query point.
    pad_p: f32,
}

impl GpuQuery {
    /// Packs one [`WelzlMinSphereQuery`] into its `std430` image, zero-filling
    /// the lanes an operation does not use.
    fn new(query: &WelzlMinSphereQuery) -> GpuQuery {
        let mut g = GpuQuery::zeroed();
        match *query {
            WelzlMinSphereQuery::Contains2 {
                center,
                radius,
                point,
            } => {
                g.op = OP_CONTAINS_2;
                g.center = [center[0], center[1], 0.0];
                g.radius = radius;
                g.point = [point[0], point[1], 0.0];
            }
            WelzlMinSphereQuery::Contains3 {
                center,
                radius,
                point,
            } => {
                g.op = OP_CONTAINS_3;
                g.center = center;
                g.radius = radius;
                g.point = point;
            }
            WelzlMinSphereQuery::CircleDiameter2 { a, b } => {
                g.op = OP_CIRCLE_DIAMETER_2;
                g.a = [a[0], a[1], 0.0];
                g.b = [b[0], b[1], 0.0];
            }
            WelzlMinSphereQuery::Circumcircle2 { a, b, c } => {
                g.op = OP_CIRCUMCIRCLE_2;
                g.a = [a[0], a[1], 0.0];
                g.b = [b[0], b[1], 0.0];
                g.c = [c[0], c[1], 0.0];
            }
            WelzlMinSphereQuery::SphereDiameter3 { a, b } => {
                g.op = OP_SPHERE_DIAMETER_3;
                g.a = a;
                g.b = b;
            }
            WelzlMinSphereQuery::Circumcircle3 { a, b, c } => {
                g.op = OP_CIRCUMCIRCLE_3;
                g.a = a;
                g.b = b;
                g.c = c;
            }
            WelzlMinSphereQuery::Circumsphere4 { a, b, c, d } => {
                g.op = OP_CIRCUMSPHERE_4;
                g.a = a;
                g.b = b;
                g.c = c;
                g.d = d;
            }
        }
        g
    }
}

/// `repr(C)` `std430` image of one result: a `(center.xyz, radius)` slot
/// followed by the `contains` and `valid` flags with two pad words — `32`
/// bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Constructed ball center (planar results ignore the `z` lane).
    center: [f32; 3],
    /// Constructed ball radius.
    radius: f32,
    /// Containment answer: `1` inside/on, `0` outside.
    contains: u32,
    /// Construction validity: `1` valid, `0` degenerate.
    valid: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable minimum-enclosing-ball primitive compute pipeline.
///
/// Provenance: twinned from this repository's
/// [`welzl_min_sphere`](prism_render_architecture::particle::welzl_min_sphere);
/// no third-party engine source or derived code.
pub struct GpuWelzlMinSphere {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWelzlMinSphere {
    /// Compiles the minimum-enclosing-ball primitive kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWelzlMinSphere {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere"),
            source: ShaderSource::Wgsl(WELZL_MIN_SPHERE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWelzlMinSphere {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`WelzlMinSphereResult`] per
    /// input, in order.
    ///
    /// Containment queries return [`WelzlMinSphereResult::Contains`] matching
    /// the reference predicate exactly; construction queries return
    /// [`WelzlMinSphereResult::Circle`] / [`WelzlMinSphereResult::Ball`] within
    /// the documented tolerance, or [`WelzlMinSphereResult::Degenerate`] when
    /// the defining determinant collapses. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[WelzlMinSphereQuery],
    ) -> Vec<WelzlMinSphereResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_output"),
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
            label: Some("prism_volumetric_welzl_min_sphere_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_bind_group"),
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
            label: Some("prism_volumetric_welzl_min_sphere_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_welzl_min_sphere_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_welzl_min_sphere_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WelzlMinSphereResult`],
/// using the originating `query` to pick the right variant (containment,
/// planar circle, spatial ball, or a degenerate construction).
fn decode_result(query: &WelzlMinSphereQuery, raw: &GpuResult) -> WelzlMinSphereResult {
    match *query {
        WelzlMinSphereQuery::Contains2 { .. } | WelzlMinSphereQuery::Contains3 { .. } => {
            WelzlMinSphereResult::Contains(raw.contains == 1)
        }
        WelzlMinSphereQuery::CircleDiameter2 { .. } | WelzlMinSphereQuery::Circumcircle2 { .. } => {
            if raw.valid == 1 {
                WelzlMinSphereResult::Circle {
                    center: [raw.center[0], raw.center[1]],
                    radius: raw.radius,
                }
            } else {
                WelzlMinSphereResult::Degenerate
            }
        }
        WelzlMinSphereQuery::SphereDiameter3 { .. }
        | WelzlMinSphereQuery::Circumcircle3 { .. }
        | WelzlMinSphereQuery::Circumsphere4 { .. } => {
            if raw.valid == 1 {
                WelzlMinSphereResult::Ball {
                    center: raw.center,
                    radius: raw.radius,
                }
            } else {
                WelzlMinSphereResult::Degenerate
            }
        }
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
