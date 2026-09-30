//! Cubic-Bézier round-curve primitive and its single-level `BVH`.
//!
//! Hair, fur, and grass are the canonical *curve* workloads: millions of thin
//! strands where tessellating each into ribbons of triangles is wasteful and
//! aliasing-prone. Production path tracers instead carry a native *curve*
//! primitive — a cubic Bézier spine swept by a linearly varying width — and
//! intersect it directly. On hardware ray tracing the curve rides the
//! DXR/Vulkan *procedural-primitive* path (an [`Aabb`]-bounded `BLAS` entry plus
//! an intersection shader); this module is the `CPU` golden reference for that
//! path.
//!
//! The intersection follows the recursive-refinement scheme of `pbrt`
//! (Pharr, Jakob & Humphreys): transform the four control points into a
//! ray-aligned frame (ray origin at the origin, ray direction along `+z`), then
//! recursively subdivide the Bézier with de Casteljau blossoming until the
//! sub-segment is nearly linear, and test that segment as a swept circle —
//! rejecting by the ray-frame `x`/`y`/`z` bounds at every level so most of the
//! curve is culled cheaply. The refinement depth is derived from the segment's
//! second-difference flatness so a straight strand tests immediately while a
//! tightly curled one subdivides more.
//!
//! Every step is add/sub/mul/div, `min`/`max`, `abs`, and `sqrt` — no
//! transcendental call — so the arithmetic is bit-reproducible on the `GPU`.
//! A [`CurveBvh`] reuses the shared binned-`SAH` [`build_linear_bvh`] over each
//! curve's conservative [`Aabb`] and the same ordered slab walk the triangle
//! [`super::bvh::Bvh`] and analytic [`super::sphere::SphereBvh`] use, so every
//! primitive kind shares one acceleration-structure contract.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// Hard cap on de Casteljau refinement depth.
///
/// A cubic segment halves its arc every level, so ten levels resolve a strand
/// to `1/1024` of its length — far below any width a renderer draws — while
/// bounding the recursion (and the mirrored `GPU` stack) to a fixed size.
const MAX_REFINEMENT_DEPTH: u32 = 10;

/// A cubic-Bézier round-curve primitive in world space.
///
/// The spine is the cubic Bézier through the four `control` points; the swept
/// radius is half of a width that interpolates linearly from `width_start` at
/// the spine's start (`u = 0`) to `width_end` at its end (`u = 1`). `primitive`
/// is the caller's stable id (mirroring [`super::bvh::Triangle`] and
/// [`super::sphere::Sphere`]): the [`CurveBvh`] builder reorders curves
/// internally but always reports hits by this id so downstream shading can look
/// up material/attributes. Widths are stored non-negative; a caller-supplied
/// negative width is folded to its magnitude so the derived [`Aabb`] stays well
/// formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    /// The four cubic-Bézier control points (`p0`, `p1`, `p2`, `p3`).
    control: [[f32; 3]; 4],
    /// Non-negative full width at the spine start (`u = 0`).
    width_start: f32,
    /// Non-negative full width at the spine end (`u = 1`).
    width_end: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Curve {
    /// Builds a curve from four `control` points with the given start/end widths
    /// (folded to their magnitudes) and stable id `primitive`.
    #[must_use]
    pub fn new(control: [[f32; 3]; 4], width_start: f32, width_end: f32, primitive: u32) -> Self {
        Self {
            control,
            width_start: width_start.abs(),
            width_end: width_end.abs(),
            primitive,
        }
    }

    /// The four cubic-Bézier control points.
    #[must_use]
    pub fn control(&self) -> [[f32; 3]; 4] {
        self.control
    }

    /// Non-negative full width at the spine start (`u = 0`).
    #[must_use]
    pub fn width_start(&self) -> f32 {
        self.width_start
    }

    /// Non-negative full width at the spine end (`u = 1`).
    #[must_use]
    pub fn width_end(&self) -> f32 {
        self.width_end
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Full width at spine parameter `u` (linear interpolation of the endpoints).
    #[must_use]
    fn width_at(&self, u: f32) -> f32 {
        lerp_f(u, self.width_start, self.width_end)
    }

    /// Conservative axis-aligned bounds: the control-point hull expanded by the
    /// larger half-width on every axis.
    ///
    /// The Bézier spine is contained in the convex hull of its control points,
    /// so padding that hull by `max(width_start, width_end) / 2` bounds the
    /// swept surface. This is the procedural-primitive `AABB` the hardware
    /// `BLAS` stores per curve.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut min = self.control[0];
        let mut max = self.control[0];
        for p in self.control.iter().skip(1) {
            min = [min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])];
            max = [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])];
        }
        let pad = 0.5 * self.width_start.max(self.width_end);
        Aabb::new(
            [min[0] - pad, min[1] - pad, min[2] - pad],
            [max[0] + pad, max[1] + pad, max[2] + pad],
        )
    }

    /// Nearest ray/curve intersection inside `ray`'s `[t_min, t_max]` interval,
    /// or `None` when the ray misses or the curve has zero width.
    ///
    /// The reported [`CurveHit::t`] is the parameter at the spine point whose
    /// projection is closest to the ray (curves are thin, so — like `pbrt` — the
    /// small radial depth offset of the swept surface is ignored), and
    /// [`CurveHit::u`] is the spine parameter there. [`CurveHit::normal`] is the
    /// unit surface normal in the plane perpendicular to the spine tangent,
    /// oriented against the incident ray, and [`CurveHit::front_face`] is `true`
    /// when the ray struck the outward side.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<CurveHit> {
        let max_width = self.width_start.max(self.width_end);
        if max_width <= 0.0 {
            // A zero-width curve has no surface to strike.
            return None;
        }
        let direction = ray.direction();
        let dir_len2 = dot(direction, direction);
        if dir_len2 <= 0.0 {
            return None;
        }
        let dir_len = dir_len2.sqrt();
        let w_axis = scale(direction, 1.0 / dir_len);
        // Any vector not near-parallel to the ray yields a stable perpendicular
        // basis via two cross products; picking the axis by the ray's smallest
        // component keeps the first cross well away from degeneracy.
        let helper = if w_axis[0].abs() > 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let u_axis = normalize(cross(helper, w_axis));
        let v_axis = cross(w_axis, u_axis);
        let origin = ray.origin();
        // Control points in the ray-aligned frame: (x, y) is the offset from the
        // ray line, z is the signed distance along the unit ray direction.
        let mut ray_cp = [[0.0f32; 3]; 4];
        for (out, p) in ray_cp.iter_mut().zip(self.control.iter()) {
            let rel = sub(*p, origin);
            *out = [dot(rel, u_axis), dot(rel, v_axis), dot(rel, w_axis)];
        }

        let ctx = SegmentCtx {
            curve: self,
            ray,
            w_axis,
            dir_len,
            max_depth: refinement_depth(&ray_cp, max_width),
        };
        intersect_recursive(&ctx, &ray_cp, 0.0, 1.0, 0, ray.t_max())
    }
}

/// A ray/curve intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Spine parameter `u ∈ [0, 1]` at the intersection.
    pub u: f32,
    /// Stable id of the curve that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck.
    pub front_face: bool,
}

/// Shared, per-ray context threaded through the recursive refinement so the
/// recursion itself stays within a small argument count.
struct SegmentCtx<'a> {
    /// The curve being intersected (source of world control points and widths).
    curve: &'a Curve,
    /// The incident ray (source of `t_min`, origin, and direction).
    ray: &'a Ray,
    /// Unit ray direction (the `+z` axis of the ray-aligned frame).
    w_axis: [f32; 3],
    /// Length of the ray direction, converting ray-frame `z` to a ray `t`.
    dir_len: f32,
    /// De Casteljau refinement depth chosen from the segment's flatness.
    max_depth: u32,
}

/// Recursively refines the ray-frame Bézier `ray_cp` (spanning spine parameters
/// `[u0, u1]`) and returns its nearest hit no farther than `t_max`.
///
/// Each level first culls by the ray-frame bounding box (padded by the local
/// half-width): if the box cannot contain the ray line (`x = y = 0`) or lies
/// outside the current `z` (depth) interval, the whole sub-curve is skipped. At
/// `max_depth` the sub-curve is tested as a single swept-circle segment;
/// otherwise it is split at its midpoint and both halves are refined, with the
/// near interval shrunk by any hit so the result is the nearest.
fn intersect_recursive(
    ctx: &SegmentCtx,
    ray_cp: &[[f32; 3]; 4],
    u0: f32,
    u1: f32,
    depth: u32,
    t_max: f32,
) -> Option<CurveHit> {
    let half_width = 0.5 * ctx.curve.width_at(u0).max(ctx.curve.width_at(u1));
    // Ray-frame bounding box of the four control points.
    let mut min = ray_cp[0];
    let mut max = ray_cp[0];
    for p in ray_cp.iter().skip(1) {
        min = [min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])];
        max = [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])];
    }
    // The ray line is the z axis (x = y = 0): cull if the padded box misses it.
    if max[0] + half_width < 0.0 || min[0] - half_width > 0.0 {
        return None;
    }
    if max[1] + half_width < 0.0 || min[1] - half_width > 0.0 {
        return None;
    }
    // Valid depth interval along the unit ray, from t_min to the current t_max.
    let z_min = ctx.ray.t_min() * ctx.dir_len;
    let z_max = t_max * ctx.dir_len;
    if max[2] + half_width < z_min || min[2] - half_width > z_max {
        return None;
    }

    if depth >= ctx.max_depth {
        return test_segment(ctx, ray_cp, u0, u1, t_max);
    }

    let (left, right) = subdivide(ray_cp);
    let u_mid = 0.5 * (u0 + u1);
    let near = intersect_recursive(ctx, &left, u0, u_mid, depth + 1, t_max);
    // Shrink the far interval so the second half only reports a closer hit.
    let far_t_max = near.map_or(t_max, |hit| hit.t);
    let far = intersect_recursive(ctx, &right, u_mid, u1, depth + 1, far_t_max);
    far.or(near)
}

/// Tests a nearly linear ray-frame sub-curve `ray_cp` (spanning spine
/// parameters `[u0, u1]`) as a single swept circle and returns its hit if the
/// ray line passes within the local half-width and inside `[t_min, t_max]`.
fn test_segment(
    ctx: &SegmentCtx,
    ray_cp: &[[f32; 3]; 4],
    u0: f32,
    u1: f32,
    t_max: f32,
) -> Option<CurveHit> {
    // Reject when the ray line projects before the start or past the end of the
    // segment (the two tangent half-planes at the endpoints), giving flat caps.
    let start_edge =
        (ray_cp[1][1] - ray_cp[0][1]) * -ray_cp[0][1] + ray_cp[0][0] * (ray_cp[0][0] - ray_cp[1][0]);
    if start_edge < 0.0 {
        return None;
    }
    let end_edge =
        (ray_cp[2][1] - ray_cp[3][1]) * -ray_cp[3][1] + ray_cp[3][0] * (ray_cp[3][0] - ray_cp[2][0]);
    if end_edge < 0.0 {
        return None;
    }

    // Parameter of the ray line's projection onto the chord (cp0 -> cp3), in xy.
    let seg = [ray_cp[3][0] - ray_cp[0][0], ray_cp[3][1] - ray_cp[0][1]];
    let denom = seg[0] * seg[0] + seg[1] * seg[1];
    if denom <= 0.0 {
        return None;
    }
    let raw_w = (-ray_cp[0][0] * seg[0] - ray_cp[0][1] * seg[1]) / denom;
    let w_param = raw_w.clamp(0.0, 1.0);

    // Global spine parameter and the point/derivative there, in the ray frame.
    let u = lerp_f(w_param, u0, u1);
    let (pc, _) = eval_bezier(ray_cp, w_param);
    let hit_width = ctx.curve.width_at(u);
    let radius = 0.5 * hit_width;
    let dist2 = pc[0] * pc[0] + pc[1] * pc[1];
    if dist2 > radius * radius {
        return None;
    }
    // Curve depth (its axis) is the reported hit; thin curves ignore the radial
    // depth offset, matching the reference intersector.
    let t = pc[2] / ctx.dir_len;
    if t < ctx.ray.t_min() || t > t_max {
        return None;
    }

    Some(build_hit(ctx, u, t))
}

/// Builds the world-space [`CurveHit`] at spine parameter `u` and ray parameter
/// `t`: the normal is the ray/curve offset projected perpendicular to the world
/// spine tangent, oriented against the incident ray.
fn build_hit(ctx: &SegmentCtx, u: f32, t: f32) -> CurveHit {
    let (center, tangent) = eval_bezier(&ctx.curve.control, u);
    let hit_point = ctx.ray.at(t);
    let offset = sub(hit_point, center);
    let tan_len2 = dot(tangent, tangent);
    let radial = if tan_len2 > 0.0 {
        sub(offset, scale(tangent, dot(offset, tangent) / tan_len2))
    } else {
        offset
    };
    let radial_len2 = dot(radial, radial);
    let mut normal = if radial_len2 > 0.0 {
        scale(radial, 1.0 / radial_len2.sqrt())
    } else {
        // Hit sits on the spine (dead-centre): face the incident ray.
        [-ctx.w_axis[0], -ctx.w_axis[1], -ctx.w_axis[2]]
    };
    let front_face = dot(ctx.ray.direction(), normal) < 0.0;
    if !front_face {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    CurveHit {
        t,
        u,
        primitive: ctx.curve.primitive,
        normal,
        front_face,
    }
}

/// Chooses a de Casteljau refinement depth from a ray-frame segment's flatness.
///
/// The larger the segment's second differences (its deviation from a straight
/// line) relative to a small fraction of the width, the more subdivisions are
/// needed before the linear swept-circle test is accurate. The depth is the
/// smallest `r` with `4^r ≥ √2·6·L / (8·eps)`, clamped to
/// [`MAX_REFINEMENT_DEPTH`], computed by repeated multiplication so no `log` is
/// used.
fn refinement_depth(ray_cp: &[[f32; 3]; 4], max_width: f32) -> u32 {
    let d0 = second_difference(ray_cp[0], ray_cp[1], ray_cp[2]);
    let d1 = second_difference(ray_cp[1], ray_cp[2], ray_cp[3]);
    let l = d0[0]
        .abs()
        .max(d0[1].abs())
        .max(d0[2].abs())
        .max(d1[0].abs())
        .max(d1[1].abs())
        .max(d1[2].abs());
    let eps = max_width * 0.05;
    if eps <= 0.0 {
        return MAX_REFINEMENT_DEPTH;
    }
    // √2·6/8 = 0.75·√2; computed at runtime to avoid an approximate constant.
    let k = 0.75 * 2.0_f32.sqrt() * l / eps;
    let mut depth = 0u32;
    let mut acc = 1.0f32;
    while acc < k && depth < MAX_REFINEMENT_DEPTH {
        acc *= 4.0;
        depth += 1;
    }
    depth
}

/// Componentwise second difference `a - 2·b + c` (curvature proxy).
fn second_difference(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    [
        a[0] - 2.0 * b[0] + c[0],
        a[1] - 2.0 * b[1] + c[1],
        a[2] - 2.0 * b[2] + c[2],
    ]
}

/// Splits a cubic Bézier at its midpoint into two cubic halves (exact de
/// Casteljau blossoming, so the two halves reproduce the original curve).
fn subdivide(cp: &[[f32; 3]; 4]) -> ([[f32; 3]; 4], [[f32; 3]; 4]) {
    let l1 = midpoint(cp[0], cp[1]);
    let m = midpoint(cp[1], cp[2]);
    let r2 = midpoint(cp[2], cp[3]);
    let l2 = midpoint(l1, m);
    let r1 = midpoint(m, r2);
    let mid = midpoint(l2, r1);
    ([cp[0], l1, l2, mid], [mid, r1, r2, cp[3]])
}

/// Evaluates a cubic Bézier at parameter `w ∈ [0, 1]`, returning the point and
/// its (unnormalized) derivative via de Casteljau.
fn eval_bezier(cp: &[[f32; 3]; 4], w: f32) -> ([f32; 3], [f32; 3]) {
    let a0 = lerp3(cp[0], cp[1], w);
    let a1 = lerp3(cp[1], cp[2], w);
    let a2 = lerp3(cp[2], cp[3], w);
    let b0 = lerp3(a0, a1, w);
    let b1 = lerp3(a1, a2, w);
    let point = lerp3(b0, b1, w);
    let deriv = scale(sub(b1, b0), 3.0);
    (point, deriv)
}

/// Midpoint of two 3-vectors.
fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        0.5 * (a[0] + b[0]),
        0.5 * (a[1] + b[1]),
        0.5 * (a[2] + b[2]),
    ]
}

/// Componentwise linear interpolation `a + t·(b - a)`.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Scalar linear interpolation `a + t·(b - a)`.
fn lerp_f(t: f32, a: f32, b: f32) -> f32 {
    a + t * (b - a)
}

/// Difference of two 3-vectors.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales a 3-vector by a scalar.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes a 3-vector; a zero-length input is returned unchanged.
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len2 = dot(a, a);
    if len2 > 0.0 {
        scale(a, 1.0 / len2.sqrt())
    } else {
        a
    }
}

/// A single-level `BVH` over cubic-Bézier [`Curve`] primitives.
///
/// Empty input yields an empty hierarchy ([`CurveBvh::is_empty`]); traversal of
/// an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`] and analytic
/// [`super::sphere::SphereBvh`] so every primitive kind shares one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Curves reordered so each leaf owns a contiguous slice.
    curves: Vec<Curve>,
}

impl CurveBvh {
    /// Builds a `BVH` over `curves` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(curves: &[Curve]) -> Self {
        Self::build_with(curves, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `curves` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each curve's [`Curve::aabb`] and then reorders the
    /// curves by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`CurveBvh::curves`].
    #[must_use]
    pub fn build_with(curves: &[Curve], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = curves.iter().map(Curve::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let curves = order.iter().map(|&i| curves[i as usize]).collect();
        Self { nodes, curves }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of curves in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.curves.len()
    }

    /// True when the hierarchy holds no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |node| node.bounds)
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// The reordered curve array (leaf slices index into this).
    #[must_use]
    pub fn curves(&self) -> &[Curve] {
        &self.curves
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<CurveHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<CurveHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for curve in &self.curves[start..end] {
                        if let Some(hit) = curve.intersect(&ray) {
                            // Tighten the interval so far subtrees are pruned.
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = node.second_child;
                    let neg = ray.direction()[node.axis as usize] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* curve intersects `ray` inside its interval.
    ///
    /// Returns on the first hit without tracking the nearest, so it is the cheap
    /// query for shadow and ambient-occlusion rays.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for curve in &self.curves[start..end] {
                        if curve.intersect(ray).is_some() {
                            return true;
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = node.second_child;
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// Pops the top node index off the traversal stack, or `None` when empty.
fn stack_pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /// A straight, axis-aligned strand at `z = -5` with constant width.
    fn straight_curve(primitive: u32) -> Curve {
        Curve::new(
            [
                [-1.0, 0.0, -5.0],
                [-0.3333, 0.0, -5.0],
                [0.3333, 0.0, -5.0],
                [1.0, 0.0, -5.0],
            ],
            0.5,
            0.5,
            primitive,
        )
    }

    #[test]
    fn straight_curve_centre_hit_has_expected_depth() {
        let curve = straight_curve(7);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let hit = curve.intersect(&ray).expect("ray must hit the strand");
        assert_eq!(hit.primitive, 7);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(approx(hit.u, 0.5, 1e-3), "u = {}", hit.u);
        assert!(hit.front_face);
        // Normal faces back toward the ray (+z) for a curve in the xz plane.
        assert!(approx(hit.normal[2], 1.0, 1e-3), "n = {:?}", hit.normal);
    }

    #[test]
    fn ray_outside_width_misses() {
        let curve = straight_curve(0);
        // Offset in y by more than the half-width (0.25).
        let ray = Ray::infinite([0.0, 0.4, 0.0], [0.0, 0.0, -1.0]);
        assert!(curve.intersect(&ray).is_none());
    }

    #[test]
    fn ray_just_inside_width_hits() {
        let curve = straight_curve(3);
        // Offset in y by less than the half-width (0.25).
        let ray = Ray::infinite([0.0, 0.2, 0.0], [0.0, 0.0, -1.0]);
        assert!(curve.intersect(&ray).is_some());
    }

    #[test]
    fn zero_width_curve_never_hits() {
        let curve = Curve::new(
            [
                [-1.0, 0.0, -5.0],
                [-0.3, 0.0, -5.0],
                [0.3, 0.0, -5.0],
                [1.0, 0.0, -5.0],
            ],
            0.0,
            0.0,
            1,
        );
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(curve.intersect(&ray).is_none());
    }

    #[test]
    fn curve_behind_origin_is_missed() {
        let curve = straight_curve(0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(curve.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let curve = straight_curve(0);
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 4.0);
        assert!(curve.intersect(&ray).is_none());
    }

    /// A random curled cubic strand near `z = -5`, in front of the origin.
    fn random_curve(rng: &mut Rng, primitive: u32) -> Curve {
        let base = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-8.0, -3.0)];
        let mut control = [[0.0f32; 3]; 4];
        for (i, cp) in control.iter_mut().enumerate() {
            let s = i as f32 / 3.0;
            *cp = [
                base[0] + rng.range(-1.5, 1.5) + s * rng.range(-1.0, 1.0),
                base[1] + rng.range(-1.5, 1.5) + s * rng.range(-1.0, 1.0),
                base[2] + rng.range(-0.5, 0.5),
            ];
        }
        let w0 = rng.range(0.05, 0.6);
        let w1 = rng.range(0.05, 0.6);
        Curve::new(control, w0, w1, primitive)
    }

    /// Brute-force nearest hit over the *original* (unordered) curve list, used
    /// as the ground truth the `BVH` must reproduce bit-for-bit.
    fn brute_closest(curves: &[Curve], ray: &Ray) -> Option<CurveHit> {
        let mut best: Option<CurveHit> = None;
        let mut ray = *ray;
        for curve in curves {
            if let Some(hit) = curve.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0c00_be00_1234_5678u64);
        let curves: Vec<Curve> = (0..200).map(|i| random_curve(&mut rng, i)).collect();
        let bvh = CurveBvh::build(&curves);
        assert_eq!(bvh.primitive_count(), curves.len());

        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(0.0, 2.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, -0.2)];
            let ray = Ray::infinite(origin, dir);
            let bvh_hit = bvh.closest_hit(&ray);
            let brute_hit = brute_closest(&curves, &ray);
            match (bvh_hit, brute_hit) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits(), "t mismatch");
                    assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                    assert_eq!(a.u.to_bits(), b.u.to_bits(), "u mismatch");
                    hits += 1;
                }
                (a, b) => panic!("hit/miss disagreement: {a:?} vs {b:?}"),
            }
        }
        assert!(hits > 50, "test scene too sparse: only {hits} hits");
    }

    #[test]
    fn bvh_any_hit_matches_brute_force() {
        let mut rng = Rng::new(0x00a7_9d3c_51de_0001);
        let curves: Vec<Curve> = (0..120).map(|i| random_curve(&mut rng, i)).collect();
        let bvh = CurveBvh::build(&curves);

        for _ in 0..3000 {
            let origin = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(0.0, 2.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, -0.2)];
            let ray = Ray::infinite(origin, dir);
            let any = bvh.any_hit(&ray);
            let brute = brute_closest(&curves, &ray).is_some();
            assert_eq!(any, brute, "any_hit disagreement");
        }
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = CurveBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn subdivision_reproduces_the_curve() {
        let cp = [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, -1.0, 1.0],
            [3.0, 0.0, 2.0],
        ];
        let (left, right) = subdivide(&cp);
        // The split point is both halves' shared endpoint and the curve at 0.5.
        let (mid, _) = eval_bezier(&cp, 0.5);
        for a in 0..3 {
            assert!(approx(left[3][a], mid[a], 1e-6));
            assert!(approx(right[0][a], mid[a], 1e-6));
        }
        // Each half, re-evaluated, matches the original curve on its subrange.
        for i in 0..=10 {
            let w = i as f32 / 10.0;
            let (whole_left, _) = eval_bezier(&cp, 0.5 * w);
            let (half_left, _) = eval_bezier(&left, w);
            for a in 0..3 {
                assert!(approx(whole_left[a], half_left[a], 1e-5));
            }
        }
    }
}
