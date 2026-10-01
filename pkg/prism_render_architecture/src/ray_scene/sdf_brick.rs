//! Sphere-traced signed-distance-field (`SDF`) brick primitive and its
//! single-level `BVH`.
//!
//! A brick stores a dense 3D grid of signed distances sampled over an
//! axis-aligned box. Trilinear interpolation reconstructs a continuous field
//! whose zero isocontour is the surface, and a ray is marched through it by
//! *sphere tracing*: at each step the interpolated distance is a conservative
//! bound on how far the ray can advance without crossing the surface, so the
//! march converges onto the first isocontour crossing. This is the `CPU` golden
//! reference for the procedural-primitive intersection shader that the
//! `DXR`/`Vulkan` `BLAS` would run inside each brick's box: the hardware stores
//! one axis-aligned box per brick and the shader refines the hit by marching
//! the field.
//!
//! Unlike the analytic primitives in this crate (sphere, ellipsoid, …) an `SDF`
//! brick can approximate *arbitrary* closed surfaces — carved, filleted, or
//! CSG-combined shapes — at the cost of storing a volume. The march here is
//! deliberately deterministic: a fixed [`SURFACE_EPS`] hit threshold, a fixed
//! [`MIN_STEP`] progress floor, and a fixed [`MAX_STEPS`] cap, using only
//! `+ − × ÷`, `sqrt`, `abs`, `min`, `max`, `clamp`, and `floor`, so it is
//! bit-reproducible on the `GPU` and free of any transcendental call. The
//! [`SdfBrickBvh`] reuses the shared binned-`SAH` [`build_linear_bvh`] over the
//! per-brick boxes and the same ordered slab walk the triangle
//! [`super::bvh::Bvh`] uses.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;
use core::fmt;

/// Distance threshold (in field units) at or below which the march reports a
/// surface hit. A small positive value keeps the hit just outside the exact
/// zero isocontour of the interpolated field.
pub const SURFACE_EPS: f32 = 1e-4;

/// Minimum ray-parameter advance per sphere-tracing step, guaranteeing forward
/// progress even where the interpolated distance underestimates the true one.
pub const MIN_STEP: f32 = 1e-4;

/// Maximum number of sphere-tracing steps before the march gives up and reports
/// a miss. Bounds the work per ray and keeps traversal terminating.
pub const MAX_STEPS: u32 = 128;

/// Why a [`SdfBrick`] could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfBrickError {
    /// A grid dimension was smaller than two, so no cell (and thus no trilinear
    /// interpolation) is possible along that axis.
    DegenerateDims,
    /// The sample count did not equal `dims.x * dims.y * dims.z`.
    SampleCountMismatch,
    /// A grid spacing component was not strictly positive.
    NonPositiveSpacing,
}

impl fmt::Display for SdfBrickError {
    /// Formats a human-readable description of the construction failure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateDims => f.write_str("SDF brick dimensions must each be at least two"),
            Self::SampleCountMismatch => {
                f.write_str("SDF brick sample count must equal the product of its dimensions")
            }
            Self::NonPositiveSpacing => {
                f.write_str("SDF brick grid spacing must be strictly positive on every axis")
            }
        }
    }
}

impl std::error::Error for SdfBrickError {}

/// A dense signed-distance-field brick in world space.
///
/// The grid stores one signed distance per node in row-major order with `x`
/// varying fastest, then `y`, then `z` (`index = x + dims.x * (y + dims.y * z)`).
/// Negative values are inside the surface, positive outside, and the zero
/// isocontour is the surface itself. The node at grid coordinate
/// `(0, 0, 0)` sits at [`SdfBrick::origin`] and successive nodes are
/// [`SdfBrick::spacing`] apart on each axis.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfBrick {
    /// Node counts along `x`, `y`, `z`; each is at least two.
    dims: [u32; 3],
    /// Signed distances, row-major (`x` fastest), length `dims.x*dims.y*dims.z`.
    data: Vec<f32>,
    /// World-space position of the `(0, 0, 0)` node.
    origin: [f32; 3],
    /// Strictly-positive node spacing along `x`, `y`, `z`.
    spacing: [f32; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl SdfBrick {
    /// Builds a brick from its `dims`, row-major `data`, world-space `origin`,
    /// per-axis `spacing`, and stable id `primitive`.
    ///
    /// # Errors
    /// Returns [`SdfBrickError`] when any dimension is below two, the spacing is
    /// not strictly positive, or `data.len()` does not match the dimensions.
    pub fn new(
        dims: [u32; 3],
        data: Vec<f32>,
        origin: [f32; 3],
        spacing: [f32; 3],
        primitive: u32,
    ) -> Result<Self, SdfBrickError> {
        if dims[0] < 2 || dims[1] < 2 || dims[2] < 2 {
            return Err(SdfBrickError::DegenerateDims);
        }
        if spacing[0] <= 0.0 || spacing[1] <= 0.0 || spacing[2] <= 0.0 {
            return Err(SdfBrickError::NonPositiveSpacing);
        }
        let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        if data.len() != expected {
            return Err(SdfBrickError::SampleCountMismatch);
        }
        Ok(Self {
            dims,
            data,
            origin,
            spacing,
            primitive,
        })
    }

    /// Node counts along `x`, `y`, `z`.
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Signed distance samples in row-major order (`x` fastest).
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// World-space position of the `(0, 0, 0)` node.
    #[must_use]
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Per-axis node spacing.
    #[must_use]
    pub fn spacing(&self) -> [f32; 3] {
        self.spacing
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Axis-aligned world bounds of the sampled grid.
    ///
    /// Spans from [`SdfBrick::origin`] to `origin + (dims - 1) * spacing`; this
    /// is the procedural-primitive box the hardware `BLAS` stores per brick.
    #[must_use]
    pub fn local_aabb(&self) -> Aabb {
        Aabb::new(
            self.origin,
            [
                self.origin[0] + (self.dims[0] - 1) as f32 * self.spacing[0],
                self.origin[1] + (self.dims[1] - 1) as f32 * self.spacing[1],
                self.origin[2] + (self.dims[2] - 1) as f32 * self.spacing[2],
            ],
        )
    }

    /// Lower grid node index and interpolation fraction for world coordinate
    /// `p` on `axis`, clamped so the returned node and its successor are both in
    /// range (edge samples extrapolate as a flat clamp).
    fn axis_coord(&self, p: f32, axis: usize) -> (usize, f32) {
        let grid = (p - self.origin[axis]) / self.spacing[axis];
        let last = (self.dims[axis] - 1) as f32;
        let clamped = grid.clamp(0.0, last);
        let max_lower = (self.dims[axis] - 2) as f32;
        let lower = clamped.floor().min(max_lower);
        (lower as usize, clamped - lower)
    }

    /// Flat row-major index of grid node `(x, y, z)`.
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;
        x + dx * (y + dy * z)
    }

    /// Trilinearly interpolated signed distance at world point `p`.
    ///
    /// Coordinates outside the grid clamp to the boundary, so the field stays
    /// defined everywhere the ray marches.
    #[must_use]
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let (x0, fx) = self.axis_coord(p[0], 0);
        let (y0, fy) = self.axis_coord(p[1], 1);
        let (z0, fz) = self.axis_coord(p[2], 2);
        let (x1, y1, z1) = (x0 + 1, y0 + 1, z0 + 1);

        let c000 = self.data[self.index(x0, y0, z0)];
        let c100 = self.data[self.index(x1, y0, z0)];
        let c010 = self.data[self.index(x0, y1, z0)];
        let c110 = self.data[self.index(x1, y1, z0)];
        let c001 = self.data[self.index(x0, y0, z1)];
        let c101 = self.data[self.index(x1, y0, z1)];
        let c011 = self.data[self.index(x0, y1, z1)];
        let c111 = self.data[self.index(x1, y1, z1)];

        // Interpolate along x, then y, then z.
        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        c0 + (c1 - c0) * fz
    }

    /// Central-difference gradient of the interpolated field at `p`, stepping by
    /// the grid spacing on each axis. The gradient points toward increasing
    /// distance, i.e. outward from the surface.
    #[must_use]
    pub fn gradient(&self, p: [f32; 3]) -> [f32; 3] {
        let [hx, hy, hz] = self.spacing;
        let gx = (self.sample([p[0] + hx, p[1], p[2]]) - self.sample([p[0] - hx, p[1], p[2]]))
            / (2.0 * hx);
        let gy = (self.sample([p[0], p[1] + hy, p[2]]) - self.sample([p[0], p[1] - hy, p[2]]))
            / (2.0 * hy);
        let gz = (self.sample([p[0], p[1], p[2] + hz]) - self.sample([p[0], p[1], p[2] - hz]))
            / (2.0 * hz);
        [gx, gy, gz]
    }

    /// Nearest ray/surface intersection inside `ray`'s `[t_min, t_max]`
    /// interval, found by sphere tracing the interpolated field, or `None` when
    /// the ray misses the brick box or never crosses the surface.
    ///
    /// The reported [`SdfBrickHit::normal`] is the unit field gradient oriented
    /// *against* the incident ray, and [`SdfBrickHit::front_face`] is `true`
    /// when the outward-facing side was struck. A zero-length ray direction
    /// never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<SdfBrickHit> {
        let direction = ray.direction();
        let dir_len2 = direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2];
        if dir_len2 <= 0.0 {
            return None;
        }
        // Steps advance by the safe distance in world units, converted to ray
        // parameter by dividing out the direction length so a non-unit ray is
        // handled correctly and the step never overshoots the surface.
        let inv_dir_len = 1.0 / dir_len2.sqrt();

        let (t0, t1) = ray.aabb_interval(&self.local_aabb(), ray.t_min(), ray.t_max())?;
        let mut t = t0;
        let mut distance = self.sample(ray.at(t));
        if distance.abs() <= SURFACE_EPS {
            // The ray enters the box already on the surface.
            let p = ray.at(t);
            return Some(self.make_hit(ray, t, p, distance));
        }
        let mut steps = 0u32;
        while steps < MAX_STEPS {
            let t_prev = t;
            let distance_prev = distance;
            // Advance by the *unsigned* safe distance in world units, converted
            // to ray parameter and floored so the march always makes progress.
            // Using the magnitude lets a ray that starts inside the solid march
            // outward toward the back-face crossing just as an exterior ray
            // marches inward toward the front face.
            t += (distance.abs() * inv_dir_len).max(MIN_STEP);
            if t > t1 {
                return None;
            }
            distance = self.sample(ray.at(t));
            let crossed = (distance_prev > 0.0) != (distance > 0.0);
            if crossed || distance.abs() <= SURFACE_EPS {
                // A sign change brackets a surface crossing in `[t_prev, t]`
                // (the trilinearly interpolated field is non-metric, so a single
                // step can overshoot it); otherwise a same-sign near-zero sample
                // is a grazing near-surface hit reported directly.
                let t_hit = if crossed {
                    self.refine_crossing(ray, t_prev, distance_prev, t)
                } else {
                    t
                };
                let p = ray.at(t_hit);
                let hit_distance = self.sample(p);
                return Some(self.make_hit(ray, t_hit, p, hit_distance));
            }
            steps += 1;
        }
        None
    }

    /// Bisects the ray-parameter bracket `[lo, hi]` onto the surface crossing,
    /// where the interpolated field at `lo` has sign `sign(distance_lo)` and the
    /// field at `hi` has the opposite sign. A fixed, deterministic iteration
    /// count keeps the result bit-reproducible.
    fn refine_crossing(&self, ray: &Ray, mut lo: f32, distance_lo: f32, mut hi: f32) -> f32 {
        let lo_positive = distance_lo > 0.0;
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if (self.sample(ray.at(mid)) > 0.0) == lo_positive {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Builds a [`SdfBrickHit`] at parameter `t`/point `p` with sampled
    /// `distance`, orienting the gradient normal against the ray.
    fn make_hit(&self, ray: &Ray, t: f32, p: [f32; 3], distance: f32) -> SdfBrickHit {
        let direction = ray.direction();
        let grad = self.gradient(p);
        let grad_len2 = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
        // Fall back to the ray direction's reverse when the gradient vanishes
        // (a flat field region) so the normal stays finite and unit.
        let outward = if grad_len2 > 0.0 {
            let inv = 1.0 / grad_len2.sqrt();
            [grad[0] * inv, grad[1] * inv, grad[2] * inv]
        } else {
            let inv = 1.0 / vec3_length(direction);
            [-direction[0] * inv, -direction[1] * inv, -direction[2] * inv]
        };
        let facing =
            direction[0] * outward[0] + direction[1] * outward[1] + direction[2] * outward[2];
        let front_face = facing < 0.0;
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };
        SdfBrickHit {
            t,
            primitive: self.primitive,
            position: p,
            normal,
            distance,
            front_face,
        }
    }
}

/// Euclidean length of a 3-vector; used only for the degenerate-gradient
/// fallback normal.
fn vec3_length(d: [f32; 3]) -> f32 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// A ray/`SDF`-brick intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfBrickHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the brick that was hit.
    pub primitive: u32,
    /// World-space hit position (`ray.at(t)`).
    pub position: [f32; 3],
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// Interpolated signed distance at the hit (at or below [`SURFACE_EPS`]).
    pub distance: f32,
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (march starting inside the surface), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// A single-level `BVH` over [`SdfBrick`] primitives.
///
/// Empty input yields an empty hierarchy ([`SdfBrickBvh::is_empty`]); traversal
/// of an empty hierarchy never reports a hit. The layout and ordered slab walk
/// mirror the triangle [`super::bvh::Bvh`] and the analytic
/// [`super::ellipsoid::EllipsoidBvh`] so every primitive kind shares one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfBrickBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Bricks reordered so each leaf owns a contiguous slice.
    bricks: Vec<SdfBrick>,
}

impl SdfBrickBvh {
    /// Builds a `BVH` over `bricks` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(bricks: &[SdfBrick]) -> Self {
        Self::build_with(bricks, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `bricks` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each brick's [`SdfBrick::local_aabb`] and then
    /// reorders the bricks by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`SdfBrickBvh::bricks`].
    #[must_use]
    pub fn build_with(bricks: &[SdfBrick], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = bricks.iter().map(SdfBrick::local_aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let bricks = order.iter().map(|&i| bricks[i as usize].clone()).collect();
        Self { nodes, bricks }
    }

    /// Number of `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of bricks in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.bricks.len()
    }

    /// True when the hierarchy has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or [`Aabb::empty`] when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or_else(Aabb::empty, |n| n.bounds)
    }

    /// Flattened `BVH` nodes (depth-first, root at index `0`).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Bricks in leaf-contiguous order.
    #[must_use]
    pub fn bricks(&self) -> &[SdfBrick] {
        &self.bricks
    }

    /// Nearest brick intersection along `ray`, or `None` if the ray hits
    /// nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<SdfBrickHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<SdfBrickHit> = None;

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
                    for brick in &self.bricks[start..end] {
                        if let Some(hit) = brick.intersect(&ray) {
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

    /// True when *any* brick intersects `ray` inside its interval.
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
                    for brick in &self.bricks[start..end] {
                        if brick.intersect(ray).is_some() {
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

    /// Samples a sphere `SDF` (`|p - center| - radius`) onto a `dims` grid that
    /// spans `[origin, origin + (dims - 1) * spacing]`.
    fn sphere_brick(
        dims: [u32; 3],
        origin: [f32; 3],
        spacing: [f32; 3],
        center: [f32; 3],
        radius: f32,
        primitive: u32,
    ) -> SdfBrick {
        let mut data = Vec::with_capacity(dims[0] as usize * dims[1] as usize * dims[2] as usize);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let p = [
                        origin[0] + x as f32 * spacing[0],
                        origin[1] + y as f32 * spacing[1],
                        origin[2] + z as f32 * spacing[2],
                    ];
                    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - radius;
                    data.push(dist);
                }
            }
        }
        SdfBrick::new(dims, data, origin, spacing, primitive).expect("valid sphere brick")
    }

    #[test]
    fn new_rejects_degenerate_inputs() {
        assert_eq!(
            SdfBrick::new([1, 2, 2], vec![0.0; 4], [0.0; 3], [1.0; 3], 0),
            Err(SdfBrickError::DegenerateDims)
        );
        assert_eq!(
            SdfBrick::new([2, 2, 2], vec![0.0; 7], [0.0; 3], [1.0; 3], 0),
            Err(SdfBrickError::SampleCountMismatch)
        );
        assert_eq!(
            SdfBrick::new([2, 2, 2], vec![0.0; 8], [0.0; 3], [1.0, 0.0, 1.0], 0),
            Err(SdfBrickError::NonPositiveSpacing)
        );
        assert!(SdfBrick::new([2, 2, 2], vec![0.0; 8], [0.0; 3], [1.0; 3], 0).is_ok());
    }

    #[test]
    fn sample_reproduces_grid_nodes_exactly() {
        let dims = [5, 4, 3];
        let origin = [-1.0, 0.5, 2.0];
        let spacing = [0.5, 0.75, 1.25];
        let brick = sphere_brick(dims, origin, spacing, [0.3, 0.4, 0.5], 0.9, 7);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let p = [
                        origin[0] + x as f32 * spacing[0],
                        origin[1] + y as f32 * spacing[1],
                        origin[2] + z as f32 * spacing[2],
                    ];
                    let expected = brick.data()[brick.index(x as usize, y as usize, z as usize)];
                    assert_eq!(brick.sample(p).to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn sphere_trace_lands_on_the_surface() {
        let dims = [41, 41, 41];
        let origin = [-2.0, -2.0, -2.0];
        let spacing = [0.1, 0.1, 0.1];
        let center = [0.0, 0.0, 0.0];
        let radius = 1.0;
        let brick = sphere_brick(dims, origin, spacing, center, radius, 3);

        let mut rng = Rng::new(0x5DF0_B41C);
        let mut hits = 0u32;
        for _ in 0..6_000 {
            let origin_p = [
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
            ];
            // Reject origins that fall inside (or very near) the sphere: an
            // interior start point legitimately produces a back-face crossing,
            // which this exterior-only assertion is not meant to cover.
            let d_origin = {
                let dx = origin_p[0] - center[0];
                let dy = origin_p[1] - center[1];
                let dz = origin_p[2] - center[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            };
            if d_origin < radius + 0.1 {
                continue;
            }
            // Aim near the sphere center so a large fraction of rays strike it.
            let target = [
                rng.range(-0.6, 0.6),
                rng.range(-0.6, 0.6),
                rng.range(-0.6, 0.6),
            ];
            let dir = [
                target[0] - origin_p[0],
                target[1] - origin_p[1],
                target[2] - origin_p[2],
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin_p, dir);
            let Some(hit) = brick.intersect(&ray) else {
                continue;
            };
            hits += 1;

            // The hit lies on the interpolated zero isocontour, which tracks the
            // true sphere radius to within the grid interpolation error.
            let r = [
                hit.position[0] - center[0],
                hit.position[1] - center[1],
                hit.position[2] - center[2],
            ];
            let dist = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!(
                (dist - radius).abs() < 0.05,
                "hit radius {dist} too far from {radius}"
            );

            // Unit normal that faces the incoming ray and points outward.
            let nlen = (hit.normal[0] * hit.normal[0]
                + hit.normal[1] * hit.normal[1]
                + hit.normal[2] * hit.normal[2])
                .sqrt();
            assert!((nlen - 1.0).abs() < 1e-3, "normal not unit: {nlen}");
            let facing =
                hit.normal[0] * dir[0] + hit.normal[1] * dir[1] + hit.normal[2] * dir[2];
            assert!(facing <= 1e-4, "normal not oriented against the ray: {facing}");
            assert!(hit.front_face, "exterior ray should strike the front face");

            // The outward normal roughly matches the exact radial direction.
            let inv = 1.0 / dist;
            let dot = hit.normal[0] * r[0] * inv
                + hit.normal[1] * r[1] * inv
                + hit.normal[2] * r[2] * inv;
            assert!(dot > 0.9, "normal not radial enough: {dot}");
        }
        assert!(hits > 1_000, "too few surface hits accumulated: {hits}");
    }

    #[test]
    fn ray_missing_the_box_reports_no_hit() {
        let brick = sphere_brick([9, 9, 9], [-1.0; 3], [0.25; 3], [0.0; 3], 0.5, 0);
        let ray = Ray::infinite([10.0, 10.0, 10.0], [0.0, 0.0, 1.0]);
        assert!(brick.intersect(&ray).is_none());
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = SdfBrickBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    /// Reference closest-hit over all bricks, mirroring the `BVH`'s interval
    /// tightening so the two must agree bit-for-bit.
    fn brute_closest(bricks: &[SdfBrick], ray: &Ray) -> Option<SdfBrickHit> {
        let mut best: Option<SdfBrickHit> = None;
        let mut ray = *ray;
        for brick in bricks {
            if let Some(hit) = brick.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<SdfBrick> {
        (0..count)
            .map(|i| {
                let center = [
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                ];
                let radius = rng.range(0.4, 1.1);
                let half = radius + 0.3;
                let dims = [13u32, 13, 13];
                let spacing = [
                    2.0 * half / (dims[0] - 1) as f32,
                    2.0 * half / (dims[1] - 1) as f32,
                    2.0 * half / (dims[2] - 1) as f32,
                ];
                let origin = [center[0] - half, center[1] - half, center[2] - half];
                sphere_brick(dims, origin, spacing, center, radius, i)
            })
            .collect()
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0xB41C_5DF0);
        let bricks = random_scene(&mut rng, 32);
        let bvh = SdfBrickBvh::build(&bricks);
        assert_eq!(bvh.primitive_count(), bricks.len());

        for _ in 0..4_000 {
            let origin = [
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let expected = brute_closest(&bricks, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive);
                    assert_eq!(e.t.to_bits(), a.t.to_bits());
                    assert_eq!(e.front_face, a.front_face);
                    for k in 0..3 {
                        assert_eq!(e.position[k].to_bits(), a.position[k].to_bits());
                        assert_eq!(e.normal[k].to_bits(), a.normal[k].to_bits());
                    }
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), brute_closest(&bricks, &ray).is_some());
        }
    }
}
