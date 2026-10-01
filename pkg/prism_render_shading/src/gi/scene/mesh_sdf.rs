//! Mesh signed-distance field on a uniform voxel lattice — CPU golden.
//!
//! Unreal Engine's *Mesh Distance Fields* (and the *Global Distance Field*
//! that Lumen marches for software ray tracing) store, per mesh, a volume
//! texture of signed distances to the surface: negative inside, positive
//! outside, zero on the surface.  Sampling that volume with trilinear
//! filtering gives a cheap, continuous distance everywhere, and sphere tracing
//! that field intersects geometry without triangle work.  This module is the
//! backend-neutral, GPU-free reference for that structure.
//!
//! [`MeshSdf`] holds a regular lattice of signed distances over an
//! axis-aligned world-space bounding box.  The lattice is *node based*: samples
//! live at the grid corners, the first node sits exactly on `min`, the last on
//! `max`, and the eight nodes around a point are trilinearly blended.  This is
//! the convention a GPU 3D texture addressed with `textureSampleLevel` and
//! clamp-to-edge wrapping reproduces, so the twin pass sees identical values.
//!
//! # Conventions
//! * **Storage.** Distances are `f32`, laid out row-major with `x` fastest,
//!   then `y`, then `z`: `index = x + y*res.x + z*res.x*res.y`.  This matches a
//!   `R32F` volume texture uploaded in the natural scan order, so the CPU
//!   buffer and the GPU twin share a byte layout.
//! * **Sign.** Negative distance is inside the surface, positive is outside,
//!   following UE / the standard SDF convention.  Analytic bakes
//!   ([`sphere_sdf`], [`box_sdf`]) use the exact closed-form distance so unit
//!   tests can cross-check against ground truth.
//! * **World <-> grid.** `world_to_grid` maps a world point to continuous
//!   lattice coordinates in `[0, res-1]`; `grid_to_world` is its inverse.  Per
//!   axis the node spacing is `(max - min) / (res - 1)`.  Both the spacing and
//!   the box extent are clamped away from zero so a degenerate (flat or
//!   single-node) axis can never divide by zero or produce `NaN`.
//! * **Boundary.** [`MeshSdf::sample_distance`] clamps the sample coordinate to
//!   the lattice (clamp-to-edge), exactly like a clamped GPU sampler.  The
//!   lower-level [`MeshSdf::distance_at`] integer read instead returns the
//!   conservative [`MeshSdf::FAR_DISTANCE`] for out-of-range indices, so code
//!   that walks raw cells treats "off the grid" as empty space rather than
//!   reading a neighbour.
//! * **Determinism.** Every function is a pure, deterministic computation: no
//!   RNG, no I/O, no GPU, no global state, and no `unsafe`.  The only
//!   allocation is the distance buffer owned by [`MeshSdf`].

use alloc::vec::Vec;
use bevy_math::{IVec3, UVec3, Vec3};

/// Exact signed distance from `point` to the surface of a sphere.
///
/// Negative inside, positive outside: `|point - center| - radius`.  `radius`
/// is clamped to be non-negative.  This is the analytic ground truth the
/// trilinear [`MeshSdf`] approximation is tested against.
#[inline]
pub fn sphere_sdf(point: Vec3, center: Vec3, radius: f32) -> f32 {
    (point - center).length() - radius.max(0.0)
}

/// Exact signed distance from `point` to the surface of an axis-aligned box.
///
/// Uses Inigo Quilez's closed form: with `q = |point - center| - half_extents`
/// the distance is `|max(q, 0)| + min(max(q.x, q.y, q.z), 0)`, which is correct
/// both outside (positive) and inside (negative) the box.  Negative half
/// extents are clamped to zero so a degenerate box collapses to its centre
/// plane rather than inverting.
#[inline]
pub fn box_sdf(point: Vec3, center: Vec3, half_extents: Vec3) -> f32 {
    let he = half_extents.max(Vec3::ZERO);
    let q = (point - center).abs() - he;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// A single sphere-trace intersection against a [`MeshSdf`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereTraceHit {
    /// Distance travelled from the ray origin to the surface, in world units.
    pub distance: f32,
    /// World-space position of the intersection (`origin + dir * distance`).
    pub position: Vec3,
    /// Number of marching iterations taken before convergence (diagnostic).
    pub steps: u32,
}

/// A uniform voxel lattice of signed distances over a world-space box.
///
/// See the module docs for the storage layout, sign convention, and boundary
/// behaviour.  Construct one from an analytic primitive with [`from_sphere`]
/// or [`from_box`], from an arbitrary distance callback with
/// [`from_distance_fn`], or from a pre-baked buffer with [`new`].
///
/// [`from_sphere`]: MeshSdf::from_sphere
/// [`from_box`]: MeshSdf::from_box
/// [`from_distance_fn`]: MeshSdf::from_distance_fn
/// [`new`]: MeshSdf::new
#[derive(Clone, Debug, PartialEq)]
pub struct MeshSdf {
    /// Per-axis node count; each component is clamped to be at least `1`.
    resolution: UVec3,
    /// World-space minimum corner of the bounding box (first lattice node).
    min: Vec3,
    /// World-space maximum corner of the bounding box (last lattice node).
    max: Vec3,
    /// Row-major signed distances, `x` fastest (see module docs).  Length is
    /// always `resolution.x * resolution.y * resolution.z`.
    distances: Vec<f32>,
}

impl MeshSdf {
    /// Conservative "far away / empty space" distance returned for reads that
    /// fall outside the lattice.  Large but finite so arithmetic on it never
    /// overflows to infinity or `NaN`.
    pub const FAR_DISTANCE: f32 = 1.0e18;

    /// Builds a field from a pre-computed distance buffer.
    ///
    /// `resolution` is clamped so every axis has at least one node, and `min`
    /// / `max` are reordered component-wise so `min <= max` holds even if the
    /// caller passes them swapped.  The buffer is resized to exactly
    /// `res.x * res.y * res.z` entries, padding any shortfall with
    /// [`FAR_DISTANCE`](Self::FAR_DISTANCE), so the length invariant always
    /// holds and no later index can be out of bounds.
    pub fn new(resolution: UVec3, min: Vec3, max: Vec3, mut distances: Vec<f32>) -> Self {
        let resolution = resolution.max(UVec3::ONE);
        let lo = min.min(max);
        let hi = min.max(max);
        let count = (resolution.x as usize) * (resolution.y as usize) * (resolution.z as usize);
        distances.resize(count, Self::FAR_DISTANCE);
        Self {
            resolution,
            min: lo,
            max: hi,
            distances,
        }
    }

    /// Bakes a field by evaluating a distance callback at every lattice node.
    ///
    /// `f` receives the world-space position of each node (via
    /// [`grid_to_world`](Self::grid_to_world)) and returns its signed distance.
    /// This is the primitive used by [`from_sphere`](Self::from_sphere) and
    /// [`from_box`](Self::from_box); supply any closed-form SDF to bake a custom
    /// primitive for tests.
    pub fn from_distance_fn(
        resolution: UVec3,
        min: Vec3,
        max: Vec3,
        f: impl Fn(Vec3) -> f32,
    ) -> Self {
        let resolution = resolution.max(UVec3::ONE);
        let lo = min.min(max);
        let hi = min.max(max);
        let count = (resolution.x as usize) * (resolution.y as usize) * (resolution.z as usize);
        let mut distances = Vec::new();
        distances.reserve(count);
        // Pre-build a scaffold so grid_to_world (which reads resolution/bounds)
        // can be reused for the node positions.
        let scaffold = Self {
            resolution,
            min: lo,
            max: hi,
            distances: Vec::new(),
        };
        for z in 0..resolution.z {
            for y in 0..resolution.y {
                for x in 0..resolution.x {
                    let world = scaffold.grid_to_world(Vec3::new(x as f32, y as f32, z as f32));
                    distances.push(f(world));
                }
            }
        }
        Self {
            resolution,
            min: lo,
            max: hi,
            distances,
        }
    }

    /// Bakes an analytic sphere into a field whose box tightly wraps the sphere
    /// plus `padding` of empty space on every side.
    ///
    /// Each node stores the exact [`sphere_sdf`] distance, so the trilinear
    /// reconstruction is a faithful approximation that unit tests compare to
    /// the closed form.  `radius` and `padding` are clamped to be non-negative;
    /// the box half-size is `radius + padding` so there is always some positive
    /// (outside) shell for sphere tracing to march through.
    pub fn from_sphere(resolution: UVec3, center: Vec3, radius: f32, padding: f32) -> Self {
        let radius = radius.max(0.0);
        let padding = padding.max(0.0);
        let half = Vec3::splat(radius + padding);
        Self::from_distance_fn(resolution, center - half, center + half, move |p| {
            sphere_sdf(p, center, radius)
        })
    }

    /// Bakes an analytic axis-aligned box into a field whose bounding box wraps
    /// the primitive plus `padding` of empty space on every side.
    ///
    /// Each node stores the exact [`box_sdf`] distance.  `half_extents` is
    /// clamped to be non-negative and `padding` likewise, so the enclosing box
    /// is `half_extents + padding` and always contains an outside shell.
    pub fn from_box(resolution: UVec3, center: Vec3, half_extents: Vec3, padding: f32) -> Self {
        let he = half_extents.max(Vec3::ZERO);
        let padding = padding.max(0.0);
        let wrap = he + Vec3::splat(padding);
        Self::from_distance_fn(resolution, center - wrap, center + wrap, move |p| {
            box_sdf(p, center, he)
        })
    }

    /// Per-axis node count (always `>= 1` on every component).
    #[inline]
    pub fn resolution(&self) -> UVec3 {
        self.resolution
    }

    /// World-space minimum corner (the first lattice node).
    #[inline]
    pub fn bounds_min(&self) -> Vec3 {
        self.min
    }

    /// World-space maximum corner (the last lattice node).
    #[inline]
    pub fn bounds_max(&self) -> Vec3 {
        self.max
    }

    /// The raw row-major distance buffer.
    #[inline]
    pub fn distances(&self) -> &[f32] {
        &self.distances
    }

    /// Total number of lattice nodes (`res.x * res.y * res.z`).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.distances.len()
    }

    /// Per-axis spacing between adjacent nodes, `(max - min) / (res - 1)`.
    ///
    /// A single-node axis (`res == 1`) and a zero-extent axis would both make
    /// the spacing meaningless, so each component is clamped to at least
    /// `f32::MIN_POSITIVE`.  The clamp only ever enlarges a would-be-zero
    /// spacing, so world/grid mapping stays finite without changing valid
    /// grids.
    #[inline]
    pub fn cell_size(&self) -> Vec3 {
        let denom = Vec3::new(
            (self.resolution.x.max(2) - 1) as f32,
            (self.resolution.y.max(2) - 1) as f32,
            (self.resolution.z.max(2) - 1) as f32,
        );
        let extent = self.max - self.min;
        let cs = extent / denom;
        Vec3::new(
            cs.x.max(f32::MIN_POSITIVE),
            cs.y.max(f32::MIN_POSITIVE),
            cs.z.max(f32::MIN_POSITIVE),
        )
    }

    /// Maps a world point to continuous lattice coordinates in `[0, res-1]`.
    ///
    /// The result is *not* clamped: callers that want in-bounds coordinates
    /// should clamp, and [`sample_distance`](Self::sample_distance) does so
    /// internally.  Exposed unclamped so `grid_to_world` is an exact inverse.
    #[inline]
    pub fn world_to_grid(&self, world: Vec3) -> Vec3 {
        (world - self.min) / self.cell_size()
    }

    /// Maps continuous lattice coordinates back to a world position; the exact
    /// inverse of [`world_to_grid`](Self::world_to_grid).
    #[inline]
    pub fn grid_to_world(&self, grid: Vec3) -> Vec3 {
        self.min + grid * self.cell_size()
    }

    /// Flattens integer node coordinates to a buffer index without bounds
    /// checks.  Callers must guarantee `0 <= i < res` on every axis; internal
    /// use always clamps first.
    #[inline]
    fn linear_index(&self, x: u32, y: u32, z: u32) -> usize {
        (x as usize)
            + (y as usize) * (self.resolution.x as usize)
            + (z as usize) * (self.resolution.x as usize) * (self.resolution.y as usize)
    }

    /// Reads the stored distance at integer node coordinates, clamping each
    /// axis into the valid range (clamp-to-edge).  Used by the sampler, where
    /// the lattice is guaranteed non-empty.
    #[inline]
    fn node_clamped(&self, x: i32, y: i32, z: i32) -> f32 {
        let cx = (x.max(0) as u32).min(self.resolution.x - 1);
        let cy = (y.max(0) as u32).min(self.resolution.y - 1);
        let cz = (z.max(0) as u32).min(self.resolution.z - 1);
        self.distances[self.linear_index(cx, cy, cz)]
    }

    /// Reads the stored distance at integer node coordinates, returning
    /// [`FAR_DISTANCE`](Self::FAR_DISTANCE) for any out-of-range index.
    ///
    /// This is the conservative raw accessor: stepping off the lattice reads as
    /// empty space rather than wrapping or clamping to a neighbour.
    #[inline]
    pub fn distance_at(&self, cell: IVec3) -> f32 {
        if cell.x < 0
            || cell.y < 0
            || cell.z < 0
            || cell.x as u32 >= self.resolution.x
            || cell.y as u32 >= self.resolution.y
            || cell.z as u32 >= self.resolution.z
        {
            return Self::FAR_DISTANCE;
        }
        self.distances[self.linear_index(cell.x as u32, cell.y as u32, cell.z as u32)]
    }

    /// Trilinearly samples the signed distance at an arbitrary world point.
    ///
    /// The sample coordinate is clamped to the lattice (clamp-to-edge), so a
    /// point outside the box reads the nearest boundary rather than
    /// extrapolating.  The eight nodes around the clamped coordinate are blended
    /// with the fractional position, matching a clamped GPU 3D-texture fetch.
    #[inline]
    pub fn sample_distance(&self, world: Vec3) -> f32 {
        let res = self.resolution;
        let max_coord = Vec3::new(
            (res.x - 1) as f32,
            (res.y - 1) as f32,
            (res.z - 1) as f32,
        );
        let g = self.world_to_grid(world).clamp(Vec3::ZERO, max_coord);
        let base = g.floor();
        let frac = (g - base).clamp(Vec3::ZERO, Vec3::ONE);

        let x0 = base.x as i32;
        let y0 = base.y as i32;
        let z0 = base.z as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let c000 = self.node_clamped(x0, y0, z0);
        let c100 = self.node_clamped(x1, y0, z0);
        let c010 = self.node_clamped(x0, y1, z0);
        let c110 = self.node_clamped(x1, y1, z0);
        let c001 = self.node_clamped(x0, y0, z1);
        let c101 = self.node_clamped(x1, y0, z1);
        let c011 = self.node_clamped(x0, y1, z1);
        let c111 = self.node_clamped(x1, y1, z1);

        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(c000, c100, frac.x);
        let c10 = lerp(c010, c110, frac.x);
        let c01 = lerp(c001, c101, frac.x);
        let c11 = lerp(c011, c111, frac.x);
        let c0 = lerp(c00, c10, frac.y);
        let c1 = lerp(c01, c11, frac.y);
        lerp(c0, c1, frac.z)
    }

    /// Estimates the signed-distance gradient at a world point by central
    /// differences, returning a unit vector.
    ///
    /// For a true SDF the gradient has unit length and points away from the
    /// surface, so it doubles as the surface normal at a hit.  The sampling step
    /// is half the smallest node spacing (floored to a tiny constant), which
    /// keeps the stencil inside one cell.  A vanishing gradient (flat region or
    /// exactly on a symmetry point) falls back to `+Y` so the result is never a
    /// zero vector or `NaN`.
    #[inline]
    pub fn gradient(&self, world: Vec3) -> Vec3 {
        let h = (self.cell_size().min_element() * 0.5).max(1.0e-4);
        let dx = self.sample_distance(world + Vec3::new(h, 0.0, 0.0))
            - self.sample_distance(world - Vec3::new(h, 0.0, 0.0));
        let dy = self.sample_distance(world + Vec3::new(0.0, h, 0.0))
            - self.sample_distance(world - Vec3::new(0.0, h, 0.0));
        let dz = self.sample_distance(world + Vec3::new(0.0, 0.0, h))
            - self.sample_distance(world - Vec3::new(0.0, 0.0, h));
        let grad = Vec3::new(dx, dy, dz);
        let len = grad.length();
        if len > f32::MIN_POSITIVE {
            grad / len
        } else {
            Vec3::Y
        }
    }

    /// Sphere-traces a ray against the field and returns the first surface hit.
    ///
    /// Standard SDF ray marching: at each step the sampled distance is a safe
    /// radius to advance along `dir`.  Convergence is declared when the sampled
    /// distance drops below a small epsilon (including when the ray starts
    /// inside the surface, where the distance is already negative).  Each
    /// advance is floored to a minimum step so a near-zero distance in open
    /// space cannot stall the march, and the loop stops once the travelled
    /// distance exceeds `max_dist` or `max_steps` is reached.
    ///
    /// Returns `None` when the ray misses within the budget, or when `dir` is
    /// too short to normalise.
    #[inline]
    pub fn sphere_trace(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_dist: f32,
        max_steps: u32,
    ) -> Option<SphereTraceHit> {
        const HIT_EPS: f32 = 1.0e-3;
        let len = dir.length();
        if len <= f32::MIN_POSITIVE {
            return None;
        }
        let dir = dir / len;
        let max_dist = max_dist.max(0.0);
        let min_step = (max_dist * 1.0e-3).max(1.0e-4);

        let mut t = 0.0f32;
        for step in 0..max_steps {
            let position = origin + dir * t;
            let d = self.sample_distance(position);
            if d < HIT_EPS {
                return Some(SphereTraceHit {
                    distance: t,
                    position,
                    steps: step,
                });
            }
            t += d.max(min_step);
            if t > max_dist {
                break;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_field() -> MeshSdf {
        MeshSdf::from_sphere(UVec3::splat(48), Vec3::ZERO, 1.0, 1.0)
    }

    #[test]
    fn new_enforces_buffer_length_invariant() {
        // Too-short buffer is padded; resolution is clamped to >= 1.
        let sdf = MeshSdf::new(UVec3::new(2, 2, 2), Vec3::ZERO, Vec3::splat(1.0), Vec::new());
        assert_eq!(sdf.node_count(), 8);
        assert!(sdf.distances().iter().all(|&d| d == MeshSdf::FAR_DISTANCE));

        let clamped = MeshSdf::new(UVec3::ZERO, Vec3::ZERO, Vec3::splat(1.0), Vec::new());
        assert_eq!(clamped.resolution(), UVec3::ONE);
        assert_eq!(clamped.node_count(), 1);
    }

    #[test]
    fn new_orders_bounds() {
        let sdf = MeshSdf::new(UVec3::splat(2), Vec3::splat(2.0), Vec3::splat(-2.0), Vec::new());
        assert_eq!(sdf.bounds_min(), Vec3::splat(-2.0));
        assert_eq!(sdf.bounds_max(), Vec3::splat(2.0));
    }

    #[test]
    fn world_grid_roundtrip_is_exact() {
        let sdf = sphere_field();
        for p in [
            Vec3::new(0.3, -0.7, 1.1),
            Vec3::new(-1.5, 0.0, 0.9),
            Vec3::ZERO,
            sdf.bounds_min(),
            sdf.bounds_max(),
        ] {
            let back = sdf.grid_to_world(sdf.world_to_grid(p));
            assert!((back - p).length() < 1.0e-4, "{p:?} -> {back:?}");
        }
    }

    #[test]
    fn grid_corners_map_to_bounds() {
        let sdf = sphere_field();
        let res = sdf.resolution();
        let last = Vec3::new(
            (res.x - 1) as f32,
            (res.y - 1) as f32,
            (res.z - 1) as f32,
        );
        assert!((sdf.grid_to_world(Vec3::ZERO) - sdf.bounds_min()).length() < 1.0e-4);
        assert!((sdf.grid_to_world(last) - sdf.bounds_max()).length() < 1.0e-4);
    }

    #[test]
    fn distance_at_is_far_out_of_range() {
        let sdf = sphere_field();
        assert_eq!(sdf.distance_at(IVec3::new(-1, 0, 0)), MeshSdf::FAR_DISTANCE);
        assert_eq!(
            sdf.distance_at(IVec3::new(10_000, 0, 0)),
            MeshSdf::FAR_DISTANCE
        );
        // In-range reads a finite value.
        assert!(sdf.distance_at(IVec3::ZERO).is_finite());
    }

    #[test]
    fn sphere_sign_is_correct() {
        let sdf = sphere_field();
        assert!(sdf.sample_distance(Vec3::ZERO) < 0.0, "centre is inside");
        // A point just outside the radius-1 sphere is positive.
        assert!(sdf.sample_distance(Vec3::new(1.5, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn sphere_surface_distance_near_zero() {
        let sdf = sphere_field();
        // Points on the analytic surface should read |d| within a cell.
        let tol = sdf.cell_size().max_element();
        for dir in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 1.0, 1.0)] {
            let p = dir.normalize(); // radius 1 -> on the surface
            assert!(sdf.sample_distance(p).abs() < tol, "{p:?}");
        }
    }

    #[test]
    fn sphere_distance_matches_analytic_away_from_surface() {
        let sdf = sphere_field();
        // Away from the surface the field is near-linear, so trilinear error
        // is tiny; compare against the closed form.
        for p in [
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(1.4, 0.2, -0.1),
        ] {
            let got = sdf.sample_distance(p);
            let want = sphere_sdf(p, Vec3::ZERO, 1.0);
            assert!((got - want).abs() < 0.05, "{p:?}: {got} vs {want}");
        }
    }

    #[test]
    fn gradient_points_radially_outward() {
        let sdf = sphere_field();
        for dir in [Vec3::X, Vec3::new(0.5, 0.8, 0.0), Vec3::new(-0.3, 0.2, 0.9)] {
            let p = dir.normalize() * 1.3; // outside, radial
            let g = sdf.gradient(p);
            assert!((g.length() - 1.0).abs() < 1.0e-3, "unit length");
            assert!(g.dot(dir.normalize()) > 0.9, "{p:?} grad {g:?}");
        }
    }

    #[test]
    fn gradient_falls_back_on_flat_field() {
        // A uniform field has zero gradient everywhere -> +Y fallback.
        let flat = MeshSdf::new(
            UVec3::splat(4),
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            alloc::vec![5.0; 64],
        );
        assert_eq!(flat.gradient(Vec3::ZERO), Vec3::Y);
    }

    #[test]
    fn sphere_trace_hits_surface() {
        let sdf = sphere_field();
        // March from +x toward the origin; expect a hit near x = radius = 1.
        let origin = Vec3::new(1.9, 0.0, 0.0);
        let hit = sdf
            .sphere_trace(origin, Vec3::NEG_X, 4.0, 128)
            .expect("ray should hit the sphere");
        let expected_t = (origin.x) - 1.0; // 0.9
        assert!((hit.distance - expected_t).abs() < 0.05, "{hit:?}");
        assert!(hit.position.length() <= 1.05, "near the surface");
    }

    #[test]
    fn sphere_trace_misses_when_aimed_away() {
        let sdf = sphere_field();
        let hit = sdf.sphere_trace(Vec3::new(1.9, 0.0, 0.0), Vec3::X, 4.0, 128);
        assert!(hit.is_none(), "ray aimed away from the sphere misses");
    }

    #[test]
    fn sphere_trace_zero_direction_is_none() {
        let sdf = sphere_field();
        assert!(sdf.sphere_trace(Vec3::ZERO, Vec3::ZERO, 4.0, 64).is_none());
    }

    #[test]
    fn box_sign_and_distance_match_analytic() {
        let he = Vec3::new(0.5, 0.5, 0.5);
        let sdf = MeshSdf::from_box(UVec3::splat(48), Vec3::ZERO, he, 1.0);
        // Inside the box the sign is negative.
        assert!(sdf.sample_distance(Vec3::ZERO) < 0.0);
        // A point outside along +x: analytic distance = x - 0.5.
        let p = Vec3::new(1.2, 0.0, 0.0);
        let want = box_sdf(p, Vec3::ZERO, he);
        assert!((sdf.sample_distance(p) - want).abs() < 0.05, "{p:?}");
    }

    #[test]
    fn box_sdf_matches_known_values() {
        let he = Vec3::splat(1.0);
        // On a face: distance 0.
        assert!(box_sdf(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, he).abs() < 1.0e-6);
        // Outside along one axis.
        assert!((box_sdf(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, he) - 2.0).abs() < 1.0e-6);
        // Interior point: negative, equals distance to nearest face.
        assert!((box_sdf(Vec3::ZERO, Vec3::ZERO, he) - (-1.0)).abs() < 1.0e-6);
    }

    #[test]
    fn sample_outside_box_clamps_to_edge() {
        let sdf = sphere_field();
        // Far outside the lattice -> finite, no NaN (clamp-to-edge).
        let d = sdf.sample_distance(Vec3::splat(1000.0));
        assert!(d.is_finite());
    }

    #[test]
    fn degenerate_bounds_are_safe() {
        // min == max: zero extent on every axis.
        let sdf = MeshSdf::from_sphere(UVec3::splat(4), Vec3::ZERO, 0.0, 0.0);
        let d = sdf.sample_distance(Vec3::ZERO);
        assert!(d.is_finite(), "no NaN from zero extent");
        let g = sdf.gradient(Vec3::ZERO);
        assert!(g.is_finite());
    }

    #[test]
    fn sampling_is_deterministic() {
        let sdf = sphere_field();
        let p = Vec3::new(0.37, -0.21, 0.88);
        assert_eq!(sdf.sample_distance(p), sdf.sample_distance(p));
    }
}
