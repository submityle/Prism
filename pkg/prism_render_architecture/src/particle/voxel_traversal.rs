//! Amanatides-Woo integer 3D voxel-grid ray traversal for the particle
//! subsystem (design §7, §10, §29).
//!
//! Several particle stages need to know *exactly which uniform grid voxels a
//! ray crosses, in the order it crosses them*: a voxel-fluid stage marches a
//! sampling ray through its `PerGridVoxel` domain; a collision probe walks the
//! occupancy grid a spark travels through between two substeps; and a
//! deterministic `CPU` reference path must reproduce the same ordered voxel
//! sequence the future `GPU` kernel will visit, cell for cell. This module owns
//! the small, self-contained contract those stages share: turning a ray plus a
//! uniform grid definition into the ordered list of `(voxel, t_enter)` pairs
//! the ray passes through, front to back.
//!
//! # The algorithm
//! This is the classic Amanatides & Woo (1987) "A Fast Voxel Traversal
//! Algorithm for Ray Tracing" incremental grid DDA. The ray origin is snapped
//! to a starting voxel with [`f32::floor`]; each of the three axes carries a
//! `step` of `+1` or `-1` (the sign of that direction component), a `t_max`
//! (the ray parameter at which the ray crosses the *next* voxel boundary on
//! that axis), and a `t_delta` (the ray-parameter distance between successive
//! boundaries on that axis). Each iteration advances along whichever axis has
//! the smallest `t_max`, emits the entered voxel, and pushes that axis' `t_max`
//! forward by its `t_delta`. The walk terminates on a maximum ray distance or a
//! maximum voxel count, whichever comes first.
//!
//! # Axis-parallel rays and division safety
//! An axis whose direction component is (near) zero never crosses a boundary on
//! that axis. Rather than divide by it, this module sets that axis' `t_max` and
//! `t_delta` to the sentinel [`f32::INFINITY`] (a finite-vs-infinite ordering
//! value, not a transcendental function), so the incremental minimum never
//! selects that axis and the voxel coordinate on it stays fixed. Every division
//! guards its denominator against the [`EPS`] floor first, so no divide by zero
//! is possible; negative directions step and seed `t_max` correctly toward the
//! lower boundary.
//!
//! # Guaranteed properties
//! For any ray and grid the returned walk (see the tests):
//! * lists voxels in strictly non-decreasing `t_enter` order,
//! * never repeats a voxel and changes exactly one axis coordinate by exactly
//!   `±1` between consecutive voxels (6-connected),
//! * begins at the voxel containing the ray origin when the origin lies inside
//!   (or on) the traversal, and
//! * is fully deterministic: identical inputs yield an identical walk.
//!
//! # Strict scope
//! This module only performs integer voxel-grid DDA from a floating-point ray.
//! It is deliberately *not* a screen-space line DDA (see
//! [`super::screen_space_reflection`]) and *not* a ray/AABB slab intersection
//! (see [`super::volume_march`]); it neither imports nor mutates any sibling
//! module. Its math is hand-rolled `f32` limited to `sqrt`/`floor`/`abs`/
//! `min`/`max`/`clamp` plus integer arithmetic, with `f32::INFINITY` used only
//! as an ordering sentinel.

use alloc::vec::Vec;

/// Magnitude tolerance below which an `f32` is treated as zero.
///
/// Used both to classify an axis as parallel (no boundary crossings) and to
/// floor division denominators so a divide by zero can never occur.
pub const EPS: f32 = 1e-6;

/// A hand-rolled 3-component `f32` vector, local to this module.
///
/// The traversal only needs component access, subtraction, and scaling, so the
/// surface is intentionally tiny. It is a plain value type with no dependency
/// on any sibling module's vector math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector `(0, 0, 0)`.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its three components.
    #[must_use]
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Builds a vector with all three components set to `v`.
    #[must_use]
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Component-wise subtraction, `self - rhs`.
    #[must_use]
    #[inline]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Component-wise addition, `self + rhs`.
    #[must_use]
    #[inline]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub/neg methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Uniform scale by `s`.
    #[must_use]
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Squared Euclidean length, `x² + y² + z²`.
    #[must_use]
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean length via [`f32::sqrt`].
    #[must_use]
    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit-length vector, or [`Vec3::ZERO`] if the length is below
    /// [`EPS`] (avoids a divide by zero and any `NaN`).
    #[must_use]
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len < EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A hand-rolled integer 3-vector naming a voxel coordinate, local to this
/// module.
///
/// Coordinates are `i32` so a grid may extend into negative octants around its
/// origin. Arithmetic in the traversal is kept in-range by construction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IVec3 {
    /// The x voxel index.
    pub x: i32,
    /// The y voxel index.
    pub y: i32,
    /// The z voxel index.
    pub z: i32,
}

impl IVec3 {
    /// Builds an integer vector from its three components.
    #[must_use]
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Builds an integer vector with all three components set to `v`.
    #[must_use]
    #[inline]
    pub const fn splat(v: i32) -> Self {
        Self { x: v, y: v, z: v }
    }
}

/// A ray with an origin and a direction, local to this module.
///
/// The direction need not be unit length; [`traverse`] normalizes it so the
/// returned `t_enter` values are true world-space distances from the origin. A
/// direction shorter than [`EPS`] is a degenerate ray and yields an empty walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The world-space ray origin.
    pub origin: Vec3,
    /// The ray direction (any non-zero length; normalized internally).
    pub dir: Vec3,
}

impl Ray {
    /// Builds a ray from an origin and a direction.
    #[must_use]
    #[inline]
    pub const fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// The point at ray parameter `t`: `origin + dir * t`.
    #[must_use]
    #[inline]
    pub fn point_at(self, t: f32) -> Vec3 {
        self.origin.add(self.dir.scale(t))
    }
}

/// A uniform voxel grid: an origin corner plus a per-axis cell size.
///
/// Voxel `(i, j, k)` occupies the world-space box
/// `[origin + (i,j,k) * cell, origin + (i+1,j+1,k+1) * cell)`. Each cell-size
/// component is floored at [`EPS`] so a zero or negative size cannot produce a
/// divide by zero or a nonsensical index; callers should pass strictly positive
/// sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    /// The world-space position of the grid's `(0, 0, 0)` corner.
    pub origin: Vec3,
    /// The per-axis voxel edge length (floored at [`EPS`]).
    pub cell: Vec3,
}

impl VoxelGrid {
    /// Builds a grid from its origin corner and per-axis cell size.
    #[must_use]
    #[inline]
    pub const fn new(origin: Vec3, cell: Vec3) -> Self {
        Self { origin, cell }
    }

    /// Builds a grid at `origin` whose cells are cubes of edge `size`.
    #[must_use]
    #[inline]
    pub const fn cubic(origin: Vec3, size: f32) -> Self {
        Self {
            origin,
            cell: Vec3::splat(size),
        }
    }

    /// Returns the per-axis cell size with each component floored at [`EPS`].
    #[must_use]
    #[inline]
    fn safe_cell(self) -> Vec3 {
        Vec3::new(
            self.cell.x.max(EPS),
            self.cell.y.max(EPS),
            self.cell.z.max(EPS),
        )
    }

    /// Converts a world-space point to the grid-space coordinate (in cell
    /// units) relative to the grid origin.
    #[must_use]
    #[inline]
    fn to_grid_space(self, p: Vec3) -> Vec3 {
        let c = self.safe_cell();
        let rel = p.sub(self.origin);
        Vec3::new(rel.x / c.x, rel.y / c.y, rel.z / c.z)
    }

    /// Returns the integer voxel containing the world-space point `p`.
    ///
    /// The floor is taken in grid space so the result is the mathematically
    /// correct floor even for points in the negative octant.
    #[must_use]
    #[inline]
    pub fn voxel_of(self, p: Vec3) -> IVec3 {
        let g = self.to_grid_space(p);
        IVec3::new(floor_i32(g.x), floor_i32(g.y), floor_i32(g.z))
    }
}

/// How the traversal is bounded: by world-space distance, by voxel count, or by
/// both (the tighter limit wins).
///
/// A [`Limit`] with no distance and no step cap would run forever on a ray that
/// never leaves the grid, so [`traverse`] requires at least one bound; passing
/// [`Limit::NONE`] yields an empty walk by contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limit {
    /// Maximum ray parameter (world distance, since the direction is
    /// normalized). `None` means unbounded in distance.
    pub max_distance: Option<f32>,
    /// Maximum number of voxels to emit. `None` means unbounded in count.
    pub max_steps: Option<usize>,
}

impl Limit {
    /// No bound at all. Passing this to [`traverse`] yields an empty walk,
    /// because an unbounded traversal cannot be safely enumerated.
    pub const NONE: Self = Self {
        max_distance: None,
        max_steps: None,
    };

    /// Bounds the traversal to at most `steps` voxels.
    #[must_use]
    #[inline]
    pub const fn steps(steps: usize) -> Self {
        Self {
            max_distance: None,
            max_steps: Some(steps),
        }
    }

    /// Bounds the traversal to a maximum world-space `distance` along the ray.
    #[must_use]
    #[inline]
    pub const fn distance(distance: f32) -> Self {
        Self {
            max_distance: Some(distance),
            max_steps: None,
        }
    }

    /// Bounds the traversal by both a world-space `distance` and a voxel
    /// `steps` cap; whichever limit is reached first ends the walk.
    #[must_use]
    #[inline]
    pub const fn both(distance: f32, steps: usize) -> Self {
        Self {
            max_distance: Some(distance),
            max_steps: Some(steps),
        }
    }
}

/// A single visited voxel and the ray parameter at which the ray *enters* it.
///
/// `t_enter` is a world-space distance from the ray origin because [`traverse`]
/// normalizes the ray direction. The first emitted hit has the smallest
/// `t_enter` and subsequent hits are non-decreasing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelHit {
    /// The integer voxel coordinate.
    pub voxel: IVec3,
    /// The ray parameter (world distance) at which the ray enters `voxel`.
    pub t_enter: f32,
}

impl VoxelHit {
    /// Builds a hit from a voxel and its entry parameter.
    #[must_use]
    #[inline]
    pub const fn new(voxel: IVec3, t_enter: f32) -> Self {
        Self { voxel, t_enter }
    }
}

/// Floors an `f32` to the largest `i32` not greater than it, saturating on
/// out-of-range or non-finite input.
///
/// Uses only [`f32::floor`] (an allowed operation) and a range clamp; it never
/// touches a transcendental function. Non-finite inputs saturate to the `i32`
/// extremes so the traversal cannot produce an undefined index.
#[must_use]
#[inline]
fn floor_i32(v: f32) -> i32 {
    let f = v.floor();
    if f >= i32::MAX as f32 {
        i32::MAX
    } else if f <= i32::MIN as f32 {
        i32::MIN
    } else {
        f as i32
    }
}

/// Per-axis DDA state: the current voxel coordinate, the step direction, the
/// ray parameter of the next boundary crossing, and the per-boundary delta.
#[derive(Clone, Copy, Debug)]
struct AxisState {
    /// Current integer voxel index on this axis.
    coord: i32,
    /// Step direction: `+1`, `-1`, or `0` for an axis-parallel (fixed) axis.
    step: i32,
    /// Ray parameter at the next boundary crossing on this axis
    /// ([`f32::INFINITY`] when the axis is parallel/fixed).
    t_max: f32,
    /// Ray-parameter distance between successive boundaries on this axis
    /// ([`f32::INFINITY`] when the axis is parallel/fixed).
    t_delta: f32,
}

impl AxisState {
    /// Seeds one axis of the DDA from the ray's grid-space origin coordinate
    /// and grid-space direction component.
    ///
    /// `pos` and `dir` are already expressed in cell units (grid space), so a
    /// unit `t` step covers one cell of travel projected onto this axis. A
    /// direction component with magnitude below [`EPS`] marks the axis parallel:
    /// `step = 0` and both `t_max`/`t_delta` become the [`f32::INFINITY`]
    /// sentinel, which the traversal's minimum-selection never picks.
    fn seed(pos: f32, dir: f32) -> Self {
        let coord = floor_i32(pos);
        if dir.abs() < EPS {
            return Self {
                coord,
                step: 0,
                t_max: f32::INFINITY,
                t_delta: f32::INFINITY,
            };
        }
        // Fractional position within the current cell, in [0, 1).
        let frac = pos - (coord as f32);
        if dir > 0.0 {
            // Distance (in grid units) to the next higher boundary is
            // `(1 - frac)`; dividing by the positive `dir` gives the ray param.
            let dist_to_boundary = 1.0 - frac;
            Self {
                coord,
                step: 1,
                t_max: dist_to_boundary / dir,
                t_delta: 1.0 / dir,
            }
        } else {
            // Negative direction: the next boundary is the lower face of this
            // cell, `frac` grid units away; `-dir` is the positive speed.
            let speed = -dir;
            Self {
                coord,
                step: -1,
                t_max: frac / speed,
                t_delta: 1.0 / speed,
            }
        }
    }
}

/// Walks the ray through the voxel grid and returns the visited voxels in the
/// order the ray crosses them, each paired with its entry ray parameter.
///
/// The direction is normalized internally, so every returned `t_enter` is a
/// world-space distance from [`Ray::origin`]. The walk starts at the voxel
/// containing the origin (its `t_enter` is `0`) and proceeds outward, advancing
/// the axis with the nearest boundary each step (Amanatides-Woo). It stops when
/// the next entry parameter would exceed `limit.max_distance`, when
/// `limit.max_steps` voxels have been emitted, or immediately for a degenerate
/// ray (direction shorter than [`EPS`]) or a fully unbounded [`Limit::NONE`].
///
/// The result is deterministic: identical inputs always produce an identical
/// [`Vec`].
#[must_use]
pub fn traverse(ray: Ray, grid: VoxelGrid, limit: Limit) -> Vec<VoxelHit> {
    let mut out = Vec::new();

    // An unbounded limit cannot be safely enumerated: refuse by contract.
    if limit.max_distance.is_none() && limit.max_steps.is_none() {
        return out;
    }
    // A zero step cap can never emit a voxel.
    if let Some(0) = limit.max_steps {
        return out;
    }
    // A non-positive distance bound admits no voxel entry beyond the origin's.
    // We still allow the origin voxel at t = 0 when distance is exactly 0, but a
    // negative distance is empty.
    if let Some(d) = limit.max_distance
        && d < 0.0
    {
        return out;
    }

    // Normalize the direction so `t` reads as world distance. A degenerate ray
    // has no defined traversal.
    let dir = ray.dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return out;
    }

    // Work in grid space (cell units) so each axis' unit step is one cell.
    let cell = grid.safe_cell();
    let origin_g = grid.to_grid_space(ray.origin);
    // The grid-space direction: world direction divided per-axis by cell size.
    // Guarded because `cell` is already floored at `EPS`.
    let dir_g = Vec3::new(dir.x / cell.x, dir.y / cell.y, dir.z / cell.z);

    let mut ax = AxisState::seed(origin_g.x, dir_g.x);
    let mut ay = AxisState::seed(origin_g.y, dir_g.y);
    let mut az = AxisState::seed(origin_g.z, dir_g.z);

    let max_distance = limit.max_distance.unwrap_or(f32::INFINITY);
    let max_steps = limit.max_steps.unwrap_or(usize::MAX);

    // Emit the origin voxel first (entered at t = 0), respecting the caps.
    let mut t_enter = 0.0_f32;
    out.push(VoxelHit::new(
        IVec3::new(ax.coord, ay.coord, az.coord),
        t_enter,
    ));

    while out.len() < max_steps {
        // Advance along whichever axis has the nearest upcoming boundary. Ties
        // are resolved x, then y, then z for determinism; a parallel axis holds
        // `INFINITY` and is never selected.
        let next_t = ax.t_max.min(ay.t_max).min(az.t_max);

        // No finite boundary remains (all axes parallel, or numerically at
        // infinity): the ray never enters another voxel.
        if next_t >= f32::INFINITY {
            break;
        }
        // Stepping past the distance bound ends the walk.
        if next_t > max_distance {
            break;
        }

        // Select the axis and step it. Because `next_t` equals at least one of
        // the three `t_max` values, exactly one branch (the first matching, in
        // x/y/z order) fires, changing a single coordinate by `±1`.
        if ax.t_max <= ay.t_max && ax.t_max <= az.t_max {
            ax.coord += ax.step;
            ax.t_max += ax.t_delta;
        } else if ay.t_max <= az.t_max {
            ay.coord += ay.step;
            ay.t_max += ay.t_delta;
        } else {
            az.coord += az.step;
            az.t_max += az.t_delta;
        }

        t_enter = next_t;
        out.push(VoxelHit::new(
            IVec3::new(ax.coord, ay.coord, az.coord),
            t_enter,
        ));
    }

    out
}

/// Returns just the ordered voxel coordinates visited by [`traverse`],
/// discarding the entry parameters.
///
/// A convenience for callers that only need the cell path (e.g. marking grid
/// occupancy) and not the distances.
#[must_use]
pub fn voxels(ray: Ray, grid: VoxelGrid, limit: Limit) -> Vec<IVec3> {
    traverse(ray, grid, limit)
        .into_iter()
        .map(|h| h.voxel)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for comparing `t` values in assertions.
    const T_EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= T_EPS
    }

    fn unit_grid() -> VoxelGrid {
        VoxelGrid::cubic(Vec3::ZERO, 1.0)
    }

    #[test]
    fn vec3_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 8.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 8.0, 11.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 4.0, 5.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(Vec3::splat(-1.0), Vec3::new(-1.0, -1.0, -1.0));
    }

    #[test]
    fn vec3_length_uses_sqrt() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        assert!(approx(a.length(), 5.0));
    }

    #[test]
    fn normalize_zero_is_zero_not_nan() {
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        let n = Vec3::new(0.0, 0.0, 7.0).normalize_or_zero();
        assert_eq!(n, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn ray_point_at_advances_along_dir() {
        let r = Ray::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(r.point_at(3.0), Vec3::new(4.0, 1.0, 1.0));
    }

    #[test]
    fn floor_i32_matches_math_floor_including_negatives() {
        assert_eq!(floor_i32(0.0), 0);
        assert_eq!(floor_i32(0.9), 0);
        assert_eq!(floor_i32(1.0), 1);
        assert_eq!(floor_i32(-0.1), -1);
        assert_eq!(floor_i32(-1.0), -1);
        assert_eq!(floor_i32(-2.5), -3);
    }

    #[test]
    fn floor_i32_saturates_on_non_finite_and_extremes() {
        assert_eq!(floor_i32(f32::INFINITY), i32::MAX);
        assert_eq!(floor_i32(f32::NEG_INFINITY), i32::MIN);
        assert_eq!(floor_i32(1e30), i32::MAX);
        assert_eq!(floor_i32(-1e30), i32::MIN);
    }

    #[test]
    fn voxel_of_snaps_world_point_to_cell() {
        let grid = unit_grid();
        assert_eq!(grid.voxel_of(Vec3::new(0.5, 0.5, 0.5)), IVec3::new(0, 0, 0));
        assert_eq!(grid.voxel_of(Vec3::new(1.2, 2.9, 0.0)), IVec3::new(1, 2, 0));
        assert_eq!(
            grid.voxel_of(Vec3::new(-0.1, -1.0, -2.5)),
            IVec3::new(-1, -1, -3)
        );
    }

    #[test]
    fn voxel_of_respects_non_unit_cell_and_origin() {
        let grid = VoxelGrid::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 5.0));
        // world (13, 5, 12) -> rel (3, 5, 12) -> grid (1.5, 1.25, 2.4)
        assert_eq!(
            grid.voxel_of(Vec3::new(13.0, 5.0, 12.0)),
            IVec3::new(1, 1, 2)
        );
    }

    #[test]
    fn axis_aligned_x_ray_visits_consecutive_cells() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::steps(4));
        let coords: Vec<IVec3> = hits.iter().map(|h| h.voxel).collect();
        assert_eq!(
            coords,
            [
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(2, 0, 0),
                IVec3::new(3, 0, 0),
            ]
        );
        // y and z never change for an x-parallel ray.
        for h in &hits {
            assert_eq!(h.voxel.y, 0);
            assert_eq!(h.voxel.z, 0);
        }
    }

    #[test]
    fn axis_aligned_y_ray_visits_consecutive_cells() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 1.0, 0.0));
        let coords = voxels(ray, grid, Limit::steps(3));
        assert_eq!(
            coords,
            [
                IVec3::new(0, 0, 0),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 2, 0),
            ]
        );
    }

    #[test]
    fn axis_aligned_z_ray_visits_consecutive_cells() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, 1.0));
        let coords = voxels(ray, grid, Limit::steps(3));
        assert_eq!(
            coords,
            [
                IVec3::new(0, 0, 0),
                IVec3::new(0, 0, 1),
                IVec3::new(0, 0, 2),
            ]
        );
    }

    #[test]
    fn t_enter_values_are_world_distances_for_axis_ray() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::steps(4));
        // Boundaries at x = 1, 2, 3 -> t = 0.5, 1.5, 2.5 from origin at x=0.5.
        assert!(approx(hits[0].t_enter, 0.0));
        assert!(approx(hits[1].t_enter, 0.5));
        assert!(approx(hits[2].t_enter, 1.5));
        assert!(approx(hits[3].t_enter, 2.5));
    }

    #[test]
    fn diagonal_ray_t_enter_is_monotonic_nondecreasing() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.1, 0.1, 0.1), Vec3::new(1.0, 1.0, 1.0));
        let hits = traverse(ray, grid, Limit::steps(30));
        assert!(hits.len() >= 2);
        for w in hits.windows(2) {
            assert!(w[1].t_enter >= w[0].t_enter - T_EPS);
        }
    }

    #[test]
    fn perfect_diagonal_crosses_corners_in_order() {
        let grid = unit_grid();
        // Start off-corner so the three boundaries at each integer coincide.
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 1.0, 1.0));
        let coords = voxels(ray, grid, Limit::steps(10));
        assert_eq!(coords[0], IVec3::new(0, 0, 0));
        // The diagonal must eventually reach (n, n, n) for growing n, and each
        // step changes exactly one coordinate by +1.
        let last = *coords.last().unwrap();
        assert!(last.x >= 2 && last.y >= 2 && last.z >= 2);
    }

    #[test]
    fn negative_direction_steps_down() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(3.5, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0));
        let coords = voxels(ray, grid, Limit::steps(4));
        assert_eq!(
            coords,
            [
                IVec3::new(3, 0, 0),
                IVec3::new(2, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, 0, 0),
            ]
        );
    }

    #[test]
    fn negative_direction_into_negative_octant() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0));
        let coords = voxels(ray, grid, Limit::steps(3));
        assert_eq!(
            coords,
            [
                IVec3::new(0, 0, 0),
                IVec3::new(-1, 0, 0),
                IVec3::new(-2, 0, 0),
            ]
        );
    }

    #[test]
    fn negative_direction_t_enter_values() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(3.5, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::steps(3));
        // Boundaries at x = 3, 2 -> t = 0.5, 1.5.
        assert!(approx(hits[0].t_enter, 0.0));
        assert!(approx(hits[1].t_enter, 0.5));
        assert!(approx(hits[2].t_enter, 1.5));
    }

    #[test]
    fn max_steps_of_one_returns_only_origin_voxel() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::steps(1));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].voxel, IVec3::new(0, 0, 0));
        assert!(approx(hits[0].t_enter, 0.0));
    }

    #[test]
    fn max_steps_zero_is_empty() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(traverse(ray, grid, Limit::steps(0)).is_empty());
    }

    #[test]
    fn origin_voxel_is_first_and_contains_origin() {
        let grid = unit_grid();
        let origin = Vec3::new(2.3, 4.7, 1.1);
        let ray = Ray::new(origin, Vec3::new(0.3, -0.7, 0.6));
        let hits = traverse(ray, grid, Limit::steps(5));
        assert_eq!(hits[0].voxel, grid.voxel_of(origin));
        assert!(approx(hits[0].t_enter, 0.0));
    }

    #[test]
    fn hits_are_sorted_by_t_enter() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.2, 0.7, 0.4), Vec3::new(0.6, 1.0, -0.3));
        let hits = traverse(ray, grid, Limit::steps(40));
        for w in hits.windows(2) {
            assert!(w[0].t_enter <= w[1].t_enter + T_EPS);
        }
    }

    #[test]
    fn no_duplicate_voxels() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.15, 0.35, 0.85), Vec3::new(0.7, 0.9, 1.3));
        let coords = voxels(ray, grid, Limit::steps(50));
        let mut seen = Vec::new();
        for c in &coords {
            assert!(!seen.contains(c), "duplicate voxel {c:?}");
            seen.push(*c);
        }
    }

    #[test]
    fn consecutive_voxels_are_six_connected() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, 0.6, 0.4));
        let coords = voxels(ray, grid, Limit::steps(40));
        for w in coords.windows(2) {
            let dx = (w[1].x - w[0].x).abs();
            let dy = (w[1].y - w[0].y).abs();
            let dz = (w[1].z - w[0].z).abs();
            assert_eq!(dx + dy + dz, 1, "step {:?} -> {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn distance_limit_bounds_the_walk() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        // Boundaries at t = 0.5, 1.5, 2.5; a 2.0 cap admits t=0.5 and t=1.5.
        let hits = traverse(ray, grid, Limit::distance(2.0));
        let coords: Vec<IVec3> = hits.iter().map(|h| h.voxel).collect();
        assert_eq!(
            coords,
            [
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(2, 0, 0),
            ]
        );
        for h in &hits {
            assert!(h.t_enter <= 2.0 + T_EPS);
        }
    }

    #[test]
    fn both_limits_tighter_one_wins_steps() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        // Distance would allow many, but the 2-step cap wins.
        let hits = traverse(ray, grid, Limit::both(100.0, 2));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn both_limits_tighter_one_wins_distance() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        // Steps allow many, but the distance cap of 1.0 admits only t<=1.0.
        let hits = traverse(ray, grid, Limit::both(1.0, 100));
        // Origin (t=0) and the x=1 boundary (t=0.5) qualify; t=1.5 does not.
        assert_eq!(hits.len(), 2);
        assert!(approx(hits[1].t_enter, 0.5));
    }

    #[test]
    fn limit_none_is_empty_by_contract() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(traverse(ray, grid, Limit::NONE).is_empty());
    }

    #[test]
    fn degenerate_zero_direction_is_empty() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO);
        assert!(traverse(ray, grid, Limit::steps(10)).is_empty());
    }

    #[test]
    fn negative_distance_is_empty() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(traverse(ray, grid, Limit::distance(-1.0)).is_empty());
    }

    #[test]
    fn zero_distance_returns_only_origin() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::distance(0.0));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].voxel, IVec3::new(0, 0, 0));
    }

    #[test]
    fn non_unit_cell_scales_t_enter() {
        // Cells of edge 2 along x: boundaries at world x = 2, 4, ...
        let grid = VoxelGrid::cubic(Vec3::ZERO, 2.0);
        let ray = Ray::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        let hits = traverse(ray, grid, Limit::steps(3));
        assert_eq!(hits[0].voxel, IVec3::new(0, 0, 0));
        assert_eq!(hits[1].voxel, IVec3::new(1, 0, 0));
        // Next boundary at world x = 2, origin x = 1 -> t = 1.0.
        assert!(approx(hits[1].t_enter, 1.0));
        // Then world x = 4 -> t = 3.0.
        assert!(approx(hits[2].t_enter, 3.0));
    }

    #[test]
    fn grid_origin_offset_is_respected() {
        let grid = VoxelGrid::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::splat(1.0));
        let ray = Ray::new(Vec3::new(-4.5, -4.5, -4.5), Vec3::new(1.0, 0.0, 0.0));
        let coords = voxels(ray, grid, Limit::steps(2));
        // world -4.5 -> rel 0.5 -> voxel 0.
        assert_eq!(coords[0], IVec3::new(0, 0, 0));
        assert_eq!(coords[1], IVec3::new(1, 0, 0));
    }

    #[test]
    fn traversal_is_deterministic() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.13, 0.71, 0.42), Vec3::new(0.5, -0.9, 1.1));
        let a = traverse(ray, grid, Limit::steps(37));
        let b = traverse(ray, grid, Limit::steps(37));
        assert_eq!(a.len(), b.len());
        for (ha, hb) in a.iter().zip(b.iter()) {
            assert_eq!(ha.voxel, hb.voxel);
            assert!(approx(ha.t_enter, hb.t_enter));
        }
    }

    #[test]
    fn axis_parallel_ray_does_not_divide_by_zero() {
        // Direction exactly along z; x and y are parallel axes (t_max = inf).
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, 1.0));
        let hits = traverse(ray, grid, Limit::steps(5));
        for h in &hits {
            assert!(h.t_enter.is_finite());
            assert_eq!(h.voxel.x, 0);
            assert_eq!(h.voxel.y, 0);
        }
    }

    #[test]
    fn near_zero_direction_component_treated_as_parallel() {
        // A tiny x component below EPS must be treated as parallel (fixed x).
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1e-9, 0.0, 1.0));
        let hits = traverse(ray, grid, Limit::steps(5));
        for h in &hits {
            assert_eq!(h.voxel.x, 0);
            assert!(h.t_enter.is_finite());
        }
    }

    #[test]
    fn voxels_helper_matches_traverse_coords() {
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.2, 0.3, 0.4), Vec3::new(1.0, 0.7, 0.2));
        let full = traverse(ray, grid, Limit::steps(20));
        let just = voxels(ray, grid, Limit::steps(20));
        let mapped: Vec<IVec3> = full.iter().map(|h| h.voxel).collect();
        assert_eq!(mapped, just);
    }

    #[test]
    fn diagonal_covers_expected_manhattan_progress() {
        // Over N steps the sum of coordinate deltas equals N-1 (each step +1).
        let grid = unit_grid();
        let ray = Ray::new(Vec3::new(0.1, 0.2, 0.05), Vec3::new(1.0, 2.0, 0.5));
        let coords = voxels(ray, grid, Limit::steps(25));
        let first = coords[0];
        let last = *coords.last().unwrap();
        let manhattan =
            (last.x - first.x).abs() + (last.y - first.y).abs() + (last.z - first.z).abs();
        assert_eq!(manhattan as usize, coords.len() - 1);
    }
}
