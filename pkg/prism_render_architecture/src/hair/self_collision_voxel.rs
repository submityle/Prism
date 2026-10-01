//! Voxel-density approximate strand self-collision (design doc §8.5 item3).
//!
//! A dense groom has far too many strands for an all-pairs `O(n^2)`
//! strand-vs-strand test, yet neglecting self-collision lets hair clumps
//! collapse into each other and lose their fluffy volume. The production answer
//! (AMD `TressFX`, and the volumetric hair work of `Petrovic` et al. 2005) is to
//! stop treating hair as discrete contacts and instead treat it as a *density
//! field*: scatter ("splat") every strand particle into a low-resolution voxel
//! grid, then let the field drive three cheap, grid-local forces that together
//! approximate large-scale self-collision in `O(n)`:
//!
//! 1. **Density-gradient repulsion** — where the splatted density `rho` exceeds
//!    a rest threshold the hair is overcrowded, so each particle is pushed along
//!    `-grad(rho)` (down the density gradient) out of the pile-up. This is the
//!    volumetric stand-in for pairwise push-apart: a particle never needs to
//!    know *which* strands it overlaps, only that it sits in a locally dense
//!    region.
//! 2. **Volume preservation** — a soft, two-sided restore toward the rest
//!    density keeps the groom's authored "poofiness": over-dense regions spread
//!    out (down the gradient) and under-dense regions clump back together (up
//!    the gradient), so the style neither pancakes nor explodes over time.
//! 3. **Hair-hair friction** — a velocity field is splatted alongside the
//!    density (mass-weighted), and each particle's velocity is blended toward
//!    the local average with a friction coefficient `mu in [0, 1]`, the
//!    volumetric approximation of the viscous drag of hair sliding against hair.
//!
//! This module is a deterministic `CPU` *golden*: array in, array out, with a
//! fixed evaluation order so identical inputs produce bit-identical outputs and
//! results can be golden-tested by hand (design §7). It performs **no**
//! transcendental math — only multiplies, `sqrt` for vector length, and integer
//! lattice quantisation — so it needs no `libm` determinism shim. Every input is
//! sanitised: a non-positive or non-finite `cell_size` or a zero `dims` is
//! repaired, out-of-grid particles contribute nothing and are left untouched,
//! and `NaN`/infinite components are scrubbed, so no groom ever panics here.
//!
//! The types here are intentionally self-contained and **disjoint** from the
//! clustering / virtual-geometry grids (e.g. [`super::cluster::ClusterGrid`]):
//! that grid quantises strand *roots* for `LOD` bucketing, whereas this grid is
//! a continuous *density accumulator*, so sharing a type would conflate two
//! different contracts. The quantisation idiom (`floor` to an integer lattice)
//! and the `sanitized()` discipline mirror those modules deliberately.

use alloc::vec::Vec;

/// Shared epsilon for zero-length / near-zero-density guards and tests.
const EPS: f32 = 1.0e-6;

// ---------------------------------------------------------------------------
// Hand-written 3-vector (the crate carries no linear-algebra dependency).
// ---------------------------------------------------------------------------

/// A minimal 3-component vector for voxel self-collision math.
///
/// All operations are plain `f32` arithmetic in a fixed order, which is what
/// makes the whole pass bit-for-bit reproducible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean length (no `sqrt`); the square is written `x*x` so no
    /// disallowed `powi` is needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector along `self`, or [`Vec3::ZERO`] when `self` is numerically
    /// zero, so normalisation never yields a `NaN` or divides by zero.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS * EPS {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }

    /// Whether every component is finite (no `NaN`, no infinity).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// This vector with every non-finite component repaired to `0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            sanitize_coord(self.x),
            sanitize_coord(self.y),
            sanitize_coord(self.z),
        )
    }

    /// Row-major component array.
    #[must_use]
    fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Builds a vector from a component array.
    #[must_use]
    fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

#[must_use]
fn sanitize_coord(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Non-negative finite repair: non-finite or negative values collapse to `0`.
#[must_use]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Clamp to `0..=1`, repairing non-finite values to `0`.
#[must_use]
fn sanitize_unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Voxel grid.
// ---------------------------------------------------------------------------

/// A low-resolution uniform voxel grid for density splatting.
///
/// The grid covers the axis-aligned box `[origin, origin + dims * cell_size]`.
/// Each cell owns one density sample located at its **centre**
/// (`origin + (i + 0.5) * cell_size` on each axis), so splat/gather use
/// trilinear weights about the cell-centre lattice. Invariant after
/// [`VoxelGrid::sanitized`]: `cell_size` is finite and `> 0`, and every `dims`
/// axis is `>= 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    /// World-space minimum corner of the grid.
    pub origin: Vec3,
    /// Edge length of one cubic cell in world units (`> 0` after sanitising).
    pub cell_size: f32,
    /// Cell counts along x, y, z (`>= 1` each after sanitising).
    pub dims: [u32; 3],
}

impl Default for VoxelGrid {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            cell_size: 1.0,
            dims: [1, 1, 1],
        }
    }
}

impl VoxelGrid {
    /// A grid from an explicit origin, cell size, and cell counts.
    #[must_use]
    pub const fn new(origin: Vec3, cell_size: f32, dims: [u32; 3]) -> Self {
        Self {
            origin,
            cell_size,
            dims,
        }
    }

    /// This grid with its `origin` finite, `cell_size` repaired to `1.0` when
    /// non-finite or non-positive, and every `dims` axis forced to at least `1`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let cell_size = if self.cell_size.is_finite() && self.cell_size > 0.0 {
            self.cell_size
        } else {
            1.0
        };
        Self {
            origin: self.origin.sanitized(),
            cell_size,
            dims: [
                self.dims[0].max(1),
                self.dims[1].max(1),
                self.dims[2].max(1),
            ],
        }
    }

    /// Total cell count (`dims.x * dims.y * dims.z`), computed on the sanitised
    /// grid so it is always `>= 1`.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        let g = self.sanitized();
        (g.dims[0] as usize) * (g.dims[1] as usize) * (g.dims[2] as usize)
    }

    /// Whether an integer cell coordinate lies inside the grid.
    #[must_use]
    pub fn in_bounds(&self, cell: [i32; 3]) -> bool {
        let g = self.sanitized();
        cell[0] >= 0
            && cell[1] >= 0
            && cell[2] >= 0
            && (cell[0] as i64) < (g.dims[0] as i64)
            && (cell[1] as i64) < (g.dims[1] as i64)
            && (cell[2] as i64) < (g.dims[2] as i64)
    }

    /// Flat row-major index of a cell, or `None` when it is out of bounds.
    ///
    /// Layout is `x + dims.x * (y + dims.y * z)`, matching the storage order of
    /// [`DensityField::cells`].
    #[must_use]
    pub fn index_of(&self, cell: [i32; 3]) -> Option<usize> {
        if !self.in_bounds(cell) {
            return None;
        }
        let g = self.sanitized();
        let x = cell[0] as usize;
        let y = cell[1] as usize;
        let z = cell[2] as usize;
        Some(x + (g.dims[0] as usize) * (y + (g.dims[1] as usize) * z))
    }

    /// World-space maximum corner (`origin + dims * cell_size`).
    #[must_use]
    fn world_max(&self) -> Vec3 {
        let g = self.sanitized();
        Vec3::new(
            g.origin.x + (g.dims[0] as f32) * g.cell_size,
            g.origin.y + (g.dims[1] as f32) * g.cell_size,
            g.origin.z + (g.dims[2] as f32) * g.cell_size,
        )
    }

    /// Clamps a world position into the grid's axis-aligned box, so a stencil
    /// sample can never step fully outside the field (the boundary clamp used by
    /// the central-difference gradient).
    #[must_use]
    fn clamp_to_bounds(&self, pos: Vec3) -> Vec3 {
        let g = self.sanitized();
        let hi = g.world_max();
        Vec3::new(
            pos.x.clamp(g.origin.x, hi.x),
            pos.y.clamp(g.origin.y, hi.y),
            pos.z.clamp(g.origin.z, hi.z),
        )
    }
}

/// Continuous cell-centre coordinate of a world position plus the trilinear
/// fraction into the base cell. `base` is the floor on each axis; `frac` is in
/// `[0, 1)` per axis. Both derive from the *sanitised* grid.
#[must_use]
fn cell_coord(grid: VoxelGrid, pos: Vec3) -> ([i32; 3], [f32; 3]) {
    let g = grid.sanitized();
    let p = pos.sanitized();
    let inv = 1.0 / g.cell_size;
    let gx = (p.x - g.origin.x) * inv - 0.5;
    let gy = (p.y - g.origin.y) * inv - 0.5;
    let gz = (p.z - g.origin.z) * inv - 0.5;
    let bx = gx.floor();
    let by = gy.floor();
    let bz = gz.floor();
    (
        [bx as i32, by as i32, bz as i32],
        [gx - bx, gy - by, gz - bz],
    )
}

/// Invokes `f(flat_index, weight)` for each of the up-to-eight trilinear
/// neighbour cells of `pos`, skipping any corner that falls outside the grid.
///
/// Across the full `2x2x2` stencil the weights sum to `1`, so a particle wholly
/// inside the grid deposits its entire mass (weights conserved); corners that
/// fall outside the grid are silently dropped, which is how out-of-bounds mass
/// is discarded without a panic. The corner order (x fastest, then y, then z) is
/// fixed, so scatter-add order is deterministic.
fn for_each_corner(grid: VoxelGrid, pos: Vec3, mut f: impl FnMut(usize, f32)) {
    let (base, frac) = cell_coord(grid, pos);
    for cz in 0..2u32 {
        let wz = if cz == 0 { 1.0 - frac[2] } else { frac[2] };
        for cy in 0..2u32 {
            let wy = if cy == 0 { 1.0 - frac[1] } else { frac[1] };
            for cx in 0..2u32 {
                let wx = if cx == 0 { 1.0 - frac[0] } else { frac[0] };
                let w = wx * wy * wz;
                let cell = [
                    base[0] + cx as i32,
                    base[1] + cy as i32,
                    base[2] + cz as i32,
                ];
                if let Some(idx) = grid.index_of(cell) {
                    f(idx, w);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hair point (the per-particle state this pass reads and writes).
// ---------------------------------------------------------------------------

/// One strand particle seen by the voxel self-collision pass.
///
/// `mass` weights both the density splat and the velocity-field average (so
/// heavier guide points dominate the local mean velocity). A non-positive mass
/// contributes nothing to the field but the particle is still advected by the
/// forces it samples. Non-finite positions are treated as invalid and left
/// untouched.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HairPoint {
    /// World-space position (updated in place by repulsion / volume terms).
    pub position: Vec3,
    /// World-space velocity (updated in place by the friction term).
    pub velocity: Vec3,
    /// Splat mass; non-positive values deposit no density.
    pub mass: f32,
}

impl HairPoint {
    /// A unit-mass particle with a given position and velocity.
    #[must_use]
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position,
            velocity,
            mass: 1.0,
        }
    }

    /// A unit-mass, zero-velocity particle at `position`.
    #[must_use]
    pub fn at(position: Vec3) -> Self {
        Self::new(position, Vec3::ZERO)
    }

    /// Whether this particle has a finite position (the gate for splatting and
    /// advecting it).
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.position.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Density + velocity field.
// ---------------------------------------------------------------------------

/// The splatted scalar density field `rho` and its companion mass-weighted
/// velocity field, both laid out row-major (see [`VoxelGrid::index_of`]).
///
/// `cells[i]` is the accumulated splat mass of cell `i`. `velocity[i]` is the
/// mass-weighted velocity *sum* of that cell (not yet divided by density);
/// [`DensityField::sample_avg_velocity`] performs the gather-and-divide to
/// recover a local average. Both vectors have exactly [`VoxelGrid::cell_count`]
/// entries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DensityField {
    /// Cell counts the field was built for (matches the grid's sanitised dims).
    pub dims: [u32; 3],
    /// Per-cell accumulated density (splat mass).
    pub cells: Vec<f32>,
    /// Per-cell mass-weighted velocity sum.
    pub velocity: Vec<Vec3>,
}

impl DensityField {
    /// An all-zero field sized for `grid` (one entry per cell).
    #[must_use]
    pub fn zeros(grid: VoxelGrid) -> Self {
        let g = grid.sanitized();
        let count = grid.cell_count();
        let mut cells = Vec::with_capacity(count);
        let mut velocity = Vec::with_capacity(count);
        for _ in 0..count {
            cells.push(0.0);
            velocity.push(Vec3::ZERO);
        }
        Self {
            dims: g.dims,
            cells,
            velocity,
        }
    }

    /// Number of cells in the field.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the field holds no cells (only for a wholly degenerate build).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Trilinearly samples the density at a world position. Corners outside the
    /// grid contribute `0` (density vanishes outside the field).
    #[must_use]
    pub fn sample_density(&self, grid: VoxelGrid, pos: Vec3) -> f32 {
        let mut sum = 0.0;
        for_each_corner(grid, pos, |idx, w| {
            if let Some(c) = self.cells.get(idx) {
                sum += w * *c;
            }
        });
        sum
    }

    /// Local mass-weighted average velocity at a world position, or
    /// [`Vec3::ZERO`] where there is no nearby mass (density `<= EPS`).
    #[must_use]
    pub fn sample_avg_velocity(&self, grid: VoxelGrid, pos: Vec3) -> Vec3 {
        let mut density = 0.0;
        let mut vel = Vec3::ZERO;
        for_each_corner(grid, pos, |idx, w| {
            if let Some(c) = self.cells.get(idx) {
                density += w * *c;
            }
            if let Some(v) = self.velocity.get(idx) {
                vel = vel.add(v.scale(w));
            }
        });
        if density > EPS {
            vel.scale(1.0 / density)
        } else {
            Vec3::ZERO
        }
    }
}

/// Splats every particle's mass (and mass-weighted velocity) into a fresh
/// [`DensityField`] with trilinear scatter-add.
///
/// Particles with a non-finite position, or a non-positive / non-finite mass,
/// deposit nothing (they cannot define a valid density). Each valid particle's
/// mass is distributed over its up-to-eight neighbour cells with trilinear
/// weights that sum to `1`, so an interior particle's mass is exactly conserved
/// and an out-of-grid particle's mass is harmlessly discarded. Particles are
/// visited in slice order and corners in a fixed order, so the result is
/// deterministic.
#[must_use]
pub fn splat_density(points: &[HairPoint], grid: VoxelGrid) -> DensityField {
    let grid = grid.sanitized();
    let mut field = DensityField::zeros(grid);
    for p in points {
        if !p.position.is_finite() {
            continue;
        }
        let mass = sanitize_nonneg(p.mass);
        if !(mass > 0.0) {
            continue;
        }
        let vel = p.velocity.sanitized();
        for_each_corner(grid, p.position, |idx, w| {
            if let Some(c) = field.cells.get_mut(idx) {
                *c += mass * w;
            }
            if let Some(v) = field.velocity.get_mut(idx) {
                *v = v.add(vel.scale(mass * w));
            }
        });
    }
    field
}

/// Central-difference density gradient `grad(rho)` at a world position, with the
/// stencil clamped to the grid's box at the boundary.
///
/// Each axis uses `(rho(p + h) - rho(p - h)) / (actual separation)` with
/// `h = cell_size`; when the clamp collapses the two sample points to within
/// [`EPS`] the derivative is reported as `0` rather than dividing by zero. The
/// returned vector points toward *increasing* density, so repulsion moves along
/// its negation.
#[must_use]
pub fn sample_gradient(field: &DensityField, grid: VoxelGrid, pos: Vec3) -> Vec3 {
    let grid = grid.sanitized();
    let h = grid.cell_size;
    let base = grid.clamp_to_bounds(pos).to_array();
    let mut g = [0.0f32; 3];
    for axis in 0..3 {
        let mut hi = base;
        let mut lo = base;
        hi[axis] += h;
        lo[axis] -= h;
        let hi_pos = grid.clamp_to_bounds(Vec3::from_array(hi)).to_array();
        let lo_pos = grid.clamp_to_bounds(Vec3::from_array(lo)).to_array();
        let d_hi = field.sample_density(grid, Vec3::from_array(hi_pos));
        let d_lo = field.sample_density(grid, Vec3::from_array(lo_pos));
        let denom = hi_pos[axis] - lo_pos[axis];
        g[axis] = if denom > EPS {
            (d_hi - d_lo) / denom
        } else {
            0.0
        };
    }
    Vec3::from_array(g)
}

// ---------------------------------------------------------------------------
// Parameters + the resolve pass.
// ---------------------------------------------------------------------------

/// Tunables for one voxel self-collision resolve.
///
/// Invariant after [`SelfCollisionParams::sanitized`]: `rest_density`,
/// `repulsion`, and `target_volume_gain` are finite and `>= 0`, and `friction`
/// is clamped to `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfCollisionParams {
    /// Target / threshold density. Density above this is "overcrowded" and
    /// drives repulsion; the volume term restores both over- and under-dense
    /// regions toward it.
    pub rest_density: f32,
    /// Strength of the one-sided density-gradient push-apart (per unit of
    /// over-density, as a displacement along `-grad(rho)`).
    pub repulsion: f32,
    /// Hair-hair friction `mu in [0, 1]`: the fraction of each particle's
    /// velocity replaced by the local average velocity (`0` = frictionless,
    /// `1` = snap to the local mean).
    pub friction: f32,
    /// Strength of the two-sided volume-preservation restore toward
    /// `rest_density` (keeps the groom's authored poofiness).
    pub target_volume_gain: f32,
}

impl Default for SelfCollisionParams {
    fn default() -> Self {
        Self {
            rest_density: 1.0,
            repulsion: 0.1,
            friction: 0.25,
            target_volume_gain: 0.05,
        }
    }
}

impl SelfCollisionParams {
    /// Explicit parameters.
    #[must_use]
    pub const fn new(
        rest_density: f32,
        repulsion: f32,
        friction: f32,
        target_volume_gain: f32,
    ) -> Self {
        Self {
            rest_density,
            repulsion,
            friction,
            target_volume_gain,
        }
    }

    /// Clamps every field into its valid range (see the type invariant).
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            rest_density: sanitize_nonneg(self.rest_density),
            repulsion: sanitize_nonneg(self.repulsion),
            friction: sanitize_unit(self.friction),
            target_volume_gain: sanitize_nonneg(self.target_volume_gain),
        }
    }
}

/// Resolves approximate strand self-collision in place over `points`.
///
/// One deterministic pass: (1) splat the current positions/velocities into a
/// density + velocity field; then, per particle, sample the field at its
/// *original* position and apply (2) density-gradient repulsion when it is
/// overcrowded, (3) a two-sided volume-preservation restore toward
/// `rest_density`, and (4) a friction blend of its velocity toward the local
/// average. The field is built once from the input state, so the result is a
/// pure function of the inputs regardless of iteration order.
///
/// Panic-free and sanitising throughout: an empty slice is a no-op, a particle
/// with a non-finite position is skipped (left exactly as-is), a particle with
/// no local mass around it (density `<= EPS`, e.g. one that splatted entirely
/// outside the grid) receives neither a push nor friction, and any displacement
/// or blended velocity that comes out non-finite is discarded rather than
/// written back.
pub fn resolve_self_collision(
    points: &mut [HairPoint],
    grid: VoxelGrid,
    params: SelfCollisionParams,
) {
    if points.is_empty() {
        return;
    }
    let grid = grid.sanitized();
    let params = params.sanitized();
    let field = splat_density(points, grid);

    for p in points.iter_mut() {
        if !p.position.is_finite() {
            continue;
        }
        // Scrub any non-finite incoming velocity to zero before it feeds the
        // friction blend, mirroring the density splat's velocity sanitising:
        // `inf * keep` would otherwise survive the finite-guard below and leak a
        // non-finite velocity back out.
        p.velocity = p.velocity.sanitized();
        let pos0 = p.position;
        let density = field.sample_density(grid, pos0);
        let gradient = sample_gradient(&field, grid, pos0);
        let dir = gradient.normalize_or_zero();
        let avg_vel = field.sample_avg_velocity(grid, pos0);

        // Signed over-density relative to the rest/target density.
        let excess = density - params.rest_density;

        // (2) Repulsion: one-sided push-apart only where overcrowded.
        let mut disp_mag = 0.0;
        if excess > 0.0 {
            disp_mag += params.repulsion * excess;
        }
        // (3) Volume preservation: two-sided soft restore toward rest density.
        // Positive excess (too dense) adds to the down-gradient push; negative
        // excess (too sparse) flips the sign below and pulls up-gradient, so a
        // thinning clump gathers back together instead of collapsing.
        disp_mag += params.target_volume_gain * excess;

        // Move down the density gradient for a positive magnitude (and up it for
        // a negative one). `dir` is unit length or zero (uniform / empty field),
        // so a flat region produces no spurious drift.
        let disp = dir.scale(-disp_mag);
        let new_pos = pos0.add(disp);
        if new_pos.is_finite() {
            p.position = new_pos;
        }

        // (4) Hair-hair friction: blend velocity toward the local average, but
        // only where there is a genuine neighbourhood (nonzero local density).
        if density > EPS {
            let keep = 1.0 - params.friction;
            let blended = p.velocity.scale(keep).add(avg_vel.scale(params.friction));
            if blended.is_finite() {
                p.velocity = blended;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    #[test]
    fn splat_single_point_conserves_mass() {
        // A particle wholly inside the grid deposits its entire mass: the
        // trilinear weights across the 8 corners sum to 1.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 4, 4]);
        let points = [HairPoint {
            position: Vec3::new(1.3, 2.1, 2.7),
            velocity: Vec3::ZERO,
            mass: 3.0,
        }];
        let field = splat_density(&points, grid);
        let total: f32 = field.cells.iter().copied().sum();
        assert!(close(total, 3.0));
        // Sampling the density back at the point is positive.
        assert!(field.sample_density(grid, points[0].position) > 0.0);
    }

    #[test]
    fn empty_particles_give_empty_field() {
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [2, 2, 2]);
        let field = splat_density(&[], grid);
        assert_eq!(field.len(), 8);
        let total: f32 = field.cells.iter().copied().sum();
        assert!(close(total, 0.0));
        for v in &field.velocity {
            assert!(close(v.length(), 0.0));
        }
        // Resolving an empty slice is a no-op and never panics.
        let mut none: [HairPoint; 0] = [];
        resolve_self_collision(&mut none, grid, SelfCollisionParams::default());
    }

    #[test]
    fn out_of_bounds_point_is_skipped() {
        // Far outside the grid: contributes no density and is left untouched.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [2, 2, 2]);
        let far = Vec3::new(100.0, 100.0, 100.0);
        let field = splat_density(&[HairPoint::at(far)], grid);
        let total: f32 = field.cells.iter().copied().sum();
        assert!(close(total, 0.0));

        let mut points = [HairPoint::new(far, Vec3::new(5.0, 0.0, 0.0))];
        resolve_self_collision(
            &mut points,
            grid,
            SelfCollisionParams::new(0.0, 1.0, 1.0, 1.0),
        );
        // Position and velocity unchanged (no local mass => no force).
        assert!(close(points[0].position.sub(far).length(), 0.0));
        assert!(close(points[0].velocity.x, 5.0));
    }

    #[test]
    fn two_near_points_repel_apart() {
        // Two particles in a 1D row of cells: the density piles up between them,
        // so each is pushed down the gradient away from the other.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
        let a = Vec3::new(1.5, 0.5, 0.5);
        let b = Vec3::new(2.0, 0.5, 0.5);
        let before = b.sub(a).length();
        let mut points = [HairPoint::at(a), HairPoint::at(b)];
        // Pure repulsion (rest_density 0 so all density is excess), no volume,
        // no friction, to isolate the push-apart.
        resolve_self_collision(
            &mut points,
            grid,
            SelfCollisionParams::new(0.0, 0.1, 0.0, 0.0),
        );
        let after = points[1].position.sub(points[0].position).length();
        assert!(after > before, "after={after} before={before}");
    }

    #[test]
    fn gradient_points_uphill() {
        // Hand-built field increasing along +x: gradient must point +x.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [3, 1, 1]);
        let mut field = DensityField::zeros(grid);
        field.cells[0] = 1.0;
        field.cells[1] = 2.0;
        field.cells[2] = 3.0;
        let g = sample_gradient(&field, grid, Vec3::new(1.5, 0.5, 0.5));
        assert!(g.x > 0.0);
        assert!(close(g.y, 0.0));
        assert!(close(g.z, 0.0));
    }

    #[test]
    fn friction_converges_velocities() {
        // Two co-located-ish particles with opposite velocities: with friction 1
        // each snaps to the local mass-weighted mean, so they converge.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
        let mut points = [
            HairPoint::new(Vec3::new(1.6, 0.5, 0.5), Vec3::new(2.0, 0.0, 0.0)),
            HairPoint::new(Vec3::new(1.9, 0.5, 0.5), Vec3::new(0.0, 0.0, 0.0)),
        ];
        let before = points[0].velocity.sub(points[1].velocity).length();
        resolve_self_collision(
            &mut points,
            grid,
            SelfCollisionParams::new(1.0e9, 0.0, 1.0, 0.0),
        );
        let after = points[0].velocity.sub(points[1].velocity).length();
        assert!(after < before, "after={after} before={before}");
    }

    #[test]
    fn volume_term_pulls_sparse_region_together() {
        // A lone particle in an under-dense field (rest_density far above its
        // local density) is pulled up-gradient toward denser hair, not pushed.
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
        // Two particles so there is a nonzero gradient to climb.
        let mut points = [
            HairPoint::at(Vec3::new(1.3, 0.5, 0.5)),
            HairPoint::at(Vec3::new(2.4, 0.5, 0.5)),
        ];
        let before = points[1].position.sub(points[0].position).length();
        // High rest_density => negative excess => up-gradient restore; no
        // repulsion so the volume term is isolated.
        resolve_self_collision(
            &mut points,
            grid,
            SelfCollisionParams::new(10.0, 0.0, 0.0, 0.05),
        );
        let after = points[1].position.sub(points[0].position).length();
        assert!(after < before, "after={after} before={before}");
    }

    #[test]
    fn nan_and_inf_are_sanitized() {
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [2, 2, 2]);
        // NaN position => skipped in splat, left untouched in resolve.
        let nan_pos = Vec3::new(f32::NAN, 0.0, 0.0);
        // Infinite velocity but finite position => velocity scrubbed in splat.
        let inf_vel = HairPoint::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(f32::INFINITY, 0.0, 0.0));
        let field = splat_density(&[HairPoint::at(nan_pos), inf_vel], grid);
        for v in &field.velocity {
            assert!(v.is_finite());
        }
        let total: f32 = field.cells.iter().copied().sum();
        assert!(total.is_finite());

        let mut points = [HairPoint::at(nan_pos), inf_vel];
        resolve_self_collision(&mut points, grid, SelfCollisionParams::default());
        // The NaN-position particle is still NaN (never advected / repaired).
        assert!(points[0].position.x.is_nan());
        // The finite particle stayed finite throughout.
        assert!(points[1].position.is_finite());
        assert!(points[1].velocity.is_finite());
    }

    #[test]
    fn degenerate_grid_and_params_do_not_panic() {
        // cell_size 0 and zero dims are repaired by sanitising.
        let grid = VoxelGrid::new(Vec3::new(f32::NAN, 0.0, 0.0), 0.0, [0, 0, 0]);
        let s = grid.sanitized();
        assert!(s.cell_size > 0.0);
        assert_eq!(s.dims, [1, 1, 1]);
        assert_eq!(grid.cell_count(), 1);

        // Non-finite / out-of-range params are clamped.
        let p = SelfCollisionParams::new(f32::NAN, -5.0, 10.0, f32::INFINITY).sanitized();
        assert!(close(p.rest_density, 0.0));
        assert!(close(p.repulsion, 0.0));
        assert!(close(p.friction, 1.0));
        assert!(close(p.target_volume_gain, 0.0));

        let mut points = [HairPoint::at(Vec3::new(0.2, 0.2, 0.2))];
        resolve_self_collision(&mut points, grid, p);
        assert!(points[0].position.is_finite());
    }

    #[test]
    fn index_of_bounds_and_layout() {
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [2, 3, 4]);
        assert_eq!(grid.index_of([0, 0, 0]), Some(0));
        assert_eq!(grid.index_of([1, 0, 0]), Some(1));
        assert_eq!(grid.index_of([0, 1, 0]), Some(2));
        assert_eq!(grid.index_of([0, 0, 1]), Some(6));
        assert_eq!(grid.index_of([-1, 0, 0]), None);
        assert_eq!(grid.index_of([2, 0, 0]), None);
        assert!(!grid.in_bounds([0, 3, 0]));
        assert_eq!(grid.cell_count(), 24);
    }

    #[test]
    fn uniform_density_has_no_drift() {
        // A spatially symmetric pair about a node: with no net gradient the
        // repulsion/volume push stays bounded and finite (never NaN).
        let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 4, 4]);
        let mut points = [HairPoint::at(Vec3::new(2.0, 2.0, 2.0))];
        resolve_self_collision(&mut points, grid, SelfCollisionParams::default());
        assert!(points[0].position.is_finite());
    }
}
