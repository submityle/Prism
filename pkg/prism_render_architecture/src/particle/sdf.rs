//! Discrete scalar signed-distance-field (`SDF`) / `VDB` grid sampling, surface
//! emission, sticking, and penetration *queries* — the `CPU` reference behind
//! [`super::modules::BuiltinModule::SpawnSdfSurface`] and the query half of
//! [`super::modules::BuiltinModule::CollisionSdf`] (design §8.3).
//!
//! # Relationship to the sibling field/collision modules
//!
//! Three modules touch signed-distance and volumetric data; they are strictly
//! orthogonal:
//!
//! * [`super::collision`] owns the *analytic-primitive* `SDF`s (plane / sphere /
//!   box) **and every collision response** (push-out, restitution, friction).
//!   It resolves contacts; it does not sample a baked grid.
//! * [`super::vector_field`] samples a discrete **vector** `Texture3d` (a flow
//!   field of `Vec3` texels) for forces. It carries directions, not distances.
//! * **This module** samples a discrete **scalar** `SDF`/`VDB` grid (one signed
//!   distance per texel, negative inside), and layers on the geometry that a
//!   scalar field enables: trilinear distance reconstruction, a
//!   gradient-derived surface normal, `Niagara`-style *spawn on `SDF` surface*
//!   emission, surface *sticking*, and a sphere-versus-field *penetration
//!   query*. The query returns a [`Contact`]; the actual response still belongs
//!   to [`super::collision`], so the two never overlap.
//!
//! # Storage and transform convention
//!
//! The grid follows the same convention as [`super::vector_field`]: a
//! row-major (`X`-fastest) buffer whose texel centres sit on integer grid
//! coordinates, mapped through an axis-aligned box (an [`Aabb`]) so the box
//! spans the full grid (the `Niagara`/`UE` volume bounds convention). Only the
//! payload differs — one `f32` distance per texel instead of a `Vec3`.
//!
//! # Determinism
//!
//! Everything here is pure and deterministic. The only floating-point
//! primitives beyond ordinary multiply-add are `f32::floor` (integer grid
//! location) and `sqrt` (through the hand-rolled [`Vec3`] math); there are no
//! transcendental calls (`sin`/`cos`/`exp`/`ln`/`pow`), so a `GPU` kernel that
//! samples the same `SDF`/`VDB` texture in the same order is bit-reproducible
//! against this reference. Randomness for surface emission is drawn from a
//! shared [`UnitCursor`] so the `CPU` and `GPU` spawn paths agree.

use alloc::vec;
use alloc::vec::Vec;

use super::emitter::UnitCursor;
use super::sort_cull::Aabb;
use super::Vec3;

/// Absolute tolerance for the `f32` comparison guards in this module
/// (degenerate-extent detection, near-zero displacement clamping, gradient
/// degeneracy). Comparisons use `(a - b).abs() < EPS` rather than a bare `==`.
pub const EPS: f32 = 1e-6;

/// Bounded fallback for the surface-emission rejection loop so a pathological
/// band (empty grid, no surface inside the box) can never spin forever; after
/// this many rejects the last candidate is projected instead. Mirrors the
/// `MAX_REJECTION_TRIES` pattern used by [`super::emitter`].
const MAX_REJECTION_TRIES: u32 = 256;

/// Half-texel finite-difference step used by [`SdfField::gradient`]. Sampling
/// the trilinear field at the midpoints on either side of a coordinate yields
/// the gradient of the reconstructed field without landing exactly on the
/// (piecewise-discontinuous) integer grid planes.
const GRAD_STEP: f32 = 0.5;

/// The boundary policy applied when a sample lands outside the `[0, dims)` grid
/// range on any axis (design §8.3, §10).
///
/// Defined locally rather than reused from [`super::vector_field`] so the
/// scalar `SDF` sampler carries no dependency on the vector-field module's
/// private index resolver.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WrapMode {
    /// Clamp the sample to the nearest edge texel, so the field extends its
    /// border distance outward indefinitely (the common default for a bounded
    /// baked `SDF`/`VDB` volume).
    Clamp,
    /// Tile the field periodically by wrapping indices with an integer modulo,
    /// so the volume repeats seamlessly in every direction.
    Tile,
}

/// A discrete, pre-baked scalar signed-distance field: `dims` texels of `f32`
/// stored row-major with `X` fastest, then `Y`, then `Z` (design §8.3).
///
/// Each texel holds a *signed distance* to the encoded surface: negative inside
/// the solid, zero on the surface, positive outside — the standard `SDF`/`VDB`
/// convention. The data is static (authored or baked once, only sampled at
/// runtime), which is what distinguishes it from the analytic primitives in
/// [`super::collision`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SdfField {
    /// Texel counts along `[X, Y, Z]`; every axis is at least `1` for a field
    /// built through [`SdfField::from_data`] or [`SdfField::zeroed`].
    dims: [u32; 3],
    /// Row-major (`X`-fastest) signed distances, `dims[0] * dims[1] * dims[2]`
    /// long.
    data: Vec<f32>,
}

impl SdfField {
    /// Builds a field with every texel set to `0.0` (a degenerate everywhere-on-
    /// surface field, useful as a scratch target).
    ///
    /// Returns `None` when any axis is `0` or the texel count would overflow a
    /// `usize`, since a zero-extent grid cannot be sampled.
    #[must_use]
    pub fn zeroed(dims: [u32; 3]) -> Option<Self> {
        let count = Self::checked_count(dims)?;
        Some(Self {
            dims,
            data: vec![0.0; count],
        })
    }

    /// Builds a field from explicit `dims` and row-major (`X`-fastest) `data`.
    ///
    /// Returns `None` unless every axis is at least `1` and
    /// `dims[0] * dims[1] * dims[2] == data.len()`, which is the storage
    /// contract the samplers rely on (no bounds surprises, no partial last
    /// row).
    #[must_use]
    pub fn from_data(dims: [u32; 3], data: Vec<f32>) -> Option<Self> {
        let count = Self::checked_count(dims)?;
        if count == data.len() {
            Some(Self { dims, data })
        } else {
            None
        }
    }

    /// Total texel count for `dims`, or `None` on a zero axis / `usize`
    /// overflow.
    #[must_use]
    fn checked_count(dims: [u32; 3]) -> Option<usize> {
        let [nx, ny, nz] = dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        let nx = nx as usize;
        let ny = ny as usize;
        let nz = nz as usize;
        nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz))
    }

    /// The `[X, Y, Z]` texel dimensions.
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// The total number of stored texels, `dims[0] * dims[1] * dims[2]`.
    #[must_use]
    pub fn texel_count(&self) -> usize {
        self.data.len()
    }

    /// Read-only view of the row-major signed distances.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Row-major (`X`-fastest) linear index of texel `(i, j, k)`.
    ///
    /// The caller supplies in-range coordinates; this is the pure index the
    /// sampler and gradient routines share.
    #[must_use]
    pub fn linear_index(&self, i: u32, j: u32, k: u32) -> usize {
        let [nx, ny, _] = self.dims;
        (((k * ny) + j) * nx + i) as usize
    }

    /// The stored signed distance at integer texel `(i, j, k)`.
    ///
    /// Coordinates are clamped into range first, so this never panics even on
    /// an out-of-range request; in-range callers get the exact stored texel.
    #[must_use]
    pub fn sample_texel(&self, i: u32, j: u32, k: u32) -> f32 {
        let [nx, ny, nz] = self.dims;
        let ci = i.min(nx - 1);
        let cj = j.min(ny - 1);
        let ck = k.min(nz - 1);
        self.data[self.linear_index(ci, cj, ck)]
    }

    /// Fetches a texel distance by signed coordinates under a [`WrapMode`],
    /// resolving out-of-range indices per the boundary policy. Used by the
    /// trilinear sampler for the eight corners.
    #[must_use]
    fn fetch(&self, i: i32, j: i32, k: i32, wrap: WrapMode) -> f32 {
        let [nx, ny, nz] = self.dims;
        let ri = resolve_index(i, nx, wrap);
        let rj = resolve_index(j, ny, wrap);
        let rk = resolve_index(k, nz, wrap);
        self.data[self.linear_index(ri, rj, rk)]
    }

    /// Trilinearly samples the signed distance at a continuous grid coordinate,
    /// where a texel centre sits on its integer coordinate (design §8.3).
    ///
    /// A coordinate exactly on an integer returns that texel; the midpoint
    /// between two texels returns their average. Out-of-range coordinates are
    /// resolved by `wrap`. Pure multiply-add plus a `floor`.
    #[must_use]
    pub fn sample_distance(&self, grid: Vec3, wrap: WrapMode) -> f32 {
        let i0f = grid.x.floor();
        let j0f = grid.y.floor();
        let k0f = grid.z.floor();
        let fx = grid.x - i0f;
        let fy = grid.y - j0f;
        let fz = grid.z - k0f;
        let i0 = i0f as i32;
        let j0 = j0f as i32;
        let k0 = k0f as i32;
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let k1 = k0 + 1;

        let c000 = self.fetch(i0, j0, k0, wrap);
        let c100 = self.fetch(i1, j0, k0, wrap);
        let c010 = self.fetch(i0, j1, k0, wrap);
        let c110 = self.fetch(i1, j1, k0, wrap);
        let c001 = self.fetch(i0, j0, k1, wrap);
        let c101 = self.fetch(i1, j0, k1, wrap);
        let c011 = self.fetch(i0, j1, k1, wrap);
        let c111 = self.fetch(i1, j1, k1, wrap);

        let gx = 1.0 - fx;
        let gy = 1.0 - fy;
        let gz = 1.0 - fz;

        let c00 = c000 * gx + c100 * fx;
        let c10 = c010 * gx + c110 * fx;
        let c01 = c001 * gx + c101 * fx;
        let c11 = c011 * gx + c111 * fx;

        let c0 = c00 * gy + c10 * fy;
        let c1 = c01 * gy + c11 * fy;

        c0 * gz + c1 * fz
    }

    /// The raw (unnormalised) central-difference gradient of the trilinear
    /// field in grid space, in distance-per-texel units.
    #[must_use]
    fn grad_grid(&self, grid: Vec3, wrap: WrapMode) -> Vec3 {
        let dx = self.sample_distance(grid.add(Vec3::new(GRAD_STEP, 0.0, 0.0)), wrap)
            - self.sample_distance(grid.sub(Vec3::new(GRAD_STEP, 0.0, 0.0)), wrap);
        let dy = self.sample_distance(grid.add(Vec3::new(0.0, GRAD_STEP, 0.0)), wrap)
            - self.sample_distance(grid.sub(Vec3::new(0.0, GRAD_STEP, 0.0)), wrap);
        let dz = self.sample_distance(grid.add(Vec3::new(0.0, 0.0, GRAD_STEP)), wrap)
            - self.sample_distance(grid.sub(Vec3::new(0.0, 0.0, GRAD_STEP)), wrap);
        Vec3::new(dx, dy, dz).scale(1.0 / (2.0 * GRAD_STEP))
    }

    /// The unit surface normal at a continuous grid coordinate, defined as the
    /// normalised central-difference gradient of the trilinear field (design
    /// §8.3).
    ///
    /// For a valid `SDF` the gradient points away from the surface (toward
    /// increasing distance), so the normalised gradient is the outward surface
    /// normal. Returns [`Vec3::ZERO`] where the gradient degenerates (a flat
    /// region or the interior of a symmetric shell), guarding against `NaN`.
    /// The result lives in *grid* space; [`SdfField::sample`] converts it to a
    /// world-space normal through the transform's per-axis scale.
    #[must_use]
    pub fn gradient(&self, grid: Vec3, wrap: WrapMode) -> Vec3 {
        self.grad_grid(grid, wrap).normalize_or_zero()
    }

    /// Samples the field at a world position through `transform`, returning the
    /// signed distance and the world-space unit surface normal (design §8.3).
    ///
    /// The distance is the trilinear reconstruction; the normal is the grid
    /// gradient rescaled by each axis's texels-per-world-unit (so a non-cubic
    /// box still yields a correct world direction) and then normalised.
    #[must_use]
    pub fn sample(&self, transform: &SdfTransform, world_pos: Vec3, wrap: WrapMode) -> SdfSample {
        let dims = self.dims;
        let grid = transform.world_to_grid(dims, world_pos);
        let distance = self.sample_distance(grid, wrap);
        let raw = self.grad_grid(grid, wrap);
        let scale = transform.grid_per_world(dims);
        let normal = raw.mul(scale).normalize_or_zero();
        SdfSample { distance, normal }
    }
}

/// Resolves a signed grid index into a valid `[0, dim)` texel index under a
/// boundary policy. `dim` is assumed non-zero (guaranteed by the field
/// constructors).
#[must_use]
fn resolve_index(i: i32, dim: u32, wrap: WrapMode) -> u32 {
    let d = dim as i32;
    match wrap {
        WrapMode::Clamp => i.clamp(0, d - 1) as u32,
        WrapMode::Tile => (((i % d) + d) % d) as u32,
    }
}

/// Maps world space into a scalar `SDF`'s continuous grid space through an
/// axis-aligned bounding box (design §8.3).
///
/// The box `bounds` (an [`Aabb`], the same type the sort/cull stage reduces to)
/// encloses the whole field. Texel centres are placed on integer grid
/// coordinates and the box edges fall half a texel outside the first/last
/// centre, matching the `Niagara`/`UE` volume bounds convention so a resampled
/// field lines up with its authored volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfTransform {
    /// The world-space axis-aligned box the field occupies.
    pub bounds: Aabb,
}

impl SdfTransform {
    /// Builds a transform from an explicit world-space box.
    #[must_use]
    pub fn new(bounds: Aabb) -> Self {
        Self { bounds }
    }

    /// Builds a transform from a minimum-corner `origin` and a positive
    /// `extent` (the box size along each axis).
    #[must_use]
    pub fn from_origin_extent(origin: Vec3, extent: Vec3) -> Self {
        Self {
            bounds: Aabb {
                min: origin,
                max: origin.add(extent),
            },
        }
    }

    /// Maps a world position into continuous grid coordinates for a field of
    /// `dims` texels.
    ///
    /// The box spans `[0, dims)` in normalised terms with texel centres on
    /// integers: the returned coordinate is
    /// `((world - min) / size) * dims - 0.5` per axis. A degenerate
    /// (zero-thickness) axis maps to `0.0` on that axis instead of dividing by
    /// zero.
    #[must_use]
    pub fn world_to_grid(&self, dims: [u32; 3], world: Vec3) -> Vec3 {
        Vec3::new(
            axis_world_to_grid(world.x, self.bounds.min.x, self.bounds.max.x, dims[0]),
            axis_world_to_grid(world.y, self.bounds.min.y, self.bounds.max.y, dims[1]),
            axis_world_to_grid(world.z, self.bounds.min.z, self.bounds.max.z, dims[2]),
        )
    }

    /// Inverse of [`SdfTransform::world_to_grid`]: maps a continuous grid
    /// coordinate back to a world position,
    /// `min + (grid + 0.5) / dims * size` per axis.
    #[must_use]
    pub fn grid_to_world(&self, dims: [u32; 3], grid: Vec3) -> Vec3 {
        Vec3::new(
            axis_grid_to_world(grid.x, self.bounds.min.x, self.bounds.max.x, dims[0]),
            axis_grid_to_world(grid.y, self.bounds.min.y, self.bounds.max.y, dims[1]),
            axis_grid_to_world(grid.z, self.bounds.min.z, self.bounds.max.z, dims[2]),
        )
    }

    /// Per-axis grid texels per world unit, `dims / size`, used to convert a
    /// grid-space gradient into a world-space direction. A degenerate axis
    /// contributes `0.0` so it drops out of the normalised normal.
    #[must_use]
    fn grid_per_world(&self, dims: [u32; 3]) -> Vec3 {
        Vec3::new(
            axis_grid_per_world(self.bounds.min.x, self.bounds.max.x, dims[0]),
            axis_grid_per_world(self.bounds.min.y, self.bounds.max.y, dims[1]),
            axis_grid_per_world(self.bounds.min.z, self.bounds.max.z, dims[2]),
        )
    }
}

/// One-axis world-to-grid map with a degenerate-extent guard.
#[must_use]
fn axis_world_to_grid(world: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    if size.abs() < EPS {
        return 0.0;
    }
    let norm = (world - lo) / size;
    norm * dim as f32 - 0.5
}

/// One-axis grid-to-world map (inverse of [`axis_world_to_grid`]).
#[must_use]
fn axis_grid_to_world(grid: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    lo + ((grid + 0.5) / dim as f32) * size
}

/// One-axis `dims / size` factor with a degenerate-extent guard.
#[must_use]
fn axis_grid_per_world(lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    if size.abs() < EPS {
        return 0.0;
    }
    dim as f32 / size
}

/// The result of sampling a [`SdfField`] at a world position: the signed
/// distance and the world-space unit surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSample {
    /// Signed distance to the surface (negative inside, positive outside).
    pub distance: f32,
    /// World-space unit outward normal (the normalised field gradient), or
    /// [`Vec3::ZERO`] where the gradient degenerates.
    pub normal: Vec3,
}

/// A point emitted onto the zero-isosurface of a [`SdfField`], with its normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// World-space position projected onto (near) the surface.
    pub position: Vec3,
    /// World-space unit outward normal at [`SurfacePoint::position`].
    pub normal: Vec3,
}

/// A sphere-versus-field penetration query result (design §8.3).
///
/// This is a pure *query*: it reports the contact geometry a colliding particle
/// would resolve against, but performs no response. The response (push-out,
/// restitution, friction) belongs to [`super::collision`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// World-space unit outward normal at the contact (the field gradient).
    pub normal: Vec3,
    /// Positive penetration depth: how far the sphere overlaps the surface.
    pub depth: f32,
}

/// Refines a world position onto the zero-isosurface with a few Newton steps
/// `p -= distance * normal`, using the exact signed distance and gradient
/// (design §8.3).
///
/// For a well-formed `SDF` (unit gradient) a single step is exact on a plane
/// and each step roughly halves the residual on a curved surface; a handful of
/// iterations drives `|distance|` to the trilinear reconstruction floor.
#[must_use]
fn project_to_surface(
    field: &SdfField,
    transform: &SdfTransform,
    wrap: WrapMode,
    start: Vec3,
    iters: u32,
) -> SurfacePoint {
    let mut p = start;
    for _ in 0..iters {
        let s = field.sample(transform, p, wrap);
        p = p.sub(s.normal.scale(s.distance));
    }
    let s = field.sample(transform, p, wrap);
    SurfacePoint {
        position: p,
        normal: s.normal,
    }
}

/// Emits one particle onto the zero-isosurface of a [`SdfField`], the `CPU`
/// reference for [`super::modules::BuiltinModule::SpawnSdfSurface`] (design
/// §8.3).
///
/// Rejection-samples uniformly random world points inside the transform's
/// [`Aabb`] using the shared deterministic [`UnitCursor`], accepting the first
/// candidate inside the narrow band `|distance| < band`, then refines it onto
/// the surface with `newton_iters` Newton steps. After [`MAX_REJECTION_TRIES`]
/// rejects it falls back to projecting the last candidate, so the emitter is
/// always total (it never loops forever and always returns a point).
#[must_use]
pub fn emit_surface(
    field: &SdfField,
    transform: &SdfTransform,
    wrap: WrapMode,
    band: f32,
    newton_iters: u32,
    cursor: &mut UnitCursor<'_>,
) -> SurfacePoint {
    let min = transform.bounds.min;
    let ext = transform.bounds.max.sub(min);
    let mut last = transform.bounds.center();
    for _ in 0..MAX_REJECTION_TRIES {
        let p = Vec3::new(
            min.x + cursor.next_unit() * ext.x,
            min.y + cursor.next_unit() * ext.y,
            min.z + cursor.next_unit() * ext.z,
        );
        last = p;
        let d = field.sample(transform, p, wrap).distance;
        if d.abs() < band {
            return project_to_surface(field, transform, wrap, p, newton_iters);
        }
    }
    project_to_surface(field, transform, wrap, last, newton_iters)
}

/// Sticks a position to the nearest surface by projecting toward the zero-
/// isosurface, clamped to a maximum pull distance `max_pull` (design §8.3).
///
/// Moves `pos` by `-distance * normal` (one Newton step toward the surface),
/// but never more than `max_pull` world units, so a particle far from the field
/// is nudged rather than teleported. A degenerate gradient or a sub-`EPS`
/// displacement leaves `pos` unchanged.
#[must_use]
pub fn stick_to_surface(
    pos: Vec3,
    field: &SdfField,
    transform: &SdfTransform,
    wrap: WrapMode,
    max_pull: f32,
) -> Vec3 {
    let s = field.sample(transform, pos, wrap);
    let disp = s.normal.scale(-s.distance);
    let len = disp.length();
    if len < EPS {
        return pos;
    }
    if len > max_pull {
        return pos.add(disp.scale(max_pull / len));
    }
    pos.add(disp)
}

/// Queries the penetration of a sphere of `radius` centred at `pos` against the
/// field, for consumption by [`super::collision`] (design §8.3).
///
/// Returns `Some(Contact)` when the sphere overlaps the surface — that is when
/// the signed distance is below `radius` (including deep interior, where the
/// distance is negative) — with `depth = radius - distance` and the field
/// gradient as the outward normal. Returns `None` when the sphere is clear of
/// the surface. This is a pure query; it applies no response.
#[must_use]
pub fn penetration(
    pos: Vec3,
    radius: f32,
    field: &SdfField,
    transform: &SdfTransform,
    wrap: WrapMode,
) -> Option<Contact> {
    let s = field.sample(transform, pos, wrap);
    if s.distance < radius {
        Some(Contact {
            normal: s.normal,
            depth: radius - s.distance,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[must_use]
    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[must_use]
    fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    /// Builds an analytic sphere `SDF` sampled onto a `dims` grid over `[-1, 1]^3`.
    #[must_use]
    fn sphere_field(dims: [u32; 3], radius: f32) -> (SdfField, SdfTransform) {
        let transform = SdfTransform::from_origin_extent(Vec3::splat(-1.0), Vec3::splat(2.0));
        let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let world =
                        transform.grid_to_world(dims, Vec3::new(i as f32, j as f32, k as f32));
                    let idx = (((k * dims[1]) + j) * dims[0] + i) as usize;
                    data[idx] = world.length() - radius;
                }
            }
        }
        (SdfField::from_data(dims, data).unwrap(), transform)
    }

    #[test]
    fn from_data_rejects_length_mismatch() {
        assert!(SdfField::from_data([2, 2, 2], vec![0.0; 7]).is_none());
        assert!(SdfField::from_data([2, 2, 2], vec![0.0; 8]).is_some());
    }

    #[test]
    fn from_data_and_zeroed_reject_zero_axis() {
        assert!(SdfField::from_data([0, 2, 2], vec![0.0; 0]).is_none());
        assert!(SdfField::from_data([2, 0, 2], vec![0.0; 0]).is_none());
        assert!(SdfField::from_data([2, 2, 0], vec![0.0; 0]).is_none());
        assert!(SdfField::zeroed([0, 1, 1]).is_none());
        assert!(SdfField::zeroed([1, 1, 1]).is_some());
    }

    #[test]
    fn zeroed_is_all_zero() {
        let f = SdfField::zeroed([2, 3, 4]).unwrap();
        assert_eq!(f.texel_count(), 24);
        assert!(f.data().iter().all(|&d| approx(d, 0.0, TOL)));
        assert_eq!(f.dims(), [2, 3, 4]);
    }

    #[test]
    fn linear_index_is_row_major_x_fastest() {
        let f = SdfField::zeroed([4, 5, 6]).unwrap();
        assert_eq!(f.linear_index(0, 0, 0), 0);
        assert_eq!(f.linear_index(1, 0, 0), 1); // X fastest
        assert_eq!(f.linear_index(0, 1, 0), 4); // then Y (nx = 4)
        assert_eq!(f.linear_index(0, 0, 1), 20); // then Z (nx * ny = 20)
        assert_eq!(f.linear_index(3, 4, 5), 3 + 4 * 4 + 5 * 20);
    }

    #[test]
    fn sample_texel_clamps_out_of_range() {
        let data = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let f = SdfField::from_data([2, 2, 2], data).unwrap();
        // In range hits the exact stored value.
        assert!(approx(f.sample_texel(1, 1, 1), 7.0, TOL));
        // Out of range clamps to the last texel on each axis.
        assert!(approx(f.sample_texel(9, 9, 9), 7.0, TOL));
        assert!(approx(f.sample_texel(9, 0, 0), 1.0, TOL));
    }

    #[test]
    fn sample_distance_hits_texel_center_value() {
        let data = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let f = SdfField::from_data([2, 2, 2], data).unwrap();
        // Texel centres sit on integer grid coordinates.
        assert!(approx(
            f.sample_distance(Vec3::ZERO, WrapMode::Clamp),
            0.0,
            TOL
        ));
        assert!(approx(
            f.sample_distance(Vec3::new(1.0, 0.0, 0.0), WrapMode::Clamp),
            1.0,
            TOL
        ));
        assert!(approx(
            f.sample_distance(Vec3::new(1.0, 1.0, 1.0), WrapMode::Clamp),
            7.0,
            TOL
        ));
    }

    #[test]
    fn sample_distance_midpoint_is_average() {
        let data = vec![2.0, 8.0]; // dims [2,1,1]
        let f = SdfField::from_data([2, 1, 1], data).unwrap();
        let mid = f.sample_distance(Vec3::new(0.5, 0.0, 0.0), WrapMode::Clamp);
        assert!(approx(mid, 5.0, TOL));
    }

    #[test]
    fn wrap_clamp_extends_edge_value() {
        let data = vec![2.0, 8.0];
        let f = SdfField::from_data([2, 1, 1], data).unwrap();
        // Beyond the max edge, Clamp holds the last texel.
        let v = f.sample_distance(Vec3::new(5.0, 0.0, 0.0), WrapMode::Clamp);
        assert!(approx(v, 8.0, TOL));
        // Below the min edge holds the first texel.
        let v = f.sample_distance(Vec3::new(-5.0, 0.0, 0.0), WrapMode::Clamp);
        assert!(approx(v, 2.0, TOL));
    }

    #[test]
    fn wrap_tile_repeats_period() {
        let data = vec![2.0, 8.0];
        let f = SdfField::from_data([2, 1, 1], data).unwrap();
        // Grid index 2 wraps to index 0 under Tile (period 2).
        let v = f.sample_distance(Vec3::new(2.0, 0.0, 0.0), WrapMode::Tile);
        assert!(approx(v, 2.0, TOL));
        // Index -1 wraps to index 1.
        let v = f.sample_distance(Vec3::new(-1.0, 0.0, 0.0), WrapMode::Tile);
        assert!(approx(v, 8.0, TOL));
    }

    #[test]
    fn trilinear_interior_blend() {
        // Corner texels 0..8; centre of the cube is the mean 3.5.
        let data = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let f = SdfField::from_data([2, 2, 2], data).unwrap();
        let c = f.sample_distance(Vec3::splat(0.5), WrapMode::Clamp);
        assert!(approx(c, 3.5, TOL));
    }

    #[test]
    fn gradient_direction_on_planar_field() {
        // A plane SDF d(p) = grid.x: distance grows +1 per texel along X.
        let dims = [4u32, 3, 3];
        let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let idx = (((k * dims[1]) + j) * dims[0] + i) as usize;
                    data[idx] = i as f32;
                }
            }
        }
        let f = SdfField::from_data(dims, data).unwrap();
        let g = f.gradient(Vec3::new(1.5, 1.0, 1.0), WrapMode::Clamp);
        assert!(approx_vec(g, Vec3::new(1.0, 0.0, 0.0), TOL));
        // Unit length.
        assert!(approx(g.length(), 1.0, TOL));
    }

    #[test]
    fn world_grid_round_trip_and_zero_axis() {
        let t =
            SdfTransform::from_origin_extent(Vec3::new(-2.0, 0.0, 1.0), Vec3::new(4.0, 3.0, 6.0));
        let dims = [8u32, 4, 5];
        let world = Vec3::new(0.3, 1.7, 3.1);
        let grid = t.world_to_grid(dims, world);
        let back = t.grid_to_world(dims, grid);
        assert!(approx_vec(back, world, TOL));

        // Zero-thickness axis maps to 0 on that axis rather than dividing by zero.
        let flat = SdfTransform::from_origin_extent(Vec3::ZERO, Vec3::new(0.0, 2.0, 2.0));
        let g = flat.world_to_grid(dims, Vec3::new(5.0, 1.0, 1.0));
        assert!(approx(g.x, 0.0, TOL));
        assert!(g.x.is_finite());
    }

    #[test]
    fn sphere_field_distance_matches_analytic() {
        let (f, t) = sphere_field([17, 17, 17], 0.5);
        // A point on the +X axis at radius 0.8: analytic distance 0.3.
        let p = Vec3::new(0.8, 0.0, 0.0);
        let s = f.sample(&t, p, WrapMode::Clamp);
        assert!(approx(s.distance, 0.3, 5e-3));
        // Outward normal points along +X there.
        assert!(approx_vec(s.normal, Vec3::new(1.0, 0.0, 0.0), 5e-2));
        assert!(approx(s.normal.length(), 1.0, TOL));
    }

    #[test]
    fn newton_projection_lands_on_surface() {
        let (f, t) = sphere_field([33, 33, 33], 0.5);
        let start = Vec3::new(0.9, 0.1, -0.2);
        let sp = project_to_surface(&f, &t, WrapMode::Clamp, start, 8);
        let d = f.sample(&t, sp.position, WrapMode::Clamp).distance;
        assert!(approx(d, 0.0, 5e-3));
        assert!(approx(sp.normal.length(), 1.0, TOL));
    }

    #[test]
    fn emit_surface_lands_on_surface() {
        let (f, t) = sphere_field([33, 33, 33], 0.5);
        let samples = [0.15f32, 0.85, 0.42, 0.6, 0.3, 0.7, 0.5, 0.2, 0.9, 0.1];
        let mut cursor = UnitCursor::new(&samples);
        let sp = emit_surface(&f, &t, WrapMode::Clamp, 0.2, 8, &mut cursor);
        let d = f.sample(&t, sp.position, WrapMode::Clamp).distance;
        assert!(approx(d, 0.0, 1e-2));
        assert!(approx(sp.normal.length(), 1.0, TOL));
        // The projected point should be near the sphere radius from the centre.
        assert!(approx(sp.position.length(), 0.5, 2e-2));
    }

    #[test]
    fn stick_clamps_to_max_pull() {
        let (f, t) = sphere_field([17, 17, 17], 0.5);
        // Far outside on +X: raw pull would be ~0.3, clamp to 0.1.
        let pos = Vec3::new(0.8, 0.0, 0.0);
        let stuck = stick_to_surface(pos, &f, &t, WrapMode::Clamp, 0.1);
        let moved = pos.distance(stuck);
        assert!(moved <= 0.1 + TOL);
        assert!(approx(moved, 0.1, 5e-3));
        // Moving toward the surface reduces the outside distance.
        let d_before = f.sample(&t, pos, WrapMode::Clamp).distance;
        let d_after = f.sample(&t, stuck, WrapMode::Clamp).distance;
        assert!(d_after < d_before);
    }

    #[test]
    fn stick_unclamped_reaches_surface() {
        let (f, t) = sphere_field([33, 33, 33], 0.5);
        let pos = Vec3::new(0.8, 0.0, 0.0);
        let stuck = stick_to_surface(pos, &f, &t, WrapMode::Clamp, 10.0);
        let d = f.sample(&t, stuck, WrapMode::Clamp).distance;
        assert!(approx(d, 0.0, 1e-2));
    }

    #[test]
    fn penetration_hit_miss_and_depth() {
        let (f, t) = sphere_field([33, 33, 33], 0.5);
        // Sphere of radius 0.2 centred at distance 0.3 outside: 0.3 > 0.2 -> miss.
        let outside = Vec3::new(0.8, 0.0, 0.0);
        assert!(penetration(outside, 0.2, &f, &t, WrapMode::Clamp).is_none());

        // Sphere of radius 0.4 at the same point: 0.3 < 0.4 -> hit, depth 0.1.
        let contact = penetration(outside, 0.4, &f, &t, WrapMode::Clamp).unwrap();
        assert!(approx(contact.depth, 0.1, 5e-3));
        assert!(approx(contact.normal.length(), 1.0, TOL));
        assert!(approx_vec(contact.normal, Vec3::new(1.0, 0.0, 0.0), 5e-2));

        // A point well inside the sphere (distance ~ -0.5) always penetrates any
        // positive radius, with depth = radius - distance > radius.
        let inside = Vec3::ZERO;
        let c = penetration(inside, 0.1, &f, &t, WrapMode::Clamp).unwrap();
        assert!(c.depth > 0.1);
    }

    #[test]
    fn normals_are_unit_length_across_the_field() {
        let (f, t) = sphere_field([17, 17, 17], 0.5);
        let probes = [
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.0),
            Vec3::new(0.0, 0.0, -0.65),
            Vec3::new(0.4, 0.4, 0.2),
        ];
        for p in probes {
            let n = f.sample(&t, p, WrapMode::Clamp).normal;
            assert!(approx(n.length(), 1.0, TOL));
        }
    }
}
