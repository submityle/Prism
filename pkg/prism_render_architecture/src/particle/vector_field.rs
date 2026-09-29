//! Discrete sampled 3D vector-field forces (baked/imported grids), the `CPU`
//! reference behind [`super::modules::BuiltinModule::VectorField`] and its
//! [`super::modules::ResourceKind::Texture3d`] binding (design §8, §10).
//!
//! This module is the *stored-data* member of the three orthogonal flow-field
//! families the engine ships:
//!
//! * [`super::noise`] synthesises a flow field **analytically** from a
//!   hash-seeded lattice — no stored grid, no simulation state — for cheap
//!   ambient turbulence and curl swirls.
//! * [`super::fluid`] reconstructs a flow field by **simulating** a voxel
//!   stable-fluids solve every frame, re-solving advection/projection so the
//!   grid evolves over time.
//! * This module samples a **static, pre-baked or imported** discrete grid of
//!   velocity/force vectors that never changes at runtime: a `Houdini`-exported
//!   volume, a `FGA`/`VF`-style vector-field asset, or a tool-authored box of
//!   directions. There is no solver and no procedural synthesis — only stored
//!   texels and interpolation.
//!
//! It is the `CPU`-verifiable contract for the vector-field force shipped by
//! Unreal `Niagara` ("Vector Field" / `UVectorField` and the `GPU` vector
//! fields), Unity `VFX Graph` ("Vector Field Force"), and `PopcornFX`'s
//! imported flow volumes. The stack is:
//!
//! 1. **Grid storage** — [`VectorField`] holds `dims` and a row-major
//!    (`X`-fastest) `Vec<Vec3>` of texel vectors, validated so the texel count
//!    always matches the dimensions; see [`VectorField::from_data`] and
//!    [`VectorField::sample_texel`].
//! 2. **World-to-grid transform** — [`VectorFieldTransform`] maps a world
//!    position through an axis-aligned box (an [`Aabb`]) into continuous grid
//!    coordinates, placing texel centres on integer grid coordinates so the
//!    box spans the full grid (the `Niagara`/`UE` bounds convention).
//! 3. **Trilinear sampling** — [`VectorField::sample`] blends the eight
//!    surrounding texels with pure multiply-add weights, honouring a
//!    [`WrapMode`] boundary policy (`Clamp` to the edge texel or `Tile` by
//!    integer modulo).
//! 4. **Force application** — [`apply`] turns a sampled field vector into a
//!    [`VectorFieldEffect`] under an [`ApplyMode`] (`Direct` set-velocity,
//!    `Force` add-acceleration, or `Velocity` converge-toward-field), scaled by
//!    an `intensity` and, for the `Velocity` mode, a `tightness` blend.
//! 5. **Diagnostics** — [`VectorField::divergence`] and [`VectorField::curl`]
//!    are central-difference field derivatives for authoring-time inspection.
//!
//! Determinism matches the sibling particle modules: the only floating-point
//! primitives beyond ordinary arithmetic are `f32::floor` (integer grid
//! location) and `sqrt` (through [`Vec3`]); there are no transcendental calls
//! (`sin`/`cos`/`exp`/`ln`/`pow`), and every interpolation is multiply-add. The
//! result is bit-reproducible against a future `GPU` kernel that samples the
//! same `Texture3d`.

use alloc::vec;
use alloc::vec::Vec;

use super::sort_cull::Aabb;
use super::Vec3;

/// Absolute tolerance for the `f32` equality guards in this module (intensity
/// gating and degenerate-extent detection). Comparisons use
/// `(a - b).abs() < EPS` rather than a bare `==`.
pub const EPS: f32 = 1e-6;

/// The boundary policy applied when a sample lands outside the `[0, dims)` grid
/// range on any axis (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WrapMode {
    /// Clamp the sample to the nearest edge texel, so the field extends its
    /// border value outward indefinitely (the common default for a bounded
    /// baked volume).
    Clamp,
    /// Tile the field periodically by wrapping indices with an integer modulo,
    /// so the volume repeats seamlessly in every direction.
    Tile,
}

/// A discrete, pre-baked 3D vector field: `dims` texels of `Vec3` stored
/// row-major with `X` fastest, then `Y`, then `Z` (design §8, §10).
///
/// This is the storage backing a `Texture3d` vector-field resource. The data is
/// static — it is authored or imported once and only sampled at runtime — which
/// is what distinguishes it from the analytic [`super::noise`] fields and the
/// per-frame [`super::fluid`] solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorField {
    /// Texel counts along `(X, Y, Z)`; every axis is at least `1` for a field
    /// built through [`VectorField::from_data`] or [`VectorField::zeroed`].
    dims: (u32, u32, u32),
    /// Row-major (`X`-fastest) texel vectors, `dims.0 * dims.1 * dims.2` long.
    data: Vec<Vec3>,
}

impl VectorField {
    /// Builds a field with every texel set to [`Vec3::ZERO`].
    ///
    /// Returns `None` when any axis is `0` or the texel count would overflow a
    /// `usize`, since a zero-extent grid cannot be sampled.
    #[must_use]
    pub fn zeroed(dims: (u32, u32, u32)) -> Option<Self> {
        let count = Self::checked_count(dims)?;
        Some(Self {
            dims,
            data: vec![Vec3::ZERO; count],
        })
    }

    /// Builds a field from explicit `dims` and row-major (`X`-fastest) `data`.
    ///
    /// Returns `None` unless every axis is at least `1` and
    /// `dims.0 * dims.1 * dims.2 == data.len()`, which is the storage contract
    /// the samplers rely on (no bounds surprises, no partial last row).
    #[must_use]
    pub fn from_data(dims: (u32, u32, u32), data: Vec<Vec3>) -> Option<Self> {
        let count = Self::checked_count(dims)?;
        if count == data.len() {
            Some(Self { dims, data })
        } else {
            None
        }
    }

    /// Total texel count for `dims`, or `None` on a zero axis / `usize`
    /// overflow.
    fn checked_count(dims: (u32, u32, u32)) -> Option<usize> {
        let (nx, ny, nz) = dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        let nx = nx as usize;
        let ny = ny as usize;
        let nz = nz as usize;
        nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz))
    }

    /// The `(X, Y, Z)` texel dimensions.
    #[must_use]
    pub fn dims(&self) -> (u32, u32, u32) {
        self.dims
    }

    /// The total number of stored texels, `dims.0 * dims.1 * dims.2`.
    #[must_use]
    pub fn texel_count(&self) -> usize {
        self.data.len()
    }

    /// Read-only view of the row-major texel vectors.
    #[must_use]
    pub fn data(&self) -> &[Vec3] {
        &self.data
    }

    /// Row-major (`X`-fastest) linear index of texel `(i, j, k)`.
    ///
    /// The caller supplies in-range coordinates; this is the pure index the
    /// sampler and derivative routines share.
    #[must_use]
    pub fn linear_index(&self, i: u32, j: u32, k: u32) -> usize {
        let (nx, ny, _) = self.dims;
        (((k * ny) + j) * nx + i) as usize
    }

    /// The stored vector at integer texel `(i, j, k)`.
    ///
    /// Coordinates are clamped into range first, so this never panics even on
    /// an out-of-range request; in-range callers get the exact stored texel.
    #[must_use]
    pub fn sample_texel(&self, i: u32, j: u32, k: u32) -> Vec3 {
        let (nx, ny, nz) = self.dims;
        let ci = i.min(nx - 1);
        let cj = j.min(ny - 1);
        let ck = k.min(nz - 1);
        self.data[self.linear_index(ci, cj, ck)]
    }

    /// Fetches a texel by signed coordinates under a [`WrapMode`], resolving
    /// out-of-range indices per the boundary policy. Used by the trilinear
    /// sampler for the eight corners.
    fn fetch(&self, i: i32, j: i32, k: i32, wrap: WrapMode) -> Vec3 {
        let (nx, ny, nz) = self.dims;
        let ri = resolve_index(i, nx, wrap);
        let rj = resolve_index(j, ny, wrap);
        let rk = resolve_index(k, nz, wrap);
        self.data[self.linear_index(ri, rj, rk)]
    }

    /// Trilinearly samples the field at a continuous grid coordinate, where a
    /// texel centre sits on its integer coordinate (design §10).
    ///
    /// A coordinate exactly on an integer returns that texel; the midpoint
    /// between two texels returns their average. Out-of-range coordinates are
    /// resolved by `wrap`. Pure multiply-add plus a `floor`.
    #[must_use]
    pub fn sample_grid(&self, grid: Vec3, wrap: WrapMode) -> Vec3 {
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

        let w000 = gx * gy * gz;
        let w100 = fx * gy * gz;
        let w010 = gx * fy * gz;
        let w110 = fx * fy * gz;
        let w001 = gx * gy * fz;
        let w101 = fx * gy * fz;
        let w011 = gx * fy * fz;
        let w111 = fx * fy * fz;

        c000.scale(w000)
            .add(c100.scale(w100))
            .add(c010.scale(w010))
            .add(c110.scale(w110))
            .add(c001.scale(w001))
            .add(c101.scale(w101))
            .add(c011.scale(w011))
            .add(c111.scale(w111))
    }

    /// Trilinearly samples the field at a world position through `transform`
    /// (design §10).
    ///
    /// Composes [`VectorFieldTransform::world_to_grid`] with
    /// [`VectorField::sample_grid`]; this is the entry point a particle uses to
    /// read the flow at its location.
    #[must_use]
    pub fn sample(
        &self,
        transform: &VectorFieldTransform,
        world_pos: Vec3,
        wrap: WrapMode,
    ) -> Vec3 {
        let grid = transform.world_to_grid(self.dims, world_pos);
        self.sample_grid(grid, wrap)
    }

    /// Central-difference divergence `∂vx/∂x + ∂vy/∂y + ∂vz/∂z` at texel
    /// `(i, j, k)`, in per-texel units (design §10).
    ///
    /// Neighbours are fetched with [`WrapMode::Clamp`] so the boundary uses a
    /// one-sided-in-effect difference against the edge texel. Diagnostic only;
    /// multiply-add on stored data.
    #[must_use]
    pub fn divergence(&self, i: u32, j: u32, k: u32) -> f32 {
        let (dvx, _, _) = self.central_x(i, j, k);
        let (_, dvy, _) = self.central_y(i, j, k);
        let (_, _, dvz) = self.central_z(i, j, k);
        (dvx + dvy + dvz) * 0.5
    }

    /// Central-difference curl `∇ × v` at texel `(i, j, k)`, in per-texel units
    /// (design §10).
    ///
    /// Neighbours are fetched with [`WrapMode::Clamp`]. Diagnostic only, used to
    /// visualise the rotational structure of an imported field; multiply-add on
    /// stored data.
    #[must_use]
    pub fn curl(&self, i: u32, j: u32, k: u32) -> Vec3 {
        let (_, dvy_dx, dvz_dx) = self.central_x(i, j, k);
        let (dvx_dy, _, dvz_dy) = self.central_y(i, j, k);
        let (dvx_dz, dvy_dz, _) = self.central_z(i, j, k);
        Vec3::new(
            (dvz_dy - dvy_dz) * 0.5,
            (dvx_dz - dvz_dx) * 0.5,
            (dvy_dx - dvx_dy) * 0.5,
        )
    }

    /// Forward-minus-backward neighbour difference along `X`, returning the
    /// `(∂vx, ∂vy, ∂vz)` components of `v(i+1) - v(i-1)`.
    fn central_x(&self, i: u32, j: u32, k: u32) -> (f32, f32, f32) {
        let ii = i as i32;
        let plus = self.fetch(ii + 1, j as i32, k as i32, WrapMode::Clamp);
        let minus = self.fetch(ii - 1, j as i32, k as i32, WrapMode::Clamp);
        let d = plus.sub(minus);
        (d.x, d.y, d.z)
    }

    /// Neighbour difference `v(j+1) - v(j-1)` along `Y`.
    fn central_y(&self, i: u32, j: u32, k: u32) -> (f32, f32, f32) {
        let jj = j as i32;
        let plus = self.fetch(i as i32, jj + 1, k as i32, WrapMode::Clamp);
        let minus = self.fetch(i as i32, jj - 1, k as i32, WrapMode::Clamp);
        let d = plus.sub(minus);
        (d.x, d.y, d.z)
    }

    /// Neighbour difference `v(k+1) - v(k-1)` along `Z`.
    fn central_z(&self, i: u32, j: u32, k: u32) -> (f32, f32, f32) {
        let kk = k as i32;
        let plus = self.fetch(i as i32, j as i32, kk + 1, WrapMode::Clamp);
        let minus = self.fetch(i as i32, j as i32, kk - 1, WrapMode::Clamp);
        let d = plus.sub(minus);
        (d.x, d.y, d.z)
    }
}

/// Resolves a signed grid index into a valid `[0, dim)` texel index under a
/// boundary policy. `dim` is assumed non-zero (guaranteed by the field
/// constructors).
fn resolve_index(i: i32, dim: u32, wrap: WrapMode) -> u32 {
    let d = dim as i32;
    match wrap {
        WrapMode::Clamp => i.clamp(0, d - 1) as u32,
        WrapMode::Tile => (((i % d) + d) % d) as u32,
    }
}

/// Maps world space into a vector field's continuous grid space through an
/// axis-aligned bounding box (design §8, §10).
///
/// The box `bounds` (an [`Aabb`], the same type the sort/cull stage reduces to)
/// encloses the whole field. Texel centres are placed on integer grid
/// coordinates and the box edges fall half a texel outside the first/last
/// centre, matching the `Niagara`/`UE` vector-field bounds convention so a
/// resampled field lines up with its authored volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorFieldTransform {
    /// The world-space axis-aligned box the field occupies.
    pub bounds: Aabb,
}

impl VectorFieldTransform {
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
    /// `((world - min) / size) * dims - 0.5` per axis. A degenerate (zero-size)
    /// axis maps to `0.0` on that axis instead of dividing by zero.
    #[must_use]
    pub fn world_to_grid(&self, dims: (u32, u32, u32), world: Vec3) -> Vec3 {
        Vec3::new(
            axis_world_to_grid(world.x, self.bounds.min.x, self.bounds.max.x, dims.0),
            axis_world_to_grid(world.y, self.bounds.min.y, self.bounds.max.y, dims.1),
            axis_world_to_grid(world.z, self.bounds.min.z, self.bounds.max.z, dims.2),
        )
    }

    /// Inverse of [`VectorFieldTransform::world_to_grid`]: maps a continuous
    /// grid coordinate back to a world position, `min + (grid + 0.5) / dims *
    /// size` per axis.
    #[must_use]
    pub fn grid_to_world(&self, dims: (u32, u32, u32), grid: Vec3) -> Vec3 {
        Vec3::new(
            axis_grid_to_world(grid.x, self.bounds.min.x, self.bounds.max.x, dims.0),
            axis_grid_to_world(grid.y, self.bounds.min.y, self.bounds.max.y, dims.1),
            axis_grid_to_world(grid.z, self.bounds.min.z, self.bounds.max.z, dims.2),
        )
    }
}

/// One-axis world-to-grid map with a degenerate-extent guard.
fn axis_world_to_grid(world: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    if size.abs() < EPS {
        return 0.0;
    }
    let norm = (world - lo) / size;
    norm * dim as f32 - 0.5
}

/// One-axis grid-to-world map (inverse of [`axis_world_to_grid`]).
fn axis_grid_to_world(grid: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    lo + ((grid + 0.5) / dim as f32) * size
}

/// How a sampled field vector is turned into a per-step effect on a particle
/// (design §8), mirroring the `Niagara`/`UE` vector-field application modes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ApplyMode {
    /// Overwrite the particle velocity with the (scaled) field vector: the
    /// particle rigidly follows the field. Reported as the velocity delta that
    /// reaches that target this step.
    Direct,
    /// Treat the (scaled) field vector as an acceleration added to the
    /// particle's force accumulator; the integrator applies it over `dt`.
    Force,
    /// Converge the particle velocity toward the (scaled) field vector by a
    /// `tightness` blend, a critically-damped-style follow that lets particles
    /// lag and lead the flow.
    Velocity,
}

/// Parameters controlling how [`apply`] maps a field sample onto a particle
/// (design §8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApplyParams {
    /// Which application mode to use.
    pub mode: ApplyMode,
    /// Boundary policy for the sample.
    pub wrap: WrapMode,
    /// Master amplitude multiplying the sampled field vector. An `intensity`
    /// of (near) `0` disables the module and yields a zero effect in every
    /// mode.
    pub intensity: f32,
    /// For [`ApplyMode::Velocity`], the blend toward the field velocity in
    /// `0..=1`: `0` leaves the particle unchanged, `1` snaps it onto the field.
    /// Clamped into range. Ignored by the other modes.
    pub tightness: f32,
}

impl ApplyParams {
    /// A [`ApplyMode::Force`] configuration with unit intensity and
    /// [`WrapMode::Clamp`] boundaries.
    #[must_use]
    pub fn force(intensity: f32) -> Self {
        Self {
            mode: ApplyMode::Force,
            wrap: WrapMode::Clamp,
            intensity,
            tightness: 0.0,
        }
    }

    /// A [`ApplyMode::Direct`] configuration with unit intensity and
    /// [`WrapMode::Clamp`] boundaries.
    #[must_use]
    pub fn direct(intensity: f32) -> Self {
        Self {
            mode: ApplyMode::Direct,
            wrap: WrapMode::Clamp,
            intensity,
            tightness: 0.0,
        }
    }

    /// A [`ApplyMode::Velocity`] configuration with the given `intensity` and
    /// `tightness` and [`WrapMode::Clamp`] boundaries.
    #[must_use]
    pub fn velocity(intensity: f32, tightness: f32) -> Self {
        Self {
            mode: ApplyMode::Velocity,
            wrap: WrapMode::Clamp,
            intensity,
            tightness,
        }
    }
}

/// The per-step outcome of applying a vector field to one particle (design §8).
///
/// `velocity_delta` is the net change to add to the particle's velocity this
/// step; `acceleration` is the force-accumulator contribution (non-zero only in
/// [`ApplyMode::Force`], where `velocity_delta` already equals
/// `acceleration * dt` for callers that integrate the delta directly).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorFieldEffect {
    /// The velocity change to apply this step.
    pub velocity_delta: Vec3,
    /// The acceleration contributed to the force accumulator (zero except in
    /// [`ApplyMode::Force`]).
    pub acceleration: Vec3,
}

impl VectorFieldEffect {
    /// The no-op effect: no velocity change and no acceleration.
    pub const ZERO: Self = Self {
        velocity_delta: Vec3::ZERO,
        acceleration: Vec3::ZERO,
    };
}

/// Applies a vector field to a particle, producing its per-step
/// [`VectorFieldEffect`] (design §8).
///
/// Samples `field` at `pos` through `transform`, scales the result by
/// `params.intensity`, and maps it onto the particle's current `vel` under
/// `params.mode`:
///
/// * [`ApplyMode::Direct`] — `velocity_delta = field * intensity - vel`, so the
///   resulting velocity is exactly the scaled field vector; `acceleration` is
///   zero.
/// * [`ApplyMode::Force`] — `acceleration = field * intensity` and
///   `velocity_delta = acceleration * dt`.
/// * [`ApplyMode::Velocity`] — `velocity_delta = (field * intensity - vel) * t`
///   with `t = clamp(tightness, 0, 1)`; `acceleration` is zero.
///
/// A near-zero `intensity` short-circuits to [`VectorFieldEffect::ZERO`] in
/// every mode (the module is disabled). All paths are multiply-add on top of
/// the trilinear sample.
#[must_use]
pub fn apply(
    field: &VectorField,
    transform: &VectorFieldTransform,
    pos: Vec3,
    vel: Vec3,
    params: ApplyParams,
    dt: f32,
) -> VectorFieldEffect {
    if params.intensity.abs() < EPS {
        return VectorFieldEffect::ZERO;
    }
    let sampled = field
        .sample(transform, pos, params.wrap)
        .scale(params.intensity);
    match params.mode {
        ApplyMode::Direct => VectorFieldEffect {
            velocity_delta: sampled.sub(vel),
            acceleration: Vec3::ZERO,
        },
        ApplyMode::Force => VectorFieldEffect {
            velocity_delta: sampled.scale(dt),
            acceleration: sampled,
        },
        ApplyMode::Velocity => {
            let t = params.tightness.clamp(0.0, 1.0);
            VectorFieldEffect {
                velocity_delta: sampled.sub(vel).scale(t),
                acceleration: Vec3::ZERO,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the assertions below.
    const T: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < T
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// A 2x1x1 field: texel 0 = (1,0,0), texel 1 = (3,0,0).
    fn two_texel_field() -> VectorField {
        VectorField::from_data(
            (2, 1, 1),
            vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)],
        )
        .expect("valid 2x1x1 field")
    }

    #[test]
    fn from_data_validates_texel_count() {
        assert!(VectorField::from_data((2, 2, 2), vec![Vec3::ZERO; 8]).is_some());
        assert!(VectorField::from_data((2, 2, 2), vec![Vec3::ZERO; 7]).is_none());
        assert!(VectorField::from_data((2, 2, 2), vec![Vec3::ZERO; 9]).is_none());
    }

    #[test]
    fn zero_extent_axis_is_rejected() {
        assert!(VectorField::from_data((0, 1, 1), Vec::new()).is_none());
        assert!(VectorField::zeroed((1, 0, 1)).is_none());
        assert!(VectorField::zeroed((1, 1, 1)).is_some());
    }

    #[test]
    fn zeroed_is_all_zero_and_right_size() {
        let f = VectorField::zeroed((3, 4, 5)).expect("valid");
        assert_eq!(f.texel_count(), 60);
        assert_eq!(f.dims(), (3, 4, 5));
        assert!(f.data().iter().all(|v| *v == Vec3::ZERO));
    }

    #[test]
    fn linear_index_is_row_major_x_fastest() {
        let f = VectorField::zeroed((4, 3, 2)).expect("valid");
        assert_eq!(f.linear_index(0, 0, 0), 0);
        assert_eq!(f.linear_index(1, 0, 0), 1);
        assert_eq!(f.linear_index(0, 1, 0), 4);
        assert_eq!(f.linear_index(0, 0, 1), 12);
        assert_eq!(f.linear_index(3, 2, 1), 23);
    }

    #[test]
    fn sample_texel_clamps_out_of_range() {
        let f = two_texel_field();
        assert_eq!(f.sample_texel(0, 0, 0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(f.sample_texel(1, 0, 0), Vec3::new(3.0, 0.0, 0.0));
        // Out of range clamps to the last texel rather than panicking.
        assert_eq!(f.sample_texel(9, 9, 9), Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn sample_grid_hits_texel_center_exactly() {
        let f = two_texel_field();
        assert!(approx_vec(
            f.sample_grid(Vec3::new(0.0, 0.0, 0.0), WrapMode::Clamp),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        assert!(approx_vec(
            f.sample_grid(Vec3::new(1.0, 0.0, 0.0), WrapMode::Clamp),
            Vec3::new(3.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn sample_grid_midpoint_is_the_mean() {
        let f = two_texel_field();
        let mid = f.sample_grid(Vec3::new(0.5, 0.0, 0.0), WrapMode::Clamp);
        assert!(approx_vec(mid, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn sample_grid_clamp_holds_boundary_value() {
        let f = two_texel_field();
        // Far negative and far positive both clamp to the edge texels.
        assert!(approx_vec(
            f.sample_grid(Vec3::new(-5.0, 0.0, 0.0), WrapMode::Clamp),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        assert!(approx_vec(
            f.sample_grid(Vec3::new(5.0, 0.0, 0.0), WrapMode::Clamp),
            Vec3::new(3.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn sample_grid_tile_wraps_around() {
        let f = two_texel_field();
        // Grid coord -1 wraps to texel 1 under Tile (dim = 2).
        assert!(approx_vec(
            f.sample_grid(Vec3::new(-1.0, 0.0, 0.0), WrapMode::Tile),
            Vec3::new(3.0, 0.0, 0.0)
        ));
        // Grid coord 2 wraps to texel 0.
        assert!(approx_vec(
            f.sample_grid(Vec3::new(2.0, 0.0, 0.0), WrapMode::Tile),
            Vec3::new(1.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn trilinear_blends_all_eight_corners() {
        // Unit cube field whose value equals the X+Y+Z corner sum.
        let data = vec![
            Vec3::new(0.0, 0.0, 0.0), // (0,0,0)
            Vec3::new(1.0, 0.0, 0.0), // (1,0,0)
            Vec3::new(1.0, 0.0, 0.0), // (0,1,0)
            Vec3::new(2.0, 0.0, 0.0), // (1,1,0)
            Vec3::new(1.0, 0.0, 0.0), // (0,0,1)
            Vec3::new(2.0, 0.0, 0.0), // (1,0,1)
            Vec3::new(2.0, 0.0, 0.0), // (0,1,1)
            Vec3::new(3.0, 0.0, 0.0), // (1,1,1)
        ];
        let f = VectorField::from_data((2, 2, 2), data).expect("valid");
        // Center of the cube averages all eight corners = 1.5.
        let center = f.sample_grid(Vec3::new(0.5, 0.5, 0.5), WrapMode::Clamp);
        assert!(approx_vec(center, Vec3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn world_to_grid_places_texel_centers_on_integers() {
        // 2 texels spanning world [0, 4]: centers at world 1 and 3 -> grid 0, 1.
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let dims = (2, 1, 1);
        let g0 = t.world_to_grid(dims, Vec3::new(1.0, 0.0, 0.0));
        let g1 = t.world_to_grid(dims, Vec3::new(3.0, 0.0, 0.0));
        assert!(approx(g0.x, 0.0));
        assert!(approx(g1.x, 1.0));
    }

    #[test]
    fn grid_to_world_inverts_world_to_grid() {
        let t = VectorFieldTransform::new(Aabb {
            min: Vec3::new(-2.0, -2.0, -2.0),
            max: Vec3::new(2.0, 2.0, 2.0),
        });
        let dims = (4, 4, 4);
        let world = Vec3::new(0.75, -1.25, 1.5);
        let grid = t.world_to_grid(dims, world);
        let back = t.grid_to_world(dims, grid);
        assert!(approx_vec(back, world));
    }

    #[test]
    fn world_to_grid_degenerate_axis_is_zero() {
        // A flat box on X (min.x == max.x) maps every X to grid 0.
        let t = VectorFieldTransform::new(Aabb {
            min: Vec3::new(1.0, 0.0, 0.0),
            max: Vec3::new(1.0, 4.0, 4.0),
        });
        let g = t.world_to_grid((2, 2, 2), Vec3::new(50.0, 1.0, 1.0));
        assert!(approx(g.x, 0.0));
    }

    #[test]
    fn sample_through_transform_matches_grid_sample() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        // World x = 2 is the midpoint between the two texel centers (1 and 3).
        let world_mid = f.sample(&t, Vec3::new(2.0, 0.0, 0.0), WrapMode::Clamp);
        assert!(approx_vec(world_mid, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn apply_zero_intensity_is_no_effect() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let vel = Vec3::new(7.0, -1.0, 2.0);
        for mode in [ApplyMode::Direct, ApplyMode::Force, ApplyMode::Velocity] {
            let params = ApplyParams {
                mode,
                wrap: WrapMode::Clamp,
                intensity: 0.0,
                tightness: 1.0,
            };
            let e = apply(&f, &t, Vec3::new(1.0, 0.0, 0.0), vel, params, 0.1);
            assert_eq!(e, VectorFieldEffect::ZERO);
        }
    }

    #[test]
    fn apply_direct_sets_velocity_to_scaled_field() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let vel = Vec3::new(10.0, 5.0, -3.0);
        // World x = 1 -> texel 0 = (1,0,0); intensity 2 -> field*intensity = (2,0,0).
        let e = apply(
            &f,
            &t,
            Vec3::new(1.0, 0.0, 0.0),
            vel,
            ApplyParams::direct(2.0),
            0.1,
        );
        let resulting = vel.add(e.velocity_delta);
        assert!(approx_vec(resulting, Vec3::new(2.0, 0.0, 0.0)));
        assert!(approx_vec(e.acceleration, Vec3::ZERO));
    }

    #[test]
    fn apply_force_reports_acceleration_and_dt_delta() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let e = apply(
            &f,
            &t,
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::ZERO,
            ApplyParams::force(2.0),
            0.5,
        );
        // Texel 1 = (3,0,0); intensity 2 -> accel (6,0,0); delta = accel*dt = (3,0,0).
        assert!(approx_vec(e.acceleration, Vec3::new(6.0, 0.0, 0.0)));
        assert!(approx_vec(e.velocity_delta, Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn apply_velocity_tightness_one_snaps_and_zero_holds() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let vel = Vec3::new(9.0, 0.0, 0.0);

        let snap = apply(
            &f,
            &t,
            Vec3::new(1.0, 0.0, 0.0),
            vel,
            ApplyParams::velocity(1.0, 1.0),
            0.1,
        );
        // Field texel 0 = (1,0,0); tightness 1 -> resulting velocity == field.
        assert!(approx_vec(
            vel.add(snap.velocity_delta),
            Vec3::new(1.0, 0.0, 0.0)
        ));

        let hold = apply(
            &f,
            &t,
            Vec3::new(1.0, 0.0, 0.0),
            vel,
            ApplyParams::velocity(1.0, 0.0),
            0.1,
        );
        assert!(approx_vec(hold.velocity_delta, Vec3::ZERO));
    }

    #[test]
    fn apply_velocity_tightness_half_is_midpoint() {
        let f = two_texel_field();
        let t = VectorFieldTransform::from_origin_extent(Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        let vel = Vec3::new(3.0, 0.0, 0.0);
        // Field texel 0 = (1,0,0); half blend -> delta = (1-3)*0.5 = -1 -> vel 2.
        let e = apply(
            &f,
            &t,
            Vec3::new(1.0, 0.0, 0.0),
            vel,
            ApplyParams::velocity(1.0, 0.5),
            0.1,
        );
        assert!(approx_vec(
            vel.add(e.velocity_delta),
            Vec3::new(2.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn divergence_of_radial_field_is_positive() {
        // 3x3x3 field pointing outward along X: vx = i - 1.
        let mut data = Vec::with_capacity(27);
        for _k in 0..3 {
            for _j in 0..3 {
                for i in 0..3 {
                    data.push(Vec3::new(i as f32 - 1.0, 0.0, 0.0));
                }
            }
        }
        let f = VectorField::from_data((3, 3, 3), data).expect("valid");
        // d(vx)/dx = 1 across the interior; divergence = 1 * 0.5 * 2 = 1.
        assert!(approx(f.divergence(1, 1, 1), 1.0));
    }

    #[test]
    fn curl_of_shear_field_is_nonzero() {
        // 3x3x3 field with vy = i - 1 (a shear): curl.z = d(vy)/dx = 1.
        let mut data = Vec::with_capacity(27);
        for _k in 0..3 {
            for _j in 0..3 {
                for i in 0..3 {
                    data.push(Vec3::new(0.0, i as f32 - 1.0, 0.0));
                }
            }
        }
        let f = VectorField::from_data((3, 3, 3), data).expect("valid");
        let c = f.curl(1, 1, 1);
        assert!(approx(c.z, 1.0));
        assert!(approx(c.x, 0.0));
        assert!(approx(c.y, 0.0));
    }

    #[test]
    fn apply_params_constructors_set_modes() {
        assert_eq!(ApplyParams::force(1.0).mode, ApplyMode::Force);
        assert_eq!(ApplyParams::direct(1.0).mode, ApplyMode::Direct);
        assert_eq!(ApplyParams::velocity(1.0, 0.5).mode, ApplyMode::Velocity);
    }
}
