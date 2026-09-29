//! `GI` Probe / Irradiance `DataInterface`: the `CPU` reference for a read-only
//! environment-lighting sampler particles query for received light (design §8.3
//! "scene" category, aligned to the shading contract of §17).
//!
//! This module is the deterministic `CPU` reference for the *receive-light* data
//! interface: given a world position and a shading normal it returns the
//! environment irradiance a particle bathes in, mirroring the "Sample
//! `GI`/Irradiance" node of Unreal `Niagara`, Unity `VFX Graph` probe sampling,
//! and the `Frostbite`/`bevy_light` probe volumes — at the algorithm level, with
//! zero borrowed code and zero external dependencies.
//!
//! It is intentionally orthogonal to its sibling modules and never overlaps
//! their responsibility:
//!
//! - [`super::shading`] routes *how* a surface responds to light (the four
//!   equal-citizen shading models); this module only *provides* the incoming
//!   irradiance value those closures consume. Shading decides response, the
//!   probe supplies the environment term.
//! - [`super::raytrace`] resolves *dynamic scene* visibility and single-bounce
//!   `GI` by casting rays into `bevy_solari`; this module is the *precomputed,
//!   cache-friendly* fallback/complement — a baked irradiance field sampled by
//!   pure arithmetic, never a scene trace.
//!
//! Everything here is spelled out as polynomials over the shared hand-rolled
//! [`Vec3`]: the real-valued spherical-harmonic (`SH`) basis functions are
//! Cartesian polynomials of a unit direction (the associated Legendre
//! polynomials are, by construction, polynomials), so band 0-2 irradiance needs
//! no transcendental call. Only `sqrt` (through [`Vec3`]) and `f32::floor` /
//! `abs` are used, every division is `EPS`-guarded, and an invalid or
//! uninitialized probe returns a finite environment constant rather than `NaN`.
//! This keeps the `CPU` reference bit-reproducible against a future `GPU`
//! kernel.

use super::sort_cull::Aabb;
use super::Vec3;
use alloc::vec::Vec;

/// Absolute tolerance for `f32` comparisons and the floor of every guarded
/// divisor, so no reciprocal ever divides by (near) zero and no comparison ever
/// relies on exact bit equality.
pub const EPS: f32 = 1e-6;

/// Archimedes' constant, taken from `core` as a literal (no transcendental call
/// is made to obtain it). Used by the cosine-lobe convolution weights.
pub const PI: f32 = core::f32::consts::PI;

// ---------------------------------------------------------------------------
// Spherical-harmonic basis (design §17): real, Cartesian, polynomial.
// ---------------------------------------------------------------------------

/// Band-0 basis constant `Y(0,0)` = `0.5 * sqrt(1/PI)`.
pub const SH_K0: f32 = 0.282_094_79;
/// Band-1 basis constant, the shared magnitude of `Y(1,-1)`, `Y(1,0)`, `Y(1,1)`
/// = `sqrt(3/(4*PI))`.
pub const SH_K1: f32 = 0.488_602_5;
/// Band-2 off-axis constant for `Y(2,-2)`, `Y(2,-1)`, `Y(2,1)` =
/// `sqrt(15/(4*PI))`.
pub const SH_K2_XY: f32 = 1.092_548_4;
/// Band-2 constant for the `Y(2,0)` zonal term = `0.25 * sqrt(5/PI)`.
pub const SH_K2_Z2: f32 = 0.315_391_57;
/// Band-2 constant for the `Y(2,2)` term = `0.25 * sqrt(15/PI)`.
pub const SH_K2_X2: f32 = 0.546_274_2;

/// Cosine-lobe (Lambert) convolution weight for band 0 = `PI`.
pub const COSINE_LOBE_L0: f32 = PI;
/// Cosine-lobe convolution weight for band 1 = `2*PI/3`.
pub const COSINE_LOBE_L1: f32 = 2.0 * PI / 3.0;
/// Cosine-lobe convolution weight for band 2 = `PI/4`.
pub const COSINE_LOBE_L2: f32 = PI / 4.0;

/// Total solid angle of the unit sphere, `4*PI`, the normalization constant for
/// a uniform-sample `SH` projection.
pub const SPHERE_SOLID_ANGLE: f32 = 4.0 * PI;

/// Per-coefficient band index for the nine `L2` slots, so a convolution weight
/// can be looked up without a `match`.
const BAND_OF_INDEX: [u8; 9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

/// Evaluates the four real `SH` basis functions of bands 0-1 for a direction.
///
/// `dir` is normalized robustly first, so a non-unit or zero input never yields
/// `NaN` (a zero direction collapses to the band-0 term only). The returned
/// order is `[Y(0,0), Y(1,-1), Y(1,0), Y(1,1)]`, matching the layout of
/// [`ShColorL1`].
#[must_use]
pub fn sh_basis_l1(dir: Vec3) -> [f32; 4] {
    let n = dir.normalize_or_zero();
    [SH_K0, SH_K1 * n.y, SH_K1 * n.z, SH_K1 * n.x]
}

/// Evaluates the nine real `SH` basis functions of bands 0-2 for a direction.
///
/// `dir` is normalized robustly first. The returned order is
/// `[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), Y(2,-1), Y(2,0), Y(2,1),
/// Y(2,2)]`, matching the layout of [`ShColorL2`]. Every entry is a polynomial
/// in `n.x`, `n.y`, `n.z`; no transcendental function is used.
#[must_use]
pub fn sh_basis_l2(dir: Vec3) -> [f32; 9] {
    let n = dir.normalize_or_zero();
    [
        SH_K0,
        SH_K1 * n.y,
        SH_K1 * n.z,
        SH_K1 * n.x,
        SH_K2_XY * (n.x * n.y),
        SH_K2_XY * (n.y * n.z),
        SH_K2_Z2 * (3.0 * n.z * n.z - 1.0),
        SH_K2_XY * (n.x * n.z),
        SH_K2_X2 * (n.x * n.x - n.y * n.y),
    ]
}

/// Cosine-lobe convolution weight for a given `SH` band `l` in `0..=2`.
///
/// These are the Lambertian transfer coefficients `A_l` that turn a radiance
/// `SH` vector into an irradiance `SH` vector: `A_0 = PI`, `A_1 = 2*PI/3`,
/// `A_2 = PI/4`. Bands above 2 (unused here) convolve to zero.
#[must_use]
pub fn cosine_lobe_weight(band: u8) -> f32 {
    match band {
        0 => COSINE_LOBE_L0,
        1 => COSINE_LOBE_L1,
        2 => COSINE_LOBE_L2,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Colored SH containers (radiance stored as per-coefficient RGB via Vec3).
// ---------------------------------------------------------------------------

/// A colored band 0-1 spherical-harmonic vector: four `RGB` coefficients stored
/// as [`Vec3`] triples (the light-weight probe representation).
///
/// This is the cheap `L1` alternative to [`ShColorL2`], adequate for smooth
/// ambient environments where band-2 directionality is not required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShColorL1 {
    /// The four `RGB` coefficients in `sh_basis_l1` order.
    pub coeffs: [Vec3; 4],
}

impl ShColorL1 {
    /// The all-zero `L1` vector (a black, unlit environment).
    pub const ZERO: Self = Self {
        coeffs: [Vec3::ZERO; 4],
    };

    /// Reconstructs the *radiance* along `dir` by dotting the coefficients with
    /// the `L1` basis.
    #[must_use]
    pub fn evaluate_radiance(&self, dir: Vec3) -> Vec3 {
        let basis = sh_basis_l1(dir);
        let mut acc = Vec3::ZERO;
        for (c, &b) in self.coeffs.iter().zip(basis.iter()) {
            acc = acc.add(c.scale(b));
        }
        acc
    }

    /// Reconstructs the *irradiance* along the surface normal `n` by applying
    /// the cosine-lobe weights before the basis dot product.
    #[must_use]
    pub fn evaluate_irradiance(&self, n: Vec3) -> Vec3 {
        let basis = sh_basis_l1(n);
        // Band 0 uses the L0 cosine lobe; the three band-1 slots share L1.
        let weights = [
            COSINE_LOBE_L0,
            COSINE_LOBE_L1,
            COSINE_LOBE_L1,
            COSINE_LOBE_L1,
        ];
        let mut acc = Vec3::ZERO;
        for ((c, &b), &w) in self.coeffs.iter().zip(basis.iter()).zip(weights.iter()) {
            acc = acc.add(c.scale(w * b));
        }
        acc
    }

    /// Widens this `L1` vector into an [`ShColorL2`] with zeroed band-2 terms.
    #[must_use]
    pub fn to_l2(&self) -> ShColorL2 {
        let mut out = ShColorL2::ZERO;
        for (dst, src) in out.coeffs.iter_mut().take(4).zip(self.coeffs.iter()) {
            *dst = *src;
        }
        out
    }
}

/// A colored band 0-2 spherical-harmonic vector: nine `RGB` coefficients stored
/// as [`Vec3`] triples — the full irradiance probe representation.
///
/// A probe stores *radiance* `SH`; call [`ShColorL2::evaluate_irradiance`] to
/// get the cosine-convolved irradiance a Lambertian surface receives, or
/// [`ShColorL2::evaluate_radiance`] for the raw environment radiance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShColorL2 {
    /// The nine `RGB` coefficients in `sh_basis_l2` order.
    pub coeffs: [Vec3; 9],
}

impl ShColorL2 {
    /// The all-zero `L2` vector (a black, unlit environment).
    pub const ZERO: Self = Self {
        coeffs: [Vec3::ZERO; 9],
    };

    /// Builds a constant-radiance environment: only the band-0 coefficient is
    /// populated so every direction reconstructs `radiance`.
    ///
    /// The band-0 coefficient of a constant `c` is `c / Y(0,0)` scaled so that
    /// `evaluate_radiance` returns `c` in every direction.
    #[must_use]
    pub fn from_constant_radiance(radiance: Vec3) -> Self {
        let mut out = Self::ZERO;
        // evaluate_radiance(dir) = coeff0 * SH_K0, so coeff0 = radiance / SH_K0.
        out.coeffs[0] = radiance.scale(1.0 / SH_K0);
        out
    }

    /// Component-wise sum of two `SH` vectors.
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::ZERO;
        for (o, (a, b)) in out
            .coeffs
            .iter_mut()
            .zip(self.coeffs.iter().zip(rhs.coeffs.iter()))
        {
            *o = a.add(*b);
        }
        out
    }

    /// Uniform scale of every coefficient by a scalar.
    #[must_use]
    pub fn scale(&self, s: f32) -> Self {
        let mut out = Self::ZERO;
        for (o, c) in out.coeffs.iter_mut().zip(self.coeffs.iter()) {
            *o = c.scale(s);
        }
        out
    }

    /// Reconstructs the *radiance* along `dir` (raw environment radiance).
    #[must_use]
    pub fn evaluate_radiance(&self, dir: Vec3) -> Vec3 {
        let basis = sh_basis_l2(dir);
        let mut acc = Vec3::ZERO;
        for (c, &b) in self.coeffs.iter().zip(basis.iter()) {
            acc = acc.add(c.scale(b));
        }
        acc
    }

    /// Reconstructs the *irradiance* along the surface normal `n`.
    ///
    /// Applies the band-dependent cosine-lobe weight to each coefficient before
    /// the basis dot product, yielding the Lambertian irradiance
    /// `E(n) = sum_l A_l sum_m L(l,m) Y(l,m)(n)`. Divide by `PI` and multiply by
    /// albedo to obtain outgoing diffuse radiance.
    #[must_use]
    pub fn evaluate_irradiance(&self, n: Vec3) -> Vec3 {
        let basis = sh_basis_l2(n);
        let mut acc = Vec3::ZERO;
        for ((c, &b), &band) in self
            .coeffs
            .iter()
            .zip(basis.iter())
            .zip(BAND_OF_INDEX.iter())
        {
            acc = acc.add(c.scale(cosine_lobe_weight(band) * b));
        }
        acc
    }

    /// Samples the six axis irradiances into a lightweight [`AmbientCube`].
    ///
    /// This is the cheap runtime representation for far-field or low-priority
    /// particles: it trades band-2 directionality for six directional lookups.
    #[must_use]
    pub fn to_ambient_cube(&self) -> AmbientCube {
        AmbientCube {
            pos_x: self.evaluate_irradiance(Vec3::new(1.0, 0.0, 0.0)),
            neg_x: self.evaluate_irradiance(Vec3::new(-1.0, 0.0, 0.0)),
            pos_y: self.evaluate_irradiance(Vec3::new(0.0, 1.0, 0.0)),
            neg_y: self.evaluate_irradiance(Vec3::new(0.0, -1.0, 0.0)),
            pos_z: self.evaluate_irradiance(Vec3::new(0.0, 0.0, 1.0)),
            neg_z: self.evaluate_irradiance(Vec3::new(0.0, 0.0, -1.0)),
        }
    }

    /// Returns `true` when every coefficient is finite (no `NaN` / infinity).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.coeffs
            .iter()
            .all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite())
    }
}

/// Accumulates directional radiance samples into a colored `L2` `SH` vector by
/// least-squares projection.
///
/// Each [`ShProjector::add_sample`] adds `radiance * Y(dir) * weight` to the
/// running coefficients and `weight` to the running mass; [`ShProjector::resolve`]
/// then normalizes for a uniform-sphere Monte-Carlo estimate by scaling with
/// `SPHERE_SOLID_ANGLE / total_weight`. This is the `CPU` reference for baking a
/// probe from an environment capture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShProjector {
    /// Running weighted sum of `radiance * basis`.
    accum: ShColorL2,
    /// Running sum of sample weights (the estimate's mass).
    total_weight: f32,
}

impl ShProjector {
    /// A fresh, empty projector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accum: ShColorL2::ZERO,
            total_weight: 0.0,
        }
    }

    /// Adds one directional radiance sample with a relative solid-angle
    /// `weight` (use `1.0` for uniform sphere sampling).
    ///
    /// A non-positive weight is ignored so a degenerate sample cannot poison the
    /// estimate.
    pub fn add_sample(&mut self, dir: Vec3, radiance: Vec3, weight: f32) {
        if weight <= EPS {
            return;
        }
        let basis = sh_basis_l2(dir);
        for (c, &b) in self.accum.coeffs.iter_mut().zip(basis.iter()) {
            *c = c.add(radiance.scale(b * weight));
        }
        self.total_weight += weight;
    }

    /// Finalizes the projection into a radiance `SH` vector.
    ///
    /// Returns [`ShColorL2::ZERO`] when no positive-weight sample was added, so
    /// an empty projection is a safe black environment rather than a division by
    /// zero.
    #[must_use]
    pub fn resolve(&self) -> ShColorL2 {
        if self.total_weight <= EPS {
            return ShColorL2::ZERO;
        }
        self.accum.scale(SPHERE_SOLID_ANGLE / self.total_weight)
    }
}

impl Default for ShProjector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Octahedral direction encoding (design §8.3 probe visibility / direction map).
// ---------------------------------------------------------------------------

/// Returns `+1.0` for non-negative inputs and `-1.0` otherwise.
///
/// Used by the octahedral fold; treating `+0.0` as positive keeps the mapping
/// continuous across the seam without a transcendental `signum`.
#[must_use]
fn sign_nonzero(v: f32) -> f32 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Encodes a direction into octahedral `[-1, 1]^2` coordinates.
///
/// The direction is projected onto the octahedron `|x| + |y| + |z| = 1` and the
/// lower hemisphere is folded outward, giving an equal-area, seam-continuous 2-D
/// parameterization used by probe direction/visibility maps. Pure arithmetic;
/// `dir` is normalized robustly first so a zero input maps to the origin.
#[must_use]
pub fn octa_encode(dir: Vec3) -> (f32, f32) {
    let n = dir.normalize_or_zero();
    let denom = n.x.abs() + n.y.abs() + n.z.abs();
    if denom <= EPS {
        return (0.0, 0.0);
    }
    let inv = 1.0 / denom;
    let px = n.x * inv;
    let py = n.y * inv;
    if n.z >= 0.0 {
        (px, py)
    } else {
        (
            (1.0 - py.abs()) * sign_nonzero(px),
            (1.0 - px.abs()) * sign_nonzero(py),
        )
    }
}

/// Decodes octahedral `[-1, 1]^2` coordinates back into a unit direction.
///
/// Inverse of [`octa_encode`]; the reconstructed vector is normalized so the
/// round trip returns a unit direction. Pure arithmetic, `NaN`-free.
#[must_use]
pub fn octa_decode(u: f32, v: f32) -> Vec3 {
    let z = 1.0 - u.abs() - v.abs();
    let t = (-z).max(0.0);
    let x = u - t * sign_nonzero(u);
    let y = v - t * sign_nonzero(v);
    Vec3::new(x, y, z).normalize_or_zero()
}

/// Remaps octahedral `[-1, 1]^2` coordinates into texture-space `[0, 1]^2`.
#[must_use]
pub fn octa_to_unorm(u: f32, v: f32) -> (f32, f32) {
    (u * 0.5 + 0.5, v * 0.5 + 0.5)
}

/// Remaps texture-space `[0, 1]^2` coordinates back into octahedral
/// `[-1, 1]^2`.
#[must_use]
pub fn octa_from_unorm(u: f32, v: f32) -> (f32, f32) {
    (u * 2.0 - 1.0, v * 2.0 - 1.0)
}

// ---------------------------------------------------------------------------
// Ambient cube: the lightweight six-face irradiance representation.
// ---------------------------------------------------------------------------

/// A six-face ambient cube: one irradiance color per major axis direction.
///
/// This is the cheapest probe representation, blending the three faces a normal
/// points toward with squared-component weights (`Valve`-style "ambient cube").
/// It is used as a low-cost substitute for a full `SH` `L2` probe on far-field
/// or low-`LOD` particles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientCube {
    /// Irradiance received from the `+X` hemisphere.
    pub pos_x: Vec3,
    /// Irradiance received from the `-X` hemisphere.
    pub neg_x: Vec3,
    /// Irradiance received from the `+Y` hemisphere.
    pub pos_y: Vec3,
    /// Irradiance received from the `-Y` hemisphere.
    pub neg_y: Vec3,
    /// Irradiance received from the `+Z` hemisphere.
    pub pos_z: Vec3,
    /// Irradiance received from the `-Z` hemisphere.
    pub neg_z: Vec3,
}

impl AmbientCube {
    /// A cube whose six faces all share `color` (a uniform environment).
    #[must_use]
    pub fn from_constant(color: Vec3) -> Self {
        Self {
            pos_x: color,
            neg_x: color,
            pos_y: color,
            neg_y: color,
            pos_z: color,
            neg_z: color,
        }
    }

    /// Samples the cube along `n`, blending the three faces the normal faces
    /// with squared-component weights (which sum to 1 for a unit normal).
    ///
    /// `n` is normalized robustly first; a zero normal averages the whole cube
    /// rather than returning `NaN`.
    #[must_use]
    pub fn sample(&self, n: Vec3) -> Vec3 {
        let d = n.normalize_or_zero();
        if d.length_squared() <= EPS {
            // Degenerate normal: return the mean of the six faces.
            let sum = self
                .pos_x
                .add(self.neg_x)
                .add(self.pos_y)
                .add(self.neg_y)
                .add(self.pos_z)
                .add(self.neg_z);
            return sum.scale(1.0 / 6.0);
        }
        let wx = d.x * d.x;
        let wy = d.y * d.y;
        let wz = d.z * d.z;
        let fx = if d.x >= 0.0 { self.pos_x } else { self.neg_x };
        let fy = if d.y >= 0.0 { self.pos_y } else { self.neg_y };
        let fz = if d.z >= 0.0 { self.pos_z } else { self.neg_z };
        fx.scale(wx).add(fy.scale(wy)).add(fz.scale(wz))
    }
}

// ---------------------------------------------------------------------------
// Probe grid: trilinearly interpolated irradiance field (design §8.3).
// ---------------------------------------------------------------------------

/// A single irradiance probe: a colored `L2` radiance `SH` vector plus a
/// validity flag.
///
/// An invalid probe (unbaked, occluded, or leaked) contributes nothing to the
/// trilinear blend, and a cell of only invalid probes falls back to the grid's
/// environment constant — never `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrradianceProbe {
    /// The stored radiance `SH` (evaluate with a cosine lobe for irradiance).
    pub sh: ShColorL2,
    /// Whether this probe holds valid baked data.
    pub valid: bool,
}

impl IrradianceProbe {
    /// An invalid, zeroed probe (the uninitialized state).
    pub const INVALID: Self = Self {
        sh: ShColorL2::ZERO,
        valid: false,
    };

    /// Builds a valid probe from a radiance `SH` vector.
    #[must_use]
    pub fn valid(sh: ShColorL2) -> Self {
        Self { sh, valid: true }
    }
}

/// The three grid dimensions and a helper to size the probe array.
///
/// Each axis must hold at least one probe; a zero dimension makes the grid
/// invalid and every sample falls back to the environment constant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GridDims {
    /// Probe count along `X` (world minimum to maximum).
    pub x: u32,
    /// Probe count along `Y`.
    pub y: u32,
    /// Probe count along `Z`.
    pub z: u32,
}

impl GridDims {
    /// Builds grid dimensions from the three axis counts.
    #[must_use]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total probe count, `x * y * z` (saturating to avoid overflow).
    #[must_use]
    pub fn probe_count(self) -> usize {
        (self.x as usize)
            .saturating_mul(self.y as usize)
            .saturating_mul(self.z as usize)
    }

    /// Returns `true` when every axis holds at least one probe.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.x >= 1 && self.y >= 1 && self.z >= 1
    }
}

/// A world-space grid of irradiance probes sampled by trilinear interpolation.
///
/// Probes are stored in `X`-fastest, then `Y`, then `Z` order. Sampling maps a
/// world position into fractional grid coordinates (clamped to the grid on the
/// boundary), gathers the eight surrounding probes, and blends their `SH` by the
/// trilinear weights of the *valid* probes only. When no valid probe surrounds a
/// point — or the grid itself is degenerate — the sampler returns the
/// [`ProbeGrid::fallback`] environment irradiance so a particle never reads a
/// `NaN`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeGrid {
    /// World-space extent the grid spans (probe 0 sits at `bounds.min`, the last
    /// probe on each axis at `bounds.max`).
    pub bounds: Aabb,
    /// Probe counts on each axis.
    pub dims: GridDims,
    /// Probe storage, length `dims.probe_count()`.
    pub probes: Vec<IrradianceProbe>,
    /// Environment irradiance returned when no valid probe is available.
    pub fallback: Vec3,
}

impl ProbeGrid {
    /// Builds a grid, padding or truncating `probes` to `dims.probe_count()` so
    /// the stored length always matches the dimensions (missing entries become
    /// [`IrradianceProbe::INVALID`]).
    #[must_use]
    pub fn new(
        bounds: Aabb,
        dims: GridDims,
        mut probes: Vec<IrradianceProbe>,
        fallback: Vec3,
    ) -> Self {
        let want = dims.probe_count();
        if probes.len() < want {
            probes.resize(want, IrradianceProbe::INVALID);
        } else if probes.len() > want {
            probes.truncate(want);
        }
        Self {
            bounds,
            dims,
            probes,
            fallback,
        }
    }

    /// Builds an all-invalid grid of the given size (every sample falls back).
    #[must_use]
    pub fn uninitialized(bounds: Aabb, dims: GridDims, fallback: Vec3) -> Self {
        let probes = alloc::vec![IrradianceProbe::INVALID; dims.probe_count()];
        Self {
            bounds,
            dims,
            probes,
            fallback,
        }
    }

    /// Linear index of the probe at integer cell coordinates `(i, j, k)`.
    ///
    /// Returns `None` when any coordinate is out of range.
    #[must_use]
    pub fn probe_index(&self, i: u32, j: u32, k: u32) -> Option<usize> {
        if i >= self.dims.x || j >= self.dims.y || k >= self.dims.z {
            return None;
        }
        let idx = (i as usize)
            + (j as usize) * (self.dims.x as usize)
            + (k as usize) * (self.dims.x as usize) * (self.dims.y as usize);
        Some(idx)
    }

    /// Fetches the probe at `(i, j, k)`, or [`IrradianceProbe::INVALID`] when
    /// the coordinates or storage are out of range.
    #[must_use]
    pub fn probe_at(&self, i: u32, j: u32, k: u32) -> IrradianceProbe {
        match self.probe_index(i, j, k) {
            Some(idx) => self
                .probes
                .get(idx)
                .copied()
                .unwrap_or(IrradianceProbe::INVALID),
            None => IrradianceProbe::INVALID,
        }
    }

    /// Maps a world position to a fractional grid coordinate on one axis.
    ///
    /// Returns the base cell index and the `[0, 1]` fraction into the next cell,
    /// clamped so positions outside the bounds stay on the last cell (a `Clamp`
    /// address mode). `cells = dim - 1`; a single-probe axis always returns cell
    /// 0 with fraction 0.
    fn axis_coord(min: f32, max: f32, pos: f32, dim: u32) -> (u32, f32) {
        if dim <= 1 {
            return (0, 0.0);
        }
        let cells = (dim - 1) as f32;
        let span = max - min;
        if span.abs() <= EPS {
            return (0, 0.0);
        }
        let t = ((pos - min) * (1.0 / span)) * cells;
        let clamped = t.max(0.0).min(cells);
        let base = clamped.floor();
        let frac = clamped - base;
        let base_u = base as u32;
        // Keep the base index in `0..=dim-2` so the +1 neighbor stays in range.
        let max_base = dim - 2;
        if base_u > max_base {
            (max_base, 1.0)
        } else {
            (base_u, frac)
        }
    }

    /// Blends the eight probes surrounding `world_pos` into a single radiance
    /// `SH` vector, weighting by the trilinear weights of the valid probes only.
    ///
    /// Returns `None` when the grid is degenerate or every surrounding probe is
    /// invalid, signalling the caller to use the environment fallback.
    #[must_use]
    pub fn blend_sh(&self, world_pos: Vec3) -> Option<ShColorL2> {
        if !self.dims.is_valid() || self.probes.is_empty() {
            return None;
        }
        let (i0, fx) = Self::axis_coord(
            self.bounds.min.x,
            self.bounds.max.x,
            world_pos.x,
            self.dims.x,
        );
        let (j0, fy) = Self::axis_coord(
            self.bounds.min.y,
            self.bounds.max.y,
            world_pos.y,
            self.dims.y,
        );
        let (k0, fz) = Self::axis_coord(
            self.bounds.min.z,
            self.bounds.max.z,
            world_pos.z,
            self.dims.z,
        );

        let step_x = if self.dims.x > 1 { 1 } else { 0 };
        let step_y = if self.dims.y > 1 { 1 } else { 0 };
        let step_z = if self.dims.z > 1 { 1 } else { 0 };

        let mut acc = ShColorL2::ZERO;
        let mut total = 0.0f32;
        for corner in 0..8u32 {
            let cx = corner & 1;
            let cy = (corner >> 1) & 1;
            let cz = (corner >> 2) & 1;
            let wx = if cx == 0 { 1.0 - fx } else { fx };
            let wy = if cy == 0 { 1.0 - fy } else { fy };
            let wz = if cz == 0 { 1.0 - fz } else { fz };
            let weight = wx * wy * wz;
            if weight <= EPS {
                continue;
            }
            let i = i0 + cx * step_x;
            let j = j0 + cy * step_y;
            let k = k0 + cz * step_z;
            let probe = self.probe_at(i, j, k);
            if !probe.valid {
                continue;
            }
            acc = acc.add(&probe.sh.scale(weight));
            total += weight;
        }
        if total <= EPS {
            return None;
        }
        // Renormalize by the valid mass so a partially valid cell stays energy
        // consistent instead of darkening toward the invalid corners.
        Some(acc.scale(1.0 / total))
    }

    /// Samples the environment *irradiance* at `world_pos` along surface normal
    /// `normal`.
    ///
    /// Blends the surrounding probes (valid corners only) and evaluates the
    /// cosine-convolved irradiance; on any fallback path returns the constant
    /// [`ProbeGrid::fallback`] irradiance. The result is always finite.
    #[must_use]
    pub fn sample_irradiance(&self, world_pos: Vec3, normal: Vec3) -> Vec3 {
        match self.blend_sh(world_pos) {
            Some(sh) => {
                let e = sh.evaluate_irradiance(normal);
                if e.x.is_finite() && e.y.is_finite() && e.z.is_finite() {
                    e
                } else {
                    self.fallback
                }
            }
            None => self.fallback,
        }
    }

    /// Samples the raw environment *radiance* at `world_pos` along `dir`.
    ///
    /// Like [`ProbeGrid::sample_irradiance`] but skips the cosine convolution,
    /// for reflection-style lookups. Falls back to the environment constant when
    /// no valid probe surrounds the point.
    #[must_use]
    pub fn sample_radiance(&self, world_pos: Vec3, dir: Vec3) -> Vec3 {
        match self.blend_sh(world_pos) {
            Some(sh) => {
                let r = sh.evaluate_radiance(dir);
                if r.x.is_finite() && r.y.is_finite() && r.z.is_finite() {
                    r
                } else {
                    self.fallback
                }
            }
            None => self.fallback,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: SH round trip, octahedral round trip, trilinear boundaries, fallback.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric six-direction sample set (the major axes) whose weighted sum
    /// integrates a constant to band 0 only, so a constant projection has no
    /// spurious higher-band energy.
    fn axis_dirs() -> [Vec3; 6] {
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn sh_basis_l1_axis_values() {
        // +Z should light only the Y(1,0) lobe; the others are the constant term
        // or zero.
        let b = sh_basis_l1(Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(b[0], SH_K0));
        assert!(approx(b[1], 0.0));
        assert!(approx(b[2], SH_K1));
        assert!(approx(b[3], 0.0));
    }

    #[test]
    fn sh_basis_l2_is_normalized_input() {
        // A non-unit direction is normalized, so scaling the input does not
        // change the basis.
        let a = sh_basis_l2(Vec3::new(0.0, 0.0, 2.0));
        let b = sh_basis_l2(Vec3::new(0.0, 0.0, 1.0));
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(approx(*x, *y));
        }
    }

    #[test]
    fn sh_constant_radiance_round_trip() {
        // Projecting a constant radiance over the symmetric axis set should
        // reconstruct that constant in every direction, and integrate to the
        // Lambertian irradiance PI * L.
        let l = Vec3::new(0.4, 0.7, 1.1);
        let mut proj = ShProjector::new();
        for d in axis_dirs() {
            proj.add_sample(d, l, 1.0);
        }
        let sh = proj.resolve();
        // Radiance reconstruction ~= L in arbitrary directions.
        assert!(vec_approx(
            sh.evaluate_radiance(Vec3::new(0.0, 0.0, 1.0)),
            l
        ));
        assert!(vec_approx(
            sh.evaluate_radiance(Vec3::new(0.3, -0.4, 0.5)),
            l
        ));
        // Irradiance ~= PI * L.
        let e = sh.evaluate_irradiance(Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e, l.scale(PI)));
    }

    #[test]
    fn sh_from_constant_helper_matches_projection() {
        let l = Vec3::new(0.2, 0.5, 0.9);
        let sh = ShColorL2::from_constant_radiance(l);
        assert!(vec_approx(
            sh.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            l
        ));
        assert!(vec_approx(
            sh.evaluate_irradiance(Vec3::new(0.0, 0.0, 1.0)),
            l.scale(PI)
        ));
    }

    #[test]
    fn sh_directional_reconstruction_sign() {
        // A single bright sample from +Z should read brighter looking toward +Z
        // than toward -Z after reconstruction.
        let mut proj = ShProjector::new();
        proj.add_sample(Vec3::new(0.0, 0.0, 1.0), Vec3::splat(1.0), 1.0);
        let sh = proj.resolve();
        let up = sh.evaluate_radiance(Vec3::new(0.0, 0.0, 1.0));
        let down = sh.evaluate_radiance(Vec3::new(0.0, 0.0, -1.0));
        assert!(up.x > down.x);
    }

    #[test]
    fn sh_empty_projection_is_zero() {
        let proj = ShProjector::new();
        let sh = proj.resolve();
        assert!(vec_approx(
            sh.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::ZERO
        ));
        assert!(sh.is_finite());
    }

    #[test]
    fn sh_projection_ignores_nonpositive_weight() {
        let mut proj = ShProjector::new();
        proj.add_sample(Vec3::new(0.0, 0.0, 1.0), Vec3::splat(5.0), 0.0);
        proj.add_sample(Vec3::new(0.0, 0.0, 1.0), Vec3::splat(5.0), -2.0);
        assert!(vec_approx(proj.resolve().coeffs[0], Vec3::ZERO));
    }

    #[test]
    fn sh_add_and_scale() {
        let a = ShColorL2::from_constant_radiance(Vec3::splat(1.0));
        let b = ShColorL2::from_constant_radiance(Vec3::splat(2.0));
        let sum = a.add(&b);
        assert!(vec_approx(
            sum.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::splat(3.0)
        ));
        let half = sum.scale(0.5);
        assert!(vec_approx(
            half.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::splat(1.5)
        ));
    }

    #[test]
    fn octa_round_trip_many_dirs() {
        let dirs = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.3, 0.7, -0.2),
            Vec3::new(0.1, -0.9, 0.4),
        ];
        for d in dirs {
            let n = d.normalize_or_zero();
            let (u, v) = octa_encode(n);
            let decoded = octa_decode(u, v);
            assert!(vec_approx(decoded, n), "octa round trip failed for {n:?}");
        }
    }

    #[test]
    fn octa_encode_axis_origin() {
        // +Z maps to the octahedron center.
        let (u, v) = octa_encode(Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(u, 0.0));
        assert!(approx(v, 0.0));
    }

    #[test]
    fn octa_zero_direction_safe() {
        let (u, v) = octa_encode(Vec3::ZERO);
        assert!(approx(u, 0.0) && approx(v, 0.0));
        let d = octa_decode(0.0, 0.0);
        assert!(d.x.is_finite() && d.y.is_finite() && d.z.is_finite());
    }

    #[test]
    fn octa_unorm_round_trip() {
        let (u, v) = octa_encode(Vec3::new(0.2, -0.5, 0.3));
        let (tu, tv) = octa_to_unorm(u, v);
        let (bu, bv) = octa_from_unorm(tu, tv);
        assert!(approx(u, bu) && approx(v, bv));
        assert!((0.0..=1.0).contains(&tu) && (0.0..=1.0).contains(&tv));
    }

    #[test]
    fn ambient_cube_axis_faces() {
        let cube = AmbientCube {
            pos_x: Vec3::new(1.0, 0.0, 0.0),
            neg_x: Vec3::new(0.0, 1.0, 0.0),
            pos_y: Vec3::new(0.0, 0.0, 1.0),
            neg_y: Vec3::new(1.0, 1.0, 0.0),
            pos_z: Vec3::new(1.0, 0.0, 1.0),
            neg_z: Vec3::new(0.0, 1.0, 1.0),
        };
        assert!(vec_approx(
            cube.sample(Vec3::new(1.0, 0.0, 0.0)),
            cube.pos_x
        ));
        assert!(vec_approx(
            cube.sample(Vec3::new(0.0, -1.0, 0.0)),
            cube.neg_y
        ));
        assert!(vec_approx(
            cube.sample(Vec3::new(0.0, 0.0, 1.0)),
            cube.pos_z
        ));
    }

    #[test]
    fn ambient_cube_constant_is_flat() {
        let cube = AmbientCube::from_constant(Vec3::splat(0.6));
        assert!(vec_approx(
            cube.sample(Vec3::new(0.3, 0.4, 0.5)),
            Vec3::splat(0.6)
        ));
        // A zero normal averages to the same constant.
        assert!(vec_approx(cube.sample(Vec3::ZERO), Vec3::splat(0.6)));
    }

    fn unit_grid(fill: IrradianceProbe) -> ProbeGrid {
        let bounds = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let dims = GridDims::new(2, 2, 2);
        let probes = alloc::vec![fill; dims.probe_count()];
        ProbeGrid::new(bounds, dims, probes, Vec3::splat(0.05))
    }

    #[test]
    fn grid_uniform_probes_reconstruct_probe() {
        let l = Vec3::new(0.3, 0.6, 0.9);
        let sh = ShColorL2::from_constant_radiance(l);
        let grid = unit_grid(IrradianceProbe::valid(sh));
        // Center of the cube: all eight corners contribute equally.
        let e = grid.sample_irradiance(Vec3::splat(0.5), Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e, l.scale(PI)));
        // A corner samples the same constant field.
        let e2 = grid.sample_irradiance(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e2, l.scale(PI)));
    }

    #[test]
    fn grid_clamps_out_of_bounds() {
        let l = Vec3::new(0.2, 0.2, 0.2);
        let grid = unit_grid(IrradianceProbe::valid(ShColorL2::from_constant_radiance(l)));
        // Far outside the bounds: Clamp address mode keeps it on the field.
        let e = grid.sample_irradiance(Vec3::splat(100.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(vec_approx(e, l.scale(PI)));
        let e2 = grid.sample_irradiance(Vec3::splat(-50.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(vec_approx(e2, l.scale(PI)));
    }

    #[test]
    fn grid_all_invalid_falls_back() {
        let grid = unit_grid(IrradianceProbe::INVALID);
        let e = grid.sample_irradiance(Vec3::splat(0.5), Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e, Vec3::splat(0.05)));
        assert!(e.x.is_finite() && e.y.is_finite() && e.z.is_finite());
    }

    #[test]
    fn grid_uninitialized_falls_back() {
        let bounds = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(2.0),
        };
        let grid = ProbeGrid::uninitialized(bounds, GridDims::new(3, 3, 3), Vec3::splat(0.1));
        let e = grid.sample_irradiance(Vec3::splat(1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(vec_approx(e, Vec3::splat(0.1)));
    }

    #[test]
    fn grid_partial_validity_renormalizes() {
        // Two valid corners with the same field must reconstruct that field even
        // though the other six corners are invalid (energy stays consistent).
        let l = Vec3::new(0.5, 0.5, 0.5);
        let sh = ShColorL2::from_constant_radiance(l);
        let bounds = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let dims = GridDims::new(2, 2, 2);
        let mut probes = alloc::vec![IrradianceProbe::INVALID; dims.probe_count()];
        probes[0] = IrradianceProbe::valid(sh);
        probes[1] = IrradianceProbe::valid(sh);
        let grid = ProbeGrid::new(bounds, dims, probes, Vec3::splat(0.0));
        // A point on the edge between probe 0 and 1 blends only valid corners.
        let e = grid.sample_irradiance(Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e, l.scale(PI)));
    }

    #[test]
    fn grid_degenerate_dims_fall_back() {
        let bounds = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let grid = ProbeGrid::new(bounds, GridDims::new(0, 2, 2), Vec::new(), Vec3::splat(0.2));
        let e = grid.sample_irradiance(Vec3::splat(0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(vec_approx(e, Vec3::splat(0.2)));
    }

    #[test]
    fn grid_new_resizes_probe_storage() {
        let bounds = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let dims = GridDims::new(2, 2, 2);
        // Fewer probes than needed: padded with INVALID.
        let grid = ProbeGrid::new(bounds, dims, Vec::new(), Vec3::splat(0.3));
        assert_eq!(grid.probes.len(), 8);
        let e = grid.sample_irradiance(Vec3::splat(0.5), Vec3::new(0.0, 1.0, 0.0));
        assert!(vec_approx(e, Vec3::splat(0.3)));
    }

    #[test]
    fn probe_index_bounds() {
        let grid = unit_grid(IrradianceProbe::INVALID);
        assert_eq!(grid.probe_index(1, 1, 1), Some(7));
        assert_eq!(grid.probe_index(2, 0, 0), None);
        assert!(!grid.probe_at(9, 9, 9).valid);
    }

    #[test]
    fn ambient_cube_from_sh_matches_axis_irradiance() {
        let l = Vec3::new(0.4, 0.4, 0.4);
        let sh = ShColorL2::from_constant_radiance(l);
        let cube = sh.to_ambient_cube();
        // A constant environment cube reads the same on every face.
        assert!(vec_approx(cube.pos_x, l.scale(PI)));
        assert!(vec_approx(cube.neg_z, l.scale(PI)));
        assert!(vec_approx(
            cube.sample(Vec3::new(0.2, 0.3, 0.4)),
            l.scale(PI)
        ));
    }

    #[test]
    fn l1_widens_to_l2() {
        let mut l1 = ShColorL1::ZERO;
        l1.coeffs[0] = Vec3::splat(1.0).scale(1.0 / SH_K0);
        let l2 = l1.to_l2();
        assert!(vec_approx(
            l2.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::splat(1.0)
        ));
        assert!(vec_approx(
            l1.evaluate_radiance(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::splat(1.0)
        ));
        assert!(vec_approx(
            l1.evaluate_irradiance(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::splat(PI)
        ));
    }
}
