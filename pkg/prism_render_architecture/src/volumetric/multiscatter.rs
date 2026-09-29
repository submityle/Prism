//! Deterministic multiple-scattering energy-compensation `LUT` and spatial
//! irradiance-probe interpolation for the volumetric cloud subsystem (design
//! section 7b).
//!
//! Single-scatter ray-marching systematically loses the energy that real
//! clouds redistribute through many scattering events, which is why lit clouds
//! read as bright, diffuse, and only weakly tinted. Production engines
//! (`Frostbite`'s pre-integrated multi-scatter `LUT`, the Wrenninge / Nubis
//! `octave`-scatter approximation) recover that energy with a small
//! pre-integrated table plus a sparse set of spatial irradiance probes. This
//! module supplies the `CPU` reference for both:
//!
//! - [`MultiScatterLut`] is a three-axis table parameterized by view-zenith
//!   cosine, accumulated `optical_depth`, and single-scatter `albedo`. Each
//!   cell stores a bounded multi-scatter energy gain in `[0, 1]`, so a lookup
//!   can only ever *return* energy toward the analytic multiple-scattering
//!   ceiling and never amplify past unit `albedo`.
//! - [`ScatterProbe`] and [`ProbeGrid`] store band-resolved irradiance sampled
//!   on a regular grid and blend it with `trilinear` weights whose eight corner
//!   contributions always sum to one.
//!
//! The table is filled from the shared [`octave_scatter`] schedule and the
//! [`hg_phase`] anisotropy already validated in the `scatter` module, so this
//! file only *consumes* those approximations. It likewise only *consumes* the
//! shared atmosphere / froxel `LUT` services described in design sections 8 and
//! 5b (aerial perspective, pre-integrated transmittance); it never reimplements
//! or rewrites them.
//!
//! Every function is a pure, deterministic function of its arguments: identical
//! inputs yield bit-identical outputs, nothing reads wall-clock time or a random
//! generator, out-of-range coordinates clamp to the table edge rather than
//! `panic`, and no result exceeds one so the compensation never manufactures
//! energy. The crate determinism policy allows only `sqrt` among the `f32`
//! intrinsics, so the diffusion reflectance uses `sqrt` directly and the
//! optical-depth ramp uses the shared [`exp_approx`] helper; the `GPU` `WESL`
//! kernels evaluate the identical algebra with native intrinsics, and this
//! `CPU` reference exists so the numeric properties can be unit-tested in the
//! sandbox where there is no `GPU`.

use alloc::vec::Vec;

use super::math::{exp_approx, lerp, saturate, EPS};
use super::scatter::{hg_phase, octave_scatter, OctaveParams};
use super::Vec3;

/// Largest accumulated `optical_depth` represented on the depth axis.
///
/// Beyond this the multi-scatter gain is effectively saturated, so the axis
/// range stops here and larger queries clamp to the final column.
const DEFAULT_MAX_OPTICAL_DEPTH: f32 = 8.0;

/// Forward anisotropy used when weighting the per-`octave` phase for the table.
///
/// The gain table only needs a representative forward `g`; the successive
/// octaves decay it toward the `isotropic` limit through [`octave_scatter`].
const DEFAULT_LUT_G: f32 = 0.7;

/// Number of directional irradiance bands stored per probe.
///
/// The six bands are the signed axis directions (`+X`, `-X`, `+Y`, `-Y`, `+Z`,
/// `-Z`), a compact ambient-cube style basis for diffuse cloud irradiance.
pub const PROBE_BANDS: usize = 6;

/// Computes the flat storage index for a three-axis grid cell.
///
/// The `cos` axis is outermost, then the `optical_depth` axis, then the
/// `albedo` axis, matching the fill order in [`MultiScatterLut::from_fn`] and
/// [`ProbeGrid::new`].
#[must_use]
fn flat_index(dims: [usize; 3], ia0: usize, ia1: usize, ia2: usize) -> usize {
    (ia0 * dims[1] + ia1) * dims[2] + ia2
}

/// Returns the sample value of axis coordinate `i` on a `dim`-cell axis
/// spanning `[min, max]`.
///
/// A degenerate single-cell axis (`dim <= 1`) collapses to `min` so the fill
/// loops never divide by zero.
#[must_use]
fn axis_value(min: f32, max: f32, dim: usize, i: usize) -> f32 {
    if dim <= 1 {
        min
    } else {
        lerp(min, max, i as f32 / (dim - 1) as f32)
    }
}

/// Maps a raw `value` on `[min, max]` to a normalized fraction in `[0, 1]`.
///
/// The span is guarded against collapse (below [`EPS`]) and the result is
/// saturated, so out-of-range inputs clamp to the axis ends rather than
/// escaping `[0, 1]` or dividing by zero.
#[must_use]
fn axis_fraction(value: f32, min: f32, max: f32) -> f32 {
    let span = max - min;
    if span.abs() < EPS {
        0.0
    } else {
        saturate((value - min) / span)
    }
}

/// Resolves a normalized fraction `t` into the two bracketing cell indices and
/// the interpolation weight between them on a `dim`-cell axis.
///
/// The returned `(i0, i1, frac)` clamp to the edge: a single-cell axis yields
/// `(0, 0, 0.0)`, and `t` at or beyond the ends resolves to the boundary cell
/// with `frac == 0`, implementing clamp-to-edge addressing without a `panic`.
#[must_use]
fn lerp_index(t: f32, dim: usize) -> (usize, usize, f32) {
    if dim <= 1 {
        return (0, 0, 0.0);
    }
    let last = dim - 1;
    let pos = saturate(t) * last as f32;
    let base = pos.floor();
    let i0 = (base as usize).min(last);
    let i1 = (i0 + 1).min(last);
    let frac = pos - base;
    (i0, i1, frac)
}

/// Builds the eight corner weights of a unit-cube `trilinear` blend.
///
/// `fx`, `fy`, and `fz` are the per-axis interpolation fractions; corner `k`
/// uses bit `2` for `fx`, bit `1` for `fy`, and bit `0` for `fz`. Because each
/// axis contributes `frac` or `1 - frac`, the eight weights sum to exactly one
/// for any fractions (a partition of unity), which both [`MultiScatterLut`] and
/// [`ProbeGrid`] rely on to keep interpolated energy bounded.
#[must_use]
pub fn trilinear_weights(fx: f32, fy: f32, fz: f32) -> [f32; 8] {
    let mut weights = [0.0f32; 8];
    let mut k = 0usize;
    while k < 8 {
        let bx = (k >> 2) & 1;
        let by = (k >> 1) & 1;
        let bz = k & 1;
        let wx = if bx == 1 { fx } else { 1.0 - fx };
        let wy = if by == 1 { fy } else { 1.0 - fy };
        let wz = if bz == 1 { fz } else { 1.0 - fz };
        weights[k] = wx * wy * wz;
        k += 1;
    }
    weights
}

/// Weights the per-`octave` forward-scatter phase into a `[0, 1]` modulation.
///
/// Each `octave` from [`octave_scatter`] contributes its scattering
/// coefficient as a weight and the ratio of its [`hg_phase`] value to that
/// phase's forward peak as the modulation, so the aggregate is a
/// scattering-weighted average of normalized phase responses. Higher octaves
/// carry a smaller `g` (more `isotropic`) and less weight, matching the
/// Wrenninge `octave` decay. The result is `albedo`-independent, always in
/// `[0, 1]`, and falls back to one when `octave_count` is zero so the table
/// fill never divides by zero.
#[must_use]
fn octave_phase_weight(cos_theta: f32, params: OctaveParams) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    let mut n = 0u32;
    while n < params.octave_count {
        let (sigma_s, _sigma_t, g) = octave_scatter(1.0, 1.0, DEFAULT_LUT_G, n, params);
        let phase = hg_phase(cos_theta, g);
        let peak = hg_phase(1.0, g);
        let ratio = if peak > EPS {
            saturate(phase / peak)
        } else {
            1.0
        };
        numerator += sigma_s * ratio;
        denominator += sigma_s;
        n += 1;
    }
    if denominator > EPS {
        saturate(numerator / denominator)
    } else {
        1.0
    }
}

/// Evaluates the bounded multi-scatter energy gain stored in one table cell.
///
/// The gain is the product of three `[0, 1]` factors and therefore never
/// exceeds one:
///
/// - an analytic `isotropic` semi-infinite diffusion reflectance
///   `(1 - s) / (1 + s)` with `s = sqrt(1 - albedo)`, which is monotonically
///   non-decreasing in `albedo` (zero at `albedo = 0`, one at `albedo = 1`) and
///   the classical multiple-scattering ceiling this table approaches;
/// - an `optical_depth` ramp `1 - exp_approx(-depth)`, rising from zero so thin
///   media receive little compensation and thick media approach the ceiling;
/// - the `albedo`-independent [`octave_phase_weight`] anisotropy modulation.
///
/// Because only the diffusion term depends on `albedo`, the whole gain is
/// monotonically non-decreasing in `albedo`, and being a product of factors in
/// `[0, 1]` it is energy-conserving (never amplifying past one).
#[must_use]
fn energy_gain(cos_theta: f32, optical_depth: f32, albedo: f32, params: OctaveParams) -> f32 {
    let a = saturate(albedo);
    let s = saturate(1.0 - a).sqrt();
    let reflectance = if 1.0 + s > EPS {
        (1.0 - s) / (1.0 + s)
    } else {
        1.0
    };
    let depth = optical_depth.max(0.0);
    let depth_ramp = saturate(1.0 - exp_approx(-depth));
    let phase_mod = octave_phase_weight(cos_theta, params);
    saturate(reflectance * depth_ramp * phase_mod)
}

/// Pre-integrated multiple-scattering energy-gain `LUT`.
///
/// The table is parameterized on three axes — view-zenith cosine in `[-1, 1]`,
/// accumulated `optical_depth` in `[0, DEFAULT_MAX_OPTICAL_DEPTH]`, and
/// single-scatter `albedo` in `[0, 1]` — and stores one bounded energy gain per
/// cell. Lookups are `trilinear` with clamp-to-edge addressing, and every
/// stored and returned value lies in `[0, 1]` so the compensation approaches
/// the analytic multiple-scattering ceiling without ever amplifying energy.
#[derive(Clone, Debug)]
pub struct MultiScatterLut {
    /// Cell counts for the `cos`, `optical_depth`, and `albedo` axes.
    dims: [usize; 3],
    /// Inclusive axis minima for the `cos`, `optical_depth`, and `albedo` axes.
    mins: [f32; 3],
    /// Inclusive axis maxima for the `cos`, `optical_depth`, and `albedo` axes.
    maxs: [f32; 3],
    /// Row-major gain cells, length `dims[0] * dims[1] * dims[2]`, each `[0,1]`.
    data: Vec<f32>,
}

impl MultiScatterLut {
    /// Creates a zero-filled table with the given per-axis cell counts.
    ///
    /// Each dimension is clamped to at least one cell and the axis ranges take
    /// the module defaults (`cos` in `[-1, 1]`, `optical_depth` in
    /// `[0, DEFAULT_MAX_OPTICAL_DEPTH]`, `albedo` in `[0, 1]`).
    #[must_use]
    pub fn new(dims: [usize; 3]) -> Self {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        let count = dims[0] * dims[1] * dims[2];
        let mut data = Vec::with_capacity(count);
        let mut n = 0usize;
        while n < count {
            data.push(0.0);
            n += 1;
        }
        Self {
            dims,
            mins: [-1.0, 0.0, 0.0],
            maxs: [1.0, DEFAULT_MAX_OPTICAL_DEPTH, 1.0],
            data,
        }
    }

    /// Fills a new table by evaluating `f(cos, optical_depth, albedo)` at every
    /// cell, saturating each result into `[0, 1]`.
    ///
    /// The closure sees the physical axis coordinates (not fractions); the
    /// `saturate` guard keeps the stored table within the energy-conserving
    /// range regardless of what the closure returns.
    #[must_use]
    pub fn from_fn<F>(dims: [usize; 3], f: F) -> Self
    where
        F: Fn(f32, f32, f32) -> f32,
    {
        let mut lut = Self::new(dims);
        let dims = lut.dims;
        let mut idx = 0usize;
        let mut ic = 0usize;
        while ic < dims[0] {
            let cos = axis_value(lut.mins[0], lut.maxs[0], dims[0], ic);
            let mut id = 0usize;
            while id < dims[1] {
                let depth = axis_value(lut.mins[1], lut.maxs[1], dims[1], id);
                let mut ia = 0usize;
                while ia < dims[2] {
                    let albedo = axis_value(lut.mins[2], lut.maxs[2], dims[2], ia);
                    lut.data[idx] = saturate(f(cos, depth, albedo));
                    idx += 1;
                    ia += 1;
                }
                id += 1;
            }
            ic += 1;
        }
        lut
    }

    /// Builds the energy-gain table from the shared `octave`-scatter schedule.
    ///
    /// Every cell is filled with [`energy_gain`], so the table is monotonically
    /// non-decreasing in `albedo`, rises with `optical_depth`, and never
    /// exceeds one.
    #[must_use]
    pub fn build_energy_gain(dims: [usize; 3], params: OctaveParams) -> Self {
        Self::from_fn(dims, |cos, depth, albedo| {
            energy_gain(cos, depth, albedo, params)
        })
    }

    /// Returns the per-axis cell counts (`cos`, `optical_depth`, `albedo`).
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Samples the table with `trilinear` interpolation and clamp-to-edge.
    ///
    /// `cos`, `depth`, and `albedo` are mapped to axis fractions (clamped into
    /// range), the eight bracketing cells are blended with
    /// [`trilinear_weights`], and the result is saturated. Because the corner
    /// weights sum to one and every cell is in `[0, 1]`, the returned gain is
    /// always in `[0, 1]`; out-of-range coordinates return the nearest edge
    /// value without a `panic`.
    #[must_use]
    pub fn sample(&self, cos: f32, depth: f32, albedo: f32) -> f32 {
        let tc = axis_fraction(cos, self.mins[0], self.maxs[0]);
        let td = axis_fraction(depth, self.mins[1], self.maxs[1]);
        let ta = axis_fraction(albedo, self.mins[2], self.maxs[2]);
        let (c0, c1, fc) = lerp_index(tc, self.dims[0]);
        let (d0, d1, fd) = lerp_index(td, self.dims[1]);
        let (a0, a1, fa) = lerp_index(ta, self.dims[2]);
        let weights = trilinear_weights(fc, fd, fa);
        let corners = [
            self.data[flat_index(self.dims, c0, d0, a0)],
            self.data[flat_index(self.dims, c0, d0, a1)],
            self.data[flat_index(self.dims, c0, d1, a0)],
            self.data[flat_index(self.dims, c0, d1, a1)],
            self.data[flat_index(self.dims, c1, d0, a0)],
            self.data[flat_index(self.dims, c1, d0, a1)],
            self.data[flat_index(self.dims, c1, d1, a0)],
            self.data[flat_index(self.dims, c1, d1, a1)],
        ];
        let mut acc = 0.0f32;
        let mut k = 0usize;
        while k < 8 {
            acc += weights[k] * corners[k];
            k += 1;
        }
        saturate(acc)
    }
}

/// One spatial multiple-scatter irradiance probe.
///
/// A probe caches band-resolved diffuse irradiance at a fixed world position,
/// the sparse spatial complement to the [`MultiScatterLut`] energy gain. The
/// bands follow the [`PROBE_BANDS`] signed-axis basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterProbe {
    /// World-space position where the irradiance was captured.
    pub position: Vec3,
    /// Directional irradiance bands (`+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`).
    pub irradiance: [f32; PROBE_BANDS],
}

impl ScatterProbe {
    /// Creates a probe at `position` carrying the given band `irradiance`.
    #[must_use]
    pub fn new(position: Vec3, irradiance: [f32; PROBE_BANDS]) -> Self {
        Self {
            position,
            irradiance,
        }
    }
}

/// A regular grid of [`ScatterProbe`] samples over an axis-aligned box.
///
/// The grid stores one probe per cell corner and blends the eight probes
/// surrounding a query point with `trilinear` weights. Queries outside the box
/// clamp to the boundary probes, so the grid never reads out of bounds or
/// `panic`s, and it only *consumes* the shared atmosphere / froxel irradiance
/// that fills the probes — it does not recompute that lighting.
#[derive(Clone, Debug)]
pub struct ProbeGrid {
    /// Probe counts along the `X`, `Y`, and `Z` axes.
    dims: [usize; 3],
    /// World-space minimum corner of the probe box.
    min_corner: Vec3,
    /// World-space maximum corner of the probe box.
    max_corner: Vec3,
    /// Row-major probes, length `dims[0] * dims[1] * dims[2]`.
    probes: Vec<ScatterProbe>,
}

impl ProbeGrid {
    /// Creates a grid of zero-irradiance probes spanning the given box.
    ///
    /// Each dimension is clamped to at least one probe, and probes are placed on
    /// a regular lattice from `min_corner` to `max_corner` in `X`, `Y`, `Z`
    /// order (matching [`Self::sample`] addressing).
    #[must_use]
    pub fn new(dims: [usize; 3], min_corner: Vec3, max_corner: Vec3) -> Self {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        let count = dims[0] * dims[1] * dims[2];
        let mut probes = Vec::with_capacity(count);
        let mut ix = 0usize;
        while ix < dims[0] {
            let px = axis_value(min_corner.x, max_corner.x, dims[0], ix);
            let mut iy = 0usize;
            while iy < dims[1] {
                let py = axis_value(min_corner.y, max_corner.y, dims[1], iy);
                let mut iz = 0usize;
                while iz < dims[2] {
                    let pz = axis_value(min_corner.z, max_corner.z, dims[2], iz);
                    probes.push(ScatterProbe::new(Vec3::new(px, py, pz), [0.0; PROBE_BANDS]));
                    iz += 1;
                }
                iy += 1;
            }
            ix += 1;
        }
        Self {
            dims,
            min_corner,
            max_corner,
            probes,
        }
    }

    /// Builds a grid by evaluating `f(position)` at every lattice point.
    ///
    /// The closure returns the band `irradiance` stored at each probe; positions
    /// are laid out exactly as in [`Self::new`].
    #[must_use]
    pub fn from_fn<F>(dims: [usize; 3], min_corner: Vec3, max_corner: Vec3, f: F) -> Self
    where
        F: Fn(Vec3) -> [f32; PROBE_BANDS],
    {
        let mut grid = Self::new(dims, min_corner, max_corner);
        let mut i = 0usize;
        while i < grid.probes.len() {
            let position = grid.probes[i].position;
            grid.probes[i].irradiance = f(position);
            i += 1;
        }
        grid
    }

    /// Returns the per-axis probe counts (`X`, `Y`, `Z`).
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Returns the probe at lattice coordinate `(i, j, k)`, clamped to the grid.
    ///
    /// Out-of-range indices clamp to the boundary probe so the accessor never
    /// `panic`s.
    #[must_use]
    pub fn probe_at(&self, i: usize, j: usize, k: usize) -> ScatterProbe {
        let ci = i.min(self.dims[0] - 1);
        let cj = j.min(self.dims[1] - 1);
        let ck = k.min(self.dims[2] - 1);
        self.probes[flat_index(self.dims, ci, cj, ck)]
    }

    /// Samples band-resolved irradiance at `pos` with `trilinear` interpolation.
    ///
    /// The position is mapped to per-axis fractions (clamped into the box), the
    /// eight surrounding probes are blended per band with [`trilinear_weights`],
    /// and points outside the box clamp to the boundary probes. The eight corner
    /// weights sum to one, so the blend is a true convex combination of the
    /// stored irradiance and never `panic`s.
    #[must_use]
    pub fn sample(&self, pos: Vec3) -> [f32; PROBE_BANDS] {
        let tx = axis_fraction(pos.x, self.min_corner.x, self.max_corner.x);
        let ty = axis_fraction(pos.y, self.min_corner.y, self.max_corner.y);
        let tz = axis_fraction(pos.z, self.min_corner.z, self.max_corner.z);
        let (x0, x1, fx) = lerp_index(tx, self.dims[0]);
        let (y0, y1, fy) = lerp_index(ty, self.dims[1]);
        let (z0, z1, fz) = lerp_index(tz, self.dims[2]);
        let weights = trilinear_weights(fx, fy, fz);
        let corner_index = [
            flat_index(self.dims, x0, y0, z0),
            flat_index(self.dims, x0, y0, z1),
            flat_index(self.dims, x0, y1, z0),
            flat_index(self.dims, x0, y1, z1),
            flat_index(self.dims, x1, y0, z0),
            flat_index(self.dims, x1, y0, z1),
            flat_index(self.dims, x1, y1, z0),
            flat_index(self.dims, x1, y1, z1),
        ];
        let mut out = [0.0f32; PROBE_BANDS];
        let mut band = 0usize;
        while band < PROBE_BANDS {
            let mut acc = 0.0f32;
            let mut k = 0usize;
            while k < 8 {
                acc += weights[k] * self.probes[corner_index[k]].irradiance[band];
                k += 1;
            }
            out[band] = acc;
            band += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts `a` and `b` agree within `tol` absolute error.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Representative in-range and out-of-range sample coordinates.
    const COS_SAMPLES: [f32; 6] = [-5.0, -1.0, -0.25, 0.25, 1.0, 5.0];
    /// Representative in-range and out-of-range optical depths.
    const DEPTH_SAMPLES: [f32; 6] = [-1.0, 0.0, 0.5, 2.0, 8.0, 100.0];
    /// Representative in-range and out-of-range single-scatter albedos.
    const ALBEDO_SAMPLES: [f32; 6] = [-1.0, 0.0, 0.3, 0.7, 1.0, 2.0];

    #[test]
    fn lut_samples_stay_in_unit_range() {
        let lut = MultiScatterLut::build_energy_gain([8, 8, 8], OctaveParams::DEFAULT);
        for &c in &COS_SAMPLES {
            for &d in &DEPTH_SAMPLES {
                for &a in &ALBEDO_SAMPLES {
                    let v = lut.sample(c, d, a);
                    assert!((0.0..=1.0).contains(&v), "gain {v} out of range");
                }
            }
        }
    }

    #[test]
    fn lut_never_amplifies_energy() {
        let lut = MultiScatterLut::build_energy_gain([6, 6, 6], OctaveParams::DEFAULT);
        for &c in &COS_SAMPLES {
            for &d in &DEPTH_SAMPLES {
                for &a in &ALBEDO_SAMPLES {
                    assert!(lut.sample(c, d, a) <= 1.0 + EPS);
                }
            }
        }
    }

    #[test]
    fn lut_monotonic_non_decreasing_in_albedo() {
        let lut = MultiScatterLut::build_energy_gain([8, 8, 16], OctaveParams::DEFAULT);
        let cos = 0.4;
        let depth = 3.0;
        let mut prev = -1.0;
        let mut i = 0;
        while i <= 20 {
            let albedo = i as f32 / 20.0;
            let v = lut.sample(cos, depth, albedo);
            assert!(v >= prev - 1e-6, "albedo {albedo} broke monotonicity");
            prev = v;
            i += 1;
        }
    }

    #[test]
    fn lut_rises_with_optical_depth() {
        let lut = MultiScatterLut::build_energy_gain([8, 16, 8], OctaveParams::DEFAULT);
        let thin = lut.sample(0.5, 0.2, 0.9);
        let thick = lut.sample(0.5, 6.0, 0.9);
        assert!(thick >= thin - 1e-6, "gain should not shrink with depth");
    }

    #[test]
    fn lut_clamp_to_edge_matches_boundary() {
        let lut = MultiScatterLut::build_energy_gain([5, 5, 5], OctaveParams::DEFAULT);
        // Far out-of-range queries must equal the in-range boundary sample.
        let low = lut.sample(-100.0, -100.0, -100.0);
        let edge_low = lut.sample(-1.0, 0.0, 0.0);
        assert!(close(low, edge_low, 1e-6));
        let high = lut.sample(100.0, 100.0, 100.0);
        let edge_high = lut.sample(1.0, DEFAULT_MAX_OPTICAL_DEPTH, 1.0);
        assert!(close(high, edge_high, 1e-6));
    }

    #[test]
    fn trilinear_weights_sum_to_one() {
        let fractions = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        for &fx in &fractions {
            for &fy in &fractions {
                for &fz in &fractions {
                    let w = trilinear_weights(fx, fy, fz);
                    let mut sum = 0.0f32;
                    let mut k = 0;
                    while k < 8 {
                        assert!(w[k] >= -1e-7, "weight must be non-negative");
                        sum += w[k];
                        k += 1;
                    }
                    assert!(close(sum, 1.0, 1e-6), "weights sum {sum} != 1");
                }
            }
        }
    }

    #[test]
    fn probe_grid_trilinear_blend_and_clamp() {
        let grid = ProbeGrid::from_fn(
            [3, 3, 3],
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 2.0),
            |p| {
                // Encode position into the first three bands so interpolation is
                // checkable; remaining bands stay constant.
                [p.x, p.y, p.z, 1.0, 0.5, 0.25]
            },
        );
        // Interior point interpolates linearly between corner probes.
        let mid = grid.sample(Vec3::new(1.0, 1.0, 1.0));
        assert!(close(mid[0], 1.0, 1e-5));
        assert!(close(mid[1], 1.0, 1e-5));
        assert!(close(mid[2], 1.0, 1e-5));
        assert!(close(mid[3], 1.0, 1e-6));
        // Out-of-bounds query clamps to the boundary probe, no panic.
        let outside = grid.sample(Vec3::new(-50.0, -50.0, -50.0));
        assert!(close(outside[0], 0.0, 1e-6));
        assert!(close(outside[3], 1.0, 1e-6));
    }

    #[test]
    fn probe_at_clamps_out_of_bounds() {
        let grid = ProbeGrid::new(
            [2, 2, 2],
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        // Oversized indices must clamp to the last probe rather than panic.
        let clamped = grid.probe_at(99, 99, 99);
        let corner = grid.probe_at(1, 1, 1);
        assert_eq!(clamped, corner);
    }

    #[test]
    fn lut_and_probe_are_deterministic() {
        let a = MultiScatterLut::build_energy_gain([6, 6, 6], OctaveParams::DEFAULT);
        let b = MultiScatterLut::build_energy_gain([6, 6, 6], OctaveParams::DEFAULT);
        assert_eq!(a.dims(), b.dims());
        let va = a.sample(0.3, 2.5, 0.6);
        let vb = b.sample(0.3, 2.5, 0.6);
        assert_eq!(va.to_bits(), vb.to_bits());

        let grid_a = ProbeGrid::from_fn([3, 3, 3], Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0), |p| {
            [p.x, p.y, p.z, 0.1, 0.2, 0.3]
        });
        let grid_b = ProbeGrid::from_fn([3, 3, 3], Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0), |p| {
            [p.x, p.y, p.z, 0.1, 0.2, 0.3]
        });
        let sa = grid_a.sample(Vec3::new(1.5, 2.5, 3.5));
        let sb = grid_b.sample(Vec3::new(1.5, 2.5, 3.5));
        let mut band = 0;
        while band < PROBE_BANDS {
            assert_eq!(sa[band].to_bits(), sb[band].to_bits());
            band += 1;
        }
    }
}
