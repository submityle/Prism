//! Dual-scattering spherical-harmonic transmittance cache (design doc §8.6
//! item19).
//!
//! A single-scattering hair closure (`Marschner`/`Chiang` `R`/`TT`/`TRT`) misses
//! almost all of the energy that makes a light-coloured groom look right: the
//! visible brightness of blond, grey, or back-lit hair is dominated by *global
//! multiple forward scattering* through the thicket of strands sitting between
//! the light and the shaded fibre. `Zinke` 2008's dual-scattering approximation
//! folds that into two cheap aggregates: a global forward term governed by the
//! average forward-scatter attenuation `a_f` (accumulated as `a_f^n` over the
//! `n` strands a light ray crosses) and a local back-scatter term governed by
//! the average back-scatter attenuation `a_b`.
//!
//! Both `a_f` and `a_b` are *directional averages of transmittance*, and the
//! transmittance field around a shaded point is smooth and low-frequency. That
//! is exactly what a low-order spherical-harmonic (`SH`) expansion captures for
//! almost no storage: this module projects a set of directional transmittance
//! samples onto a second-order real `SH` basis (nine coefficients, `l = 0..=2`),
//! so the shading side can look the cached transmittance up in any direction
//! from nine numbers instead of re-marching the occluders every frame. The §8.5
//! near-/far-field adaptive path feeds this cache: near field re-projects often,
//! far field reuses the coefficients across frames and clusters.
//!
//! Design is a *material-independent, deterministic* architecture-side mapping,
//! in the same spirit as [`crate::hair::scatter_lod`] and
//! [`crate::hair::line_coverage`]: array in, array out, stable ordering,
//! golden-comparable, panic-free on empty / `NaN` / infinite input. It is fully
//! self-contained — it defines its own [`Vec3`] and [`TransmittanceSample`] and
//! shares no types with the other hair modules, keeping the module graph
//! disjoint.
//!
//! No transcendental math is used anywhere. The real `SH` basis is written in
//! its polynomial (Cartesian) form in the direction components `(x, y, z)`, so
//! there are no trigonometric calls; the only floating-point primitive beyond
//! add/mul is `sqrt` (for vector normalisation), and integer powers (`a_f^n`)
//! are evaluated with an explicit multiply loop rather than `powi`/`powf`.

use core::f32::consts::PI;

/// Number of real spherical-harmonic coefficients retained: a full second-order
/// expansion, bands `l = 0`, `l = 1`, `l = 2` (`1 + 3 + 5 = 9` terms). Second
/// order is the standard choice for smooth low-frequency irradiance/transmittance
/// caches: it reproduces a constant plus a linear plus a quadratic directional
/// trend, which is all a diffuse transmittance field carries, while staying tiny.
pub const SH_COEFFS: usize = 9;

/// Squared-length threshold below which a direction is treated as degenerate
/// (normalises to zero) instead of dividing by a near-zero length.
const NORM_EPS: f32 = 1e-12;

/// Replace a non-finite value (`NaN`/±∞) with `0`, leaving finite values intact.
#[must_use]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// A minimal 3-component direction/vector with hand-written math (no `glam`, no
/// `std`), kept local so this module shares no types with its siblings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component. By convention `+z` is the forward (light-facing) axis used
    /// by [`dual_scatter_factors`].
    pub z: f32,
}

impl Vec3 {
    /// A vector from explicit components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Euclidean dot product.
    #[must_use]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Unit-length copy, or the zero vector when the input is degenerate
    /// (zero-length, `NaN`, or infinite). Components are sanitised first so a
    /// stray `NaN`/∞ can never produce a `NaN` direction or panic.
    #[must_use]
    pub fn normalize_or_zero(self) -> Vec3 {
        let x = sanitize(self.x);
        let y = sanitize(self.y);
        let z = sanitize(self.z);
        let len_sq = x * x + y * y + z * z;
        if len_sq.is_finite() && len_sq > NORM_EPS {
            let len = len_sq.sqrt();
            Vec3::new(x / len, y / len, z / len)
        } else {
            Vec3::zero()
        }
    }
}

/// One directional transmittance sample: the (unnormalised is fine) direction
/// the sample was taken along, and the transmittance `value` seen along it. The
/// direction is normalised internally and `value` is sanitised/clamped where it
/// is consumed, so callers never have to pre-condition the input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmittanceSample {
    /// Sample direction (normalised internally; `+z` is forward).
    pub dir: Vec3,
    /// Transmittance along `dir`, nominally in `[0, 1]`.
    pub value: f32,
}

impl TransmittanceSample {
    /// A transmittance sample from a direction and value.
    #[must_use]
    pub const fn new(dir: Vec3, value: f32) -> Self {
        Self { dir, value }
    }
}

/// Nine real second-order `SH` coefficients, ordered band-major:
/// `[Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22]`. An all-zero set is the
/// identity "no transmittance cached", returned for empty input.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShCoeffs {
    /// Band-major coefficient array.
    pub c: [f32; SH_COEFFS],
}

impl ShCoeffs {
    /// Borrow the coefficients as a fixed-size array.
    #[must_use]
    pub const fn as_array(&self) -> &[f32; SH_COEFFS] {
        &self.c
    }
}

/// `Zinke` dual-scattering averaged attenuation factors, both in `[0, 1]`.
///
/// `forward` (`a_f`) is the mean transmittance over the forward (`+z`)
/// hemisphere and drives the global multiple-scatter term `a_f^n`; `backward`
/// (`a_b`) is the mean over the backward (`-z`) hemisphere and drives the local
/// back-scatter term. A hemisphere with no samples contributes `0`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScatterFactors {
    /// Forward-scatter factor `a_f` ∈ `[0, 1]`.
    pub forward: f32,
    /// Back-scatter factor `a_b` ∈ `[0, 1]`.
    pub backward: f32,
}

/// Evaluate the nine real second-order `SH` basis functions along `dir`.
///
/// The direction is normalised first (degenerate directions collapse to the
/// pole, leaving only the constant and the `Y20` term non-zero). The basis is
/// written in Cartesian polynomial form — no trigonometry — with the standard
/// orthonormal real-`SH` constants:
///
/// ```text
/// Y00  = 0.282095
/// Y1-1 = 0.488603 y     Y10 = 0.488603 z     Y11 = 0.488603 x
/// Y2-2 = 1.092548 xy    Y2-1 = 1.092548 yz   Y20 = 0.315392 (3 z^2 - 1)
/// Y21  = 1.092548 xz    Y22  = 0.546274 (x^2 - y^2)
/// ```
#[must_use]
pub fn sh_basis(dir: Vec3) -> [f32; SH_COEFFS] {
    let d = dir.normalize_or_zero();
    let x = d.x;
    let y = d.y;
    let z = d.z;
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Project directional transmittance samples onto the second-order real `SH`
/// basis.
///
/// Monte-Carlo estimate of `c_lm = ∫ T(ω) Y_lm(ω) dω`, approximated as
/// `Σ T·Y_lm·w` with the per-sample solid-angle weight `w = 4π / N` (samples are
/// assumed roughly uniform over the sphere). Sample values are sanitised, the
/// resulting coefficients are sanitised, and empty input returns the all-zero
/// [`ShCoeffs`]. Ordering-independent and panic-free.
#[must_use]
pub fn project_sh(samples: &[TransmittanceSample]) -> ShCoeffs {
    if samples.is_empty() {
        return ShCoeffs::default();
    }
    let weight = (4.0 * PI) / samples.len() as f32;
    let mut c = [0.0_f32; SH_COEFFS];
    for sample in samples {
        let value = sanitize(sample.value) * weight;
        let basis = sh_basis(sample.dir);
        for (accum, basis_value) in c.iter_mut().zip(basis.iter()) {
            *accum += value * basis_value;
        }
    }
    for accum in c.iter_mut() {
        *accum = sanitize(*accum);
    }
    ShCoeffs { c }
}

/// Reconstruct the cached transmittance along `dir` from its `SH` coefficients:
/// `eval_sh(coeffs, dir) = Σ c_lm Y_lm(dir)`, clamped to `>= 0` (transmittance is
/// non-negative) and sanitised so a `NaN`/∞ coefficient can never leak out.
#[must_use]
pub fn eval_sh(coeffs: ShCoeffs, dir: Vec3) -> f32 {
    let basis = sh_basis(dir);
    let mut accum = 0.0_f32;
    for (coeff, basis_value) in coeffs.c.iter().zip(basis.iter()) {
        accum += coeff * basis_value;
    }
    sanitize(accum).max(0.0)
}

/// Compute the `Zinke` forward/back dual-scattering factors directly from the
/// directional transmittance samples.
///
/// Splits samples by the sign of their normalised `z`: `+z` samples average into
/// `forward` (`a_f`), `-z` samples into `backward` (`a_b`). Each value is
/// sanitised and clamped to `[0, 1]` before averaging, so both factors land in
/// `[0, 1]`; a hemisphere with no samples (or empty input) yields `0`.
/// Ordering-independent and panic-free.
#[must_use]
pub fn dual_scatter_factors(samples: &[TransmittanceSample]) -> ScatterFactors {
    let mut forward_sum = 0.0_f32;
    let mut forward_count = 0_u32;
    let mut backward_sum = 0.0_f32;
    let mut backward_count = 0_u32;
    for sample in samples {
        let dir = sample.dir.normalize_or_zero();
        let value = sanitize(sample.value).clamp(0.0, 1.0);
        if dir.z > 0.0 {
            forward_sum += value;
            forward_count += 1;
        } else if dir.z < 0.0 {
            backward_sum += value;
            backward_count += 1;
        }
    }
    let forward = if forward_count > 0 {
        (forward_sum / forward_count as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let backward = if backward_count > 0 {
        (backward_sum / backward_count as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ScatterFactors { forward, backward }
}

/// Accumulated global forward-scatter transmittance `a_f^n` over `n` crossed
/// strands, the `Zinke` dual-scattering global multiplier `Ψ`.
///
/// `a_f` is sanitised (non-finite → `0`); the power is an explicit integer
/// multiply loop rather than `powi`/`powf`, so it is exact and
/// determinism-safe. `n = 0` returns `1`, `n = 1` returns `a_f`, and larger `n`
/// multiplies `a_f` into the accumulator `n` times.
#[must_use]
pub fn forward_scatter_power(a_f: f32, n: u32) -> f32 {
    let base = sanitize(a_f);
    let mut accum = 1.0_f32;
    let mut i = 0_u32;
    while i < n {
        accum *= base;
        i += 1;
    }
    accum
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn close_tol(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// `splitmix64` output finaliser: scrambles a counter into a well-mixed
    /// 64-bit value using only integer xor/shift/multiply. Applied per draw so
    /// consecutive coordinates do not share the low-dimensional lattice
    /// structure a bare LCG leaves in consecutive outputs (Marsaglia), which
    /// would otherwise bias the "uniform" sphere samples.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Deterministic uniform-on-sphere directions via rejection sampling of a
    /// cube, drawing each coordinate from an independent `splitmix64` draw (no
    /// trigonometry, no external rng). The finaliser decorrelates successive
    /// draws so the cube samples — and hence the retained sphere samples — are
    /// genuinely uniform rather than lying on a coarse LCG lattice.
    fn uniform_dirs(n: usize) -> Vec<Vec3> {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut out: Vec<Vec3> = Vec::new();
        while out.len() < n {
            let mut comp = [0.0_f32; 3];
            for channel in comp.iter_mut() {
                let bits = splitmix64(&mut state);
                let unit = ((bits >> 40) as u32) as f32 / ((1u32 << 24) as f32);
                *channel = unit * 2.0 - 1.0;
            }
            let v = Vec3::new(comp[0], comp[1], comp[2]);
            let len_sq = v.dot(v);
            if len_sq <= 1.0 && len_sq > 1e-4 {
                out.push(v.normalize_or_zero());
            }
        }
        out
    }

    #[test]
    fn basis_constant_term_is_fixed() {
        for dir in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-0.3, 0.7, 0.2),
        ] {
            let basis = sh_basis(dir);
            assert!(close(basis[0], 0.282095));
        }
    }

    #[test]
    fn basis_is_invariant_to_input_length() {
        let short = Vec3::new(0.3, -0.4, 0.5);
        let long = Vec3::new(3.0, -4.0, 5.0);
        let a = sh_basis(short);
        let b = sh_basis(long);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!(close(*va, *vb));
        }
    }

    #[test]
    fn basis_on_degenerate_dir_is_finite() {
        let basis = sh_basis(Vec3::zero());
        for v in basis {
            assert!(v.is_finite());
        }
        // Only the constant and the Y20 pole term survive at the origin.
        assert!(close(basis[0], 0.282095));
        assert!(close(basis[6], 0.315392 * (0.0 - 1.0)));
        for v in [
            basis[1], basis[2], basis[3], basis[4], basis[5], basis[7], basis[8],
        ] {
            assert!(close(v, 0.0));
        }
    }

    #[test]
    fn basis_is_orthonormal_under_monte_carlo() {
        // ∫ Y_i Y_j dω ≈ (4π/N) Σ Y_i Y_j = δ_ij for an orthonormal basis.
        let dirs = uniform_dirs(60_000);
        let weight = (4.0 * PI) / dirs.len() as f32;
        let mut gram = [[0.0_f32; SH_COEFFS]; SH_COEFFS];
        for dir in &dirs {
            let basis = sh_basis(*dir);
            for i in 0..SH_COEFFS {
                for j in 0..SH_COEFFS {
                    gram[i][j] += basis[i] * basis[j] * weight;
                }
            }
        }
        for (i, row) in gram.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(close_tol(val, expected, 0.08), "gram[{i}][{j}] = {val}");
            }
        }
    }

    #[test]
    fn constant_projection_l0_is_exact_and_reconstructs_constant() {
        let dirs = uniform_dirs(16_000);
        let value = 0.6_f32;
        let samples: Vec<TransmittanceSample> = dirs
            .iter()
            .map(|d| TransmittanceSample::new(*d, value))
            .collect();
        let coeffs = project_sh(&samples);
        // Y00 is constant, so c[0] = 4π · value · 0.282095 regardless of the
        // sample distribution — an exact, deterministic golden.
        let expected_l0 = 4.0 * PI * value * 0.282095;
        assert!(close_tol(coeffs.c[0], expected_l0, 1e-3));
        // Reconstruction of a constant field returns the constant everywhere.
        for dir in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-0.5, 0.5, 0.5),
        ] {
            assert!(close_tol(eval_sh(coeffs, dir), value, 0.05));
        }
    }

    #[test]
    fn reconstruction_of_linear_field_is_bounded() {
        // f(dir) = a + b·z is exactly in the span of {Y00, Y10}; the projection
        // should reconstruct it to within Monte-Carlo error.
        let dirs = uniform_dirs(16_000);
        let a = 0.5_f32;
        let b = 0.3_f32;
        let samples: Vec<TransmittanceSample> = dirs
            .iter()
            .map(|d| TransmittanceSample::new(*d, a + b * d.z))
            .collect();
        let coeffs = project_sh(&samples);
        for dir in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.7, 0.0, 0.3),
        ] {
            let d = dir.normalize_or_zero();
            let reference = (a + b * d.z).max(0.0);
            assert!(close_tol(eval_sh(coeffs, dir), reference, 0.05));
        }
    }

    #[test]
    fn eval_clamps_negative_reconstruction_to_zero() {
        // A pure negative Y10 coefficient makes the +z lobe negative; it must
        // clamp to 0 rather than return a negative transmittance.
        let mut coeffs = ShCoeffs::default();
        coeffs.c[2] = -1.0;
        assert!(close(eval_sh(coeffs, Vec3::new(0.0, 0.0, 1.0)), 0.0));
        // The opposite pole is positive and passes through.
        assert!(eval_sh(coeffs, Vec3::new(0.0, 0.0, -1.0)) > 0.0);
    }

    #[test]
    fn empty_samples_project_to_zero() {
        let coeffs = project_sh(&[]);
        for v in coeffs.as_array() {
            assert!(close(*v, 0.0));
        }
        // Reconstruction of the empty cache is zero everywhere.
        assert!(close(eval_sh(coeffs, Vec3::new(0.2, 0.3, 0.4)), 0.0));
    }

    #[test]
    fn scatter_factors_are_in_unit_range_and_match_means() {
        let samples = [
            TransmittanceSample::new(Vec3::new(0.0, 0.0, 1.0), 0.8),
            TransmittanceSample::new(Vec3::new(0.1, 0.0, 1.0), 0.6),
            TransmittanceSample::new(Vec3::new(0.0, 0.0, -1.0), 0.2),
            TransmittanceSample::new(Vec3::new(0.0, 0.1, -1.0), 0.4),
        ];
        let factors = dual_scatter_factors(&samples);
        assert!((0.0..=1.0).contains(&factors.forward));
        assert!((0.0..=1.0).contains(&factors.backward));
        assert!(close(factors.forward, 0.7));
        assert!(close(factors.backward, 0.3));
    }

    #[test]
    fn scatter_factors_clamp_out_of_range_values() {
        let samples = [
            TransmittanceSample::new(Vec3::new(0.0, 0.0, 1.0), 5.0),
            TransmittanceSample::new(Vec3::new(0.0, 0.0, -1.0), -3.0),
        ];
        let factors = dual_scatter_factors(&samples);
        assert!(close(factors.forward, 1.0));
        assert!(close(factors.backward, 0.0));
    }

    #[test]
    fn scatter_factors_empty_hemispheres_are_zero() {
        let factors = dual_scatter_factors(&[]);
        assert!(close(factors.forward, 0.0));
        assert!(close(factors.backward, 0.0));
        // Only a forward sample -> backward stays 0, forward is its value.
        let only_forward = [TransmittanceSample::new(Vec3::new(0.0, 0.0, 2.0), 0.5)];
        let f = dual_scatter_factors(&only_forward);
        assert!(close(f.forward, 0.5));
        assert!(close(f.backward, 0.0));
    }

    #[test]
    fn forward_scatter_power_matches_integer_powers() {
        let a_f = 0.7_f32;
        assert!(close(forward_scatter_power(a_f, 0), 1.0));
        assert!(close(forward_scatter_power(a_f, 1), a_f));
        assert!(close(forward_scatter_power(a_f, 2), a_f * a_f));
        assert!(close(forward_scatter_power(a_f, 3), a_f * a_f * a_f));
        // Monotone non-increasing in n for a_f in [0,1], and stays in [0,1].
        let mut prev = 1.0_f32;
        for n in 0..8 {
            let p = forward_scatter_power(a_f, n);
            assert!((0.0..=1.0).contains(&p));
            assert!(p <= prev + EPS);
            prev = p;
        }
    }

    #[test]
    fn forward_scatter_power_sanitizes_non_finite_base() {
        assert!(close(forward_scatter_power(f32::NAN, 0), 1.0));
        assert!(close(forward_scatter_power(f32::NAN, 3), 0.0));
        assert!(close(forward_scatter_power(f32::INFINITY, 2), 0.0));
    }

    #[test]
    fn non_finite_input_is_sanitized_without_panic() {
        let samples = [
            TransmittanceSample::new(Vec3::new(f32::NAN, 0.0, 1.0), 0.5),
            TransmittanceSample::new(Vec3::new(0.0, 0.0, 1.0), f32::INFINITY),
            TransmittanceSample::new(Vec3::new(0.0, f32::INFINITY, -1.0), 0.3),
            TransmittanceSample::new(Vec3::zero(), 0.4),
        ];
        let coeffs = project_sh(&samples);
        for v in coeffs.as_array() {
            assert!(v.is_finite());
        }
        let factors = dual_scatter_factors(&samples);
        assert!((0.0..=1.0).contains(&factors.forward));
        assert!((0.0..=1.0).contains(&factors.backward));
        let reconstruction = eval_sh(coeffs, Vec3::new(0.0, 0.0, 1.0));
        assert!(reconstruction.is_finite());
        assert!(reconstruction >= 0.0);
    }

    #[test]
    fn normalize_or_zero_handles_degenerate_vectors() {
        assert_eq!(Vec3::zero().normalize_or_zero(), Vec3::zero());
        let tiny = Vec3::new(1e-20, 0.0, 0.0).normalize_or_zero();
        assert_eq!(tiny, Vec3::zero());
        let nan = Vec3::new(f32::NAN, 1.0, 0.0).normalize_or_zero();
        assert!(nan.x.is_finite() && nan.y.is_finite() && nan.z.is_finite());
        let unit = Vec3::new(0.0, 3.0, 0.0).normalize_or_zero();
        assert!(close(unit.y, 1.0));
        assert!(close(unit.dot(unit), 1.0));
    }
}
