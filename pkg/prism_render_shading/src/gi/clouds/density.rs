//! Cloud density modelling: height gradients, coverage remapping, and detail
//! erosion that turn raw noise into a physical `[0, 1]` density field.
//!
//! The modelling pipeline mirrors the Horizon "Nubis" / Guerrilla approach: a
//! low-frequency base-shape noise is first weighted by a *height gradient* that
//! encodes the vertical silhouette of a cloud genus (stratus, cumulus, or
//! cumulonimbus), then thresholded by a *coverage* control, and finally
//! sharpened at the edges by subtracting high-frequency *detail* noise
//! (erosion). The whole chain is expressed through the single [`remap`] utility
//! so it stays monotone and clamp-safe.
//!
//! * [`remap`] — affine range remap with output clamping and degenerate-span
//!   fallback; the shared building block for the rest of the module.
//! * [`CloudType`] — the three modelled genera with distinct vertical profiles.
//! * [`height_gradient`] — the vertical density envelope for a genus, in
//!   `[0, 1]`.
//! * [`coverage_remap`] — raises the base density by a coverage control so more
//!   of the sky fills in as coverage grows.
//! * [`erosion`] — carves detail noise out of the cloud, rounding billows and
//!   wispy edges.
//! * [`cloud_density`] — the full base-shape -> gradient -> coverage ->
//!   erosion pipeline, returning a final `[0, 1]` density.
//!
//! # Conventions
//! * All inputs are treated defensively: normalised heights, noise samples, and
//!   coverage are clamped to `[0, 1]`; every output lies in `[0, 1]` and is
//!   finite (never `NaN`).
//! * Density grows monotonically with coverage and with the base-shape noise,
//!   and the erosion step can only *reduce* density — never increase it.
//! * Pure deterministic functions, no RNG / I/O / GPU / `unsafe`, and no
//!   transcendental maths (so no [`bevy_math::ops`] dependency is needed here).

/// Affine remap of `v` from `[in_lo, in_hi]` onto `[out_lo, out_hi]`.
///
/// The normalised position of `v` within the input span is clamped to `[0, 1]`
/// before being scaled into the output span, so the result never leaves
/// `[min(out_lo, out_hi), max(out_lo, out_hi)]`. A degenerate or non-finite
/// input span collapses to `out_lo`, and any non-finite result falls back to
/// `out_lo`, guaranteeing a finite output.
#[inline]
pub fn remap(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    if !span.is_finite() || !(span > f32::EPSILON) {
        return out_lo;
    }
    let t = ((v - in_lo) / span).clamp(0.0, 1.0);
    let out = out_lo + (out_hi - out_lo) * t;
    if out.is_finite() { out } else { out_lo }
}

/// Clamps a value expected in `[0, 1]`, mapping non-finite inputs to `0`.
#[inline]
fn unit_clamp(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Modelled cloud genera, each with a characteristic vertical profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudType {
    /// Low, flat sheet clouds hugging the bottom of the layer.
    Stratus,
    /// Mid-level heaped clouds with a rounded, bulging mid-section.
    Cumulus,
    /// Towering storm clouds filling almost the whole vertical extent.
    Cumulonimbus,
}

impl CloudType {
    /// The four control heights `(a, b, c, d)` of the trapezoidal envelope:
    /// density ramps up over `[a, b]`, plateaus over `[b, c]`, and ramps down
    /// over `[c, d]` (all as normalised heights in `[0, 1]`).
    #[inline]
    fn profile(self) -> (f32, f32, f32, f32) {
        match self {
            // Low, thin, bottom-hugging sheet.
            CloudType::Stratus => (0.0, 0.05, 0.15, 0.30),
            // Rounded heap biased to the lower-middle of the layer.
            CloudType::Cumulus => (0.0, 0.20, 0.45, 0.80),
            // Tall tower spanning nearly the whole layer.
            CloudType::Cumulonimbus => (0.0, 0.10, 0.80, 1.0),
        }
    }
}

/// Vertical density envelope for a cloud genus at normalised height
/// `height01 in [0, 1]`, returning a weight in `[0, 1]`.
///
/// The envelope is a trapezoid built from the genus' four control heights: it
/// rises from `0` to `1` across the lower edge, holds at `1` across the body,
/// and falls back to `0` across the upper edge. Heights outside the layer
/// evaluate to `0`. This is the factor that gives stratus their flat base and
/// cumulonimbus their towering bulk.
#[inline]
pub fn height_gradient(height01: f32, cloud_type: CloudType) -> f32 {
    let h = unit_clamp(height01);
    let (a, b, c, d) = cloud_type.profile();
    let rise = remap(h, a, b, 0.0, 1.0).clamp(0.0, 1.0);
    let fall = remap(h, c, d, 1.0, 0.0).clamp(0.0, 1.0);
    (rise * fall).clamp(0.0, 1.0)
}

/// Raises a base density by a coverage control, in `[0, 1]`.
///
/// `coverage` in `[0, 1]` shifts the lower edge of the density window: at
/// `coverage = 0` the window is `[1, 1]` so almost nothing survives, while at
/// `coverage = 1` the window is `[0, 1]` so the base density passes through
/// unchanged. The result increases monotonically with both `base` and
/// `coverage`. The classic Nubis coverage multiply `d * coverage` is folded in
/// to keep thin clouds from over-filling.
#[inline]
pub fn coverage_remap(base: f32, coverage: f32) -> f32 {
    let base = unit_clamp(base);
    let coverage = unit_clamp(coverage);
    let widened = remap(base, 1.0 - coverage, 1.0, 0.0, 1.0).clamp(0.0, 1.0);
    (widened * coverage).clamp(0.0, 1.0)
}

/// Erodes a cloud density by subtracting high-frequency `detail` noise.
///
/// `detail` in `[0, 1]` is scaled by `strength` in `[0, 1]` and used as the new
/// lower edge of a remap, carving wisps and billows out of the cloud's edges
/// while leaving its dense core intact. The result is always `<= density`
/// (erosion never adds mass) and stays in `[0, 1]`. With `strength = 0` the
/// density passes through unchanged.
#[inline]
pub fn erosion(density: f32, detail: f32, strength: f32) -> f32 {
    let density = unit_clamp(density);
    let detail = unit_clamp(detail);
    let strength = unit_clamp(strength);
    let threshold = detail * strength;
    let eroded = remap(density, threshold, 1.0, 0.0, 1.0).clamp(0.0, 1.0);
    // Never exceed the input density: erosion is strictly subtractive.
    eroded.min(density)
}

/// Full cloud density pipeline, returning a final density in `[0, 1]`.
///
/// Combines the stages in order:
/// 1. weight the `base_shape` noise by the genus [`height_gradient`];
/// 2. threshold by [`coverage_remap`] against `coverage`;
/// 3. carve edges with [`erosion`] using `detail` noise at `erosion_strength`.
///
/// All inputs are clamped to `[0, 1]`; the output is finite and in `[0, 1]`,
/// grows monotonically with `coverage` and `base_shape`, and never increases
/// under stronger erosion.
#[inline]
pub fn cloud_density(
    height01: f32,
    cloud_type: CloudType,
    base_shape: f32,
    coverage: f32,
    detail: f32,
    erosion_strength: f32,
) -> f32 {
    let gradient = height_gradient(height01, cloud_type);
    let shaped = unit_clamp(base_shape) * gradient;
    let covered = coverage_remap(shaped, coverage);
    erosion(covered, detail, erosion_strength).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_affine_and_clamped() {
        assert!((remap(0.5, 0.0, 1.0, 0.0, 10.0) - 5.0).abs() < 1e-6);
        assert!((remap(0.0, 0.0, 1.0, 2.0, 4.0) - 2.0).abs() < 1e-6);
        // Clamps below and above the input span.
        assert!((remap(-1.0, 0.0, 1.0, 0.0, 1.0) - 0.0).abs() < 1e-6);
        assert!((remap(2.0, 0.0, 1.0, 0.0, 1.0) - 1.0).abs() < 1e-6);
        // Inverted output span works.
        assert!((remap(1.0, 0.0, 1.0, 1.0, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn remap_degenerate_span_falls_back() {
        assert_eq!(remap(0.5, 1.0, 1.0, 3.0, 9.0), 3.0);
        assert_eq!(remap(0.5, 1.0, 0.0, 3.0, 9.0), 3.0);
        assert!(remap(f32::NAN, 0.0, 1.0, 0.0, 1.0).is_finite());
    }

    #[test]
    fn height_gradient_in_unit_range_and_zero_outside() {
        for &ct in &[CloudType::Stratus, CloudType::Cumulus, CloudType::Cumulonimbus] {
            for i in 0..=100 {
                let h = i as f32 / 100.0;
                let g = height_gradient(h, ct);
                assert!((0.0..=1.0).contains(&g), "gradient out of range: {g}");
            }
            // Below and above the layer the gradient is zero.
            assert_eq!(height_gradient(-0.5, ct), 0.0);
            assert_eq!(height_gradient(1.5, ct), 0.0);
        }
    }

    #[test]
    fn height_gradient_rises_then_falls() {
        // On the rising edge the gradient is non-decreasing.
        let ct = CloudType::Cumulus;
        let (a, b, _, _) = (0.0f32, 0.20f32, 0.0f32, 0.0f32);
        let _ = (a, b);
        let mut prev = height_gradient(0.0, ct);
        for i in 0..=20 {
            let h = i as f32 / 100.0; // 0.0 .. 0.20, within the rise.
            let g = height_gradient(h, ct);
            assert!(g + 1e-6 >= prev, "not rising at h={h}: {g} < {prev}");
            prev = g;
        }
        // Peak in the plateau exceeds the tails.
        assert!(height_gradient(0.3, ct) > height_gradient(0.0, ct));
        assert!(height_gradient(0.3, ct) > height_gradient(0.95, ct));
    }

    #[test]
    fn cumulonimbus_is_taller_than_stratus() {
        // High in the layer only the towering genus retains density.
        let high = 0.7;
        assert!(height_gradient(high, CloudType::Cumulonimbus) > 0.0);
        assert_eq!(height_gradient(high, CloudType::Stratus), 0.0);
    }

    #[test]
    fn coverage_is_monotone_increasing() {
        let base = 0.6;
        let mut prev = coverage_remap(base, 0.0);
        for i in 0..=100 {
            let c = i as f32 / 100.0;
            let v = coverage_remap(base, c);
            assert!(v + 1e-6 >= prev, "coverage not monotone at c={c}: {v} < {prev}");
            assert!((0.0..=1.0).contains(&v));
            prev = v;
        }
        // Zero coverage removes the cloud; full coverage keeps the most.
        assert!(coverage_remap(base, 0.0) <= coverage_remap(base, 1.0));
    }

    #[test]
    fn coverage_monotone_in_base() {
        let coverage = 0.7;
        let mut prev = coverage_remap(0.0, coverage);
        for i in 0..=100 {
            let b = i as f32 / 100.0;
            let v = coverage_remap(b, coverage);
            assert!(v + 1e-6 >= prev, "not monotone in base at b={b}");
            prev = v;
        }
    }

    #[test]
    fn erosion_is_subtractive() {
        for i in 0..=20 {
            let d = i as f32 / 20.0;
            for j in 0..=20 {
                let detail = j as f32 / 20.0;
                let eroded = erosion(d, detail, 0.5);
                assert!(eroded <= d + 1e-6, "erosion added mass: {eroded} > {d}");
                assert!((0.0..=1.0).contains(&eroded));
            }
        }
        // Zero strength is a no-op.
        assert!((erosion(0.6, 0.9, 0.0) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn cloud_density_bounded_and_monotone_in_coverage() {
        let mut prev = cloud_density(0.3, CloudType::Cumulus, 0.8, 0.0, 0.3, 0.4);
        for i in 0..=100 {
            let c = i as f32 / 100.0;
            let v = cloud_density(0.3, CloudType::Cumulus, 0.8, c, 0.3, 0.4);
            assert!((0.0..=1.0).contains(&v), "density out of range: {v}");
            assert!(v + 1e-6 >= prev, "density not monotone in coverage at c={c}");
            prev = v;
        }
    }

    #[test]
    fn cloud_density_never_nan_on_bad_inputs() {
        let v = cloud_density(
            f32::NAN,
            CloudType::Cumulonimbus,
            f32::INFINITY,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::NAN,
        );
        assert!(v.is_finite());
        assert!((0.0..=1.0).contains(&v));
    }
}
