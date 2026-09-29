//! Procedural cloud modelling: `coverage` / cloud-type / `height` gradient
//! modulation plus energy-preserving `detail erosion` `remap` (design section
//! 4, `Nubis`-style `Perlin-Worley` pipeline).
//!
//! This module owns the *shape* half of the density field. It takes the raw
//! base-shape and detail values (low-frequency `Perlin-Worley` and
//! high-frequency `Worley`/`fBm` noise produced by the sibling `noise` module)
//! as plain `f32` arguments and folds in the four authored/weather modulators:
//!
//! - the `coverage` gradient (weather-map `R` channel) — how much cloud fills a
//!   cell, applied through [`coverage_remap`] so more `coverage` monotonically
//!   grows the cloud;
//! - the cloud-type gradient — interpolating the flat stratus base and the
//!   puffy cumulus base through [`cloud_type_shape`];
//! - the `height` gradient — the vertical density profile of each
//!   [`CloudKind`] within its own altitude band via [`height_gradient`], zero
//!   at both band edges and positive in between;
//! - the `detail erosion` step — high-frequency noise eating fluffy detail out
//!   of the cloud edges via [`detail_erosion`], which uses `remap` rather than
//!   subtraction so total energy is preserved (no brightening or darkening
//!   drift, design section 6).
//!
//! Everything here is a pure, deterministic, allocation-free `f32` function.
//! Out-of-range inputs are saturated or clamped rather than trusted, so the
//! functions never panic and never emit a value outside `0..=1`. Only the
//! shared hand-rolled math from [`super::math`] is used (no float intrinsics
//! beyond the workspace-permitted `sqrt`, which this module does not even
//! need); the `GPU` `WESL` kernels mirror these formulas with native
//! intrinsics, and these `CPU` versions are the verification target.

use super::math::{lerp, remap, saturate, smoothstep};
use super::{CloudKind, CloudModeling};

/// Default `detail erosion` strength used when an authored [`CloudModeling`]
/// leaves it unspecified; a moderate bite that fluffs cumulus edges without
/// dissolving the core (design section 4).
pub const DEFAULT_EROSION_STRENGTH: f32 = 0.35;

/// Default base `coverage` applied before the weather-map `R` channel scales
/// it; a partly-cloudy sky at rest.
pub const DEFAULT_COVERAGE: f32 = 0.5;

/// The four control edges of one [`CloudKind`]'s vertical `height` profile,
/// all expressed as normalised band fractions in `0..=1`.
///
/// The profile rises from zero across `[rise_lo, rise_hi]` (the rounded cloud
/// base) and falls back to zero across `[fall_lo, fall_hi]` (the cloud top /
/// `anvil`). Between the two ramps the profile is a plateau of full weight.
/// Keeping the edges inside `0..=1` guarantees the gradient vanishes at both
/// band boundaries, matching the `Nubis` height-gradient contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightProfile {
    /// Fraction where the base ramp starts rising from zero.
    pub rise_lo: f32,
    /// Fraction where the base ramp reaches full weight.
    pub rise_hi: f32,
    /// Fraction where the top ramp starts falling from full weight.
    pub fall_lo: f32,
    /// Fraction where the top ramp reaches zero again.
    pub fall_hi: f32,
}

impl HeightProfile {
    /// The canonical vertical profile for a [`CloudKind`], describing how that
    /// kind distributes density *within its own altitude band*:
    ///
    /// - `Stratus`: a bottom-heavy flat sheet that fills the lower band and
    ///   fades out through the upper-middle.
    /// - `Cumulus`: a rounded cauliflower bulge peaking in the mid-to-upper
    ///   band with rounded base and top.
    /// - `Cirrus`: a thin wispy layer centred in the band's middle.
    /// - `Cumulonimbus`: deep vertical development filling nearly the whole
    ///   band, with the fall ramp pushed to the very top to model the spreading
    ///   `anvil` (design section 9b).
    #[must_use]
    pub fn for_kind(kind: CloudKind) -> Self {
        match kind {
            CloudKind::Stratus => Self {
                rise_lo: 0.00,
                rise_hi: 0.10,
                fall_lo: 0.55,
                fall_hi: 0.75,
            },
            CloudKind::Cumulus => Self {
                rise_lo: 0.08,
                rise_hi: 0.30,
                fall_lo: 0.65,
                fall_hi: 0.95,
            },
            CloudKind::Cirrus => Self {
                rise_lo: 0.05,
                rise_hi: 0.25,
                fall_lo: 0.55,
                fall_hi: 0.85,
            },
            CloudKind::Cumulonimbus => Self {
                rise_lo: 0.03,
                rise_hi: 0.12,
                fall_lo: 0.85,
                fall_hi: 1.00,
            },
        }
    }

    /// Evaluates the profile at a normalised band `height` fraction, returning
    /// a weight in `0..=1`.
    ///
    /// The result is the product of a rising `smoothstep` (the base ramp) and a
    /// falling `smoothstep` (the top ramp), so it is zero at the band edges,
    /// positive on the plateau, and never leaves `0..=1`.
    #[must_use]
    pub fn weight(self, height_fraction: f32) -> f32 {
        let h = saturate(height_fraction);
        let rising = smoothstep(self.rise_lo, self.rise_hi, h);
        let falling = 1.0 - smoothstep(self.fall_lo, self.fall_hi, h);
        saturate(rising * falling)
    }
}

/// Density weight of a [`CloudKind`] at a normalised `height` fraction.
///
/// `height_fraction` is the `0..=1` position inside the cloud layer's own
/// altitude band (see [`super::CloudGeometry::height_fraction`]). The returned
/// weight is zero at both band edges (`0.0` and `1.0`), positive in the
/// interior, and bounded to `0..=1`. Each kind uses its canonical
/// [`HeightProfile`] so the vertical shapes differ (flat stratus sheet, rounded
/// cumulus bulge, thin cirrus band, deep cumulonimbus column with a high
/// `anvil`) while all obeying the boundary and range contract.
#[must_use]
pub fn height_gradient(height_fraction: f32, kind: CloudKind) -> f32 {
    HeightProfile::for_kind(kind).weight(height_fraction)
}

/// Applies the `coverage` gradient to a base-shape value via the `Nubis`
/// `remap`.
///
/// `base_shape` is the raw low-frequency `Perlin-Worley` shape in `0..=1` and
/// `coverage` is the weather-driven fill amount in `0..=1`. The mapping
/// `remap(base, 1 - coverage, 1, 0, 1)` (then saturated) rescales the shape so
/// that raising `coverage` lifts more of the noise field above the cloud
/// threshold: the output is monotonically non-decreasing in `coverage` for any
/// fixed `base_shape`. At `coverage == 0` the collapsed input span makes the
/// shared `remap` return zero (clear sky); at `coverage == 1` the shape passes
/// through unchanged.
#[must_use]
pub fn coverage_remap(base_shape: f32, coverage: f32) -> f32 {
    let base = saturate(base_shape);
    let cov = saturate(coverage);
    saturate(remap(base, 1.0 - cov, 1.0, 0.0, 1.0))
}

/// Erodes fluffy `detail` out of a cloud's edges, preserving energy.
///
/// `base_density` is the coverage/height-shaped density in `0..=1`,
/// `detail_noise` is the high-frequency `Worley`/`fBm` value in `0..=1`, and
/// `erosion_strength` in `0..=1` scales how hard the detail bites. Following
/// the `Nubis` model this uses `remap(base, detail * strength, 1, 0, 1)` — a
/// rescale, *not* a subtraction — so the dynamic range is renormalised instead
/// of uniformly dimmed: dense cores stay dense while thin edges dissolve. The
/// result is clamped to `0..=1`, so it never goes negative and never overshoots
/// even when `detail * strength` approaches one (the collapsed span then yields
/// zero). The output is monotonically non-increasing in both `detail_noise`
/// and `erosion_strength`.
#[must_use]
pub fn detail_erosion(base_density: f32, detail_noise: f32, erosion_strength: f32) -> f32 {
    let base = saturate(base_density);
    let detail = saturate(detail_noise);
    let strength = saturate(erosion_strength);
    let eaten = detail * strength;
    saturate(remap(base, eaten, 1.0, 0.0, 1.0))
}

/// Interpolates the cloud-type gradient between the stratus and cumulus base
/// shapes.
///
/// `base_low` is the flat-sheet (stratus) base shape, `base_high` is the puffy
/// (cumulus) base shape, and `cloud_type` in `0..=1` blends stratus (`0`) to
/// cumulus (`1`), matching the weather-map `G` channel and
/// [`CloudModeling::cloud_type`]. Inputs are saturated and the blended result
/// is clamped to `0..=1`.
#[must_use]
pub fn cloud_type_shape(base_low: f32, base_high: f32, cloud_type: f32) -> f32 {
    let low = saturate(base_low);
    let high = saturate(base_high);
    let t = saturate(cloud_type);
    saturate(lerp(low, high, t))
}

/// Combines an authored base `coverage` with a sampled weather-map `coverage`
/// into a single effective fill in `0..=1`.
///
/// The product keeps both controls meaningful (either one at zero clears the
/// sky) and is monotonically non-decreasing in each argument, so the composed
/// density inherits the same `coverage` monotonicity.
#[must_use]
pub fn combine_coverage(base_coverage: f32, weather_coverage: f32) -> f32 {
    saturate(saturate(base_coverage) * saturate(weather_coverage))
}

/// Composes the final cloud density from the four modulators.
///
/// Pipeline (design section 4): apply the `coverage` `remap` to the base shape,
/// multiply by the `height` gradient to place it vertically, then erode fluffy
/// `detail` off the edges. Every stage stays in `0..=1` and every combination
/// is either a `remap` or a product of unit-range factors, so the result is
/// bounded to `0..=1` with no additive brightening or darkening drift (energy
/// conserving). Deterministic and panic-free for any input.
#[must_use]
pub fn compose_density(
    base_shape: f32,
    coverage: f32,
    height_weight: f32,
    detail_noise: f32,
    erosion_strength: f32,
) -> f32 {
    let covered = coverage_remap(base_shape, coverage);
    let shaped = saturate(covered * saturate(height_weight));
    detail_erosion(shaped, detail_noise, erosion_strength)
}

/// Convenience: composes density directly from a [`CloudModeling`] contract.
///
/// Wires the authored [`CloudModeling`] fields into the [`compose_density`]
/// pipeline: [`cloud_type_shape`] blends the stratus/cumulus bases by
/// [`CloudModeling::cloud_type`], [`combine_coverage`] folds the authored
/// [`CloudModeling::coverage`] with the sampled weather-map `coverage`,
/// [`height_gradient`] supplies the per-[`CloudKind`] vertical profile, and
/// [`CloudModeling::detail_erosion`] drives the erosion strength. Callers pass
/// the raw noise values (`base_low`, `base_high`, `detail_noise`) so this
/// module stays independent of the `noise` module's write set.
#[must_use]
pub fn compose_from_modeling(
    modeling: CloudModeling,
    kind: CloudKind,
    base_low: f32,
    base_high: f32,
    height_fraction: f32,
    weather_coverage: f32,
    detail_noise: f32,
) -> f32 {
    let base_shape = cloud_type_shape(base_low, base_high, modeling.cloud_type);
    let coverage = combine_coverage(modeling.coverage, weather_coverage);
    let height_weight = height_gradient(height_fraction, kind);
    compose_density(
        base_shape,
        coverage,
        height_weight,
        detail_noise,
        modeling.detail_erosion,
    )
}

#[cfg(test)]
mod tests {
    use super::super::math::EPS;
    use super::*;

    /// All four kinds, for exhaustive property scans.
    const KINDS: [CloudKind; 4] = [
        CloudKind::Cumulus,
        CloudKind::Stratus,
        CloudKind::Cirrus,
        CloudKind::Cumulonimbus,
    ];

    #[test]
    fn height_gradient_vanishes_at_edges_and_is_positive_in_the_middle() {
        for kind in KINDS {
            let at_bottom = height_gradient(0.0, kind);
            let at_top = height_gradient(1.0, kind);
            let at_mid = height_gradient(0.5, kind);
            assert!(at_bottom.abs() < EPS, "{kind:?} not ~0 at fraction 0");
            assert!(at_top.abs() < EPS, "{kind:?} not ~0 at fraction 1");
            assert!(at_mid > 0.0, "{kind:?} vanished at mid band");
        }
    }

    #[test]
    fn height_gradient_stays_in_unit_range_over_the_band() {
        for kind in KINDS {
            let mut h = 0.0;
            while h <= 1.0 {
                let w = height_gradient(h, kind);
                assert!((0.0..=1.0).contains(&w), "{kind:?} out of range at {h}");
                h += 0.02;
            }
        }
        // Out-of-band inputs saturate rather than panic or escape the range.
        for kind in KINDS {
            assert!(height_gradient(-3.0, kind).abs() < EPS);
            assert!(height_gradient(9.0, kind).abs() < EPS);
        }
    }

    #[test]
    fn height_gradient_orders_kinds_by_vertical_development_high_in_the_band() {
        // High in the band the deep convective column dominates, then cumulus,
        // then the thin cirrus tail, and the low flat stratus sheet has faded.
        let cnb = height_gradient(0.8, CloudKind::Cumulonimbus);
        let cumulus = height_gradient(0.8, CloudKind::Cumulus);
        let cirrus = height_gradient(0.8, CloudKind::Cirrus);
        let stratus = height_gradient(0.8, CloudKind::Stratus);
        assert!(cnb > cumulus, "cumulonimbus should exceed cumulus high up");
        assert!(cumulus > cirrus, "cumulus should exceed cirrus high up");
        assert!(
            cirrus > stratus,
            "cirrus should exceed faded stratus high up"
        );
    }

    #[test]
    fn coverage_remap_increases_monotonically_with_coverage() {
        for &base in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut prev = coverage_remap(base, 0.0);
            let mut cov = 0.0;
            while cov <= 1.0 {
                let out = coverage_remap(base, cov);
                assert!((0.0..=1.0).contains(&out), "coverage_remap out of range");
                assert!(
                    out + EPS >= prev,
                    "coverage_remap decreased at base={base} cov={cov}"
                );
                prev = out;
                cov += 0.05;
            }
        }
        // Zero coverage clears the sky; full coverage passes the shape through.
        assert!(coverage_remap(0.7, 0.0).abs() < EPS);
        assert!((coverage_remap(0.7, 1.0) - 0.7).abs() < 1e-4);
    }

    #[test]
    fn detail_erosion_stays_in_range_and_never_goes_negative() {
        let mut base = 0.0;
        while base <= 1.0 {
            let mut detail = 0.0;
            while detail <= 1.0 {
                let mut s = 0.0;
                while s <= 1.0 {
                    let out = detail_erosion(base, detail, s);
                    assert!(out >= 0.0, "erosion produced a negative density");
                    assert!(out <= 1.0, "erosion overshot unit range");
                    s += 0.25;
                }
                detail += 0.25;
            }
            base += 0.25;
        }
    }

    #[test]
    fn detail_erosion_is_monotonic_in_strength() {
        let base = 0.8;
        let detail = 0.6;
        let mut prev = detail_erosion(base, detail, 0.0);
        let mut s = 0.0;
        while s <= 1.0 {
            let out = detail_erosion(base, detail, s);
            assert!(
                out <= prev + EPS,
                "stronger erosion must not increase density at s={s}"
            );
            prev = out;
            s += 0.05;
        }
        // A dense core with no erosion is untouched; full erosion of pure noise
        // dissolves it completely.
        assert!((detail_erosion(1.0, 1.0, 0.0) - 1.0).abs() < 1e-4);
        assert!(detail_erosion(0.5, 1.0, 1.0).abs() < EPS);
    }

    #[test]
    fn cloud_type_shape_interpolates_stratus_to_cumulus() {
        assert!((cloud_type_shape(0.2, 0.9, 0.0) - 0.2).abs() < 1e-4);
        assert!((cloud_type_shape(0.2, 0.9, 1.0) - 0.9).abs() < 1e-4);
        assert!((cloud_type_shape(0.2, 0.9, 0.5) - 0.55).abs() < 1e-4);
        // Out-of-range type saturates.
        assert!((cloud_type_shape(0.2, 0.9, -1.0) - 0.2).abs() < 1e-4);
        assert!((cloud_type_shape(0.2, 0.9, 2.0) - 0.9).abs() < 1e-4);
    }

    #[test]
    fn combine_coverage_is_monotonic_and_bounded() {
        assert!(combine_coverage(0.0, 1.0).abs() < EPS);
        assert!(combine_coverage(1.0, 0.0).abs() < EPS);
        assert!((combine_coverage(1.0, 1.0) - 1.0).abs() < EPS);
        let a = combine_coverage(0.5, 0.5);
        let b = combine_coverage(0.5, 0.8);
        assert!(
            b + EPS >= a,
            "coverage combine not monotonic in weather term"
        );
    }

    #[test]
    fn compose_density_stays_in_unit_range() {
        for kind in KINDS {
            let mut h = 0.0;
            while h <= 1.0 {
                let hw = height_gradient(h, kind);
                for &cov in &[0.0, 0.4, 0.8, 1.0] {
                    for &detail in &[0.0, 0.5, 1.0] {
                        let d = compose_density(0.6, cov, hw, detail, 0.4);
                        assert!(
                            (0.0..=1.0).contains(&d),
                            "compose out of range at kind={kind:?} h={h} cov={cov}"
                        );
                    }
                }
                h += 0.05;
            }
        }
    }

    #[test]
    fn compose_from_modeling_wires_the_contract_and_stays_bounded() {
        let modeling = CloudModeling {
            coverage: 0.7,
            cloud_type: 0.8,
            base_frequency: 4.0,
            detail_frequency: 16.0,
            detail_erosion: DEFAULT_EROSION_STRENGTH,
            curl_strength: 12.0,
        };
        let d = compose_from_modeling(modeling, CloudKind::Cumulus, 0.3, 0.7, 0.5, 0.9, 0.4);
        assert!((0.0..=1.0).contains(&d), "modeling compose out of range");
        // Clearing the weather-map coverage clears the cloud regardless of the
        // authored base coverage.
        let clear = compose_from_modeling(modeling, CloudKind::Cumulus, 0.3, 0.7, 0.5, 0.0, 0.4);
        assert!(clear.abs() < EPS, "zero weather coverage should clear sky");
    }

    #[test]
    fn modelling_is_deterministic() {
        for kind in KINDS {
            let a = height_gradient(0.42, kind);
            let b = height_gradient(0.42, kind);
            assert_eq!(a, b, "height_gradient not deterministic for {kind:?}");
        }
        let x = compose_density(0.55, 0.65, 0.7, 0.35, 0.4);
        let y = compose_density(0.55, 0.65, 0.7, 0.35, 0.4);
        assert_eq!(x, y, "compose_density not deterministic");
    }
}
