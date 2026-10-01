//! Roughness-driven blur LOD and chromatic dispersion for rough refraction.
//!
//! Perfectly smooth glass transmits a sharp image of the background; a frosted
//! or micro-rough surface scatters the refracted rays into a cone, so the
//! transmitted image is *blurred*.  Screen-space refraction approximates that
//! blur by sampling a pre-filtered (mip) pyramid of the background buffer: the
//! rougher the surface, the higher the mip level — and the wider the gather
//! kernel — that is read.  This module is the CPU golden reference for that
//! roughness -> level-of-detail mapping.
//!
//! It also models **chromatic dispersion**.  A real dielectric's index of
//! refraction varies with wavelength (`n_blue > n_red`), so the three colour
//! channels refract by slightly different amounts and the background is
//! sampled at three slightly different UVs — the prismatic colour fringing seen
//! through a cut gem or a thick lens edge.  This module derives the per-channel
//! indices and the per-channel displacement scales that drive that split.
//!
//! It is disjoint from the geometric bend of [`crate::gi::refraction::bend`]
//! (which turns a single refracted ray into a UV offset) and from the
//! radiometric absorption of [`crate::gi::refraction::absorption`]: here we
//! only decide *how blurry* and *how colour-separated* the sampling is.
//!
//! # Conventions
//! * `roughness` is perceptual roughness in `[0, 1]`; `0` is a mirror-smooth
//!   interface and `1` is fully diffuse.  It is clamped into range on entry.
//! * `max_lod` is the top mip index of the background pyramid (e.g. a 1024-wide
//!   buffer has ~10 mips); LODs are returned in `[0, max_lod]`.
//! * `dispersion` is a unitless strength in `[0, 1]`; `0` disables chromatic
//!   splitting (all channels share the base index) and `1` is the strongest
//!   fringing the model allows.
//! * `no_std`: math via `bevy_math`; the power function via
//!   `bevy_math::ops::powf` and base-2 log via `bevy_math::ops::log2` (never
//!   `f32::powf`).  Nothing is allocated.
//! * All inputs are defensively clamped; every result is finite, non-negative,
//!   and the LOD / kernel mappings are monotonically non-decreasing in
//!   roughness.  No `NaN`, no infinity.
//! * Every function is a deterministic, allocation-free pure function.

use bevy_math::{ops, Vec2, Vec3};

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Perceptual-roughness exponent used to shape the roughness -> LOD curve.
///
/// A value `> 1` keeps low roughness sharp (little blur) and lets the blur ramp
/// up quickly only as the surface becomes clearly frosted, matching how GGX
/// specular lobes widen with roughness.
const LOD_SHAPE: f32 = 1.5;

/// Fraction of the base index of refraction spanned by the full dispersion
/// range at `dispersion == 1`.  A typical crown glass drifts by well under a
/// percent across the visible band; `0.04` gives an artistically visible —
/// but still plausible — fringe at maximum strength.
const MAX_DISPERSION_SPREAD: f32 = 0.04;

/// Clamps a perceptual roughness to `[0, 1]`; non-finite inputs become `0`.
#[inline]
fn clamp_roughness(roughness: f32) -> f32 {
    if roughness.is_finite() {
        roughness.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a `[0, 1]` strength parameter; non-finite inputs become `0`.
#[inline]
fn clamp_unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Sanitises a non-negative magnitude; non-finite inputs become `0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Maps perceptual roughness to a background-pyramid mip level.
///
/// Returns `max_lod * roughness^LOD_SHAPE`, a monotonically non-decreasing
/// curve on `[0, 1]` that is `0` at `roughness == 0` (sample the sharp mip 0)
/// and `max_lod` at `roughness == 1` (sample the blurriest mip).  The convex
/// shape keeps near-smooth glass crisp and ramps the blur up as the surface
/// frosts over.  `max_lod` is clamped non-negative.
#[inline]
pub fn roughness_to_lod(roughness: f32, max_lod: f32) -> f32 {
    let r = clamp_roughness(roughness);
    let max_lod = sanitize_nonneg(max_lod);
    max_lod * ops::powf(r, LOD_SHAPE)
}

/// Maps perceptual roughness to a gather-kernel radius in texels.
///
/// A linear ramp `max_radius * roughness`, clamped and monotonically
/// non-decreasing, used when the blur is produced by an explicit tap gather
/// rather than (or in addition to) a mip fetch.  `0` at `roughness == 0`.
#[inline]
pub fn roughness_to_kernel_radius(roughness: f32, max_radius: f32) -> f32 {
    clamp_roughness(roughness) * sanitize_nonneg(max_radius)
}

/// A trilinear mip blend: the two bracketing integer levels and the fraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MipBlend {
    /// Lower (sharper) integer mip level.
    pub lo: u32,
    /// Upper (blurrier) integer mip level; `lo + 1` unless already at the top.
    pub hi: u32,
    /// Interpolation fraction from `lo` toward `hi`, in `[0, 1]`.
    pub frac: f32,
}

/// Decomposes a fractional LOD into a trilinear mip blend.
///
/// Given a continuous `lod` (as returned by [`roughness_to_lod`]) and the top
/// mip index `max_lod_index`, returns the integer levels bracketing it plus the
/// fractional weight between them.  The LOD is clamped to
/// `[0, max_lod_index]`, so at the top level `lo == hi == max_lod_index` and
/// `frac == 0`.
#[inline]
pub fn mip_blend(lod: f32, max_lod_index: u32) -> MipBlend {
    let clamped = if lod.is_finite() {
        lod.clamp(0.0, max_lod_index as f32)
    } else {
        0.0
    };
    let lo = clamped as u32;
    if lo >= max_lod_index {
        MipBlend {
            lo: max_lod_index,
            hi: max_lod_index,
            frac: 0.0,
        }
    } else {
        MipBlend {
            lo,
            hi: lo + 1,
            frac: (clamped - lo as f32).clamp(0.0, 1.0),
        }
    }
}

/// Converts a background pyramid base width into its top mip index.
///
/// A `width`-texel buffer has `floor(log2(width)) + 1` mip levels, so the top
/// index is `floor(log2(width))`.  Degenerate widths (`< 1`) yield `0`.
#[inline]
pub fn max_lod_index_for_width(width: u32) -> u32 {
    if width < 1 {
        0
    } else {
        ops::log2(width as f32) as u32
    }
}

/// Per-channel indices of refraction for RGB dispersion.
///
/// Spreads the `base_ior` into three indices around it so that red refracts
/// least and blue most — the physical ordering `n_red < n_green < n_blue`.  The
/// green channel keeps the base index; red and blue are offset by
/// `dispersion * MAX_DISPERSION_SPREAD * base_ior / 2`.  At `dispersion == 0`
/// all three equal the (clamped) base index.  Every component stays `>= 1`.
#[inline]
pub fn dispersive_iors(base_ior: f32, dispersion: f32) -> Vec3 {
    let base = clamp_ior(base_ior);
    let spread = clamp_unit(dispersion) * MAX_DISPERSION_SPREAD * base * 0.5;
    Vec3::new(
        (base - spread).max(MIN_IOR),
        base,
        (base + spread).max(MIN_IOR),
    )
}

/// Per-channel displacement scales for chromatic refraction.
///
/// Given the medium's `base_ior`, the surrounding `medium_ior` (the incident
/// side, usually air), and a `dispersion` strength, returns a per-channel
/// multiplier `(r, g, b)` normalised so the green channel is exactly `1`.  Each
/// channel's scale is proportional to its index contrast `|n_ch - medium|`, so
/// the three refracted samples spread apart as dispersion grows.  At
/// `dispersion == 0` the result is [`Vec3::ONE`] (no separation).
#[inline]
pub fn channel_offset_scales(base_ior: f32, medium_ior: f32, dispersion: f32) -> Vec3 {
    let medium = clamp_ior(medium_ior);
    let iors = dispersive_iors(base_ior, dispersion);
    let contrast = Vec3::new(
        (iors.x - medium).abs(),
        (iors.y - medium).abs(),
        (iors.z - medium).abs(),
    );
    // Normalise against the green channel; guard a vanishing reference.
    if contrast.y > f32::MIN_POSITIVE {
        contrast / contrast.y
    } else {
        Vec3::ONE
    }
}

/// Per-channel UV offsets for a dispersive refraction.
///
/// Scales a single `base_offset` (the green-channel background-UV displacement
/// produced by [`crate::gi::refraction::bend`]) into three channel offsets
/// using [`channel_offset_scales`].  Returns `(red, green, blue)` offsets; at
/// `dispersion == 0` all three equal `base_offset`, and as dispersion grows the
/// red and blue samples pull apart from green, producing the colour fringe.
#[inline]
pub fn dispersive_offsets(
    base_offset: Vec2,
    base_ior: f32,
    medium_ior: f32,
    dispersion: f32,
) -> [Vec2; 3] {
    let s = channel_offset_scales(base_ior, medium_ior, dispersion);
    [base_offset * s.x, base_offset * s.y, base_offset * s.z]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    #[test]
    fn lod_is_zero_at_mirror_and_max_at_diffuse() {
        assert!(roughness_to_lod(0.0, 8.0).abs() < EPS);
        assert!((roughness_to_lod(1.0, 8.0) - 8.0).abs() < EPS);
    }

    #[test]
    fn lod_is_monotonic_in_roughness() {
        let mut prev = -1.0;
        for i in 0..=20 {
            let r = i as f32 / 20.0;
            let lod = roughness_to_lod(r, 10.0);
            assert!(lod >= prev - EPS, "non-monotonic at r={r}: {lod} < {prev}");
            assert!(lod >= 0.0 && lod <= 10.0 + EPS);
            prev = lod;
        }
    }

    #[test]
    fn lod_clamps_out_of_range_roughness() {
        assert!(roughness_to_lod(-5.0, 8.0).abs() < EPS);
        assert!((roughness_to_lod(5.0, 8.0) - 8.0).abs() < EPS);
        assert!(roughness_to_lod(f32::NAN, 8.0).abs() < EPS);
    }

    #[test]
    fn kernel_radius_is_monotonic_and_bounded() {
        assert!(roughness_to_kernel_radius(0.0, 16.0).abs() < EPS);
        assert!((roughness_to_kernel_radius(1.0, 16.0) - 16.0).abs() < EPS);
        assert!(roughness_to_kernel_radius(0.5, 16.0) < roughness_to_kernel_radius(0.6, 16.0));
    }

    #[test]
    fn mip_blend_brackets_the_lod() {
        let b = mip_blend(3.25, 9);
        assert_eq!(b.lo, 3);
        assert_eq!(b.hi, 4);
        assert!((b.frac - 0.25).abs() < EPS);
    }

    #[test]
    fn mip_blend_saturates_at_top_level() {
        let b = mip_blend(100.0, 9);
        assert_eq!(b.lo, 9);
        assert_eq!(b.hi, 9);
        assert!(b.frac.abs() < EPS);
    }

    #[test]
    fn mip_blend_handles_nan() {
        let b = mip_blend(f32::NAN, 9);
        assert_eq!(b.lo, 0);
        assert_eq!(b.hi, 1);
        assert!(b.frac.abs() < EPS);
    }

    #[test]
    fn max_lod_index_matches_power_of_two_width() {
        assert_eq!(max_lod_index_for_width(1024), 10);
        assert_eq!(max_lod_index_for_width(1), 0);
        assert_eq!(max_lod_index_for_width(0), 0);
    }

    #[test]
    fn dispersion_orders_channels_red_green_blue() {
        let iors = dispersive_iors(1.5, 0.8);
        assert!(iors.x < iors.y, "red should refract less than green");
        assert!(iors.y < iors.z, "blue should refract more than green");
        assert!(iors.x >= 1.0 && iors.z >= 1.0);
    }

    #[test]
    fn zero_dispersion_collapses_channels() {
        let iors = dispersive_iors(1.5, 0.0);
        assert!((iors.x - iors.y).abs() < EPS && (iors.y - iors.z).abs() < EPS);
        let scales = channel_offset_scales(1.5, 1.0, 0.0);
        assert!((scales - Vec3::ONE).length() < EPS);
    }

    #[test]
    fn dispersion_strength_widens_the_spread() {
        let soft = dispersive_iors(1.5, 0.2);
        let hard = dispersive_iors(1.5, 0.9);
        assert!((hard.z - hard.x) > (soft.z - soft.x));
    }

    #[test]
    fn channel_scales_are_green_normalised_and_separated() {
        let s = channel_offset_scales(1.5, 1.0, 0.7);
        assert!((s.y - 1.0).abs() < EPS, "green must be unit scale");
        // Red contrasts less with air than blue, so scales straddle 1.
        assert!(s.x < 1.0 && s.z > 1.0, "{s:?}");
    }

    #[test]
    fn dispersive_offsets_separate_the_channels() {
        let base = Vec2::new(0.02, -0.01);
        let offs = dispersive_offsets(base, 1.5, 1.0, 0.7);
        // Green stays put; red and blue diverge.
        assert!((offs[1] - base).length() < EPS);
        assert!((offs[0] - base).length() > EPS);
        assert!((offs[2] - base).length() > EPS);
        // Red shorter, blue longer than the green reference offset.
        assert!(offs[0].length() < base.length());
        assert!(offs[2].length() > base.length());
    }

    #[test]
    fn dispersive_offsets_collapse_without_dispersion() {
        let base = Vec2::new(0.03, 0.04);
        let offs = dispersive_offsets(base, 1.5, 1.0, 0.0);
        assert!((offs[0] - base).length() < EPS);
        assert!((offs[1] - base).length() < EPS);
        assert!((offs[2] - base).length() < EPS);
    }
}
