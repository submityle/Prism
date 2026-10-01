//! PCSS penumbra estimation — blocker search → soft shadow (CPU golden).
//!
//! Percentage-Closer Soft Shadows (PCSS) turn a hard shadow map into a
//! physically-plausible soft shadow whose penumbra widens with the receiver's
//! distance from its occluder — the hallmark of an area light.  The algorithm
//! has three stages, each represented here as an independent, tested function so
//! the GPU twin can reproduce them bit-for-bit:
//!
//! 1. **Blocker search** ([`blocker_search`]): sample a neighbourhood of the
//!    shadow map and average the depths of the texels *closer* to the light than
//!    the receiver.  These are the occluders casting onto the point.
//! 2. **Penumbra estimate** ([`penumbra_width`]): from the similar-triangles
//!    area-light model, `w = (d_receiver - d_blocker) / d_blocker * light_size`.
//!    When the blocker sits right under the receiver `w → 0` (contact hardening);
//!    when it is far above, `w` grows toward the full light size.
//! 3. **Filtered compare** ([`pcf_visibility`]): run a PCF (or Chebyshev-softened)
//!    depth comparison over a kernel whose radius is driven by the penumbra, then
//!    normalise to a visibility in `[0, 1]`.
//!
//! [`pcss_visibility`] chains all three using the shared low-discrepancy sampler
//! so the blocker-search and PCF taps are well stratified and temporally stable.
//!
//! # Conventions
//! * Depths are light-space receiver distances with the *same orientation* as
//!   the shadow map: smaller depth = closer to the light.  A texel is a blocker
//!   when its depth `< receiver_depth - bias`.
//! * `light_size` and all returned radii are in shadow-map **UV** units, so they
//!   feed a UV-space kernel directly; the GPU twin shares that space.
//! * Visibility is `1.0` fully lit, `0.0` fully shadowed, and always clamped.
//! * Every function is a deterministic pure function whose only external input
//!   is the caller's depth-sampling callback; all guard against zero blockers,
//!   zero/negative depths and non-finite inputs, never returning `NaN`.

use bevy_math::{ops, Vec2};

use crate::gi::sample::sample_2d;
use crate::gi::sample::sobol::to_unit_f32;
use crate::gi::world_space::visibility::chebyshev_weight;

/// Result of a PCSS blocker search over a shadow-map neighbourhood.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockerResult {
    /// Average light-space depth of the texels found closer than the receiver.
    pub average_depth: f32,
    /// How many of the sampled texels qualified as blockers.
    pub blocker_count: u32,
    /// Total number of texels sampled (the search budget actually used).
    pub sample_count: u32,
}

impl BlockerResult {
    /// Returns whether any blocker was found.
    #[inline]
    pub fn has_blocker(&self) -> bool {
        self.blocker_count > 0
    }

    /// Fraction of samples that were blockers, in `[0, 1]`.
    #[inline]
    pub fn coverage(&self) -> f32 {
        if self.sample_count == 0 {
            0.0
        } else {
            (self.blocker_count as f32 / self.sample_count as f32).clamp(0.0, 1.0)
        }
    }
}

/// Averages the depth of occluders around a receiver (PCSS stage 1).
///
/// Draws `samples` low-discrepancy taps on a disk of radius `search_radius` (UV
/// units) around `center`, queries each through `depth_at`, and averages the
/// depths that are closer to the light than `receiver_depth - bias`.  `seed`
/// decorrelates the disk pattern per pixel/frame.
///
/// Returns a [`BlockerResult`]; when no blocker is found `average_depth` is left
/// at `0.0` and [`BlockerResult::has_blocker`] is `false`.  A non-positive radius
/// or sample count, or a non-finite receiver depth, yields an empty result.
pub fn blocker_search<F>(
    center: Vec2,
    receiver_depth: f32,
    search_radius: f32,
    bias: f32,
    samples: u32,
    seed: u32,
    depth_at: F,
) -> BlockerResult
where
    F: Fn(Vec2) -> f32,
{
    let empty = BlockerResult {
        average_depth: 0.0,
        blocker_count: 0,
        sample_count: 0,
    };
    if !receiver_depth.is_finite() || !(search_radius > 0.0) || samples == 0 {
        return empty;
    }
    let bias = if bias.is_finite() { bias.max(0.0) } else { 0.0 };
    let threshold = receiver_depth - bias;

    let mut sum = 0.0_f32;
    let mut count = 0u32;
    for i in 0..samples {
        let (u, v) = sample_2d(i, seed);
        // Map the unit square to a disk via concentric mapping's polar form.
        let offset = disk_offset(u, v, search_radius);
        let d = depth_at(center + offset);
        if d.is_finite() && d < threshold {
            sum += d;
            count += 1;
        }
    }
    if count == 0 {
        return BlockerResult {
            average_depth: 0.0,
            blocker_count: 0,
            sample_count: samples,
        };
    }
    BlockerResult {
        average_depth: sum / count as f32,
        blocker_count: count,
        sample_count: samples,
    }
}

/// Maps a `[0,1)^2` sample to a disk offset of the given radius (polar form).
#[inline]
fn disk_offset(u: f32, v: f32, radius: f32) -> Vec2 {
    let r = radius * u.max(0.0).sqrt();
    let theta = core::f32::consts::TAU * v;
    Vec2::new(r * ops::cos(theta), r * ops::sin(theta))
}

/// Similar-triangles penumbra width from an area-light blocker model (stage 2).
///
/// Implements `w = (d_receiver - d_blocker) / d_blocker * light_size`, the
/// width (in UV units) of the penumbra cast by an area light of angular extent
/// `light_size` when a blocker at depth `d_blocker` shadows a receiver at depth
/// `d_receiver`.  The result is clamped non-negative; a blocker at or behind the
/// receiver (`d_blocker >= d_receiver`) or a non-positive/non-finite blocker
/// depth yields `0.0` (a hard, contact-hardened edge).
#[inline]
pub fn penumbra_width(receiver_depth: f32, blocker_depth: f32, light_size: f32) -> f32 {
    if !(blocker_depth > 0.0)
        || !blocker_depth.is_finite()
        || !receiver_depth.is_finite()
        || !light_size.is_finite()
    {
        return 0.0;
    }
    if blocker_depth >= receiver_depth {
        return 0.0;
    }
    let ratio = (receiver_depth - blocker_depth) / blocker_depth;
    (ratio * light_size.max(0.0)).max(0.0)
}

/// Maps a penumbra width to the PCF kernel radius in UV units (stage 2→3).
///
/// The penumbra width is a world-ish extent at the receiver; projecting it back
/// through the light frustum scales it by `near_ratio = near / d_receiver` so
/// close receivers get a wider UV kernel.  The radius is clamped into
/// `[min_radius, max_radius]` so a degenerate penumbra still runs at least a
/// 1-tap-wide PCF and a huge one cannot blow the kernel budget.
#[inline]
pub fn penumbra_filter_radius(
    penumbra: f32,
    receiver_depth: f32,
    near: f32,
    min_radius: f32,
    max_radius: f32,
) -> f32 {
    let lo = min_radius.max(0.0);
    let hi = max_radius.max(lo);
    if !penumbra.is_finite() || penumbra <= 0.0 {
        return lo;
    }
    let near_ratio = if receiver_depth > 0.0 && receiver_depth.is_finite() && near.is_finite() {
        (near.max(0.0) / receiver_depth).clamp(0.0, 1.0)
    } else {
        1.0
    };
    (penumbra * near_ratio).clamp(lo, hi)
}

/// Filtered depth comparison over a disk kernel (PCSS stage 3).
///
/// Draws `samples` low-discrepancy taps on a disk of `radius` UV units and, for
/// each, compares the stored depth against `receiver_depth - bias`.  When
/// `use_chebyshev` is `false` this is a plain PCF (fraction of taps the receiver
/// is in front of); when `true`, each tap is softened by [`chebyshev_weight`]
/// using the tap depth as a single-sample mean with a small variance floor,
/// giving a smoother, less banded transition.
///
/// Returns visibility in `[0, 1]` (`1` lit).  A non-finite receiver depth or
/// zero samples returns fully lit `1.0`; a zero radius degenerates to a single
/// centre tap.
pub fn pcf_visibility<F>(
    center: Vec2,
    receiver_depth: f32,
    radius: f32,
    bias: f32,
    samples: u32,
    seed: u32,
    use_chebyshev: bool,
    depth_at: F,
) -> f32
where
    F: Fn(Vec2) -> f32,
{
    if !receiver_depth.is_finite() || samples == 0 {
        return 1.0;
    }
    let bias = if bias.is_finite() { bias.max(0.0) } else { 0.0 };
    let radius = if radius.is_finite() { radius.max(0.0) } else { 0.0 };
    let threshold = receiver_depth - bias;

    let mut vis = 0.0_f32;
    for i in 0..samples {
        let (u, v) = sample_2d(i, seed ^ 0x68e3_1da4);
        let offset = disk_offset(u, v, radius);
        let d = depth_at(center + offset);
        if !d.is_finite() {
            // Treat a missing sample as unoccluded (lit) for stability.
            vis += 1.0;
            continue;
        }
        if use_chebyshev {
            // Single-sample moments: mean = d, var floored so equal depths read
            // as lit rather than hard-cutting.
            let mean = d;
            let mean_sq = d * d + 1.0e-5;
            // The receiver is lit where it is in front of the mean occluder.
            vis += chebyshev_weight(mean, mean_sq, threshold);
        } else if threshold <= d {
            vis += 1.0;
        }
    }
    (vis / samples as f32).clamp(0.0, 1.0)
}

/// Immutable PCSS configuration shared by [`pcss_visibility`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcssParams {
    /// Area-light extent in UV units (drives both search radius and penumbra).
    pub light_size: f32,
    /// Blocker-search disk radius in UV units.
    pub search_radius: f32,
    /// Light-frustum near distance (for the penumbra→UV projection).
    pub near: f32,
    /// Depth-compare bias (blocker threshold & PCF bias).
    pub bias: f32,
    /// Blocker-search sample budget.
    pub blocker_samples: u32,
    /// PCF sample budget.
    pub pcf_samples: u32,
    /// Minimum PCF kernel radius in UV units.
    pub min_radius: f32,
    /// Maximum PCF kernel radius in UV units.
    pub max_radius: f32,
    /// Whether to soften the PCF with Chebyshev weighting.
    pub use_chebyshev: bool,
}

impl Default for PcssParams {
    #[inline]
    fn default() -> Self {
        Self {
            light_size: 0.05,
            search_radius: 0.04,
            near: 0.1,
            bias: 0.002,
            blocker_samples: 16,
            pcf_samples: 24,
            min_radius: 0.001,
            max_radius: 0.08,
            use_chebyshev: false,
        }
    }
}

/// Full PCSS visibility: blocker search → penumbra → filtered compare.
///
/// Runs [`blocker_search`], and if no blocker is found returns fully lit `1.0`
/// (no occluder casts onto the point).  Otherwise it computes the
/// [`penumbra_width`], maps it to a [`penumbra_filter_radius`], and runs
/// [`pcf_visibility`] with that kernel.  `seed` decorrelates both sampling
/// stages (the PCF stage perturbs the seed internally).
///
/// Returns visibility in `[0, 1]`.  Because the kernel radius shrinks as the
/// blocker approaches the receiver, the soft edge hardens on contact — the
/// defining PCSS behaviour.
pub fn pcss_visibility<F>(
    center: Vec2,
    receiver_depth: f32,
    params: PcssParams,
    seed: u32,
    depth_at: &F,
) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let search = blocker_search(
        center,
        receiver_depth,
        params.search_radius,
        params.bias,
        params.blocker_samples,
        seed,
        depth_at,
    );
    if !search.has_blocker() {
        return 1.0;
    }
    let penumbra = penumbra_width(receiver_depth, search.average_depth, params.light_size);
    let radius = penumbra_filter_radius(
        penumbra,
        receiver_depth,
        params.near,
        params.min_radius,
        params.max_radius,
    );
    pcf_visibility(
        center,
        receiver_depth,
        radius,
        params.bias,
        params.pcf_samples,
        seed,
        params.use_chebyshev,
        depth_at,
    )
}

/// Converts a 24-bit low-discrepancy integer to a jitter in `[-0.5, 0.5)`.
///
/// A small helper for callers that want to dither the shadow lookup themselves;
/// reuses the shared [`to_unit_f32`] quantiser so the GPU twin matches exactly.
#[inline]
pub fn centered_jitter(raw: u32) -> f32 {
    to_unit_f32(raw) - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penumbra_grows_with_receiver_blocker_gap() {
        let near = penumbra_width(2.0, 1.9, 0.1);
        let far = penumbra_width(4.0, 1.0, 0.1);
        assert!(far > near, "far {far} should exceed near {near}");
        // Exact similar-triangles value.
        let w = penumbra_width(3.0, 1.0, 0.2);
        assert!((w - (3.0 - 1.0) / 1.0 * 0.2).abs() < 1e-6);
    }

    #[test]
    fn penumbra_hardens_on_contact() {
        // Blocker essentially at the receiver -> ~zero penumbra.
        let w = penumbra_width(2.0, 2.0 - 1e-6, 0.1);
        assert!(w < 1e-5);
        // Blocker behind receiver -> zero.
        assert_eq!(penumbra_width(2.0, 3.0, 0.1), 0.0);
    }

    #[test]
    fn penumbra_rejects_degenerate_depths() {
        assert_eq!(penumbra_width(2.0, 0.0, 0.1), 0.0);
        assert_eq!(penumbra_width(2.0, -1.0, 0.1), 0.0);
        assert_eq!(penumbra_width(f32::NAN, 1.0, 0.1), 0.0);
        assert_eq!(penumbra_width(2.0, f32::NAN, 0.1), 0.0);
    }

    #[test]
    fn filter_radius_clamped_and_scaled() {
        // Zero penumbra -> min radius.
        assert_eq!(penumbra_filter_radius(0.0, 2.0, 0.1, 0.002, 0.08), 0.002);
        // Huge penumbra -> clamped to max.
        assert_eq!(penumbra_filter_radius(100.0, 2.0, 0.1, 0.002, 0.08), 0.08);
        // Closer receiver => larger near_ratio => larger radius.
        let close = penumbra_filter_radius(0.1, 0.2, 0.1, 0.0, 1.0);
        let farr = penumbra_filter_radius(0.1, 2.0, 0.1, 0.0, 1.0);
        assert!(close > farr);
    }

    #[test]
    fn blocker_search_finds_near_occluders() {
        // A uniform occluder plane at depth 1.0 under a receiver at depth 3.0.
        let res = blocker_search(
            Vec2::splat(0.5),
            3.0,
            0.05,
            0.001,
            32,
            7,
            |_uv| 1.0,
        );
        assert!(res.has_blocker());
        assert_eq!(res.blocker_count, res.sample_count);
        assert!((res.average_depth - 1.0).abs() < 1e-5);
        assert!((res.coverage() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blocker_search_reports_none_when_all_behind() {
        // Everything is farther than the receiver -> no blockers.
        let res = blocker_search(
            Vec2::splat(0.5),
            1.0,
            0.05,
            0.001,
            16,
            3,
            |_uv| 5.0,
        );
        assert!(!res.has_blocker());
        assert_eq!(res.average_depth, 0.0);
        assert_eq!(res.sample_count, 16);
    }

    #[test]
    fn blocker_search_degenerate_inputs() {
        let r = blocker_search(Vec2::ZERO, f32::NAN, 0.05, 0.0, 8, 0, |_| 1.0);
        assert_eq!(r.sample_count, 0);
        let r2 = blocker_search(Vec2::ZERO, 2.0, 0.0, 0.0, 8, 0, |_| 1.0);
        assert_eq!(r2.sample_count, 0);
        let r3 = blocker_search(Vec2::ZERO, 2.0, 0.05, 0.0, 0, 0, |_| 1.0);
        assert_eq!(r3.sample_count, 0);
    }

    #[test]
    fn pcf_fully_lit_when_receiver_in_front() {
        // Receiver closer than everything -> fully lit.
        let v = pcf_visibility(Vec2::splat(0.5), 1.0, 0.02, 0.001, 16, 11, false, |_| 5.0);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pcf_fully_shadowed_when_receiver_behind() {
        // Receiver behind everything -> fully shadowed.
        let v = pcf_visibility(Vec2::splat(0.5), 5.0, 0.02, 0.001, 16, 11, false, |_| 1.0);
        assert!(v < 1e-6);
    }

    #[test]
    fn pcf_chebyshev_is_softer_than_hard() {
        // On a half-shadowed step edge, Chebyshev should not be a hard 0/1.
        let depth_fn = |uv: Vec2| if uv.x < 0.5 { 1.0 } else { 10.0 };
        let hard = pcf_visibility(Vec2::splat(0.5), 5.0, 0.1, 0.0, 64, 5, false, depth_fn);
        let cheb = pcf_visibility(Vec2::splat(0.5), 5.0, 0.1, 0.0, 64, 5, true, depth_fn);
        assert!((0.0..=1.0).contains(&hard));
        assert!((0.0..=1.0).contains(&cheb));
        // Chebyshev stays strictly interior where the hard test may cut sharply.
        assert!(cheb > 0.0 && cheb < 1.0);
    }

    #[test]
    fn pcf_degenerate_is_lit() {
        assert_eq!(
            pcf_visibility(Vec2::ZERO, f32::NAN, 0.02, 0.0, 8, 0, false, |_| 1.0),
            1.0
        );
        assert_eq!(
            pcf_visibility(Vec2::ZERO, 2.0, 0.02, 0.0, 0, 0, false, |_| 1.0),
            1.0
        );
    }

    #[test]
    fn pcss_lit_without_blocker() {
        let params = PcssParams::default();
        let v = pcss_visibility(Vec2::splat(0.5), 1.0, params, 1, &|_uv| 5.0);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pcss_shadowed_with_blocker() {
        // Occluder plane in front of a far receiver -> shadowed.
        let params = PcssParams::default();
        let v = pcss_visibility(Vec2::splat(0.5), 3.0, params, 1, &|_uv| 1.0);
        assert!(v < 0.5, "expected shadow, got {v}");
    }

    #[test]
    fn pcss_contact_hardening_narrows_kernel() {
        // Blocker just under the receiver -> tiny penumbra -> near-min radius.
        let params = PcssParams::default();
        let penumbra = penumbra_width(2.0, 1.99, params.light_size);
        let radius = penumbra_filter_radius(
            penumbra,
            2.0,
            params.near,
            params.min_radius,
            params.max_radius,
        );
        assert!(radius < 0.01, "contact kernel should be small, got {radius}");
    }

    #[test]
    fn pcss_is_deterministic() {
        let params = PcssParams::default();
        let f = |uv: Vec2| if uv.x < 0.5 { 1.0 } else { 4.0 };
        let a = pcss_visibility(Vec2::splat(0.5), 3.0, params, 9, &f);
        let b = pcss_visibility(Vec2::splat(0.5), 3.0, params, 9, &f);
        assert_eq!(a, b);
    }

    #[test]
    fn centered_jitter_in_range() {
        for raw in [0u32, 1, 1234, u32::MAX] {
            let j = centered_jitter(raw);
            assert!((-0.5..0.5).contains(&j), "jitter {j} out of range");
        }
    }

    #[test]
    fn blocker_coverage_bounds() {
        let r = BlockerResult {
            average_depth: 1.0,
            blocker_count: 3,
            sample_count: 4,
        };
        assert!((r.coverage() - 0.75).abs() < 1e-6);
        let empty = BlockerResult {
            average_depth: 0.0,
            blocker_count: 0,
            sample_count: 0,
        };
        assert_eq!(empty.coverage(), 0.0);
    }
}
