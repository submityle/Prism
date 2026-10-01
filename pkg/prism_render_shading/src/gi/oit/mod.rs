//! Order-independent transparency (OIT) — CPU golden reference.
//!
//! This module is the backend-neutral, GPU-free numerical reference for the
//! render engine's transparency resolve.  Correct transparency requires
//! back-to-front "over" compositing, which in turn requires a per-pixel depth
//! sort that is prohibitively expensive for arbitrary geometry.  OIT techniques
//! sidestep the sort by accumulating *order-independent* statistics and
//! reconstructing an approximate composite at resolve time.  Two complementary
//! families are implemented:
//!
//! * [`weighted_blended`] — McGuire & Bavoil 2013 weighted-blended OIT: a
//!   depth-weighted colour sum plus a revealage product.  Cheap, single pass,
//!   and exact for a single layer; the go-to default.
//! * [`moment_based`] — Münstermann et al. 2018 moment-based OIT: a compact
//!   power-moment summary of the absorbance-versus-depth distribution from which
//!   per-fragment front transmittance is reconstructed.  More storage, far
//!   better quality under heavy overlap.
//! * [`blend`] — the shared compositing algebra (premultiplied "over", coverage
//!   conversions) plus a sorted ground-truth compositor used to *verify* order
//!   independence.
//!
//! The high-level [`WeightedBlendedAccumulator`] and [`MomentOit`] types stitch
//! the primitives into ready-to-use resolvers driven by a single [`OitParams`]
//! bundle, so callers can composite a set of transparent fragments over an
//! opaque background without touching the low-level accumulators.
//!
//! # Conventions
//! * Colours are **straight** linear RGB with a separate `alpha ∈ [0, 1]`;
//!   premultiplication happens inside the accumulators.
//! * Depth is a positive, view-space distance from the eye (larger = farther).
//!   Moment-based OIT warps depth to `[-1, 1]` via [`OitParams::depth_near`] /
//!   [`OitParams::depth_far`]; weighted-blended OIT consumes the raw distance.
//! * Every public entry point is finite-clamped: degenerate fragments never
//!   produce `NaN`/`inf` output.
//!
//! # References
//! * McGuire & Bavoil 2013, *Weighted Blended Order-Independent Transparency*.
//! * Münstermann, Krüger, Bavoil & Wyman 2018, *Moment-Based Order-Independent
//!   Transparency*.

pub mod blend;
pub mod moment_based;
pub mod weighted_blended;

use bevy_math::Vec3;

pub use blend::{
    composite_average, composite_over_opaque, composite_sorted, is_order_independent,
    max_channel_error, over_premultiplied, over_straight, premultiply,
    revealage_to_coverage, transmittance_to_coverage, unpremultiply, SortableFragment,
};
pub use moment_based::{
    alpha_to_absorbance, generate_moments, reconstruct_transmittance, warp_depth, PowerMoments4,
    PowerMoments6,
};
pub use weighted_blended::{
    weight, ResolvedTransparency, WeightScheme, WeightedBlendTarget, DEFAULT_RESOLVE_EPS,
    MAX_WEIGHT, MIN_WEIGHT,
};

/// Shared authoring parameters for both OIT resolvers.
///
/// Defaults are production-safe: the general-purpose [`WeightScheme::Equation8`]
/// depth weight, the paper's resolve epsilon, a small moment bias to suppress
/// ringing, and a `[0.1, 1000]` view-depth range used to warp depths for
/// moment-based OIT.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OitParams {
    /// Depth-weighting function for weighted-blended OIT.
    pub weight_scheme: WeightScheme,
    /// Divide-by-zero guard for the weighted-blended resolve average.
    pub resolve_eps: f32,
    /// Moment regularisation `∈ [0, 1]` for moment-based reconstruction.
    pub moment_bias: f32,
    /// Near plane used to warp depths into `[-1, 1]` for moment-based OIT.
    pub depth_near: f32,
    /// Far plane used to warp depths into `[-1, 1]` for moment-based OIT.
    pub depth_far: f32,
}

impl Default for OitParams {
    #[inline]
    fn default() -> Self {
        Self {
            weight_scheme: WeightScheme::Equation8,
            resolve_eps: DEFAULT_RESOLVE_EPS,
            moment_bias: 5.0e-3,
            depth_near: 0.1,
            depth_far: 1000.0,
        }
    }
}

impl OitParams {
    /// Warps a view-space depth into `[-1, 1]` using the configured planes.
    #[inline]
    pub fn warp(&self, z: f32) -> f32 {
        warp_depth(z, self.depth_near, self.depth_far)
    }
}

/// High-level weighted-blended OIT resolver.
///
/// Wraps a [`WeightedBlendTarget`] and the [`OitParams`] weighting policy.
/// Feed transparent fragments through [`step`](Self::step) in any order, then
/// read the composite over an opaque background with
/// [`resolve`](Self::resolve).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedBlendedAccumulator {
    target: WeightedBlendTarget,
    params: OitParams,
}

impl WeightedBlendedAccumulator {
    /// Creates an empty accumulator driven by `params`.
    #[inline]
    pub fn new(params: OitParams) -> Self {
        Self {
            target: WeightedBlendTarget::new(),
            params,
        }
    }

    /// Accumulates one transparent fragment (`color`/`alpha`) at view depth `z`.
    #[inline]
    pub fn step(&mut self, color: Vec3, alpha: f32, z: f32) {
        self.target
            .accumulate(self.params.weight_scheme, color, alpha, z);
    }

    /// Resolves the accumulated transparency over an opaque `background`.
    #[inline]
    pub fn resolve(&self, background: Vec3) -> Vec3 {
        self.target
            .resolve(self.params.resolve_eps)
            .over_background(background)
    }

    /// Borrows the underlying raw accumulator (render-target view).
    #[inline]
    pub fn target(&self) -> &WeightedBlendTarget {
        &self.target
    }
}

/// High-level moment-based OIT resolver (two-pass).
///
/// Pass one feeds every fragment through [`accumulate`](Self::accumulate) to
/// build the power moments.  Pass two replays the same fragments through
/// [`composite`](Self::composite), which weights each fragment by its
/// reconstructed front transmittance and folds the opaque background in by the
/// total transmittance — an order-independent approximation of back-to-front
/// "over".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentOit {
    moments: PowerMoments4,
    params: OitParams,
}

impl MomentOit {
    /// Creates an empty moment-based resolver driven by `params`.
    #[inline]
    pub fn new(params: OitParams) -> Self {
        Self {
            moments: PowerMoments4::new(),
            params,
        }
    }

    /// Pass one: accumulates one fragment's absorbance at view depth `z`.
    #[inline]
    pub fn accumulate(&mut self, alpha: f32, z: f32) {
        self.moments.add(self.params.warp(z), alpha);
    }

    /// Front transmittance of fragments closer than view depth `z` (in `[0, 1]`).
    #[inline]
    pub fn transmittance(&self, z: f32) -> f32 {
        self.moments
            .reconstruct_transmittance(self.params.warp(z), self.params.moment_bias)
    }

    /// Pass two: composites `fragments` (`color`, `alpha`, view `depth`) over
    /// an opaque `background`.
    ///
    /// Each fragment contributes `colorᵢ·alphaᵢ·T_front(zᵢ)`; the background is
    /// attenuated by the whole-stack transmittance `exp(-b₀)`.  Fragment order
    /// does not affect the result.
    pub fn composite(&self, fragments: &[SortableFragment], background: Vec3) -> Vec3 {
        let mut accum = Vec3::ZERO;
        for frag in fragments {
            let t_front = self.transmittance(frag.depth);
            accum += frag.color * frag.alpha * t_front;
        }
        let total_t = self.moments.total_transmittance();
        accum + background * total_t
    }

    /// Borrows the accumulated power moments.
    #[inline]
    pub fn moments(&self) -> &PowerMoments4 {
        &self.moments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_default_is_sane() {
        let p = OitParams::default();
        assert_eq!(p.weight_scheme, WeightScheme::Equation8);
        assert!(p.depth_far > p.depth_near);
        assert!((0.0..=1.0).contains(&p.moment_bias));
    }

    #[test]
    fn weighted_accumulator_single_layer_is_exact() {
        let mut acc = WeightedBlendedAccumulator::new(OitParams::default());
        acc.step(Vec3::new(0.8, 0.2, 0.1), 1.0, 5.0);
        let out = acc.resolve(Vec3::ZERO);
        assert!(max_channel_error(out, Vec3::new(0.8, 0.2, 0.1)) < 1e-5, "out={out:?}");
    }

    #[test]
    fn weighted_accumulator_is_order_independent() {
        let params = OitParams::default();
        let frags = [
            (Vec3::X, 0.5_f32, 2.0_f32),
            (Vec3::Y, 0.4, 8.0),
            (Vec3::Z, 0.6, 4.0),
        ];
        let mut a = WeightedBlendedAccumulator::new(params);
        for &(c, al, z) in frags.iter() {
            a.step(c, al, z);
        }
        let mut b = WeightedBlendedAccumulator::new(params);
        for &(c, al, z) in frags.iter().rev() {
            b.step(c, al, z);
        }
        let bg = Vec3::splat(0.1);
        assert!(max_channel_error(a.resolve(bg), b.resolve(bg)) < 1e-6);
    }

    #[test]
    fn moment_oit_is_order_independent() {
        let params = OitParams::default();
        let frags = [
            SortableFragment::new(Vec3::X, 0.5, 3.0),
            SortableFragment::new(Vec3::Y, 0.4, 50.0),
            SortableFragment::new(Vec3::Z, 0.6, 400.0),
        ];
        let bg = Vec3::splat(0.1);

        let mut forward = MomentOit::new(params);
        for f in frags.iter() {
            forward.accumulate(f.alpha, f.depth);
        }
        let mut shuffled = MomentOit::new(params);
        for f in frags.iter().rev() {
            shuffled.accumulate(f.alpha, f.depth);
        }
        let out_a = forward.composite(&frags, bg);
        let mut rev = frags;
        rev.reverse();
        let out_b = shuffled.composite(&rev, bg);
        assert!(max_channel_error(out_a, out_b) < 1e-5, "a={out_a:?} b={out_b:?}");
    }

    #[test]
    fn moment_oit_approximates_sorted_ground_truth() {
        // Well-separated layers are the easy case moment OIT should nail.
        let params = OitParams::default();
        let bg = Vec3::new(0.02, 0.02, 0.05);
        let mut frags = [
            SortableFragment::new(Vec3::new(1.0, 0.2, 0.2), 0.6, 5.0),
            SortableFragment::new(Vec3::new(0.2, 1.0, 0.2), 0.5, 60.0),
        ];
        let truth = composite_sorted(&mut frags, bg);

        let mut oit = MomentOit::new(params);
        for f in frags.iter() {
            oit.accumulate(f.alpha, f.depth);
        }
        let est = oit.composite(&frags, bg);
        assert!(max_channel_error(est, truth) < 0.1, "est={est:?} truth={truth:?}");
    }

    #[test]
    fn empty_resolvers_return_background() {
        let bg = Vec3::new(0.3, 0.5, 0.7);
        let wb = WeightedBlendedAccumulator::new(OitParams::default());
        assert!(max_channel_error(wb.resolve(bg), bg) < 1e-6);
        let mb = MomentOit::new(OitParams::default());
        assert!(max_channel_error(mb.composite(&[], bg), bg) < 1e-6);
    }

    #[test]
    fn resolvers_never_produce_nan() {
        let mut wb = WeightedBlendedAccumulator::new(OitParams::default());
        wb.step(Vec3::splat(f32::NAN), f32::NAN, f32::NAN);
        assert!(wb.resolve(Vec3::ZERO).is_finite());

        let mut mb = MomentOit::new(OitParams::default());
        mb.accumulate(f32::NAN, f32::NAN);
        let frag = [SortableFragment::new(Vec3::splat(f32::NAN), f32::NAN, f32::NAN)];
        assert!(mb.composite(&frag, Vec3::ZERO).is_finite());
    }
}
