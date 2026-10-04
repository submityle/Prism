//! Moment-based order-independent transparency (`MBOIT`).
//!
//! This is the constant-storage transparency path referenced by
//! [`crate::transparency::TransparencyPath::MomentOit`]. Unlike the exact
//! `A-buffer` [`crate::transparency::sorted_oit`] (unbounded per-pixel lists)
//! and the single-weight [`crate::transparency::weighted_oit`] (one blended
//! bucket), `MBOIT` stores a small fixed vector of statistical *power moments*
//! of the per-pixel absorbance measure and reconstructs each fragment's
//! transmittance from those moments. Storage and bandwidth are independent of
//! overdraw, which is what makes it practical for dense foliage, particles, and
//! layered glass at `AAA` scene complexity.
//!
//! The pipeline is two passes, matching the hardware design:
//! 1. **Moment generation** — splat every transparent fragment's
//!    `(view_depth, alpha)` into [`reconstruct::PowerMoments`] to accumulate
//!    `b0..=b4`.
//! 2. **Resolve** — for each fragment, reconstruct the transmittance of
//!    everything in front of it from the moments and composite
//!    `premultiplied_color * transmittance` over the background.
//!
//! Everything is deterministic add/multiply/compare plus the reproducible
//! `ln`/`exp`/`Cholesky` in [`math`]; no `f32` transcendental intrinsic is used.

mod math;
pub mod reconstruct;

use crate::transparency::weighted_oit::OitFragment;
pub use reconstruct::PowerMoments;

/// Which power-moment order the resolve reconstructs with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MomentOrder {
    /// Two power moments (Chebyshev–Cantelli). Cheapest, coarsest.
    Two,
    /// Four power moments (Peters canonical). Tighter, tracks the exact
    /// depth-sorted result closely for multi-layer pixels.
    Four,
}

/// Generates the per-pixel power moments from a fragment list.
///
/// `near`/`far` bound the view-depth range the moments are warped against; they
/// should enclose every fragment so the warp stays inside `[-1, 1]`.
#[must_use]
pub fn generate_moments(fragments: &[OitFragment], near: f32, far: f32) -> PowerMoments {
    let mut moments = PowerMoments::new(near, far);
    for frag in fragments {
        moments.add_fragment(frag.view_depth, frag.alpha);
    }
    moments
}

/// Reconstructs the transmittance in front of `view_depth` at the chosen order.
#[must_use]
pub fn transmittance_at(moments: &PowerMoments, view_depth: f32, order: MomentOrder) -> f32 {
    match order {
        MomentOrder::Two => moments.transmittance2(view_depth),
        MomentOrder::Four => moments.transmittance4(view_depth),
    }
}

/// Resolves a pixel's transparent fragments over `background` using `MBOIT`.
///
/// Each fragment contributes `premultiplied_color * transmittance_in_front`,
/// where the transmittance is reconstructed from the shared moments. This is
/// order-independent: the result does not depend on the order fragments appear
/// in `fragments`. Returns the composited `RGB` radiance.
#[must_use]
pub fn resolve(
    fragments: &[OitFragment],
    background: [f32; 3],
    near: f32,
    far: f32,
    order: MomentOrder,
) -> [f32; 3] {
    let moments = generate_moments(fragments, near, far);
    resolve_with_moments(fragments, &moments, background, order)
}

/// Resolve variant reusing already-generated moments (the second hardware pass).
#[must_use]
pub fn resolve_with_moments(
    fragments: &[OitFragment],
    moments: &PowerMoments,
    background: [f32; 3],
    order: MomentOrder,
) -> [f32; 3] {
    // Accumulate each fragment's contribution weighted by the transmittance of
    // everything strictly in front of it, then attenuate the background by the
    // total transmittance across all fragments.
    let mut acc = [0.0_f32; 3];
    for frag in fragments {
        let t = transmittance_at(moments, frag.view_depth, order);
        // `frag.color` is straight (non-premultiplied) radiance.
        for (a, &col) in acc.iter_mut().zip(frag.color.iter()) {
            *a += col * frag.alpha * t;
        }
    }
    let bg_t = exp_total(moments.total_absorbance());
    [
        acc[0] + background[0] * bg_t,
        acc[1] + background[1] * bg_t,
        acc[2] + background[2] * bg_t,
    ]
}

/// Total transmittance through all transparent layers, `exp(-b0)`.
#[must_use]
fn exp_total(total_absorbance: f32) -> f32 {
    math::exp(-total_absorbance).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transparency::sorted_oit::SortedResolve;
    use alloc::vec;
    use alloc::vec::Vec;

    fn frag(color: [f32; 3], alpha: f32, view_depth: f32) -> OitFragment {
        OitFragment {
            color,
            alpha,
            view_depth,
        }
    }

    #[test]
    fn empty_returns_background() {
        let bg = [0.2, 0.3, 0.4];
        let out = resolve(&[], bg, 0.0, 1.0, MomentOrder::Four);
        assert_eq!(out, bg);
    }

    #[test]
    fn order_independent() {
        let a = frag([1.0, 0.0, 0.0], 0.5, 0.2);
        let b = frag([0.0, 1.0, 0.0], 0.4, 0.5);
        let c = frag([0.0, 0.0, 1.0], 0.6, 0.8);
        let bg = [0.1, 0.1, 0.1];
        let forward = resolve(&[a, b, c], bg, 0.0, 1.0, MomentOrder::Four);
        let shuffled = resolve(&[c, a, b], bg, 0.0, 1.0, MomentOrder::Four);
        for i in 0..3 {
            assert!(
                (forward[i] - shuffled[i]).abs() < 1e-5,
                "channel {i}: {} vs {}",
                forward[i],
                shuffled[i]
            );
        }
    }

    #[test]
    fn four_moment_tracks_exact_sorted_resolve() {
        let frags = vec![
            frag([0.9, 0.1, 0.1], 0.4, 0.15),
            frag([0.1, 0.9, 0.1], 0.5, 0.35),
            frag([0.1, 0.1, 0.9], 0.3, 0.6),
            frag([0.8, 0.8, 0.1], 0.45, 0.85),
        ];
        let bg = [0.05, 0.05, 0.05];

        let mut sorted = SortedResolve::new();
        for f in &frags {
            sorted.push(*f);
        }
        let exact = sorted.resolve(bg);

        let mboit = resolve(&frags, bg, 0.0, 1.0, MomentOrder::Four);
        let two = resolve(&frags, bg, 0.0, 1.0, MomentOrder::Two);

        let err4: f32 = (0..3).map(|i| (mboit[i] - exact[i]).abs()).sum::<f32>() / 3.0;
        let err2: f32 = (0..3).map(|i| (two[i] - exact[i]).abs()).sum::<f32>() / 3.0;
        assert!(err4 <= err2 + 1e-4, "err4 {err4} should beat err2 {err2}");
        assert!(err4 < 0.1, "four-moment composite error too large: {err4}");
    }

    #[test]
    fn generate_then_resolve_matches_single_shot() {
        let frags = vec![
            frag([0.5, 0.2, 0.7], 0.5, 0.3),
            frag([0.2, 0.6, 0.1], 0.4, 0.7),
        ];
        let bg = [0.0, 0.0, 0.0];
        let one_shot = resolve(&frags, bg, 0.0, 1.0, MomentOrder::Four);
        let moments = generate_moments(&frags, 0.0, 1.0);
        let two_pass = resolve_with_moments(&frags, &moments, bg, MomentOrder::Four);
        for i in 0..3 {
            assert!((one_shot[i] - two_pass[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn total_transmittance_attenuates_background() {
        // Two opaque-ish layers: background should be heavily attenuated.
        let frags = vec![frag([0.0; 3], 0.9, 0.3), frag([0.0; 3], 0.9, 0.6)];
        let bg = [1.0, 1.0, 1.0];
        let out = resolve(&frags, bg, 0.0, 1.0, MomentOrder::Four);
        // Expected background transmittance ~ exp(-(a1+a2)) with a=-ln(0.1).
        for c in &out {
            assert!(*c < 0.05, "background not attenuated: {c}");
        }
    }
}
