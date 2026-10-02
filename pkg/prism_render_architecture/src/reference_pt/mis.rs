//! Multiple importance sampling (`MIS`) weights for the reference tracer.
//!
//! When the same integral (for example the direct lighting reflected off an
//! area emitter) can be estimated by more than one sampling strategy — drawing
//! a point on the light versus sampling the surface `BSDF` lobe — neither
//! strategy alone has low variance everywhere: light sampling wins for broad,
//! near-diffuse surfaces while `BSDF` sampling wins for sharp, near-specular
//! lobes. `MIS` combines both into a single unbiased estimator by weighting each
//! strategy's samples so that the combined density is a provably good blend
//! (Veach 1995).
//!
//! This module provides the two classical heuristics. Both take the solid-angle
//! densities of the two strategies evaluated at the *same* sampled direction and
//! return the weight applied to the strategy whose density is the first
//! argument. A single sample is drawn per strategy, so the sample counts cancel
//! and only the densities remain. The arithmetic is multiply-and-divide only, so
//! it stays within the transcendental budget of the reference path tracer.

/// The balance heuristic weight `pdf_a / (pdf_a + pdf_b)`.
///
/// This is the minimum-variance linear combination of unbiased estimators and
/// the baseline every production tracer falls back to. Returns zero when both
/// densities vanish so a dead direction contributes nothing.
#[must_use]
pub fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = f64::from(pdf_a);
    let b = f64::from(pdf_b);
    let denom = a + b;
    if denom <= 0.0 {
        0.0
    } else {
        (a / denom) as f32
    }
}

/// The power heuristic weight `pdf_a^2 / (pdf_a^2 + pdf_b^2)` (Veach's beta = 2).
///
/// Squaring sharpens the balance heuristic so the strategy that is locally a far
/// better match dominates, which lowers variance further on glossy highlights
/// where one density towers over the other. The densities are promoted to `f64`
/// before squaring so a large light-sampling density (`dist^2 / area`) cannot
/// overflow. Returns zero when both densities vanish.
#[must_use]
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = f64::from(pdf_a);
    let b = f64::from(pdf_b);
    let a2 = a * a;
    let b2 = b * b;
    let denom = a2 + b2;
    if denom <= 0.0 {
        0.0
    } else {
        (a2 / denom) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both heuristics are symmetric: the two strategy weights sum to one when
    /// at least one density is positive, so the combined estimator is unbiased.
    #[test]
    fn weights_sum_to_one() {
        for (a, b) in [(1.0f32, 3.0f32), (0.25, 0.75), (10.0, 0.1), (2.0, 2.0)] {
            let wb = balance_heuristic(a, b) + balance_heuristic(b, a);
            let wp = power_heuristic(a, b) + power_heuristic(b, a);
            assert!((wb - 1.0).abs() < 1e-6, "balance sum {wb}");
            assert!((wp - 1.0).abs() < 1e-6, "power sum {wp}");
        }
    }

    /// Equal densities split the weight evenly under both heuristics.
    #[test]
    fn equal_densities_split_evenly() {
        assert!((balance_heuristic(2.0, 2.0) - 0.5).abs() < 1e-6);
        assert!((power_heuristic(2.0, 2.0) - 0.5).abs() < 1e-6);
    }

    /// The power heuristic favours the dominant strategy more aggressively than
    /// the balance heuristic.
    #[test]
    fn power_sharpens_the_dominant_strategy() {
        let a = 9.0f32;
        let b = 1.0f32;
        assert!(power_heuristic(a, b) > balance_heuristic(a, b));
        // Balance gives 0.9, power gives 81/82 ~ 0.9878.
        assert!((balance_heuristic(a, b) - 0.9).abs() < 1e-6);
        assert!((power_heuristic(a, b) - (81.0 / 82.0)).abs() < 1e-5);
    }

    /// Two dead strategies contribute no weight rather than dividing by zero.
    #[test]
    fn zero_densities_yield_zero_weight() {
        assert!(balance_heuristic(0.0, 0.0).abs() < 1e-9);
        assert!(power_heuristic(0.0, 0.0).abs() < 1e-9);
    }
}
