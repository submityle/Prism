//! Adaptive single-scattering `raymarch` integration (design section 6).
//!
//! The primary view integration walks a ray through the cloud density field,
//! accumulating extinction (`transmittance` decay) and in-scattered light. Two
//! throughput optimisations keep the step budget affordable without visible
//! banding: *empty-space skipping* takes a large stride wherever the density is
//! (near) zero, and the step shrinks to a fine stride only inside the cloud so
//! the medium is never under-sampled; *early termination* stops the walk once
//! the accumulated `transmittance` has dropped below a cutoff, since anything
//! behind an almost-opaque column contributes nothing. The step length is a
//! monotone function of local density — denser medium yields a shorter step —
//! and is always clamped into `[min_step, max_step]`.
//!
//! In-scattering is integrated with the energy-conserving analytic segment
//! form used by production volumetric engines: over a homogeneous segment the
//! integral of `T(s) sigma_s` is `sigma_s * (1 - exp(-sigma_t * step)) /
//! sigma_t`, so the scattered radiance is exact for the segment and never
//! over- or under-counts energy (the medium neither goes black nor blows out).
//! The light visibility feeding each segment (the `AVSM` self-shadow term or a
//! shadow-map lookup) and the `HG` phase are supplied by the caller as plain
//! values / closures, so this module stays a pure, deterministic reference for
//! the `GPU` `WESL` kernel and needs no transcendental intrinsic beyond the
//! shared [`super::math::exp_approx`].

use super::math::{clamp, exp_approx, lerp, saturate, EPS};
use super::scatter::powder;

/// Tunable thresholds controlling the adaptive `raymarch`.
///
/// The step bounds and cutoffs are all in the caller's world units except
/// `transmittance_cutoff`, which is a unit fraction. A sensible default set is
/// provided; callers typically scale `base_step` / `max_step` to the cloud
/// domain size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaymarchConfig {
    /// Step length used at the density threshold (the in-cloud starting step).
    pub base_step: f32,
    /// Largest stride, taken while skipping empty space.
    pub max_step: f32,
    /// Smallest stride, reached in the densest medium.
    pub min_step: f32,
    /// Density at or above which a sample is treated as inside the cloud.
    pub density_threshold: f32,
    /// `transmittance` below which the walk stops early (opaque column).
    pub transmittance_cutoff: f32,
    /// Hard cap on marching steps so the walk always terminates.
    pub max_steps: u32,
    /// Nubis `powder` dark-edge intensity in `[0, 1]`. `0` disables the term
    /// (in-scatter is unmodulated); `1` applies the full
    /// `1 - exp(-2 * view_optical_depth)` darkening near cloud edges. Values
    /// are clamped, so the modulation always lives in `[0, 1]` and can only
    /// remove energy (never amplify or blow out).
    pub powder_strength: f32,
}

impl Default for RaymarchConfig {
    fn default() -> Self {
        Self {
            base_step: 8.0,
            max_step: 64.0,
            min_step: 1.0,
            density_threshold: 1.0e-3,
            transmittance_cutoff: 1.0e-2,
            max_steps: 256,
            powder_strength: 0.0,
        }
    }
}

/// Mutable accumulator threaded through the `raymarch`.
///
/// `transmittance` starts at `1` (nothing occluded) and decays monotonically;
/// `optical_depth` is the accumulated `sigma_t * step`; `scattered` is the
/// integrated in-scattered radiance; `steps_taken` counts marched segments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaymarchState {
    /// Fraction of light surviving from the eye to the current position.
    pub transmittance: f32,
    /// Accumulated optical thickness `integral sigma_t ds`.
    pub optical_depth: f32,
    /// Accumulated in-scattered radiance (energy-conserving).
    pub scattered: f32,
    /// Number of marching steps consumed so far.
    pub steps_taken: u32,
}

impl RaymarchState {
    /// A fresh state at the eye: full `transmittance`, nothing accumulated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transmittance: 1.0,
            optical_depth: 0.0,
            scattered: 0.0,
            steps_taken: 0,
        }
    }
}

impl Default for RaymarchState {
    fn default() -> Self {
        Self::new()
    }
}

/// Chooses the next step length for the current sample.
///
/// Outside the cloud (not `in_cloud`, or density below
/// [`RaymarchConfig::density_threshold`]) the largest stride
/// [`RaymarchConfig::max_step`] is returned to skip empty space. Inside the
/// cloud the step interpolates from [`RaymarchConfig::base_step`] toward
/// [`RaymarchConfig::min_step`] as the saturated density rises, so the step is
/// monotone non-increasing in density. The result is always clamped into
/// `[min_step, max_step]`.
#[must_use]
pub fn adaptive_step(current_density: f32, in_cloud: bool, cfg: RaymarchConfig) -> f32 {
    if !in_cloud || current_density < cfg.density_threshold {
        return clamp(cfg.max_step, cfg.min_step, cfg.max_step);
    }
    let t = saturate(current_density);
    let step = lerp(cfg.base_step, cfg.min_step, t);
    clamp(step, cfg.min_step, cfg.max_step)
}

/// `true` once the accumulated `transmittance` has fallen below the cutoff.
#[must_use]
pub fn should_early_terminate(transmittance: f32, cfg: RaymarchConfig) -> bool {
    transmittance < cfg.transmittance_cutoff
}

/// Integrates one homogeneous segment into `state` (energy conserving).
///
/// `sigma_t` / `sigma_s` are the segment extinction / scattering coefficients,
/// `phase` the (already evaluated) `HG` phase for the light/view geometry,
/// `step` the segment length, and `light_transmittance` the light visibility at
/// the segment (for example the `AVSM` self-shadow term). Negative coefficients
/// are clamped to zero and `light_transmittance` is saturated, so the update
/// never amplifies energy or produces `NaN`. The scattered contribution uses
/// the analytic segment integral `sigma_s * (1 - exp(-sigma_t*step))/sigma_t`,
/// degrading to `sigma_s * step` as `sigma_t` approaches zero. `powder_factor`
/// is the Nubis dark-edge modulation for this segment (`1` = no darkening); it
/// is saturated into `[0, 1]` so it can only remove in-scattered energy, never
/// amplify it. `steps_taken` is incremented by one.
pub fn integrate_segment(
    state: &mut RaymarchState,
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    step: f32,
    light_transmittance: f32,
    powder_factor: f32,
) {
    let sigma_t = if sigma_t > 0.0 { sigma_t } else { 0.0 };
    let sigma_s = if sigma_s > 0.0 { sigma_s } else { 0.0 };
    let phase = if phase > 0.0 { phase } else { 0.0 };
    let step = if step > 0.0 { step } else { 0.0 };
    let light = saturate(light_transmittance);
    let powder_factor = saturate(powder_factor);

    let seg_optical = sigma_t * step;
    let seg_trans = exp_approx(-seg_optical);
    // Analytic in-scatter integral over the segment; the limit as sigma_t -> 0
    // is `step`, which keeps the thin-medium case energy correct.
    let integral = if sigma_t > EPS {
        (1.0 - seg_trans) / sigma_t
    } else {
        step
    };
    state.scattered += state.transmittance * light * sigma_s * phase * integral * powder_factor;
    state.transmittance = saturate(state.transmittance * seg_trans);
    state.optical_depth += seg_optical;
    state.steps_taken += 1;
}

/// Marches a full primary ray, returning the accumulated [`RaymarchState`].
///
/// `density_fn` samples normalized density at a distance along the ray,
/// `sigma_fn` maps that density to `(sigma_t, sigma_s)`, `phase` is the
/// evaluated `HG` phase, and `light_fn` returns the light visibility at a
/// distance (sampled at each segment midpoint). The walk skips empty space with
/// large strides, shrinks inside the cloud, terminates early once
/// `transmittance` drops below [`RaymarchConfig::transmittance_cutoff`], and is
/// hard-capped at [`RaymarchConfig::max_steps`]. It never panics: the step is
/// clamped so it neither overshoots `distance` nor stalls below [`EPS`].
#[must_use]
pub fn march<D, S, L>(
    density_fn: D,
    sigma_fn: S,
    phase: f32,
    light_fn: L,
    distance: f32,
    cfg: RaymarchConfig,
) -> RaymarchState
where
    D: Fn(f32) -> f32,
    S: Fn(f32) -> (f32, f32),
    L: Fn(f32) -> f32,
{
    let mut state = RaymarchState::new();
    let distance = if distance > 0.0 { distance } else { 0.0 };
    let mut t = 0.0;

    while t < distance && state.steps_taken < cfg.max_steps {
        if should_early_terminate(state.transmittance, cfg) {
            break;
        }
        let density = density_fn(t);
        let in_cloud = density >= cfg.density_threshold;
        let mut step = adaptive_step(density, in_cloud, cfg);
        // Never overshoot the ray end.
        let remaining = distance - t;
        if step > remaining {
            step = remaining;
        }
        if step <= EPS {
            break;
        }

        if in_cloud {
            let (sigma_t, sigma_s) = sigma_fn(density);
            let light = light_fn(t + step * 0.5);
            // `state.optical_depth` here is the density accumulated along the
            // view ray up to (not including) this segment, exactly the
            // `density_along_view` the Nubis `powder` curve consumes.
            let powder_factor = if cfg.powder_strength > 0.0 {
                lerp(1.0, powder(state.optical_depth, 1.0), cfg.powder_strength)
            } else {
                1.0
            };
            integrate_segment(
                &mut state,
                sigma_t,
                sigma_s,
                phase,
                step,
                light,
                powder_factor,
            );
        } else {
            // Empty space: no medium to scatter, only advance and count a step.
            state.steps_taken += 1;
        }

        t += step;
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a config with a wide skip stride for the step-count tests.
    fn cfg() -> RaymarchConfig {
        RaymarchConfig::default()
    }

    #[test]
    fn adaptive_step_is_monotone_in_density() {
        let c = cfg();
        let mut prev = adaptive_step(0.0, true, c);
        let mut d = 0.0;
        while d <= 1.0 {
            let step = adaptive_step(d, true, c);
            assert!(
                step <= prev + EPS,
                "step must not grow with density at d={d}: {step} > {prev}"
            );
            assert!((c.min_step..=c.max_step).contains(&step));
            prev = step;
            d += 0.05;
        }
        // Empty space uses the largest stride.
        assert_eq!(adaptive_step(0.0, false, c), c.max_step);
    }

    #[test]
    fn empty_space_uses_far_fewer_steps_than_dense() {
        let c = cfg();
        let empty = march(|_| 0.0, |_| (0.0, 0.0), 0.5, |_| 1.0, 1024.0, c);
        let dense = march(|_| 1.0, |_| (0.1, 0.05), 0.5, |_| 1.0, 1024.0, c);
        assert!(
            empty.steps_taken < dense.steps_taken,
            "empty {} should be far fewer than dense {}",
            empty.steps_taken,
            dense.steps_taken
        );
        // Empty field never occludes.
        assert_eq!(empty.transmittance, 1.0);
        assert_eq!(empty.scattered, 0.0);
    }

    #[test]
    fn early_termination_stops_on_low_transmittance() {
        let c = cfg();
        // Very dense, strongly extinguishing medium collapses transmittance.
        let state = march(|_| 1.0, |_| (2.0, 1.0), 0.5, |_| 1.0, 4096.0, c);
        assert!(state.transmittance < c.transmittance_cutoff);
        // It stopped well before the hard step cap thanks to early-out.
        assert!(state.steps_taken < c.max_steps);
    }

    #[test]
    fn transmittance_is_unit_ranged_and_non_increasing() {
        // Drive integrate_segment directly and watch the invariant.
        let mut state = RaymarchState::new();
        let mut prev = state.transmittance;
        for _ in 0..64 {
            integrate_segment(&mut state, 0.05, 0.02, 0.3, 2.0, 0.8, 1.0);
            assert!((0.0..=1.0).contains(&state.transmittance));
            assert!(state.transmittance <= prev + EPS);
            assert!(state.scattered >= 0.0 && state.scattered.is_finite());
            prev = state.transmittance;
        }
        // Full march also stays in range.
        let m = march(|_| 0.6, |_| (0.2, 0.1), 0.4, |_| 0.9, 512.0, cfg());
        assert!((0.0..=1.0).contains(&m.transmittance));
        assert!(m.scattered >= 0.0 && m.scattered.is_finite());
    }

    #[test]
    fn scatter_energy_is_bounded_not_blown_out() {
        // With light <= 1, phase small, and normalized single scattering, the
        // accumulated scatter stays finite and bounded by the incoming budget.
        let m = march(|_| 0.8, |_| (0.3, 0.3), 1.0, |_| 1.0, 2048.0, cfg());
        assert!(m.scattered.is_finite());
        // Single-scatter albedo <= 1 and light <= 1 bound scattered by 1.
        assert!(m.scattered <= 1.0 + 1.0e-3, "over-exposed: {}", m.scattered);
    }

    #[test]
    fn march_is_deterministic() {
        let c = cfg();
        let a = march(
            |t| saturate(t * 0.01),
            |_| (0.15, 0.07),
            0.35,
            |_| 0.85,
            700.0,
            c,
        );
        let b = march(
            |t| saturate(t * 0.01),
            |_| (0.15, 0.07),
            0.35,
            |_| 0.85,
            700.0,
            c,
        );
        assert_eq!(a, b);
        assert_eq!(a.transmittance.to_bits(), b.transmittance.to_bits());
        assert_eq!(a.scattered.to_bits(), b.scattered.to_bits());
    }

    #[test]
    fn out_of_range_inputs_do_not_panic() {
        let c = cfg();
        // Zero and negative distance produce an untouched state, no loop.
        let z = march(|_| 1.0, |_| (1.0, 1.0), 0.5, |_| 1.0, 0.0, c);
        assert_eq!(z.steps_taken, 0);
        let n = march(|_| 1.0, |_| (1.0, 1.0), 0.5, |_| 1.0, -50.0, c);
        assert_eq!(n.steps_taken, 0);
        // Negative / huge coefficients and out-of-range light are clamped.
        let mut s = RaymarchState::new();
        integrate_segment(&mut s, -1.0, -1.0, -1.0, -1.0, 5.0, 1.0);
        assert_eq!(s.transmittance, 1.0);
        assert_eq!(s.scattered, 0.0);
        integrate_segment(&mut s, 1.0e6, 1.0e6, 10.0, 10.0, 2.0, 1.0);
        assert!((0.0..=1.0).contains(&s.transmittance));
        assert!(s.scattered.is_finite());
        // Inverted step bounds still terminate and clamp.
        let bad = RaymarchConfig {
            min_step: 10.0,
            max_step: 1.0,
            ..c
        };
        let step = adaptive_step(0.5, true, bad);
        assert!(step.is_finite());
    }

    #[test]
    fn powder_darkens_dense_march_and_stays_energy_bounded() {
        // A dense uniform column with fine, non-adaptive steps so both marches
        // walk the identical step sequence; the only difference is the Nubis
        // `powder` modulation. Powder can only remove in-scattered energy, so
        // enabling it must not increase `scattered` and must keep it finite and
        // bounded, while leaving `transmittance` (pure extinction) untouched.
        let base = RaymarchConfig {
            base_step: 0.5,
            max_step: 0.5,
            min_step: 0.5,
            density_threshold: 1.0e-4,
            transmittance_cutoff: 0.0,
            max_steps: 64,
            powder_strength: 0.0,
        };
        let powdered = RaymarchConfig {
            powder_strength: 1.0,
            ..base
        };
        let density_fn = |_t: f32| 0.6_f32;
        let sigma_fn = |d: f32| (0.4 * d, 0.3 * d);
        let phase = 0.3;
        let light_fn = |_t: f32| 0.9_f32;

        let m0 = march(density_fn, sigma_fn, phase, light_fn, 16.0, base);
        let m1 = march(density_fn, sigma_fn, phase, light_fn, 16.0, powdered);

        assert!(m1.scattered.is_finite());
        assert!(m1.scattered >= 0.0);
        // Dark-edge modulation strictly removes energy on a lit dense column.
        assert!(
            m1.scattered < m0.scattered,
            "powder must darken: powdered={} plain={}",
            m1.scattered,
            m0.scattered
        );
        // Extinction (transmittance) and marched geometry are unaffected.
        assert_eq!(m0.transmittance.to_bits(), m1.transmittance.to_bits());
        assert_eq!(m0.optical_depth.to_bits(), m1.optical_depth.to_bits());
        assert_eq!(m0.steps_taken, m1.steps_taken);

        // Deterministic: identical inputs reproduce bit-identical radiance.
        let m1b = march(density_fn, sigma_fn, phase, light_fn, 16.0, powdered);
        assert_eq!(m1.scattered.to_bits(), m1b.scattered.to_bits());
    }
}
