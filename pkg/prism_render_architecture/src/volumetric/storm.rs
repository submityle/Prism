#![forbid(unsafe_code)]
//! Storm system and `Cumulonimbus` vertical development (design section 9b).
//!
//! Deep convective clouds are not static shapes: an updraft column punches
//! upward, spreads a flat `anvil` at the tropopause, occasionally pushes an
//! `overshooting` top through it, ripples the stable layer above with a
//! `gravity wave`, trails a `virga` precipitation veil beneath the base, and —
//! over a heat source — bootstraps a `pyrocumulus` buoyant plume. This module
//! owns the deterministic state machine and the authored profile curves that
//! feed those signals into the density field the sibling modelling / ray-march
//! modules consume; the `GPU` `WESL` kernels mirror the same curves with native
//! intrinsics.
//!
//! Everything here is pure and allocation-free. The only stateful object is
//! [`StormState`], whose components are all kept in `0..=1` by construction:
//! [`StormState::advance_storm`] clamps every update and grows the `anvil`
//! spread monotonically so a maturing storm never "un-spreads". Every profile
//! function saturates its output and its inputs, so out-of-range altitudes or
//! energies never panic and never escape `0..=1`. Float transcendentals route
//! through the shared hand-rolled [`super::math`] (`sin_approx` / `exp_approx`);
//! the only permitted intrinsic is `f32::sqrt`, which this module does not use.

use super::math::{exp_approx, fract, lerp, saturate, sin_approx, smoothstep, TWO_PI};

/// Per-second relaxation rate of the updraft toward its driving energy.
const UPDRAFT_RESPONSE: f32 = 0.6;

/// Per-second `anvil` spreading gain, scaled by the current maturity.
const ANVIL_SPREAD_RATE: f32 = 0.25;

/// Updraft fraction above which an `overshooting` top begins to form.
const OVERSHOOT_THRESHOLD: f32 = 0.6;

/// Gaussian sharpness of the `overshooting` bump around the band top.
const OVERSHOOT_SHARPNESS: f32 = 12.0;

/// `gravity wave` phase advance (cycles per second) of the stable layer ripple.
const GRAVITY_WAVE_FREQUENCY: f32 = 0.2;

/// Per-second relaxation rate of the `virga` veil toward the `anvil` maturity.
const VIRGA_RESPONSE: f32 = 0.4;

/// Per-second relaxation rate of the `pyrocumulus` plume toward its buoyancy.
const PYRO_RESPONSE: f32 = 0.5;

/// Buoyancy gain converting a normalized heat proxy into `pyrocumulus` lift.
const PYRO_BUOYANCY_GAIN: f32 = 2.5;

/// The evolving state of one deep convective (`Cumulonimbus`) cell.
///
/// Every field is a normalized `0..=1` signal. The struct derives [`Default`]
/// (a calm, zeroed cell) instead of a hand-written constructor so it composes
/// like the sibling contract structs in the parent module.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StormState {
    /// Updraft column strength `0..=1`; the driver of vertical development.
    pub updraft: f32,
    /// `anvil` spread `0..=1`; monotonically non-decreasing as the cell matures.
    pub anvil_spread: f32,
    /// `overshooting` top prominence `0..=1` punching above the `anvil`.
    pub overshooting_top: f32,
    /// `gravity wave` phase in `0..=1` (one full ripple cycle per unit).
    pub gravity_wave_phase: f32,
    /// `virga` precipitation-veil strength `0..=1` trailing beneath the base.
    pub virga: f32,
    /// `pyrocumulus` buoyant-plume strength `0..=1` above a heat source.
    pub pyrocumulus: f32,
}

impl StormState {
    /// Advances the storm by `dt` seconds under a normalized `energy_input`.
    ///
    /// `dt` is floored at zero and `energy_input` is saturated, so adversarial
    /// inputs cannot destabilize the machine. The updraft relaxes toward the
    /// driving energy; the `anvil` spread accrues at a rate proportional to the
    /// current maturity and therefore never decreases; the `overshooting` top,
    /// `virga` veil, `gravity wave` phase, and `pyrocumulus` plume follow. Every
    /// field stays in `0..=1`.
    ///
    /// Mutates `self`, so it is intentionally not `#[must_use]`.
    pub fn advance_storm(&mut self, dt: f32, energy_input: f32) {
        let dt = dt.max(0.0);
        let energy = saturate(energy_input);

        // Updraft relaxes toward the driving convective energy.
        let updraft_rate = saturate(UPDRAFT_RESPONSE * dt);
        self.updraft = saturate(lerp(self.updraft, energy, updraft_rate));

        // Maturity drives an irreversible anvil spread (monotonic non-decreasing).
        let maturity = self.updraft;
        self.anvil_spread = saturate(self.anvil_spread + maturity * ANVIL_SPREAD_RATE * dt);

        // Overshooting top forms only once the updraft exceeds the threshold.
        let headroom = (1.0 - OVERSHOOT_THRESHOLD).max(super::EPS);
        self.overshooting_top = saturate((self.updraft - OVERSHOOT_THRESHOLD) / headroom);

        // Gravity-wave phase advances and wraps into the unit cycle.
        self.gravity_wave_phase = fract(self.gravity_wave_phase + GRAVITY_WAVE_FREQUENCY * dt);

        // Virga veil relaxes toward the current anvil maturity.
        let virga_rate = saturate(VIRGA_RESPONSE * dt);
        self.virga = saturate(lerp(self.virga, self.anvil_spread, virga_rate));

        // Pyrocumulus plume relaxes toward the buoyancy of the heat proxy.
        let pyro_rate = saturate(PYRO_RESPONSE * dt);
        let target = pyrocumulus_buoyancy(energy);
        self.pyrocumulus = saturate(lerp(self.pyrocumulus, target, pyro_rate));
    }

    /// Composite vertical density weight of the storm cloud *body* at a
    /// normalized band `height_fraction` (`0` at the cloud base, `1` at the
    /// tropopause).
    ///
    /// This is the seam that folds the state machine's evolving fields back
    /// through this module's own authored curves into a single `0..=1` weight
    /// the modelling / ray-march density field multiplies the deep-convective
    /// `Cumulonimbus` column by. It unions the spreading [`anvil_profile`]
    /// (driven by [`StormState::anvil_spread`]) with the [`overshooting_bump`]
    /// dome (driven by [`StormState::overshooting_top`], reinforced by the
    /// [`StormState::pyrocumulus`] plume) via a probabilistic OR
    /// `a + b - a*b`, so two overlapping contributions never sum past one.
    ///
    /// The result is bounded to `0..=1` and monotonically non-decreasing in
    /// every driving field (each factor is monotone and the union has
    /// non-negative partials for inputs in `0..=1`), so a maturing storm only
    /// ever adds vertical development, never removes it.
    #[must_use]
    pub fn vertical_profile(self, height_fraction: f32) -> f32 {
        let anvil = anvil_profile(height_fraction, self.anvil_spread);
        // The pyrocumulus plume reinforces the overshooting dome's prominence
        // without letting the combined drive escape 0..=1.
        let dome_drive =
            saturate(self.overshooting_top + self.pyrocumulus * (1.0 - self.overshooting_top));
        let dome = overshooting_bump(height_fraction, dome_drive);
        saturate(anvil + dome - anvil * dome)
    }

    /// `virga` precipitation-veil density beneath the storm base at a
    /// normalized `veil_fraction` (`1` at the cloud base, `0` at the trailing
    /// tip).
    ///
    /// Scales the authored [`virga_fade`] curve by the current
    /// [`StormState::virga`] veil strength, so a young storm (no veil) yields
    /// zero and a mature storm trails a fading precipitation curtain. Bounded
    /// to `0..=1` and monotonically non-decreasing in both the veil strength
    /// and the height fraction.
    #[must_use]
    pub fn virga_veil(self, veil_fraction: f32) -> f32 {
        saturate(self.virga * virga_fade(veil_fraction))
    }
}

/// `anvil` density weight at a normalized `height_fraction` for a given spread.
///
/// The flat `anvil` concentrates in the upper band; a more mature `spread`
/// flares it lower and stronger. The result is saturated into `0..=1`: it is
/// near zero low in the column and rises toward `spread` at the band top.
#[must_use]
pub fn anvil_profile(height_fraction: f32, spread: f32) -> f32 {
    let h = saturate(height_fraction);
    let s = saturate(spread);
    // A wider anvil starts flaring from a lower base altitude.
    let base = lerp(0.85, 0.5, s);
    saturate(s * smoothstep(base, 1.0, h))
}

/// `overshooting` top bump at a normalized `height_fraction`.
///
/// Models the dome that a strong updraft punches through the `anvil`: a narrow
/// Gaussian centred on the band top (`height_fraction == 1`) scaled by
/// `overshooting_top`. Inputs above `1` (above the band) are allowed so the
/// dome can bulge past the tropopause; the output stays in `0..=1`.
#[must_use]
pub fn overshooting_bump(height_fraction: f32, overshooting_top: f32) -> f32 {
    let top = saturate(overshooting_top);
    let d = height_fraction - 1.0;
    saturate(top * exp_approx(-OVERSHOOT_SHARPNESS * d * d))
}

/// `gravity wave` ripple value for a wave `phase` at horizontal coordinate `x`.
///
/// A bounded sinusoid (via [`sin_approx`]) modelling the stable-layer undulation
/// above a mature storm. The `phase` is treated as a unit cycle and `x` as a
/// horizontal phase offset in radians. The result stays within roughly
/// `[-1, 1]`.
#[must_use]
pub fn gravity_wave(phase: f32, x: f32) -> f32 {
    sin_approx(TWO_PI * phase + x)
}

/// `virga` precipitation-veil weight at a normalized `height_fraction`.
///
/// The veil is densest at the cloud base (`height_fraction == 1`) and
/// evaporates toward its trailing tip (`height_fraction == 0`); the smooth,
/// monotonic profile stays in `0..=1`.
#[must_use]
pub fn virga_fade(height_fraction: f32) -> f32 {
    smoothstep(0.0, 1.0, saturate(height_fraction))
}

/// `pyrocumulus` buoyancy from a normalized `heat` proxy.
///
/// A saturating exponential ramp: more `heat` yields more buoyant lift with
/// diminishing returns, monotonically increasing and bounded to `0..=1`.
/// Negative `heat` is floored at zero.
#[must_use]
pub fn pyrocumulus_buoyancy(heat: f32) -> f32 {
    let q = heat.max(0.0);
    saturate(1.0 - exp_approx(-PYRO_BUOYANCY_GAIN * q))
}

#[cfg(test)]
mod tests {
    use super::super::math::EPS;
    use super::*;

    #[test]
    fn storm_state_components_stay_in_unit_range() {
        let mut storm = StormState::default();
        // Drive with a mix of in-range, negative, and over-unit inputs.
        let drivers = [0.8_f32, 1.5, -0.3, 0.4, 2.0, 0.0, 0.9];
        for &energy in &drivers {
            storm.advance_storm(0.1, energy);
            for v in [
                storm.updraft,
                storm.anvil_spread,
                storm.overshooting_top,
                storm.gravity_wave_phase,
                storm.virga,
                storm.pyrocumulus,
            ] {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "storm component escaped range: {v}"
                );
            }
        }
    }

    #[test]
    fn anvil_spread_is_monotonically_non_decreasing() {
        let mut storm = StormState::default();
        let mut prev = storm.anvil_spread;
        let mut step = 0;
        while step < 200 {
            // Even with fluctuating energy the anvil must never shrink.
            let energy = if step % 3 == 0 { 0.9 } else { 0.2 };
            storm.advance_storm(0.05, energy);
            assert!(
                storm.anvil_spread + EPS >= prev,
                "anvil spread decreased at step {step}"
            );
            prev = storm.anvil_spread;
            step += 1;
        }
    }

    #[test]
    fn anvil_profile_is_bounded_and_stronger_at_the_top() {
        for &spread in &[0.0, 0.3, 0.6, 1.0] {
            let mut h = 0.0;
            while h <= 1.0 {
                let w = anvil_profile(h, spread);
                assert!((0.0..=1.0).contains(&w), "anvil profile out of range");
                h += 0.05;
            }
            // The anvil is stronger high in the band than at its floor.
            assert!(anvil_profile(1.0, spread) + EPS >= anvil_profile(0.2, spread));
        }
        // Out-of-range inputs saturate rather than panic.
        assert!((0.0..=1.0).contains(&anvil_profile(-5.0, 3.0)));
        assert!((0.0..=1.0).contains(&anvil_profile(9.0, -3.0)));
    }

    #[test]
    fn overshooting_bump_peaks_at_band_top_and_stays_bounded() {
        let at_top = overshooting_bump(1.0, 1.0);
        let below = overshooting_bump(0.5, 1.0);
        assert!(at_top + EPS >= below, "overshoot should peak at the top");
        let mut h = -0.5;
        while h <= 1.5 {
            let b = overshooting_bump(h, 0.7);
            assert!((0.0..=1.0).contains(&b), "overshoot out of range at {h}");
            h += 0.05;
        }
    }

    #[test]
    fn gravity_wave_is_bounded() {
        let mut phase = 0.0;
        while phase <= 1.0 {
            let mut x = -10.0;
            while x <= 10.0 {
                let v = gravity_wave(phase, x);
                assert!(v.abs() <= 1.0 + 1e-4, "gravity wave unbounded: {v}");
                x += 0.25;
            }
            phase += 0.05;
        }
    }

    #[test]
    fn virga_fade_is_monotonic_and_bounded() {
        let mut prev = virga_fade(0.0);
        let mut h = 0.0;
        while h <= 1.0 {
            let v = virga_fade(h);
            assert!((0.0..=1.0).contains(&v), "virga out of range");
            assert!(v + EPS >= prev, "virga fade not monotonic at {h}");
            prev = v;
            h += 0.05;
        }
        assert!((0.0..=1.0).contains(&virga_fade(-3.0)));
        assert!((0.0..=1.0).contains(&virga_fade(4.0)));
    }

    #[test]
    fn pyrocumulus_buoyancy_is_monotonic_and_bounded() {
        let mut prev = pyrocumulus_buoyancy(0.0);
        let mut q = 0.0;
        while q <= 4.0 {
            let b = pyrocumulus_buoyancy(q);
            assert!((0.0..=1.0).contains(&b), "buoyancy out of range");
            assert!(b + EPS >= prev, "buoyancy not monotonic at {q}");
            prev = b;
            q += 0.1;
        }
        assert!(
            pyrocumulus_buoyancy(-2.0).abs() < EPS,
            "negative heat clamps to zero"
        );
    }

    #[test]
    fn advance_storm_is_deterministic() {
        let mut a = StormState::default();
        let mut b = StormState::default();
        for step in 0..50 {
            let energy = 0.3 + 0.01 * step as f32;
            a.advance_storm(0.1, energy);
            b.advance_storm(0.1, energy);
        }
        assert_eq!(a, b, "storm advance is not deterministic");
    }

    #[test]
    fn vertical_profile_stays_bounded_and_grows_with_maturity() {
        // A young cell (barely developed) versus a mature cell driven hard.
        let mut young = StormState::default();
        young.advance_storm(0.1, 0.7);
        let mut mature = StormState::default();
        for _ in 0..80 {
            mature.advance_storm(0.1, 0.95);
        }

        let mut h = 0.0_f32;
        while h <= 1.0 {
            let wy = young.vertical_profile(h);
            let wm = mature.vertical_profile(h);
            assert!(
                (0.0..=1.0).contains(&wy),
                "young profile out of range at {h}"
            );
            assert!(
                (0.0..=1.0).contains(&wm),
                "mature profile out of range at {h}"
            );
            // A more developed storm never has *less* vertical development at
            // any height than a younger one (monotone in the driving fields).
            assert!(
                wm + EPS >= wy,
                "mature profile weaker than young at {h}: {wm} < {wy}"
            );
            h += 0.05;
        }
        // A calm (zeroed) cell contributes no vertical development anywhere.
        let calm = StormState::default();
        assert!(calm.vertical_profile(0.5).abs() < EPS);
        // Out-of-band heights saturate instead of panicking.
        assert!((0.0..=1.0).contains(&mature.vertical_profile(-2.0)));
        assert!((0.0..=1.0).contains(&mature.vertical_profile(3.0)));
    }

    #[test]
    fn virga_veil_scales_with_veil_strength_and_fades_to_the_tip() {
        let mut storm = StormState::default();
        for _ in 0..80 {
            storm.advance_storm(0.1, 0.9);
        }
        assert!(
            storm.virga > 0.0,
            "a mature storm should trail a virga veil"
        );

        // Densest at the base (fraction 1), fading to nothing at the tip (0).
        let base = storm.virga_veil(1.0);
        let tip = storm.virga_veil(0.0);
        assert!(base + EPS >= tip, "veil should be densest at the base");
        assert!(tip.abs() < EPS, "veil should vanish at the trailing tip");

        // Monotone in the height fraction and always bounded.
        let mut prev = 0.0_f32;
        let mut f = 0.0_f32;
        while f <= 1.0 {
            let v = storm.virga_veil(f);
            assert!((0.0..=1.0).contains(&v), "virga veil out of range at {f}");
            assert!(v + EPS >= prev, "virga veil not monotonic at {f}");
            prev = v;
            f += 0.05;
        }

        // No veil strength means no precipitation curtain regardless of height.
        let calm = StormState::default();
        assert!(calm.virga_veil(1.0).abs() < EPS);
    }
}
