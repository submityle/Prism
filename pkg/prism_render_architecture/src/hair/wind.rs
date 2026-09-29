//! Wind-field coupling for guide-strand dynamics.
//!
//! Ambient wind is what sells hair as *alive* rather than merely falling: a
//! steady breeze plus turbulent gusts push the free particles so the groom
//! drifts and flutters coherently with the rest of the scene (design §6.3,
//! "与布料/粒子共享风数据"). This module exposes the wind as a deterministic
//! external acceleration field evaluated per particle, so the same groom in the
//! same wind at the same time always deforms identically and can be
//! golden-tested (design §9).
//!
//! Wind is applied as an external pre-pass ([`apply_wind`]) before the XPBD
//! constraint solve: it nudges each free particle's position by the wind
//! displacement over the frame, which the Verlet integrator then reads back as
//! injected velocity. Pinned particles (the skinned root) are never moved, and
//! a zero field is a no-op, so callers can always run the pass unconditionally.
//! The turbulence is a hand-written deterministic sine of position and time —
//! never `f32::sin`, whose result is not bit-reproducible across platforms.

use core::f32::consts::{FRAC_PI_2, PI, TAU};

use super::dynamics::{StrandParticle, Vec3};

/// A steady wind plus turbulent gusts, sampled as an acceleration field.
///
/// `speed` scales the steady push along `direction`; `gust_amplitude` and
/// `gust_frequency` add a coherent along-wind pulsing; `turbulence` adds a
/// small spatially varying cross-wind flutter. All magnitudes are accelerations
/// (world units/s^2). A field with zero `speed`, `gust_amplitude`, and
/// `turbulence` produces no force.
#[derive(Clone, Copy, Debug)]
pub struct WindField {
    /// Wind heading; normalized internally, so any non-zero length is fine.
    pub direction: Vec3,
    /// Steady acceleration magnitude along `direction`.
    pub speed: f32,
    /// Peak extra along-wind acceleration of the gust pulse.
    pub gust_amplitude: f32,
    /// Gust cycles per (world unit + second); drives the spatial/temporal
    /// phase. Clamped non-negative.
    pub gust_frequency: f32,
    /// Cross-wind flutter magnitude (world units/s^2); the small chaotic
    /// component that keeps a groom from moving as one rigid clump.
    pub turbulence: f32,
}

impl WindField {
    /// A dead-calm field that exerts no force.
    pub const CALM: Self = Self {
        direction: Vec3::ZERO,
        speed: 0.0,
        gust_amplitude: 0.0,
        gust_frequency: 0.0,
        turbulence: 0.0,
    };
}

/// Fixed swirl axis mixing the three position components into one gust phase,
/// so the pulse varies across the groom instead of moving it rigidly.
const GUST_SWIRL: Vec3 = Vec3::new(0.7, 1.3, 0.5);

/// Evaluates the wind acceleration at `position` and `time` (seconds).
///
/// The steady term is `direction * speed`; the gust term pulses that along-wind
/// push by `gust_amplitude * sin(phase)` where the phase mixes position and
/// time; the turbulence term adds a small phase-shifted per-axis flutter. The
/// result is fully deterministic and finite for finite inputs.
#[must_use]
pub fn wind_acceleration(field: WindField, position: Vec3, time: f32) -> Vec3 {
    let dir = field.direction.normalize_or_zero();
    let freq = field.gust_frequency.max(0.0);

    let steady = dir.scale(field.speed);

    // One scalar gust phase (in turns) shared by the whole along-wind pulse.
    let gust_phase = freq * (position.dot(GUST_SWIRL) + time);
    let gust_scalar = sin_turns(gust_phase * TAU);
    let gust = dir.scale(field.gust_amplitude * gust_scalar);

    // Per-axis flutter: phase-shifted sines of position and time give a small
    // divergent cross-wind component without needing a real noise field.
    let base = freq * time;
    let flutter = Vec3::new(
        sin_turns((position.x + base) * TAU),
        sin_turns((position.y * 1.3 + base + 0.37) * TAU),
        sin_turns((position.z * 0.7 + base + 0.71) * TAU),
    )
    .scale(field.turbulence);

    steady.add(gust).add(flutter)
}

/// Applies `field` to every free particle as an external pre-pass over `dt`.
///
/// Each free particle gains the wind displacement `acceleration * dt^2`, matching
/// the semi-implicit scaling the dynamics integrator uses for gravity, so the
/// next solve reads it back as injected velocity. Pinned particles are skipped.
/// A non-positive or non-finite `dt` is a no-op, as is [`WindField::CALM`]-like
/// input (it simply adds zero).
pub fn apply_wind(particles: &mut [StrandParticle], field: WindField, time: f32, dt: f32) {
    if dt <= 0.0 || !dt.is_finite() {
        return;
    }
    let dt_sq = dt * dt;
    for particle in particles.iter_mut() {
        if particle.is_pinned() {
            continue;
        }
        let accel = wind_acceleration(field, particle.position, time);
        particle.position = particle.position.add(accel.scale(dt_sq));
    }
}

/// Deterministic sine of an angle in radians, hand-written to avoid `f32::sin`
/// (banned for its non-reproducible cross-platform result). Range-reduces to
/// `[-PI/2, PI/2]` and evaluates a 9th-order Taylor polynomial in Horner form;
/// worst-case error over a period stays well under `1e-4`.
fn sin_turns(x: f32) -> f32 {
    let mut a = x - (x / TAU).round() * TAU;
    if a > FRAC_PI_2 {
        a = PI - a;
    } else if a < -FRAC_PI_2 {
        a = -PI - a;
    }
    let x2 = a * a;
    let poly = x2
        .mul_add(1.0 / 362_880.0, -1.0 / 5_040.0)
        .mul_add(x2, 1.0 / 120.0)
        .mul_add(x2, -1.0 / 6.0)
        .mul_add(x2, 1.0);
    a * poly
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady_x(speed: f32) -> WindField {
        WindField {
            direction: Vec3::new(1.0, 0.0, 0.0),
            speed,
            gust_amplitude: 0.0,
            gust_frequency: 0.0,
            turbulence: 0.0,
        }
    }

    #[test]
    fn calm_field_exerts_no_force() {
        let a = wind_acceleration(WindField::CALM, Vec3::new(1.0, 2.0, 3.0), 5.0);
        assert_eq!(a, Vec3::ZERO);
    }

    #[test]
    fn steady_wind_pushes_along_direction() {
        let a = wind_acceleration(steady_x(4.0), Vec3::ZERO, 0.0);
        assert!((a.x - 4.0).abs() < 1.0e-6);
        assert!(a.y.abs() < 1.0e-6);
        assert!(a.z.abs() < 1.0e-6);
    }

    #[test]
    fn apply_wind_moves_free_and_skips_pinned() {
        let mut particles = [
            StrandParticle::pinned(Vec3::ZERO),
            StrandParticle::free(Vec3::ZERO),
        ];
        apply_wind(&mut particles, steady_x(10.0), 0.0, 0.5);
        // Pinned root stays put.
        assert_eq!(particles[0].position, Vec3::ZERO);
        // Free particle displaced by accel * dt^2 = 10 * 0.25 = 2.5 along +X.
        assert!((particles[1].position.x - 2.5).abs() < 1.0e-5);
    }

    #[test]
    fn nonpositive_dt_is_noop() {
        let mut particles = [StrandParticle::free(Vec3::new(1.0, 1.0, 1.0))];
        apply_wind(&mut particles, steady_x(10.0), 0.0, 0.0);
        assert_eq!(particles[0].position, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn empty_particles_is_noop() {
        let mut particles: [StrandParticle; 0] = [];
        apply_wind(&mut particles, steady_x(1.0), 0.0, 0.016);
        assert!(particles.is_empty());
    }

    #[test]
    fn gust_oscillates_around_steady_and_stays_bounded() {
        let field = WindField {
            direction: Vec3::new(1.0, 0.0, 0.0),
            speed: 2.0,
            gust_amplitude: 1.0,
            gust_frequency: 0.5,
            turbulence: 0.0,
        };
        // Over a sweep of times the along-wind accel stays within speed +/- amp.
        for k in 0..64 {
            let t = k as f32 * 0.1;
            let a = wind_acceleration(field, Vec3::new(0.3, 0.4, 0.5), t);
            assert!(a.x >= 2.0 - 1.0 - 1.0e-3 && a.x <= 2.0 + 1.0 + 1.0e-3);
        }
    }

    #[test]
    fn wind_is_deterministic() {
        let field = WindField {
            direction: Vec3::new(0.2, 1.0, -0.3),
            speed: 3.0,
            gust_amplitude: 0.8,
            gust_frequency: 1.2,
            turbulence: 0.4,
        };
        let p = Vec3::new(0.11, 0.22, 0.33);
        let a = wind_acceleration(field, p, 2.5);
        let b = wind_acceleration(field, p, 2.5);
        assert_eq!(a, b);
    }

    #[test]
    fn sine_matches_known_values() {
        assert!(sin_turns(0.0).abs() < 1.0e-4);
        assert!((sin_turns(FRAC_PI_2) - 1.0).abs() < 1.0e-4);
        assert!(sin_turns(PI).abs() < 1.0e-4);
        assert!((sin_turns(-FRAC_PI_2) + 1.0).abs() < 1.0e-4);
    }
}
