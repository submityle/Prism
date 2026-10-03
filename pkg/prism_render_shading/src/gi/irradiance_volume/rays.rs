//! DDGI per-probe ray generation and ray-hit aggregation (CPU golden).
//!
//! The probe-update GPU pass traces a small fixed budget of rays per probe over
//! a Fibonacci sphere, classifies each as a front-face hit, back-face hit, or
//! miss, and folds the results into:
//!
//! * the cosine-weighted octahedral irradiance field
//!   ([`super::ddgi_probe::cosine_weighted_irradiance`]),
//! * the sharpened two-moment depth field
//!   ([`super::visibility::sharpened_depth_moments`]), and
//! * the per-probe relocation / classification statistics
//!   ([`ProbeRayStats`]).
//!
//! This module is the CPU golden reference for the first and third of those: it
//! produces the exact same ray directions the WESL kernel generates
//! ([`fibonacci_sphere_dir`]) and the exact same statistics reduction
//! ([`probe_ray_stats`]), so the GPU probe-update pass can be verified against a
//! deterministic CPU twin.
//!
//! # Conventions
//! * Directions are unit `Vec3` in the probe's local world frame (the probe
//!   position is the ray origin; directions are not offset by probe motion).
//! * A [`ProbeRay`] carries its direction, hit distance, and a [`RayHit`]
//!   classification. Miss rays carry no useful distance and are ignored by the
//!   statistics reduction except that they still count toward the ray budget
//!   for the back-face fraction denominator's hit total.
//! * Every function is deterministic: no RNG, no I/O, no GPU, no `unsafe`, no
//!   heap allocation; transcendental use goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};

use super::relocation::ProbeRayStats;

/// Classification of a single probe ray's intersection result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayHit {
    /// The ray hit a surface from the outside (front face): usable radiance and
    /// depth sample.
    Front,
    /// The ray hit a surface from the inside (back face): the probe is (partly)
    /// embedded in geometry along this direction.
    Back,
    /// The ray left the scene without hitting geometry (sky / out of bounds).
    Miss,
}

/// A single traced probe ray: direction, hit distance, and classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeRay {
    /// Unit direction of the ray in world space (ray origin is the probe).
    pub dir: Vec3,
    /// Distance from the probe to the hit, in world units. Meaningful only when
    /// `hit` is [`RayHit::Front`] or [`RayHit::Back`]; ignored for misses.
    pub distance: f32,
    /// Front / back / miss classification of this ray.
    pub hit: RayHit,
}

impl ProbeRay {
    /// Convenience constructor.
    #[inline]
    pub fn new(dir: Vec3, distance: f32, hit: RayHit) -> Self {
        Self { dir, distance, hit }
    }
}

/// The `index`-th of `count` directions on a Fibonacci (golden-spiral) sphere.
///
/// This is the canonical low-discrepancy spherical point set used by RTXGI for
/// probe ray budgets. The sequence is deterministic and matches the WESL
/// kernel's `fibonacci_sphere_dir` bit-for-bit (same operation order, same
/// `golden = PI * (3 - sqrt(5))` constant). Returns [`Vec3::Z`] for a
/// degenerate `count == 0` so callers never divide by zero.
#[inline]
pub fn fibonacci_sphere_dir(index: usize, count: usize) -> Vec3 {
    if count == 0 {
        return Vec3::Z;
    }
    let golden = core::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let i = index as f32;
    let n = count as f32;
    let z = 1.0 - 2.0 * (i + 0.5) / n;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let theta = golden * i;
    Vec3::new(r * ops::cos(theta), r * ops::sin(theta), z)
}

/// Aggregates a probe's traced rays into [`ProbeRayStats`] for relocation and
/// classification.
///
/// Convention (matches the GPU probe-update reduction):
/// * `backface_fraction = back_hits / max(total_hits, 1)` where
///   `total_hits = front_hits + back_hits` (misses do not count toward the
///   denominator; a probe seeing only sky is treated as fully open, fraction 0).
/// * The closest / farthest *front*-face hits (by distance) populate
///   `closest_frontface_*` / `farthest_frontface_*`; with no front hit these are
///   `INFINITY` / `0.0` with `Vec3::ZERO` directions.
/// * The closest *back*-face hit (by distance) populates `closest_backface_*`;
///   with no back hit it is `INFINITY` with a `Vec3::ZERO` direction.
///
/// Hit distances are treated as `max(0, distance)`; directions are stored
/// unnormalised (the relocation layer normalises defensively).
#[inline]
pub fn probe_ray_stats(rays: &[ProbeRay]) -> ProbeRayStats {
    let mut stats = ProbeRayStats::open();

    let mut front_hits: u32 = 0;
    let mut back_hits: u32 = 0;

    for ray in rays {
        let dist = ray.distance.max(0.0);
        match ray.hit {
            RayHit::Front => {
                front_hits += 1;
                if dist < stats.closest_frontface_distance {
                    stats.closest_frontface_distance = dist;
                    stats.closest_frontface_dir = ray.dir;
                }
                if dist > stats.farthest_frontface_distance {
                    stats.farthest_frontface_distance = dist;
                    stats.farthest_frontface_dir = ray.dir;
                }
            }
            RayHit::Back => {
                back_hits += 1;
                if dist < stats.closest_backface_distance {
                    stats.closest_backface_distance = dist;
                    stats.closest_backface_dir = ray.dir;
                }
            }
            RayHit::Miss => {}
        }
    }

    let total_hits = front_hits + back_hits;
    stats.backface_fraction = if total_hits == 0 {
        0.0
    } else {
        back_hits as f32 / total_hits as f32
    };

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_directions_are_unit_and_cover_both_poles() {
        let n = 64;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for i in 0..n {
            let d = fibonacci_sphere_dir(i, n);
            assert!(
                (d.length() - 1.0).abs() < 1e-5,
                "dir {i} not unit: len {}",
                d.length()
            );
            min_z = min_z.min(d.z);
            max_z = max_z.max(d.z);
        }
        // The spiral spans nearly the full z range.
        assert!(min_z < -0.9, "min_z {min_z}");
        assert!(max_z > 0.9, "max_z {max_z}");
    }

    #[test]
    fn fibonacci_matches_reference_formula() {
        // Reference uses std cos/sin; parity to float tolerance confirms the
        // op-order and constant match.
        let n = 32;
        let golden = core::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f32;
            #[allow(clippy::disallowed_methods)]
            let reference = Vec3::new(r * theta.cos(), r * theta.sin(), z);
            let got = fibonacci_sphere_dir(i, n);
            assert!((got - reference).length() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn degenerate_count_is_safe() {
        assert_eq!(fibonacci_sphere_dir(0, 0), Vec3::Z);
    }

    #[test]
    fn stats_from_only_sky_is_fully_open() {
        let rays = [
            ProbeRay::new(Vec3::X, 0.0, RayHit::Miss),
            ProbeRay::new(Vec3::Y, 0.0, RayHit::Miss),
        ];
        let s = probe_ray_stats(&rays);
        assert_eq!(s.backface_fraction, 0.0);
        assert_eq!(s.closest_frontface_distance, f32::INFINITY);
        assert_eq!(s.closest_backface_distance, f32::INFINITY);
    }

    #[test]
    fn stats_pick_closest_and_farthest_front() {
        let rays = [
            ProbeRay::new(Vec3::X, 3.0, RayHit::Front),
            ProbeRay::new(Vec3::Y, 1.0, RayHit::Front),
            ProbeRay::new(Vec3::Z, 7.0, RayHit::Front),
        ];
        let s = probe_ray_stats(&rays);
        assert_eq!(s.backface_fraction, 0.0);
        assert_eq!(s.closest_frontface_distance, 1.0);
        assert_eq!(s.closest_frontface_dir, Vec3::Y);
        assert_eq!(s.farthest_frontface_distance, 7.0);
        assert_eq!(s.farthest_frontface_dir, Vec3::Z);
    }

    #[test]
    fn stats_backface_fraction_counts_only_hits() {
        // 1 back, 1 front, 2 misses -> fraction = 1 / (1 + 1) = 0.5.
        let rays = [
            ProbeRay::new(Vec3::X, 0.5, RayHit::Back),
            ProbeRay::new(Vec3::Y, 2.0, RayHit::Front),
            ProbeRay::new(Vec3::Z, 0.0, RayHit::Miss),
            ProbeRay::new(Vec3::NEG_Z, 0.0, RayHit::Miss),
        ];
        let s = probe_ray_stats(&rays);
        assert!((s.backface_fraction - 0.5).abs() < 1e-6);
        assert_eq!(s.closest_backface_distance, 0.5);
        assert_eq!(s.closest_backface_dir, Vec3::X);
    }

    #[test]
    fn stats_negative_distance_clamped() {
        let rays = [ProbeRay::new(Vec3::X, -5.0, RayHit::Front)];
        let s = probe_ray_stats(&rays);
        assert_eq!(s.closest_frontface_distance, 0.0);
    }
}
