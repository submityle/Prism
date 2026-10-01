//! Multi-proxy soft-shadow and ambient-occlusion accumulation.
//!
//! Characters are built from several capsule proxies, so a receiver may be
//! shadowed or occluded by more than one at once.  This module combines the
//! per-proxy fractions from
//! [`capsule_soft_shadow`](crate::gi::capsule_shadow::capsule::capsule_soft_shadow)
//! and
//! [`capsule_ambient_occlusion`](crate::gi::capsule_shadow::capsule::capsule_ambient_occlusion)
//! into a single occlusion for the whole set:
//!
//! * [`accumulate_soft_shadow`] treats each proxy as an independent
//!   *transmittance* `1 - occ_i` and multiplies them, so the combined shadow is
//!   `1 - prod(1 - occ_i)`.  This is the standard independent-blocker model: it
//!   is exact for a single proxy, never brightens as proxies are added, and
//!   saturates smoothly toward full shadow.
//! * [`accumulate_soft_shadow_conservative`] takes the single strongest
//!   blocker, `max_i occ_i` — the conservative minimum-transmittance choice
//!   that avoids the slight over-darkening of the product when proxies overlap.
//! * [`accumulate_ambient_occlusion`] combines AO the same multiplicative way,
//!   `1 - prod(1 - ao_i)`.
//!
//! The penumbra of each proxy is driven by the light's angular size through
//! [`penumbra_angular_width`]; a wider light widens the transition band inside
//! the sphere model, and this helper exposes that width for callers that size
//! filter kernels from it.
//!
//! # Conventions
//! * Proxy sets are passed as borrowed slices `&[Capsule]`; the empty slice is
//!   valid and means *no occluder*, so shadow and AO are `0`.
//! * All returned fractions are in `[0, 1]` (`0` lit/open, `1`
//!   shadowed/occluded) and are always finite.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no allocation, no global state.

use crate::gi::capsule_shadow::capsule::{
    capsule_ambient_occlusion, capsule_soft_shadow, Capsule,
};
use crate::gi::capsule_shadow::sphere::DiskLight;
use bevy_math::{ops, Vec3};
use core::f32::consts::FRAC_PI_2;

/// Distance below which a length is treated as zero (world units).
const EPS: f32 = 1.0e-6;

/// Combined soft-shadow occlusion of a capsule set via independent
/// transmittance, in `[0, 1]`.
///
/// Returns `1 - prod_i (1 - occ_i)` where `occ_i` is each capsule's
/// [`capsule_soft_shadow`].  An empty slice returns `0` (fully lit).  Equal to
/// the single-proxy value for one capsule, and monotonically non-decreasing as
/// capsules are added.
pub fn accumulate_soft_shadow(receiver: Vec3, light: DiskLight, capsules: &[Capsule]) -> f32 {
    let mut transmittance = 1.0f32;
    for &capsule in capsules {
        let occ = capsule_soft_shadow(receiver, light, capsule).clamp(0.0, 1.0);
        transmittance *= 1.0 - occ;
    }
    (1.0 - transmittance).clamp(0.0, 1.0)
}

/// Conservative combined soft-shadow occlusion: the single strongest blocker,
/// `max_i occ_i`, in `[0, 1]`.
///
/// This is the minimum-transmittance combine.  It avoids the over-darkening the
/// product model can show when proxies occlude the same part of the light, at
/// the cost of ignoring additional partial blockers.  An empty slice returns
/// `0`.
pub fn accumulate_soft_shadow_conservative(
    receiver: Vec3,
    light: DiskLight,
    capsules: &[Capsule],
) -> f32 {
    let mut max_occ = 0.0f32;
    for &capsule in capsules {
        let occ = capsule_soft_shadow(receiver, light, capsule).clamp(0.0, 1.0);
        if occ > max_occ {
            max_occ = occ;
        }
    }
    max_occ.clamp(0.0, 1.0)
}

/// Combined ambient occlusion of a capsule set via independent transmittance,
/// in `[0, 1]`.
///
/// Returns `1 - prod_i (1 - ao_i)` where `ao_i` is each capsule's
/// [`capsule_ambient_occlusion`].  An empty slice returns `0` (fully open),
/// and the result is monotonically non-decreasing as capsules are added.
pub fn accumulate_ambient_occlusion(
    receiver: Vec3,
    normal: Vec3,
    capsules: &[Capsule],
) -> f32 {
    let mut visibility = 1.0f32;
    for &capsule in capsules {
        let ao = capsule_ambient_occlusion(receiver, normal, capsule).clamp(0.0, 1.0);
        visibility *= 1.0 - ao;
    }
    (1.0 - visibility).clamp(0.0, 1.0)
}

/// Angular half-width of the penumbra a disk light produces at `distance`, in
/// radians.
///
/// This is the light's angular radius `asin(light_radius / distance)`, the same
/// quantity that widens the circle–circle transition band inside the sphere
/// soft-shadow model.  A point light (`radius == 0`) gives `0` (a hard edge);
/// a receiver on the light (`distance <= 0`) saturates to `PI/2`.  The result
/// is clamped to `[0, PI/2]`.
#[inline]
pub fn penumbra_angular_width(light_radius: f32, distance: f32) -> f32 {
    if distance <= EPS {
        return FRAC_PI_2;
    }
    let r = light_radius.max(0.0);
    ops::asin((r / distance).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::capsule_shadow::capsule::capsule_soft_shadow as single_shadow;
    use crate::gi::capsule_shadow::capsule::capsule_ambient_occlusion as single_ao;

    const TOL: f32 = 1.0e-5;

    fn light() -> DiskLight {
        DiskLight::new(Vec3::new(0.0, 0.0, 15.0), 0.6)
    }

    fn blocker(x: f32) -> Capsule {
        Capsule::new(
            Vec3::new(x, -4.0, 3.0),
            Vec3::new(x, 4.0, 3.0),
            0.7,
        )
    }

    #[test]
    fn empty_set_casts_no_shadow() {
        assert_eq!(accumulate_soft_shadow(Vec3::ZERO, light(), &[]), 0.0);
        assert_eq!(
            accumulate_soft_shadow_conservative(Vec3::ZERO, light(), &[]),
            0.0
        );
    }

    #[test]
    fn empty_set_casts_no_ao() {
        assert_eq!(accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &[]), 0.0);
    }

    #[test]
    fn single_proxy_matches_direct_shadow() {
        let caps = [blocker(0.0)];
        let acc = accumulate_soft_shadow(Vec3::ZERO, light(), &caps);
        let direct = single_shadow(Vec3::ZERO, light(), caps[0]);
        assert!((acc - direct).abs() < TOL, "{acc} {direct}");
    }

    #[test]
    fn single_proxy_matches_direct_ao() {
        let caps = [Capsule::new(
            Vec3::new(-2.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            0.8,
        )];
        let acc = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &caps);
        let direct = single_ao(Vec3::ZERO, Vec3::Y, caps[0]);
        assert!((acc - direct).abs() < TOL, "{acc} {direct}");
    }

    #[test]
    fn conservative_single_matches_direct() {
        let caps = [blocker(0.0)];
        let acc = accumulate_soft_shadow_conservative(Vec3::ZERO, light(), &caps);
        let direct = single_shadow(Vec3::ZERO, light(), caps[0]);
        assert!((acc - direct).abs() < TOL, "{acc} {direct}");
    }

    #[test]
    fn more_proxies_never_brighten_shadow() {
        let one = [blocker(0.0)];
        let two = [blocker(0.0), blocker(0.4)];
        let three = [blocker(0.0), blocker(0.4), blocker(-0.4)];
        let a = accumulate_soft_shadow(Vec3::ZERO, light(), &one);
        let b = accumulate_soft_shadow(Vec3::ZERO, light(), &two);
        let c = accumulate_soft_shadow(Vec3::ZERO, light(), &three);
        assert!(b >= a - TOL && c >= b - TOL, "{a} {b} {c}");
    }

    #[test]
    fn more_proxies_never_brighten_ao() {
        let near = Capsule::new(
            Vec3::new(-2.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            0.8,
        );
        let extra = Capsule::new(
            Vec3::new(-2.0, 2.5, 1.0),
            Vec3::new(2.0, 2.5, 1.0),
            0.6,
        );
        let a = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &[near]);
        let b = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &[near, extra]);
        assert!(b >= a - TOL, "{a} {b}");
    }

    #[test]
    fn product_is_at_least_conservative() {
        // The independent-blocker product can only equal or exceed the single
        // strongest blocker.
        let caps = [blocker(0.0), blocker(0.5), blocker(-0.5)];
        let product = accumulate_soft_shadow(Vec3::ZERO, light(), &caps);
        let conservative = accumulate_soft_shadow_conservative(Vec3::ZERO, light(), &caps);
        assert!(product >= conservative - TOL, "{product} {conservative}");
    }

    #[test]
    fn accumulated_values_stay_in_unit_range() {
        let caps = [blocker(0.0), blocker(0.3), blocker(-0.6)];
        let s = accumulate_soft_shadow(Vec3::ZERO, light(), &caps);
        let c = accumulate_soft_shadow_conservative(Vec3::ZERO, light(), &caps);
        let ao = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &caps);
        for v in [s, c, ao] {
            assert!((0.0..=1.0).contains(&v) && v.is_finite(), "v = {v}");
        }
    }

    #[test]
    fn penumbra_grows_with_light_size() {
        let narrow = penumbra_angular_width(0.1, 10.0);
        let wide = penumbra_angular_width(1.0, 10.0);
        assert!(wide > narrow, "{narrow} {wide}");
        assert!((0.0..=FRAC_PI_2).contains(&wide));
    }

    #[test]
    fn penumbra_point_light_is_hard() {
        assert_eq!(penumbra_angular_width(0.0, 10.0), 0.0);
    }

    #[test]
    fn penumbra_degenerate_distance_saturates() {
        let w = penumbra_angular_width(0.5, 0.0);
        assert!((w - FRAC_PI_2).abs() < TOL, "w = {w}");
    }

    #[test]
    fn conservative_equals_strongest_individual_blocker() {
        // The conservative combine must reproduce the single largest per-proxy
        // occlusion exactly.
        let caps = [blocker(0.0), blocker(0.4), blocker(-0.8)];
        let conservative = accumulate_soft_shadow_conservative(Vec3::ZERO, light(), &caps);
        let strongest = caps
            .iter()
            .map(|&c| single_shadow(Vec3::ZERO, light(), c))
            .fold(0.0f32, f32::max);
        assert!((conservative - strongest).abs() < TOL, "{conservative} {strongest}");
    }

    #[test]
    fn two_full_blockers_saturate_to_one() {
        // Two independently fully-occluding proxies combine to full shadow
        // under the product model (1 - 0 * 0 = 1).
        let big = DiskLight::new(Vec3::new(0.0, 0.0, 40.0), 0.2);
        let caps = [
            Capsule::new(Vec3::new(0.0, -4.0, 3.0), Vec3::new(0.0, 4.0, 3.0), 2.0),
            Capsule::new(Vec3::new(0.0, -4.0, 5.0), Vec3::new(0.0, 4.0, 5.0), 2.0),
        ];
        let occ = accumulate_soft_shadow(Vec3::ZERO, big, &caps);
        assert!((occ - 1.0).abs() < TOL, "occ = {occ}");
    }

    #[test]
    fn accumulation_is_deterministic() {
        let caps = [blocker(0.0), blocker(0.3), blocker(-0.3)];
        let a = accumulate_soft_shadow(Vec3::ZERO, light(), &caps);
        let b = accumulate_soft_shadow(Vec3::ZERO, light(), &caps);
        assert_eq!(a, b);
        let c = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &caps);
        let d = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &caps);
        assert_eq!(c, d);
    }

    #[test]
    fn ao_product_matches_manual_combine() {
        // Verify the AO combine equals the hand-computed 1 - prod(1 - ao_i).
        let caps = [
            Capsule::new(Vec3::new(-2.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.8),
            Capsule::new(Vec3::new(-2.0, 2.5, 1.0), Vec3::new(2.0, 2.5, 1.0), 0.5),
        ];
        let manual = 1.0
            - caps
                .iter()
                .map(|&c| 1.0 - single_ao(Vec3::ZERO, Vec3::Y, c))
                .product::<f32>();
        let acc = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &caps);
        assert!((acc - manual).abs() < TOL, "{acc} {manual}");
    }
}
