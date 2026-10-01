//! Capsule/sphere proxy soft-shadow CPU golden references.
//!
//! Deterministic, GPU-free analytic soft shadows and ambient occlusion from
//! capsule and sphere proxies (character self-shadowing).  Distinct from the
//! shadow-map pipeline in [`crate::gi::shadow`] and the hemispherical AO in
//! [`crate::gi::occlusion`]: here occluders are analytic primitives.
//!
//! * [`sphere`] — analytic sphere soft-shadow fraction from a disk light plus
//!   closed-form sphere ambient occlusion.
//! * [`capsule`] — closest-point-on-segment reduction of a capsule to an
//!   effective sphere for soft shadow and AO.
//! * [`accumulate`] — multi-proxy occlusion accumulation (min / penumbra-aware
//!   combine) with light-size driven penumbra.

pub mod accumulate;
pub mod capsule;
pub mod sphere;

pub use accumulate::{
    accumulate_ambient_occlusion, accumulate_soft_shadow, accumulate_soft_shadow_conservative,
    penumbra_angular_width,
};
pub use capsule::{
    capsule_ao_sphere, capsule_shadow_sphere, closest_point_on_segment,
    closest_point_on_segment_to_line, Capsule,
};
pub use sphere::{
    disk_overlap_fraction, sphere_ambient_occlusion, sphere_soft_shadow, DiskLight, Sphere,
};

use bevy_math::Vec3;

/// High-level soft-shadow query for a character made of capsule proxies.
///
/// Evaluates the combined soft-shadow occlusion that `capsules` cast onto
/// `receiver` from `light`, in `[0, 1]` (`0` fully lit, `1` fully shadowed),
/// using the independent-blocker product model in
/// [`accumulate_soft_shadow`].  An empty slice is fully lit.
#[inline]
pub fn capsule_soft_shadow(receiver: Vec3, light: DiskLight, capsules: &[Capsule]) -> f32 {
    accumulate_soft_shadow(receiver, light, capsules)
}

/// High-level ambient-occlusion query for a character made of capsule proxies.
///
/// Evaluates the combined AO that `capsules` contribute at `receiver` for the
/// surface `normal`, in `[0, 1]` (`0` fully open, `1` fully occluded), using
/// the product model in [`accumulate_ambient_occlusion`].  An empty slice is
/// fully open.
#[inline]
pub fn capsule_ambient_occlusion(receiver: Vec3, normal: Vec3, capsules: &[Capsule]) -> f32 {
    accumulate_ambient_occlusion(receiver, normal, capsules)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn high_level_shadow_matches_accumulate() {
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 15.0), 0.6);
        let capsules = [
            Capsule::new(Vec3::new(0.0, -4.0, 3.0), Vec3::new(0.0, 4.0, 3.0), 0.7),
            Capsule::new(Vec3::new(0.5, -4.0, 4.0), Vec3::new(0.5, 4.0, 4.0), 0.5),
        ];
        let via_api = capsule_soft_shadow(Vec3::ZERO, light, &capsules);
        let via_acc = accumulate_soft_shadow(Vec3::ZERO, light, &capsules);
        assert!((via_api - via_acc).abs() < TOL, "{via_api} {via_acc}");
    }

    #[test]
    fn high_level_ao_matches_accumulate() {
        let capsules = [
            Capsule::new(Vec3::new(-2.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.8),
            Capsule::new(Vec3::new(-2.0, 2.5, 1.0), Vec3::new(2.0, 2.5, 1.0), 0.5),
        ];
        let via_api = capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, &capsules);
        let via_acc = accumulate_ambient_occlusion(Vec3::ZERO, Vec3::Y, &capsules);
        assert!((via_api - via_acc).abs() < TOL, "{via_api} {via_acc}");
    }

    #[test]
    fn empty_character_is_fully_lit_and_open() {
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 15.0), 0.6);
        assert_eq!(capsule_soft_shadow(Vec3::ZERO, light, &[]), 0.0);
        assert_eq!(capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, &[]), 0.0);
    }

    #[test]
    fn single_capsule_shadows_and_occludes() {
        // A capsule between a +Y receiver and the light both shadows the light
        // and occludes the hemisphere.
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 12.0), 0.4);
        let shadow_caps =
            [Capsule::new(Vec3::new(0.0, -3.0, 3.0), Vec3::new(0.0, 3.0, 3.0), 0.8)];
        let occ = capsule_soft_shadow(Vec3::ZERO, light, &shadow_caps);
        assert!(occ > 0.0 && occ <= 1.0, "occ = {occ}");

        let ao_caps = [Capsule::new(
            Vec3::new(-2.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            1.0,
        )];
        let ao = capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, &ao_caps);
        assert!(ao > 0.0 && ao <= 1.0, "ao = {ao}");
    }
}
