//! Margin-aware (speculative) contact query between two convex shapes.
//!
//! A speculative contact lets a constraint solver react to a collision slightly
//! before the shapes actually touch, which prevents tunnelling for fast bodies
//! without a full continuous sweep. Each shape carries a collision *margin* (a
//! thin rounded skin); the query reports the signed separation between those
//! skins together with the closest witness points and the contact normal.
//! Internally it measures the exact core separation/penetration and shifts it
//! by the combined margin rather than inflating the shapes (which would round
//! their surfaces and make EPA underestimate penetration depth).
//!
//! The sign convention mirrors the rest of the narrow phase: the `normal`
//! points from shape `a` toward shape `b`, a positive `separation` means the
//! shells are still apart (a predicted/within-reach contact), and a negative
//! `separation` means the shells already interpenetrate by that amount. The
//! unrounded cores are `separation + margin_a + margin_b` apart.
//!
//! This is a clean-room implementation built on the crate's own GJK/EPA
//! primitives and contains no Unreal Engine source or derived code.

use glam::Vec3;

use crate::narrow::distance::gjk_closest_points;
use crate::narrow::epa::gjk_contact;
use crate::narrow::support::SupportMap;

/// A speculative contact between two margin-inflated convex shapes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SpeculativeContact {
    /// Unit contact normal pointing from shape `a` toward shape `b`.
    pub normal: Vec3,
    /// Witness point on the inflated (skin) surface of shape `a`.
    pub point_a: Vec3,
    /// Witness point on the inflated (skin) surface of shape `b`.
    pub point_b: Vec3,
    /// Signed separation between the inflated shells. Positive when the shells
    /// are still apart; negative when they interpenetrate by that distance.
    pub separation: f32,
}

/// Returns the speculative contact between convex shapes `a` and `b`, each
/// inflated by its own non-negative collision `margin`, or [`None`] when the
/// shapes interpenetrate so deeply that penetration recovery cannot pin down a
/// well-defined contact normal.
///
/// When the inflated shells are separated the witness points lie on those
/// shells and `separation` is their positive gap. When the shells overlap the
/// result comes from penetration recovery (EPA) and `separation` is the
/// negative overlap depth. Passing `margin_a == margin_b == 0` reduces the
/// query to the plain closest-points / penetration pair.
pub fn speculative_contact<A: SupportMap, B: SupportMap>(
    a: &A,
    b: &B,
    margin_a: f32,
    margin_b: f32,
) -> Option<SpeculativeContact> {
    let ma = margin_a.max(0.0);
    let mb = margin_b.max(0.0);
    let total = ma + mb;

    // The query runs against the *cores*, not the inflated shells. Inflating a
    // shape rounds its surface, and EPA undershoots penetration depth on curved
    // shapes, so inflating before the query would make the overlap case
    // inaccurate. Instead we take the exact core separation/penetration and
    // shift it by the combined margin analytically. The contact normal is
    // unaffected by a uniform inflation, and each witness point moves outward
    // along that normal by its own margin onto the inflated (skin) surface.
    match gjk_closest_points(a, b) {
        // Cores are apart: the shells are `total` closer, which may still leave
        // a positive gap or push them into overlap (negative separation).
        Some(cp) => Some(SpeculativeContact {
            normal: cp.normal,
            point_a: cp.point_a + cp.normal * ma,
            point_b: cp.point_b - cp.normal * mb,
            separation: cp.distance - total,
        }),
        // Cores overlap: EPA recovers the core penetration (exact for polytopes)
        // and the shells overlap by that depth plus both margins.
        None => gjk_contact(a, b).map(|c| SpeculativeContact {
            normal: c.normal,
            point_a: c.point_a + c.normal * ma,
            point_b: c.point_b - c.normal * mb,
            separation: -(c.depth + total),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::speculative_contact;
    use crate::bounding::{BoundingSphere, Obb};
    use glam::{Quat, Vec3};

    #[test]
    fn separated_shells_report_positive_gap() {
        // Unit spheres 5 apart, 0.5 margin each: core-surface gap is 3, shells
        // sit 0.5 closer on each side, so the shell separation is 2.
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let b = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let c = speculative_contact(&a, &b, 0.5, 0.5).expect("separated contact");
        assert!((c.separation - 2.0).abs() < 1e-4, "sep = {}", c.separation);
        assert!(c.normal.x > 0.999, "normal = {:?}", c.normal);
        // Witness on a's shell: core radius 1 plus 0.5 margin along +x.
        assert!(c.point_a.abs_diff_eq(Vec3::new(1.5, 0.0, 0.0), 1e-4), "pa = {:?}", c.point_a);
        assert!(c.point_b.abs_diff_eq(Vec3::new(3.5, 0.0, 0.0), 1e-4), "pb = {:?}", c.point_b);
    }

    #[test]
    fn margins_bring_cores_into_speculative_contact() {
        // Boxes whose cores are 0.5 apart (half-extent 1 each, centres 2.5) but
        // whose 0.5 collision margins inflate the shells until their flat +x/-x
        // faces overlap by 0.5. A polytope core keeps EPA's depth exact.
        let a = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let b = Obb::new(Vec3::new(2.5, 0.0, 0.0), Vec3::splat(1.0), Quat::IDENTITY);
        let c = speculative_contact(&a, &b, 0.5, 0.5).expect("speculative contact");
        assert!((c.separation + 0.5).abs() < 1e-3, "sep = {}", c.separation);
        assert!(c.normal.x.abs() > 0.99, "normal = {:?}", c.normal);
    }

    #[test]
    fn zero_margins_match_core_distance() {
        // With no margins the separation is just the core-surface distance.
        let a = BoundingSphere::new(Vec3::ZERO, 1.0);
        let b = BoundingSphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let c = speculative_contact(&a, &b, 0.0, 0.0).expect("separated contact");
        assert!((c.separation - 3.0).abs() < 1e-4, "sep = {}", c.separation);
    }

    #[test]
    fn overlapping_cores_report_negative_separation() {
        // Overlapping unit boxes (half-extent 1, centres 1 apart) interpenetrate
        // by 1 along x; with no margins the shell overlap equals the core depth.
        let a = Obb::new(Vec3::ZERO, Vec3::splat(1.0), Quat::IDENTITY);
        let b = Obb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(1.0), Quat::IDENTITY);
        let c = speculative_contact(&a, &b, 0.0, 0.0).expect("penetrating contact");
        assert!(c.separation < 0.0, "sep = {}", c.separation);
        assert!((c.separation + 1.0).abs() < 1e-3, "sep = {}", c.separation);
        assert!(c.normal.x.abs() > 0.99, "normal = {:?}", c.normal);
    }
}
