//! Static boundary container generators for granular scenes.
//!
//! A single [`HalfSpace`] is the atom of static-boundary collision, but real
//! granular demos are posed inside *containers* assembled from several planes:
//! a box that holds a settling pile, an open-topped bin that a stream of grains
//! rains into, or a converging hopper that discharges through a slot. Hand-
//! writing the plane equations — and in particular getting every inward normal
//! to point into the region the grains occupy — is fiddly and easy to get
//! wrong by a sign.
//!
//! This module centralises that geometry. Each generator returns a `Vec` of
//! [`HalfSpace`]s whose unit normals all point **into the interior**, exactly
//! the orientation [`SphereBoundaryDriver`](super::sphere_boundary_driver::SphereBoundaryDriver)
//! expects, so a container composes with the boundary driver with no further
//! bookkeeping. The containers are pure geometry with no coupling to the time
//! integrator, and nothing here is derived from Unreal Engine source.
//!
//! All boxes are axis-aligned; the wedge hopper is centred on the world origin
//! in `XY` with its base at `z = 0` and opening upward along `+Z`.

use crate::collider::rotational_boundary_contact::HalfSpace;
use glam::Vec3;

/// Builds the six inward-facing planes of a closed axis-aligned box.
///
/// Returns `None` unless `min` and `max` are finite and `max` is strictly
/// greater than `min` on every axis. The planes are ordered floor (`+Z`),
/// ceiling (`−Z`), `−X` wall (`+X` normal), `+X` wall (`−X` normal), `−Y` wall
/// (`+Y` normal), `+Y` wall (`−Y` normal).
#[must_use]
pub fn closed_box(min: Vec3, max: Vec3) -> Option<Vec<HalfSpace>> {
    if !valid_box(min, max) {
        return None;
    }
    Some(vec![
        HalfSpace::new(min, Vec3::Z)?,
        HalfSpace::new(max, -Vec3::Z)?,
        HalfSpace::new(min, Vec3::X)?,
        HalfSpace::new(max, -Vec3::X)?,
        HalfSpace::new(min, Vec3::Y)?,
        HalfSpace::new(max, -Vec3::Y)?,
    ])
}

/// Builds the five inward-facing planes of an open-topped axis-aligned box:
/// the closed box without its `−Z` ceiling, so grains can rain in from above.
///
/// Returns `None` under the same validity conditions as [`closed_box`].
#[must_use]
pub fn open_top_box(min: Vec3, max: Vec3) -> Option<Vec<HalfSpace>> {
    if !valid_box(min, max) {
        return None;
    }
    Some(vec![
        HalfSpace::new(min, Vec3::Z)?,
        HalfSpace::new(min, Vec3::X)?,
        HalfSpace::new(max, -Vec3::X)?,
        HalfSpace::new(min, Vec3::Y)?,
        HalfSpace::new(max, -Vec3::Y)?,
    ])
}

/// Builds a prismatic (wedge) hopper: two slanted walls converging in `X`
/// toward a bottom slot, plus two vertical end walls spanning `Y`.
///
/// The hopper is centred on the origin in `XY`. The `X` half-width is
/// `half_width_bottom` at `z = 0` and flares linearly to `half_width_top` at
/// `z = height`; the `Y` half-extent is `half_depth` at every height. There is
/// no floor — the bottom slot of width `2·half_width_bottom` is the discharge
/// opening — matching how a real hopper drains.
///
/// Returns `None` unless every argument is finite, `half_width_bottom > 0`,
/// `half_width_top ≥ half_width_bottom`, `half_depth > 0`, and `height > 0`.
/// The returned planes are ordered `+X` slant, `−X` slant, `+Y` end, `−Y` end.
#[must_use]
pub fn wedge_hopper(
    half_width_bottom: f32,
    half_width_top: f32,
    half_depth: f32,
    height: f32,
) -> Option<Vec<HalfSpace>> {
    let finite = half_width_bottom.is_finite()
        && half_width_top.is_finite()
        && half_depth.is_finite()
        && height.is_finite();
    if !finite {
        return None;
    }
    if half_width_bottom <= 0.0 || half_width_top < half_width_bottom || half_depth <= 0.0 {
        return None;
    }
    if height <= 0.0 {
        return None;
    }

    let flare = half_width_top - half_width_bottom;
    // Inward normal of the +X slant: the in-plane directions are the Y axis and
    // the slope vector (flare, 0, height); their cross product, negated, points
    // back toward the interior and up.
    let plus_x = HalfSpace::new(
        Vec3::new(half_width_bottom, 0.0, 0.0),
        Vec3::new(-height, 0.0, flare),
    )?;
    let minus_x = HalfSpace::new(
        Vec3::new(-half_width_bottom, 0.0, 0.0),
        Vec3::new(height, 0.0, flare),
    )?;
    let plus_y = HalfSpace::new(Vec3::new(0.0, half_depth, 0.0), -Vec3::Y)?;
    let minus_y = HalfSpace::new(Vec3::new(0.0, -half_depth, 0.0), Vec3::Y)?;
    Some(vec![plus_x, minus_x, plus_y, minus_y])
}

fn valid_box(min: Vec3, max: Vec3) -> bool {
    min.is_finite() && max.is_finite() && min.x < max.x && min.y < max.y && min.z < max.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::hertz_contact::HertzModel;
    use crate::collider::sphere_boundary_driver::SphereBoundaryDriver;
    use crate::collider::sphere_packing::{pack_spheres, SpherePackingParams};

    const EPS: f32 = 1.0e-5;

    fn interior_positive(planes: &[HalfSpace], point: Vec3) -> bool {
        planes.iter().all(|p| p.signed_distance(point) > -EPS)
    }

    #[test]
    fn closed_box_has_six_inward_planes() {
        let min = Vec3::new(-1.0, -2.0, 0.0);
        let max = Vec3::new(3.0, 2.0, 5.0);
        let planes = closed_box(min, max).expect("box");
        assert_eq!(planes.len(), 6);
        let center = 0.5 * (min + max);
        assert!(interior_positive(&planes, center));
        // Every normal must point toward the interior centroid.
        for p in &planes {
            assert!((center - p.point()).dot(p.normal()) > 0.0);
        }
    }

    #[test]
    fn closed_box_rejects_outside_points() {
        let min = Vec3::ZERO;
        let max = Vec3::splat(4.0);
        let planes = closed_box(min, max).expect("box");
        // A point just below the floor and one just above the ceiling each fail
        // exactly one plane.
        assert!(!interior_positive(&planes, Vec3::new(2.0, 2.0, -0.1)));
        assert!(!interior_positive(&planes, Vec3::new(2.0, 2.0, 4.1)));
        assert!(!interior_positive(&planes, Vec3::new(-0.1, 2.0, 2.0)));
    }

    #[test]
    fn open_top_box_omits_the_ceiling() {
        let min = Vec3::ZERO;
        let max = Vec3::splat(4.0);
        let planes = open_top_box(min, max).expect("box");
        assert_eq!(planes.len(), 5);
        // No plane has a downward (-Z dominant) normal, i.e. no ceiling.
        assert!(planes.iter().all(|p| p.normal().z > -0.5));
        // A point high above the open top is still "interior" to all 5 planes.
        assert!(interior_positive(&planes, Vec3::new(2.0, 2.0, 100.0)));
        // But a point outside a side wall is rejected.
        assert!(!interior_positive(&planes, Vec3::new(4.1, 2.0, 2.0)));
    }

    #[test]
    fn box_rejects_degenerate_extents() {
        assert!(closed_box(Vec3::ZERO, Vec3::new(0.0, 1.0, 1.0)).is_none());
        assert!(open_top_box(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)).is_none());
        assert!(closed_box(Vec3::splat(f32::NAN), Vec3::ONE).is_none());
    }

    #[test]
    fn wedge_hopper_flares_outward_and_points_inward() {
        let planes = wedge_hopper(1.0, 2.0, 1.5, 3.0).expect("hopper");
        assert_eq!(planes.len(), 4);
        // Interior sample on the axis is inside every wall.
        assert!(interior_positive(&planes, Vec3::new(0.0, 0.0, 1.5)));
        // At z = 1.5 the half-width is 1 + (2-1)*(1.5/3) = 1.5. A point just
        // inside is interior; one just outside the +X slant is rejected.
        assert!(interior_positive(&planes, Vec3::new(1.4, 0.0, 1.5)));
        assert!(!interior_positive(&planes, Vec3::new(1.6, 0.0, 1.5)));
        // Symmetry: the -X slant rejects the mirror point.
        assert!(!interior_positive(&planes, Vec3::new(-1.6, 0.0, 1.5)));
    }

    #[test]
    fn wedge_hopper_rejects_bad_parameters() {
        assert!(wedge_hopper(0.0, 2.0, 1.0, 3.0).is_none());
        assert!(wedge_hopper(2.0, 1.0, 1.0, 3.0).is_none()); // top narrower
        assert!(wedge_hopper(1.0, 2.0, 0.0, 3.0).is_none());
        assert!(wedge_hopper(1.0, 2.0, 1.0, 0.0).is_none());
        assert!(wedge_hopper(1.0, f32::INFINITY, 1.0, 3.0).is_none());
    }

    #[test]
    fn packed_grains_feel_no_force_from_their_container() {
        let min = Vec3::ZERO;
        let max = Vec3::splat(10.0);
        let params = SpherePackingParams {
            min,
            max,
            radius_min: 0.4,
            radius_max: 0.6,
            target_count: 40,
            max_attempts: 20_000,
            separation: 0.05,
            seed: 7,
        };
        let packing = pack_spheres(&params).expect("packing");
        let walls = open_top_box(min, max).expect("walls");
        let model = HertzModel::new(1.0e7, 0.3, 10.0, 10.0, 0.5).expect("model");
        let driver = SphereBoundaryDriver::with_boundaries(model, walls);
        let velocities = vec![Vec3::ZERO; packing.len()];
        let res = driver
            .resolve(packing.positions(), packing.radii(), &velocities)
            .expect("resolve");
        // pack_spheres keeps every centre at least one radius from each face, so
        // no grain penetrates its container at rest.
        assert_eq!(res.contact_count(), 0);
        assert!(res.total_force().length() <= EPS);
    }
}
