//! Sphere narrow phase against a resident world SDF snapshot.

use glam::Vec3;
use prism_physics_core::{BodyHandle, ContactManifold, ContactPoint};
use prism_world_residency::SdfQueryResult;

use super::{SdfBridgeError, SdfPhysicsSnapshot};

impl SdfPhysicsSnapshot {
    /// Generates the solver manifold for a sphere against this world SDF.
    ///
    /// The sphere center and physics origin are signed world millimetres. The
    /// manifold witnesses are metres relative to `physics_origin_mm`, so callers
    /// should use a nearby floating origin. `body_a` is the static SDF/world
    /// body and `body_b` is the sphere. Missing SDF samples stay `Unknown`; a
    /// complete sample outside the sphere radius returns `Value(None)`.
    ///
    /// The contact normal is the normalized world-space gradient. Witnesses use
    /// a first-order SDF projection and satisfy the shared
    /// `point_a - point_b = penetration * normal` convention.
    pub fn sphere_contact_manifold(
        &self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        sphere_center_world_mm: [i64; 3],
        physics_origin_mm: [i64; 3],
        sphere_radius_mm: u32,
        mip: u8,
    ) -> Result<SdfQueryResult<Option<ContactManifold>>, SdfBridgeError> {
        if sphere_radius_mm == 0 {
            return Err(SdfBridgeError::InvalidContactData);
        }
        let query = self.query_world_distance_gradient_mm(sphere_center_world_mm, mip)?;
        let SdfQueryResult::Value(distance_gradient) = query else {
            return Ok(match query {
                SdfQueryResult::Miss => SdfQueryResult::Miss,
                SdfQueryResult::Unknown => SdfQueryResult::Unknown,
                SdfQueryResult::Value(_) => unreachable!(),
            });
        };

        let distance_m = distance_gradient.distance_mm / 1_000.0;
        let radius_m = f64::from(sphere_radius_mm) / 1_000.0;
        if distance_m > radius_m {
            return Ok(SdfQueryResult::Value(None));
        }
        let normal = distance_gradient
            .unit_normal()
            .ok_or(SdfBridgeError::InvalidContactData)?;

        let local_center_m = std::array::from_fn::<_, 3, _>(|axis| {
            (i128::from(sphere_center_world_mm[axis]) - i128::from(physics_origin_mm[axis])) as f64
                / 1_000.0
        });
        let center = Vec3::new(
            local_center_m[0] as f32,
            local_center_m[1] as f32,
            local_center_m[2] as f32,
        );
        let normal = Vec3::new(normal[0] as f32, normal[1] as f32, normal[2] as f32);
        let distance = distance_m as f32;
        let radius = radius_m as f32;
        let penetration = radius - distance;
        let point_a = center - normal * distance;
        let point_b = center - normal * radius;
        if !center.is_finite()
            || !normal.is_finite()
            || !distance.is_finite()
            || !radius.is_finite()
            || !penetration.is_finite()
            || !point_a.is_finite()
            || !point_b.is_finite()
            || penetration < 0.0
        {
            return Err(SdfBridgeError::InvalidContactData);
        }

        let mut manifold = ContactManifold::new(body_a, body_b, normal);
        let pushed = manifold.push(ContactPoint::new(point_a, point_b, penetration));
        debug_assert!(
            pushed,
            "an empty manifold always has one inline contact slot"
        );
        Ok(SdfQueryResult::Value(Some(manifold)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::collide::ContactPoint;
    use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

    fn plane_snapshot(origin: [i64; 3], sample_offset: i16) -> SdfPhysicsSnapshot {
        let mut grid = SparseSdfBrickGrid::new(2, 1).expect("valid grid");
        let samples = (0..8)
            .map(|index| {
                if index & 1 == 0 {
                    -1 + sample_offset
                } else {
                    1 + sample_offset
                }
            })
            .collect();
        grid.insert_brick(BrickCoord::new(0, 0, 0, 0), samples)
            .expect("plane brick");
        super::super::WorldSdfPhysicsBridge::new(
            super::super::SdfPhysicsBridgeBudget::new(1, 8, 80).expect("budget"),
        )
        .stage(super::super::WorldSdfSnapshot::new(
            super::super::WorldSdfTransform::new(origin, 100, 100).expect("transform"),
            1,
            grid,
        ))
        .expect("snapshot")
    }

    fn contact(result: SdfQueryResult<Option<ContactManifold>>) -> ContactManifold {
        match result {
            SdfQueryResult::Value(Some(manifold)) => manifold,
            other => panic!("expected an SDF contact, got {other:?}"),
        }
    }

    #[test]
    fn plane_sdf_generates_manifold_with_solver_witness_convention() {
        let snapshot = plane_snapshot([0; 3], 0);
        let world_body = BodyHandle::INVALID;
        let sphere_body = BodyHandle::INVALID;
        let manifold = contact(
            snapshot
                .sphere_contact_manifold(world_body, sphere_body, [50, 50, 50], [0; 3], 100, 0)
                .expect("distance query"),
        );
        assert_eq!(manifold.body_a, world_body);
        assert_eq!(manifold.body_b, sphere_body);
        assert_eq!(manifold.normal, Vec3::X);
        assert_eq!(manifold.len(), 1);
        let point = manifold.points()[0];
        assert_eq!(
            point,
            ContactPoint::new(
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(-0.05, 0.05, 0.05),
                0.1
            )
        );
        let separation = point.point_a - point.point_b;
        assert!((separation - manifold.normal * point.penetration).length() < 1.0e-6);
    }

    #[test]
    fn missing_or_separated_samples_do_not_fabricate_contacts() {
        let empty = {
            let grid = SparseSdfBrickGrid::new(2, 1).expect("valid grid");
            super::super::WorldSdfPhysicsBridge::new(
                super::super::SdfPhysicsBridgeBudget::new(1, 8, 80).expect("budget"),
            )
            .stage(super::super::WorldSdfSnapshot::new(
                super::super::WorldSdfTransform::new([0; 3], 100, 100).expect("transform"),
                1,
                grid,
            ))
            .expect("empty snapshot")
        };
        assert_eq!(
            empty
                .sphere_contact_manifold(
                    BodyHandle::INVALID,
                    BodyHandle::INVALID,
                    [50, 50, 50],
                    [0; 3],
                    100,
                    0,
                )
                .expect("unknown remains a query result"),
            SdfQueryResult::Unknown
        );

        let snapshot = plane_snapshot([0; 3], 2);
        assert_eq!(
            snapshot
                .sphere_contact_manifold(
                    BodyHandle::INVALID,
                    BodyHandle::INVALID,
                    [50, 50, 50],
                    [0; 3],
                    100,
                    0,
                )
                .expect("separated query"),
            SdfQueryResult::Value(None)
        );
    }

    #[test]
    fn touching_sphere_keeps_a_zero_penetration_contact() {
        let snapshot = plane_snapshot([0; 3], 1);
        let manifold = contact(
            snapshot
                .sphere_contact_manifold(
                    BodyHandle::INVALID,
                    BodyHandle::INVALID,
                    [50, 50, 50],
                    [0; 3],
                    100,
                    0,
                )
                .expect("touching query"),
        );
        assert_eq!(manifold.len(), 1);
        assert_eq!(manifold.points()[0].penetration, 0.0);
    }

    #[test]
    fn sphere_center_inside_sdf_surface_uses_outward_gradient() {
        let snapshot = plane_snapshot([0; 3], -1);
        let manifold = contact(
            snapshot
                .sphere_contact_manifold(
                    BodyHandle::INVALID,
                    BodyHandle::INVALID,
                    [50, 50, 50],
                    [0; 3],
                    100,
                    0,
                )
                .expect("interior query"),
        );
        assert_eq!(manifold.normal, Vec3::X);
        assert_eq!(manifold.points()[0].penetration, 0.2);
        let point = manifold.points()[0];
        assert!(
            (point.point_a - point.point_b - manifold.normal * point.penetration).length() < 1e-6
        );
    }

    #[test]
    fn world_origin_subtraction_precedes_float_conversion_and_bad_inputs_fail() {
        let origin = [i64::MAX - 500, i64::MIN + 500, 9_000_000_000_000_000_000];
        let snapshot = plane_snapshot(origin, 0);
        let manifold = contact(
            snapshot
                .sphere_contact_manifold(
                    BodyHandle::INVALID,
                    BodyHandle::INVALID,
                    [origin[0] + 50, origin[1] + 50, origin[2] + 50],
                    origin,
                    100,
                    0,
                )
                .expect("large-origin query"),
        );
        assert_eq!(manifold.points()[0].point_a.x, 0.05);
        assert!(matches!(
            snapshot.sphere_contact_manifold(
                BodyHandle::INVALID,
                BodyHandle::INVALID,
                [origin[0] + 50, origin[1] + 50, origin[2] + 50],
                origin,
                0,
                0,
            ),
            Err(SdfBridgeError::InvalidContactData)
        ));
    }
}
