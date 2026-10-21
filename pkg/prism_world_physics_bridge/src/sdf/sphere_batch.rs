//! Bounded, deterministic sphere-body adapter for the World SDF physics consumer.

use core::mem::size_of;
use std::vec::Vec;

use glam::Vec3;
use prism_physics_core::{BodyHandle, ContactManifold, PhysicsWorld};
use prism_world_residency::SdfQueryResult;

use super::{SdfBridgeError, SdfPhysicsSnapshot};

/// A body center and spherical radius represented by one live rigid-body handle.
///
/// This describes only a sphere centered at the body's origin. It does not
/// infer or support other collider shapes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfSphereBody {
    /// Rigid-body whose origin is the sphere center.
    pub body: BodyHandle,
    /// Sphere radius in millimetres.
    pub radius_mm: u32,
}

/// Hard limits for one immutable sphere batch and its reusable contact scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfSphereBatchLimits {
    /// Maximum number of canonical sphere descriptors and per-call queries.
    pub max_spheres: usize,
    /// Maximum manifolds accepted in one call. One sphere can emit at most one.
    pub max_contacts: usize,
    /// Maximum SDF queries performed per call.
    pub max_work: usize,
    /// Maximum logical bytes reserved for reusable `ContactManifold` scratch.
    pub max_scratch_bytes: usize,
}

impl SdfSphereBatchLimits {
    /// Creates explicit per-batch limits. All limits may be zero for an empty batch.
    #[must_use]
    pub const fn new(
        max_spheres: usize,
        max_contacts: usize,
        max_work: usize,
        max_scratch_bytes: usize,
    ) -> Self {
        Self {
            max_spheres,
            max_contacts,
            max_work,
            max_scratch_bytes,
        }
    }
}

/// Construction or generation failure for [`SdfSphereContactBatch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfSphereBatchError {
    /// A descriptor has an invalid sentinel handle.
    InvalidSphereHandle,
    /// The descriptor list contains the same body handle more than once.
    DuplicateSphereHandle(BodyHandle),
    /// A sphere radius is zero.
    ZeroRadius(BodyHandle),
    /// The configured SDF body is not a valid, live physics body.
    InvalidWorldBody,
    /// The live SDF body has nonzero inverse mass or inertia.
    WorldBodyMovable,
    /// The descriptor count exceeds its configured maximum.
    SphereLimitExceeded {
        /// Number of configured spheres.
        required: usize,
        /// Maximum accepted sphere count.
        limit: usize,
    },
    /// The query-work limit cannot cover every sphere in the immutable batch.
    WorkLimitExceeded {
        /// Minimum queries required for the immutable batch.
        required: usize,
        /// Maximum queries allowed per generation call.
        limit: usize,
    },
    /// The contact limit cannot cover the batch's worst-case one contact per sphere.
    ContactLimitExceeded {
        /// Worst-case or observed contact count required.
        required: usize,
        /// Maximum contacts allowed per generation call.
        limit: usize,
    },
    /// Reusable scratch would exceed its configured logical byte limit.
    ScratchBudgetExceeded {
        /// Logical scratch bytes required.
        required: usize,
        /// Maximum logical scratch bytes allowed.
        limit: usize,
    },
    /// Checked logical-byte arithmetic overflowed.
    ArithmeticOverflow,
    /// A bounded allocation could not be reserved.
    AllocationFailed,
    /// A configured sphere handle is no longer live in this world.
    InvalidSphereBody(BodyHandle),
    /// A body position is not finite.
    InvalidPosition(BodyHandle),
    /// A local f32 position could not be converted to a global i64 millimetre coordinate.
    PositionOutOfRange(BodyHandle),
    /// A sphere query failed; caller output has not been modified.
    Query(SdfBridgeError),
}

impl core::fmt::Display for SdfSphereBatchError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "SDF sphere batch error: {self:?}")
    }
}

impl std::error::Error for SdfSphereBatchError {}

/// Counts from one successfully published sphere batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SdfSphereBatchReport {
    /// Number of configured spheres considered.
    pub spheres: usize,
    /// Number of snapshot queries performed.
    pub queries: usize,
    /// Number of manifolds appended to the caller's output.
    pub contacts: usize,
    /// Number of complete SDF samples outside the sphere radius.
    pub separated: usize,
    /// Number of queries outside the snapshot bounds.
    pub misses: usize,
    /// Number of queries whose required SDF samples are not resident.
    pub unknown: usize,
    /// Logical bytes reserved by this adapter's reusable contact scratch.
    pub required_scratch_bytes: usize,
}

/// Immutable, canonical sphere set with bounded reusable SDF contact scratch.
///
/// Sphere handles are sorted by `(index, generation)` at construction. The
/// configured SDF body must be a live static body. The core API available to
/// this adapter exposes mass properties but not body kind, so generation
/// verifies that the world body is live and immovable; callers must ensure it
/// is specifically static. Sphere descriptors identify body centers and radii
/// only; non-sphere collider shapes are not inferred or supported.
#[derive(Debug)]
pub struct SdfSphereContactBatch {
    spheres: Vec<SdfSphereBody>,
    world_body: BodyHandle,
    physics_origin_mm: [i64; 3],
    mip: u8,
    limits: SdfSphereBatchLimits,
    required_scratch_bytes: usize,
    scratch: Vec<ContactManifold>,
}

impl SdfSphereContactBatch {
    /// Builds a canonical immutable descriptor set and preallocates its bounded scratch.
    pub fn new(
        mut spheres: Vec<SdfSphereBody>,
        world_body: BodyHandle,
        physics_origin_mm: [i64; 3],
        mip: u8,
        limits: SdfSphereBatchLimits,
    ) -> Result<Self, SdfSphereBatchError> {
        if spheres.len() > limits.max_spheres {
            return Err(SdfSphereBatchError::SphereLimitExceeded {
                required: spheres.len(),
                limit: limits.max_spheres,
            });
        }
        if spheres.len() > limits.max_work {
            return Err(SdfSphereBatchError::WorkLimitExceeded {
                required: spheres.len(),
                limit: limits.max_work,
            });
        }
        if spheres.len() > limits.max_contacts {
            return Err(SdfSphereBatchError::ContactLimitExceeded {
                required: spheres.len(),
                limit: limits.max_contacts,
            });
        }
        let required_scratch_bytes = spheres
            .len()
            .checked_mul(size_of::<ContactManifold>())
            .ok_or(SdfSphereBatchError::ArithmeticOverflow)?;
        if required_scratch_bytes > limits.max_scratch_bytes {
            return Err(SdfSphereBatchError::ScratchBudgetExceeded {
                required: required_scratch_bytes,
                limit: limits.max_scratch_bytes,
            });
        }
        for sphere in &spheres {
            if sphere.body == BodyHandle::INVALID {
                return Err(SdfSphereBatchError::InvalidSphereHandle);
            }
            if sphere.radius_mm == 0 {
                return Err(SdfSphereBatchError::ZeroRadius(sphere.body));
            }
            if sphere.body == world_body {
                return Err(SdfSphereBatchError::DuplicateSphereHandle(sphere.body));
            }
        }
        spheres.sort_unstable_by_key(|sphere| (sphere.body.index(), sphere.body.generation()));
        for pair in spheres.windows(2) {
            if pair[0].body == pair[1].body {
                return Err(SdfSphereBatchError::DuplicateSphereHandle(pair[0].body));
            }
        }
        if world_body == BodyHandle::INVALID {
            return Err(SdfSphereBatchError::InvalidWorldBody);
        }
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(spheres.len())
            .map_err(|_| SdfSphereBatchError::AllocationFailed)?;
        Ok(Self {
            spheres,
            world_body,
            physics_origin_mm,
            mip,
            limits,
            required_scratch_bytes,
            scratch,
        })
    }

    /// Returns the sorted immutable sphere descriptors.
    #[must_use]
    pub fn spheres(&self) -> &[SdfSphereBody] {
        &self.spheres
    }

    /// Returns the configured logical scratch requirement in bytes.
    #[must_use]
    pub const fn required_scratch_bytes(&self) -> usize {
        self.required_scratch_bytes
    }

    /// Queries every configured sphere and atomically appends the contact batch.
    ///
    /// `Miss` and `Unknown` are counted separately and never create contacts.
    /// All queries complete before `output` is reserved or modified; any error
    /// therefore leaves its elements unchanged. The scratch vector is retained
    /// and reused on subsequent calls.
    pub fn generate(
        &mut self,
        snapshot: &SdfPhysicsSnapshot,
        world: &PhysicsWorld,
        output: &mut Vec<ContactManifold>,
    ) -> Result<SdfSphereBatchReport, SdfSphereBatchError> {
        let count = self.spheres.len();
        if count > self.limits.max_spheres {
            return Err(SdfSphereBatchError::SphereLimitExceeded {
                required: count,
                limit: self.limits.max_spheres,
            });
        }
        if count > self.limits.max_work {
            return Err(SdfSphereBatchError::WorkLimitExceeded {
                required: count,
                limit: self.limits.max_work,
            });
        }
        if count > self.limits.max_contacts {
            return Err(SdfSphereBatchError::ContactLimitExceeded {
                required: count,
                limit: self.limits.max_contacts,
            });
        }
        if self.required_scratch_bytes > self.limits.max_scratch_bytes {
            return Err(SdfSphereBatchError::ScratchBudgetExceeded {
                required: self.required_scratch_bytes,
                limit: self.limits.max_scratch_bytes,
            });
        }
        let Some(properties) = world.bodies.mass_properties(self.world_body) else {
            return Err(SdfSphereBatchError::InvalidWorldBody);
        };
        if properties.inv_mass != 0.0 || properties.inv_inertia != Vec3::ZERO {
            return Err(SdfSphereBatchError::WorldBodyMovable);
        }

        self.scratch.clear();
        let mut report = SdfSphereBatchReport {
            spheres: count,
            required_scratch_bytes: self.required_scratch_bytes,
            ..SdfSphereBatchReport::default()
        };
        for sphere in &self.spheres {
            let Some(position) = world.bodies.position(sphere.body) else {
                return Err(SdfSphereBatchError::InvalidSphereBody(sphere.body));
            };
            if !position.is_finite() {
                return Err(SdfSphereBatchError::InvalidPosition(sphere.body));
            }
            let center = position_to_world_mm(position, self.physics_origin_mm)
                .ok_or(SdfSphereBatchError::PositionOutOfRange(sphere.body))?;
            report.queries += 1;
            match snapshot
                .sphere_contact_manifold(
                    self.world_body,
                    sphere.body,
                    center,
                    self.physics_origin_mm,
                    sphere.radius_mm,
                    self.mip,
                )
                .map_err(SdfSphereBatchError::Query)?
            {
                SdfQueryResult::Miss => report.misses += 1,
                SdfQueryResult::Unknown => report.unknown += 1,
                SdfQueryResult::Value(None) => report.separated += 1,
                SdfQueryResult::Value(Some(manifold)) => {
                    if self.scratch.len() == self.limits.max_contacts {
                        return Err(SdfSphereBatchError::ContactLimitExceeded {
                            required: self.scratch.len().saturating_add(1),
                            limit: self.limits.max_contacts,
                        });
                    }
                    self.scratch.push(manifold);
                    report.contacts += 1;
                }
            }
        }
        output
            .try_reserve(report.contacts)
            .map_err(|_| SdfSphereBatchError::AllocationFailed)?;
        output.extend_from_slice(&self.scratch);
        Ok(report)
    }
}

fn position_to_world_mm(position: Vec3, origin_mm: [i64; 3]) -> Option<[i64; 3]> {
    if !position.is_finite() {
        return None;
    }
    let mut result = [0_i64; 3];
    for axis in 0..3 {
        // `round` has a stable, specified half-away-from-zero rule. Convert
        // through i128 so adding a large floating-origin offset is checked.
        let local_mm = (f64::from(position[axis]) * 1_000.0).round();
        if !local_mm.is_finite() || local_mm < i128::MIN as f64 || local_mm >= i128::MAX as f64 {
            return None;
        }
        let global = i128::from(origin_mm[axis]).checked_add(local_mm as i128)?;
        result[axis] = i64::try_from(global).ok()?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use prism_physics_core::{BodyDesc, ContactPoint};
    use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
    use std::vec;

    use crate::sdf::{
        SdfPhysicsBridgeBudget, WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform,
    };

    fn snapshot() -> SdfPhysicsSnapshot {
        let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
        grid.insert_brick(
            BrickCoord::new(0, 0, 0, 0),
            vec![-1, 1, -1, 1, -1, 1, -1, 1],
        )
        .expect("brick");
        WorldSdfPhysicsBridge::new(SdfPhysicsBridgeBudget::new(1, 8, 80).expect("budget"))
            .stage(WorldSdfSnapshot::new(
                WorldSdfTransform::new([0; 3], 100, 100).expect("transform"),
                1,
                grid,
            ))
            .expect("snapshot")
    }

    fn snapshot_with_known_separation_and_missing_region() -> SdfPhysicsSnapshot {
        let mut grid = SparseSdfBrickGrid::new(4, 1).expect("grid");
        for brick_x in 0..=1 {
            let samples = (0..64)
                .map(|index| {
                    let local_x = index % 4;
                    let x = brick_x * 4 + local_x;
                    if x == 0 {
                        -1
                    } else {
                        1
                    }
                })
                .collect();
            grid.insert_brick(BrickCoord::new(brick_x, 0, 0, 0), samples)
                .expect("brick");
        }
        WorldSdfPhysicsBridge::new(SdfPhysicsBridgeBudget::new(2, 128, 336).expect("budget"))
            .stage(WorldSdfSnapshot::new(
                WorldSdfTransform::new([0; 3], 100, 100).expect("transform"),
                1,
                grid,
            ))
            .expect("snapshot")
    }

    fn world_with_spheres(positions: &[Vec3]) -> (PhysicsWorld, BodyHandle, Vec<BodyHandle>) {
        let mut world = PhysicsWorld::default();
        let world_body = world.spawn(BodyDesc::static_at(Vec3::ZERO));
        let spheres = positions
            .iter()
            .map(|position| world.spawn(BodyDesc::dynamic_at(*position)))
            .collect();
        (world, world_body, spheres)
    }

    fn adapter(
        handles: &[BodyHandle],
        world_body: BodyHandle,
        limits: SdfSphereBatchLimits,
    ) -> SdfSphereContactBatch {
        SdfSphereContactBatch::new(
            handles
                .iter()
                .map(|body| SdfSphereBody {
                    body: *body,
                    radius_mm: 100,
                })
                .collect(),
            world_body,
            [0; 3],
            0,
            limits,
        )
        .expect("adapter")
    }

    fn limits(count: usize) -> SdfSphereBatchLimits {
        SdfSphereBatchLimits::new(count, count, count, count * size_of::<ContactManifold>())
    }

    #[test]
    fn construction_sorts_canonically_and_rejects_duplicates_and_zero_radius() {
        let (world, world_body, handles) = world_with_spheres(&[Vec3::ZERO; 2]);
        let batch = SdfSphereContactBatch::new(
            vec![
                SdfSphereBody {
                    body: handles[1],
                    radius_mm: 20,
                },
                SdfSphereBody {
                    body: handles[0],
                    radius_mm: 10,
                },
            ],
            world_body,
            [0; 3],
            0,
            limits(2),
        )
        .expect("canonical batch");
        assert_eq!(
            batch
                .spheres()
                .iter()
                .map(|sphere| sphere.body)
                .collect::<Vec<_>>(),
            handles
        );

        assert!(matches!(
            SdfSphereContactBatch::new(
                vec![SdfSphereBody { body: handles[0], radius_mm: 1 }; 2],
                world_body,
                [0; 3],
                0,
                limits(2),
            ),
            Err(SdfSphereBatchError::DuplicateSphereHandle(body)) if body == handles[0]
        ));
        assert!(matches!(
            SdfSphereContactBatch::new(
                vec![SdfSphereBody { body: handles[0], radius_mm: 0 }],
                world_body,
                [0; 3],
                0,
                limits(1),
            ),
            Err(SdfSphereBatchError::ZeroRadius(body)) if body == handles[0]
        ));
        assert!(matches!(
            SdfSphereContactBatch::new(
                vec![SdfSphereBody { body: world_body, radius_mm: 10 }],
                world_body,
                [0; 3],
                0,
                limits(1),
            ),
            Err(SdfSphereBatchError::DuplicateSphereHandle(body)) if body == world_body
        ));
        assert!(matches!(
            SdfSphereContactBatch::new(
                vec![SdfSphereBody {
                    body: BodyHandle::INVALID,
                    radius_mm: 10,
                }],
                world_body,
                [0; 3],
                0,
                limits(1),
            ),
            Err(SdfSphereBatchError::InvalidSphereHandle)
        ));
        assert!(world.bodies.contains(world_body));
    }

    #[test]
    fn construction_enforces_work_contact_and_scratch_budgets() {
        let (_world, world_body, handles) = world_with_spheres(&[Vec3::ZERO; 2]);
        let descriptors = || {
            handles
                .iter()
                .map(|body| SdfSphereBody {
                    body: *body,
                    radius_mm: 10,
                })
                .collect()
        };
        assert!(matches!(
            SdfSphereContactBatch::new(
                descriptors(),
                world_body,
                [0; 3],
                0,
                SdfSphereBatchLimits::new(2, 2, 1, 2 * size_of::<ContactManifold>()),
            ),
            Err(SdfSphereBatchError::WorkLimitExceeded {
                required: 2,
                limit: 1
            })
        ));
        assert!(matches!(
            SdfSphereContactBatch::new(
                descriptors(),
                world_body,
                [0; 3],
                0,
                SdfSphereBatchLimits::new(2, 1, 2, 2 * size_of::<ContactManifold>()),
            ),
            Err(SdfSphereBatchError::ContactLimitExceeded {
                required: 2,
                limit: 1
            })
        ));
        assert!(matches!(
            SdfSphereContactBatch::new(
                descriptors(),
                world_body,
                [0; 3],
                0,
                SdfSphereBatchLimits::new(2, 2, 2, size_of::<ContactManifold>()),
            ),
            Err(SdfSphereBatchError::ScratchBudgetExceeded { .. })
        ));
    }

    #[test]
    fn generates_contacts_and_counts_separation_unknown_and_miss() {
        let (world, world_body, handles) = world_with_spheres(&[
            Vec3::new(0.05, 0.05, 0.05),
            Vec3::new(0.5, 0.05, 0.05),
            Vec3::new(1.0, 0.05, 0.05),
        ]);
        let mut batch = SdfSphereContactBatch::new(
            vec![
                SdfSphereBody {
                    body: handles[0],
                    radius_mm: 100,
                },
                SdfSphereBody {
                    body: handles[1],
                    radius_mm: 10,
                },
                SdfSphereBody {
                    body: handles[2],
                    radius_mm: 100,
                },
            ],
            world_body,
            [0; 3],
            0,
            limits(3),
        )
        .expect("batch");
        let mut output = Vec::new();
        let report = batch
            .generate(
                &snapshot_with_known_separation_and_missing_region(),
                &world,
                &mut output,
            )
            .expect("batch");
        assert_eq!(
            (
                report.queries,
                report.contacts,
                report.separated,
                report.unknown,
                report.misses
            ),
            (3, 1, 1, 1, 0)
        );
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].body_b, handles[0]);
    }

    #[test]
    fn stale_body_position_overflow_and_query_error_leave_output_unchanged() {
        let (mut world, world_body, handles) =
            world_with_spheres(&[Vec3::new(0.05, 0.05, 0.05); 2]);
        let mut batch = SdfSphereContactBatch::new(
            handles
                .iter()
                .map(|body| SdfSphereBody {
                    body: *body,
                    radius_mm: 100,
                })
                .collect(),
            world_body,
            [0; 3],
            1,
            limits(2),
        )
        .expect("invalid-mip probe batch");
        let sentinel = ContactManifold::new(handles[0], handles[1], Vec3::X);
        let mut output = vec![sentinel];
        assert!(matches!(
            batch.generate(&snapshot(), &world, &mut output),
            Err(SdfSphereBatchError::Query(SdfBridgeError::GridQuery(
                prism_world_residency::SdfError::InvalidMip
            )))
        ));
        assert_eq!(output, vec![sentinel]);

        batch.mip = 0;
        let _ = world.remove_body(handles[1]);
        assert!(matches!(
            batch.generate(&snapshot(), &world, &mut output),
            Err(SdfSphereBatchError::InvalidSphereBody(body)) if body == handles[1]
        ));
        assert_eq!(output, vec![sentinel]);

        let (mut world, world_body, handles) = world_with_spheres(&[Vec3::MAX]);
        let mut batch = adapter(&handles, world_body, limits(1));
        assert!(matches!(
            batch.generate(&snapshot(), &world, &mut output),
            Err(SdfSphereBatchError::PositionOutOfRange(body)) if body == handles[0]
        ));
        assert_eq!(output, vec![sentinel]);
        world.bodies.set_position(handles[0], Vec3::splat(f32::NAN));
        assert!(matches!(
            batch.generate(&snapshot(), &world, &mut output),
            Err(SdfSphereBatchError::InvalidPosition(body)) if body == handles[0]
        ));
        assert_eq!(output, vec![sentinel]);
    }

    #[test]
    fn live_immovable_world_body_and_repeated_calls_reuse_scratch_and_output_capacity() {
        let (world, world_body, handles) = world_with_spheres(&[Vec3::new(0.05, 0.05, 0.05)]);
        let mut batch = adapter(&handles, world_body, limits(1));
        let scratch_ptr = batch.scratch.as_ptr();
        let mut output = Vec::with_capacity(4);
        let capacity = output.capacity();
        let first = batch
            .generate(&snapshot(), &world, &mut output)
            .expect("first");
        let second = batch
            .generate(&snapshot(), &world, &mut output)
            .expect("second");
        assert_eq!(first.contacts, 1);
        assert_eq!(second.contacts, 1);
        assert_eq!(batch.scratch.as_ptr(), scratch_ptr);
        assert_eq!(output.capacity(), capacity);
        assert_eq!(first.required_scratch_bytes, size_of::<ContactManifold>());
        assert_eq!(
            output[0].points()[0],
            ContactPoint::new(
                Vec3::new(0.05, 0.05, 0.05),
                Vec3::new(-0.05, 0.05, 0.05),
                0.1,
            )
        );
    }

    #[test]
    fn adapter_manifolds_resolve_overlapping_sphere_through_xpbd_hook() {
        use prism_physics_core::XpbdSolver;

        let snapshot = snapshot();
        let (mut world, world_body, handles) = world_with_spheres(&[Vec3::new(0.05, 0.05, 0.05)]);
        let mut batch = adapter(&handles, world_body, limits(1));
        let solver = XpbdSolver::new();
        let mut generated = 0;
        solver.step_with_contact_generator(&mut world, 1.0 / 60.0, 2, |physics, manifolds| {
            generated += 1;
            batch
                .generate(&snapshot, physics, manifolds)
                .expect("sphere batch generates contacts");
        });

        assert_eq!(generated, 2, "adapter is invoked once per XPBD substep");
        let resolved = world
            .bodies
            .position(handles[0])
            .expect("dynamic sphere remains live");
        assert!(
            resolved.x >= 0.149,
            "adapter contact must push the sphere outward: {resolved:?}"
        );
    }
}
