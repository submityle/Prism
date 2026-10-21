//! Reports produced when a breakable assembly is split by the solver boundary.

use super::*;

/// A rejected solver boundary retained for retry before further integration.
/// The completed substep's reaction loads must not be lost or measured twice.
#[derive(Clone, Debug, PartialEq)]
pub struct JointBoundaryFailure {
    /// Duration used to convert these impulses to forces.
    pub dt: Real,
    /// Exact measured impulses from the completed substep.
    pub loads: Vec<SolverJointLoad>,
    /// Most recent validation failure.
    pub error: BoundaryLoadError,
}

impl PhysicsWorld {
    /// Last rejected automatic boundary. While present, XPBD does not advance
    /// another substep; repair the reported world conflict and retry.
    pub fn joint_boundary_failure(&self) -> Option<&JointBoundaryFailure> {
        self.joint_boundary_failure.as_ref()
    }

    /// Retries retained loads without integrating or applying forces again.
    /// A failed retry preserves the loads and exposes its current error.
    pub fn retry_joint_boundary(&mut self) -> Result<(), BoundaryLoadError> {
        let Some(mut failure) = self.joint_boundary_failure.take() else {
            return Ok(());
        };
        match self.submit_joint_loads(failure.dt, &failure.loads) {
            Ok(_) => Ok(()),
            Err(error) => {
                failure.error = error;
                self.joint_boundary_failure = Some(failure);
                Err(error)
            }
        }
    }

    pub(crate) fn submit_solver_joint_loads(&mut self, dt: Real, loads: Vec<SolverJointLoad>) {
        if let Err(error) = self.submit_joint_loads(dt, &loads) {
            self.joint_boundary_failure = Some(JointBoundaryFailure { dt, loads, error });
        }
    }
}

/// One original assembly member and the fresh generational handle allocated
/// for it after an automatic break split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssemblyMemberRemap {
    /// The handle captured before the assembly was merged.
    pub original: BodyHandle,
    /// The live handle allocated by the post-break split.
    pub current: BodyHandle,
}

/// Ownership handoff emitted after a breakable assembly is split by
/// [`PhysicsWorld::submit_joint_loads`]. Members are kept in the same sorted
/// order used by the core merge/split transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssemblySplitReport {
    /// The retired merged proxy.
    pub proxy: BodyHandle,
    /// Exact original-to-current member mapping.
    pub members: Vec<AssemblyMemberRemap>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::assembly::{JointLoad, SolverJointLoad};
    use crate::dynamics::field::{FieldRegistry, FieldSample};
    use crate::joint::DistanceJoint;
    use crate::snapshot::world_hash::hash_world;
    use crate::solver::xpbd::FieldStepError;
    use crate::solver::{Solver, XpbdSolver};

    struct CountingSampler(usize);
    impl FieldBatchSampler for CountingSampler {
        type Error = ();
        fn sample_batch(
            &mut self,
            _: &FieldRegistry,
            bodies: &[FieldBody],
        ) -> Result<Vec<FieldSample>, Self::Error> {
            self.0 += 1;
            Ok(vec![FieldSample::default(); bodies.len()])
        }
    }

    fn rejected_boundary() -> (PhysicsWorld, BodyHandle) {
        let mut world = PhysicsWorld::with_gravity(-Vec3::Y);
        let members = [-1.0, 1.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::X * x)));
        let desc = JointDesc::distance(
            members[0],
            Vec3::ZERO,
            members[1],
            Vec3::ZERO,
            DistanceJoint::rigid(2.0),
        );
        let joint = world.joints.insert(desc);
        world
            .connect_assembly(
                members[0],
                members[1],
                Some(joint),
                desc.kind,
                Some(BreakLimit::new(1.0, 1.0)),
            )
            .unwrap();
        let proxy = world.merge_assembly(&members).unwrap();
        assert!(world.apply_force(proxy, Vec3::Z * 6.0, Vec3::ZERO));
        assert!(world.assemblies.remove_body(proxy));
        world.submit_solver_joint_loads(
            0.1,
            vec![SolverJointLoad {
                joint,
                load: JointLoad {
                    linear_impulse: Vec3::X,
                    angular_impulse: Vec3::ZERO,
                    substep_dt: 0.1,
                },
            }],
        );
        assert!(world.joint_boundary_failure().is_some());
        (world, proxy)
    }

    #[test]
    fn blocked_boundary_stops_all_xpbd_entry_points_before_force_or_provider_mutation() {
        let (mut world, _) = rejected_boundary();
        let before = format!("{world:?}");
        let hash = hash_world(&world).unwrap();
        let mut solver = XpbdSolver::new();
        let mut sampler = CountingSampler(0);
        for _ in 0..3 {
            let expected = world.joint_boundary_failure().unwrap().error;
            assert_eq!(world.retry_joint_boundary(), Err(expected));
            solver.step(&mut world, 0.3, 3);
            let mut contact_calls = 0;
            solver.step_with_contact_generator(&mut world, 0.3, 3, |_, _| contact_calls += 1);
            assert_eq!(contact_calls, 0);
            assert_eq!(
                solver.step_with_field_sampler(&mut world, 0.3, 3, &mut sampler),
                Err(FieldStepError::JointBoundary {
                    completed_substeps: 0,
                    source: expected
                })
            );
            assert_eq!(sampler.0, 0);
            assert_eq!(hash_world(&world).unwrap(), hash);
            assert_eq!(
                format!("{world:?}"),
                before,
                "same impulses and no repeated fatigue"
            );
        }
    }

    #[test]
    fn repaired_boundary_retries_exactly_once_before_resuming_integration() {
        let (mut world, proxy) = rejected_boundary();
        world.assemblies.add_body(proxy);
        let mut explicit = world.clone();
        assert_eq!(explicit.retry_joint_boundary(), Ok(()));
        assert!(explicit.joint_boundary_failure().is_none());
        let after_retry = format!("{explicit:?}");
        assert_eq!(explicit.retry_joint_boundary(), Ok(()));
        assert_eq!(format!("{explicit:?}"), after_retry);
        let mut solver = XpbdSolver::new();
        solver.step(&mut explicit, 0.3, 3);
        solver.step(&mut world, 0.3, 3);
        assert!(world.joint_boundary_failure().is_none());
        assert_eq!(hash_world(&world).unwrap(), hash_world(&explicit).unwrap());
        assert_eq!(world.drain_assembly_split_reports().len(), 1);
        assert!(world.drain_assembly_split_reports().is_empty());
        assert!(world.external_forces.is_empty());
        assert!(!world.bodies.contains(proxy));
    }
}
