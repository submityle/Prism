//! Pose-local articulated inverse mass. A tree is factorized once, then each
//! right-hand side is solved in two linear sweeps without cloning its model.

use super::*;
use core::cell::RefCell;

struct ImpulseSolveScratch {
    forces: Vec<SpatialForce>,
    bias: Vec<Vec6>,
    rhs: Vec<[Real; THREE]>,
    accelerations: Vec<Vec6>,
    generalized_torques: Vec<[Real; THREE]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_storage(solver: &ArticulationImpulseSolver) -> [(usize, usize); 5] {
        let scratch = solver.scratch.borrow();
        [
            (scratch.forces.as_ptr() as usize, scratch.forces.capacity()),
            (scratch.bias.as_ptr() as usize, scratch.bias.capacity()),
            (scratch.rhs.as_ptr() as usize, scratch.rhs.capacity()),
            (
                scratch.accelerations.as_ptr() as usize,
                scratch.accelerations.capacity(),
            ),
            (
                scratch.generalized_torques.as_ptr() as usize,
                scratch.generalized_torques.capacity(),
            ),
        ]
    }

    #[test]
    fn world_impulses_into_reuses_storage_and_resets_each_rhs() {
        for mode in [ArticulationBase::Fixed, ArticulationBase::Floating] {
            let inertia = SpatialInertia::diagonal(2.0, Vec3::splat(0.1), Vec3::ONE);
            let mut builder = ArticulationBuilder::new(mode, inertia).with_base_pose(
                Isometry::new(Vec3::new(1.0, -2.0, 3.0), Quat::from_rotation_y(0.4)),
            );
            for (parent, joint) in [
                (None, ArticulationJoint::revolute(Vec3::Z)),
                (Some(0), ArticulationJoint::spherical()),
                (Some(0), ArticulationJoint::prismatic(Vec3::X)),
                (None, ArticulationJoint::fixed()),
            ] {
                builder
                    .add_link(
                        parent,
                        SpatialTransform::new(Quat::from_rotation_x(0.3), Vec3::Y),
                        inertia,
                        joint,
                    )
                    .unwrap();
            }
            let model = builder.build().unwrap();
            let solver = ArticulationImpulseSolver::new(&model);
            let storage = scratch_storage(&solver);
            let impulses = [
                (Some(1), Vec3::new(2.0, 1.0, -1.0), Vec3::X),
                (None, Vec3::Y, Vec3::Z),
                (Some(1), Vec3::Z, -Vec3::Y),
                (Some(2), Vec3::X, Vec3::ONE),
                (Some(3), Vec3::ONE, -Vec3::Z),
            ];
            // Compare against the original world-wrench conversion plus solve.
            let mut forces = vec![SpatialForce::default(); model.link_count()];
            let mut base_force = SpatialForce::default();
            for &(link, point, impulse) in &impulses {
                let inverse = solver.body_pose(link).inverse();
                let force = SpatialForce::at_point(
                    inverse.transform_point(point),
                    inverse.transform_vector(impulse),
                );
                let target = link.map(|i| &mut forces[i]).unwrap_or(&mut base_force);
                target.torque += force.torque;
                target.force += force.force;
            }
            let expected = solver.solve(&forces, base_force);
            let mut result = ContactImpulseResponse {
                generalized_velocity_change: Vec::with_capacity(model.dof_count()),
                base_velocity_change: SpatialMotion::default(),
            };
            let output = (
                result.generalized_velocity_change.as_ptr(),
                result.generalized_velocity_change.capacity(),
            );
            for _ in 0..8 {
                solver.solve_world_impulses_into(&impulses, &mut result);
                assert_eq!(result, expected);
                assert_eq!(solver.solve_world_impulses(&impulses), expected);
                // A biased dynamics solve must not contaminate later contacts.
                solver.solve_biased(&forces[..1], base_force, &[[0.25; SIX]], &[[0.5; THREE]]);
                result.generalized_velocity_change.fill(Real::NAN);
                solver.solve_world_impulses_into(&[], &mut result);
                assert_eq!(result, ContactImpulseResponse::zero(model.dof_count()));
                assert_eq!(scratch_storage(&solver), storage);
                assert_eq!(
                    (
                        result.generalized_velocity_change.as_ptr(),
                        result.generalized_velocity_change.capacity(),
                    ),
                    output,
                );
                // Resizing within existing capacity also needs no allocation.
                result.generalized_velocity_change.clear();
            }
        }
    }

    #[test]
    fn world_impulses_into_handles_zero_dofs_and_empty_trees() {
        for mode in [ArticulationBase::Fixed, ArticulationBase::Floating] {
            for with_link in [false, true] {
                let inertia = SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::ONE);
                let mut builder = ArticulationBuilder::new(mode, inertia);
                if with_link {
                    builder
                        .add_link(
                            None,
                            SpatialTransform::IDENTITY,
                            inertia,
                            ArticulationJoint::fixed(),
                        )
                        .unwrap();
                }
                let solver = ArticulationImpulseSolver::new(&builder.build().unwrap());
                let mut result = ContactImpulseResponse::zero(3);
                let capacity = result.generalized_velocity_change.capacity();
                for _ in 0..2 {
                    let link = if with_link { Some(0) } else { None };
                    solver.solve_world_impulses_into(&[(link, Vec3::ZERO, Vec3::X)], &mut result);
                    assert!(result.generalized_velocity_change.is_empty());
                    let expected_linear = if mode == ArticulationBase::Fixed {
                        Vec3::ZERO
                    } else {
                        Vec3::X / if with_link { 4.0 } else { 2.0 }
                    };
                    assert_eq!(
                        result.base_velocity_change,
                        SpatialMotion::new(Vec3::ZERO, expected_linear)
                    );
                    solver.solve_world_impulses_into(&[], &mut result);
                    assert_eq!(result, ContactImpulseResponse::zero(0));
                    assert_eq!(result.generalized_velocity_change.capacity(), capacity);
                }
            }
        }
    }

    #[test]
    fn generalized_impulses_into_reuses_scratch_and_output_storage() {
        for mode in [ArticulationBase::Fixed, ArticulationBase::Floating] {
            let inertia = SpatialInertia::diagonal(2.0, Vec3::splat(0.1), Vec3::ONE);
            let mut builder = ArticulationBuilder::new(mode, inertia);
            for (parent, joint) in [
                (None, ArticulationJoint::revolute(Vec3::Z)),
                (Some(0), ArticulationJoint::spherical()),
                (Some(0), ArticulationJoint::prismatic(Vec3::X)),
                (None, ArticulationJoint::fixed()),
            ] {
                builder
                    .add_link(
                        parent,
                        SpatialTransform::new(Quat::from_rotation_x(0.3), Vec3::Y),
                        inertia,
                        joint,
                    )
                    .unwrap();
            }
            let model = builder.build().unwrap();
            let solver = ArticulationImpulseSolver::new(&model);
            let impulses = [0.25, -0.5, 0.75, -1.0, 1.25];
            let torques = [
                [impulses[0], 0.0, 0.0],
                [impulses[1], impulses[2], impulses[3]],
                [impulses[4], 0.0, 0.0],
                [0.0; THREE],
            ];
            let expected = solver.solve_biased(&[], SpatialForce::default(), &[], &torques);
            let storage = scratch_storage(&solver);
            let mut result = ContactImpulseResponse {
                generalized_velocity_change: Vec::with_capacity(model.dof_count()),
                base_velocity_change: SpatialMotion::default(),
            };
            let output = (
                result.generalized_velocity_change.as_ptr(),
                result.generalized_velocity_change.capacity(),
            );

            for _ in 0..8 {
                solver.solve_generalized_impulses_into(&impulses, &mut result);
                assert_eq!(result, expected);
                assert_eq!(scratch_storage(&solver), storage);
                assert_eq!(
                    (
                        result.generalized_velocity_change.as_ptr(),
                        result.generalized_velocity_change.capacity(),
                    ),
                    output,
                );

                result.generalized_velocity_change.fill(Real::NAN);
                solver.solve_generalized_impulses_into(&[], &mut result);
                assert_eq!(result, ContactImpulseResponse::zero(model.dof_count()));
                assert_eq!(scratch_storage(&solver), storage);
                assert_eq!(
                    (
                        result.generalized_velocity_change.as_ptr(),
                        result.generalized_velocity_change.capacity(),
                    ),
                    output,
                );
                result.generalized_velocity_change.clear();
            }
        }
    }
}

pub(crate) struct ArticulationImpulseSolver {
    parents: Vec<Option<usize>>,
    offsets: Vec<usize>,
    axes: Vec<JointAxes>,
    transforms: Vec<SpatialTransform>,
    u: Vec<[[Real; THREE]; SIX]>,
    inverse_d: Vec<Mat3N>,
    reduced: Vec<Mat6>,
    inverse_root: Mat6,
    poses: Vec<Isometry>,
    base_pose: Isometry,
    dofs: usize,
    // Pose-local, single-threaded workspace, reused for every contact RHS.
    scratch: RefCell<ImpulseSolveScratch>,
}

impl ArticulationImpulseSolver {
    pub(crate) fn body_pose(&self, link: Option<usize>) -> Isometry {
        link.map(|i| self.poses[i]).unwrap_or(self.base_pose)
    }

    pub(crate) fn new(model: &Articulation) -> Self {
        Self::new_with_implicit_damping(model, 0.0)
    }

    /// Builds the integration operator `M + dt * C`, where `C` is the
    /// diagonal joint-space viscous damping matrix. Contact and arbitrary
    /// impulse queries continue to use [`Self::new`] and therefore see the
    /// physical mass matrix `M`, including armature but excluding damping.
    pub(super) fn new_for_integration(model: &Articulation, dt: Real) -> Self {
        Self::new_with_implicit_damping(model, dt)
    }

    fn new_with_implicit_damping(model: &Articulation, dt: Real) -> Self {
        let n = model.link_count();
        let transforms: Vec<_> = (0..n).map(|i| model.xup(i)).collect();
        let axes: Vec<_> = (0..n).map(|i| model.joint_axes(i)).collect();
        let mut inertia: Vec<_> = model.links.iter().map(|l| l.inertia.matrix()).collect();
        let mut root = model.base_inertia.matrix();
        let mut u = vec![[[0.0; THREE]; SIX]; n];
        let mut inverse_d = vec![[[0.0; THREE]; THREE]; n];
        let mut reduced = vec![zero_mat6(); n];
        for i in (0..n).rev() {
            let count = axes[i].count;
            for col in 0..count {
                let column = mat6_vec(&inertia[i], axes[i].cols[col]);
                for row in 0..SIX {
                    u[i][row][col] = column[row];
                }
            }
            let mut d = [[0.0; THREE]; THREE];
            for row in 0..count {
                for col in 0..count {
                    for k in 0..SIX {
                        d[row][col] += axes[i].cols[row][k] * u[i][k][col];
                    }
                }
                d[row][row] += model.links[i].joint.armature + dt * model.links[i].joint.damping;
            }
            inverse_d[i] = invert_small(d, count);
            reduced[i] = mat6_sub(&inertia[i], &outer_from_cols(&u[i], inverse_d[i], count));
            let x = transforms[i].motion_matrix();
            let propagated = mat6_mul(&mat6_transpose(&x), &mat6_mul(&reduced[i], &x));
            if let Some(parent) = model.links[i].parent {
                mat6_add_in_place(&mut inertia[parent], &propagated);
            } else {
                mat6_add_in_place(&mut root, &propagated);
            }
        }
        Self {
            parents: model.links.iter().map(|l| l.parent).collect(),
            offsets: model.dof_offsets.clone(),
            axes,
            transforms,
            u,
            inverse_d,
            reduced,
            inverse_root: if model.base_mode == ArticulationBase::Floating {
                invert6(root)
            } else {
                zero_mat6()
            },
            poses: model.forward_kinematics(),
            base_pose: model.base_pose,
            dofs: model.dof_count(),
            scratch: RefCell::new(ImpulseSolveScratch {
                forces: vec![SpatialForce::default(); n],
                bias: vec![zero6(); n],
                rhs: vec![[0.0; THREE]; n],
                accelerations: vec![zero6(); n],
                generalized_torques: vec![[0.0; THREE]; n],
            }),
        }
    }

    pub(crate) fn solve(
        &self,
        forces: &[SpatialForce],
        base_force: SpatialForce,
    ) -> ContactImpulseResponse {
        self.solve_biased(forces, base_force, &[], &[])
    }

    /// `forces` includes external force minus the velocity-product bias.
    /// `c` is the kinematic acceleration bias; motors are present only when
    /// explicitly supplied, never in a contact inverse-mass evaluation.
    pub(super) fn solve_biased(
        &self,
        forces: &[SpatialForce],
        base_force: SpatialForce,
        c: &[Vec6],
        torques: &[[Real; THREE]],
    ) -> ContactImpulseResponse {
        let mut result = ContactImpulseResponse::zero(self.dofs);
        let mut scratch = self.scratch.borrow_mut();
        for (i, bias) in scratch.bias.iter_mut().enumerate() {
            *bias = vec6_scale(forces.get(i).copied().unwrap_or_default().as_array(), -1.0);
        }
        let ImpulseSolveScratch {
            bias,
            rhs,
            accelerations,
            ..
        } = &mut *scratch;
        self.solve_bias_into(
            bias,
            rhs,
            accelerations,
            vec6_scale(base_force.as_array(), -1.0),
            c,
            torques,
            &mut result,
        );
        result
    }

    /// Solve an initialized negative-force bias using the existing ABA sweeps.
    fn solve_bias_into(
        &self,
        bias: &mut [Vec6],
        rhs: &mut [[Real; THREE]],
        accelerations: &mut [Vec6],
        mut root_bias: Vec6,
        c: &[Vec6],
        torques: &[[Real; THREE]],
        result: &mut ContactImpulseResponse,
    ) {
        let n = self.parents.len();
        rhs.fill([0.0; THREE]);
        result.generalized_velocity_change.resize(self.dofs, 0.0);
        result.generalized_velocity_change.fill(0.0);
        for i in (0..n).rev() {
            let projected = self.axes[i].transpose_vec(bias[i]);
            let torque = torques.get(i).copied().unwrap_or([0.0; THREE]);
            for k in 0..self.axes[i].count {
                rhs[i][k] = torque[k] - projected[k];
            }
            let solved = multiply_small_vec(self.inverse_d[i], rhs[i], self.axes[i].count);
            let mut propagated = vec6_add(
                bias[i],
                mat6_vec(&self.reduced[i], c.get(i).copied().unwrap_or_else(zero6)),
            );
            for row in 0..SIX {
                for col in 0..self.axes[i].count {
                    propagated[row] += self.u[i][row][col] * solved[col];
                }
            }
            // Transform the complete bias, including U D^-1 u, into the
            // parent frame. Mixing these two frames breaks reciprocity.
            let propagated = self.transforms[i]
                .transform_force_transpose(SpatialForce::from_array(propagated))
                .as_array();
            if let Some(parent) = self.parents[i] {
                vec6_add_in_place(&mut bias[parent], propagated);
            } else {
                vec6_add_in_place(&mut root_bias, propagated);
            }
        }
        let base = vec6_scale(mat6_vec(&self.inverse_root, root_bias), -1.0);
        result.base_velocity_change = SpatialMotion::from_array(base);
        // Every acceleration is overwritten in parent-before-child order.
        for i in 0..n {
            let parent = self.parents[i].map(|p| accelerations[p]).unwrap_or(base);
            let mut acceleration = vec6_add(
                self.transforms[i]
                    .transform_motion(SpatialMotion::from_array(parent))
                    .as_array(),
                c.get(i).copied().unwrap_or_else(zero6),
            );
            let mut local_rhs = rhs[i];
            for col in 0..self.axes[i].count {
                for row in 0..SIX {
                    local_rhs[col] -= self.u[i][row][col] * acceleration[row];
                }
            }
            let solved = multiply_small_vec(self.inverse_d[i], local_rhs, self.axes[i].count);
            for k in 0..self.axes[i].count {
                result.generalized_velocity_change[self.offsets[i] + k] = solved[k];
            }
            vec6_add_in_place(&mut acceleration, self.axes[i].multiply_array(solved));
            accelerations[i] = acceleration;
        }
    }

    pub(crate) fn solve_world_impulses(
        &self,
        impulses: &[(Option<usize>, Vec3, Vec3)],
    ) -> ContactImpulseResponse {
        let mut result = ContactImpulseResponse::zero(self.dofs);
        self.solve_world_impulses_into(impulses, &mut result);
        result
    }

    /// Overwrite a response without allocating when its generalized-velocity
    /// vector already has capacity for this solver's DOFs (including first use).
    pub(crate) fn solve_world_impulses_into(
        &self,
        impulses: &[(Option<usize>, Vec3, Vec3)],
        result: &mut ContactImpulseResponse,
    ) {
        let mut scratch = self.scratch.borrow_mut();
        scratch.forces.fill(SpatialForce::default());
        let mut base = SpatialForce::default();
        for &(link, point, impulse) in impulses {
            let pose = link.map(|i| self.poses[i]).unwrap_or(self.base_pose);
            let inverse = pose.inverse();
            let force = SpatialForce::at_point(
                inverse.transform_point(point),
                inverse.transform_vector(impulse),
            );
            let target = link.map(|i| &mut scratch.forces[i]).unwrap_or(&mut base);
            target.torque += force.torque;
            target.force += force.force;
        }
        for i in 0..self.parents.len() {
            scratch.bias[i] = vec6_scale(scratch.forces[i].as_array(), -1.0);
        }
        let ImpulseSolveScratch {
            bias,
            rhs,
            accelerations,
            ..
        } = &mut *scratch;
        self.solve_bias_into(
            bias,
            rhs,
            accelerations,
            vec6_scale(base.as_array(), -1.0),
            &[],
            &[],
            result,
        );
    }

    /// Overwrite a response for impulses expressed directly in the
    /// articulation's generalized-coordinate order.  This is used by
    /// unilateral joint-limit rows which share the same articulated inverse
    /// mass as contact rows.
    pub(crate) fn solve_generalized_impulses_into(
        &self,
        impulses: &[Real],
        result: &mut ContactImpulseResponse,
    ) {
        let mut scratch = self.scratch.borrow_mut();
        scratch.bias.fill(zero6());
        scratch.generalized_torques.fill([0.0; THREE]);
        for (link, &offset) in self.offsets.iter().enumerate() {
            let count = self.axes[link].count;
            for local in 0..count {
                if let Some(&value) = impulses.get(offset + local) {
                    scratch.generalized_torques[link][local] = value;
                }
            }
        }
        let ImpulseSolveScratch {
            bias,
            rhs,
            accelerations,
            generalized_torques,
            ..
        } = &mut *scratch;
        self.solve_bias_into(
            bias,
            rhs,
            accelerations,
            zero6(),
            &[],
            generalized_torques,
            result,
        );
    }

    /// World velocity due to a generalized state at a frozen contact pose.
    pub(crate) fn point_velocity(
        &self,
        state: &ContactImpulseResponse,
        link: Option<usize>,
        point: Vec3,
    ) -> Vec3 {
        let base = state.base_velocity_change;
        let mut velocity = self.base_pose.rotation
            * (base.linear
                + base
                    .angular
                    .cross(self.base_pose.inverse().transform_point(point)));
        // Each ancestor contributes its world-space Jacobian column at the
        // same witness point. Reverse traversal needs no temporary stack.
        let mut cursor = link;
        while let Some(i) = cursor {
            let mut local = [0.0; THREE];
            for k in 0..self.axes[i].count {
                local[k] = state.generalized_velocity_change[self.offsets[i] + k];
            }
            let motion = SpatialMotion::from_array(self.axes[i].multiply_array(local));
            let pose = self.poses[i];
            velocity += pose.rotation
                * (motion.linear + motion.angular.cross(pose.inverse().transform_point(point)));
            cursor = self.parents[i];
        }
        velocity
    }
}
