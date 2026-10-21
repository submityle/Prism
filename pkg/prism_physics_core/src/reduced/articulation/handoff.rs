//! Atomic handoff from a degraded reduced articulation to maximum coordinates.

use crate::collider::compound::CompoundLeafPolicy;
use crate::collider::{
    ColliderHandle, ColliderShape, CompoundChild, CompoundColliderError, PhysicsMaterial,
};
use crate::collision_filter::CollisionExclusionError;
use crate::joint::{
    AngleLimit, BoundedPd, JointAnchor, JointDesc, JointHandle, JointKind, LinearLimit,
    PrismaticJoint, RevoluteJoint, RevoluteJointState, SphericalJoint,
};
use crate::math::transform::Isometry;
use crate::reduced::articulation::ArticulationBodyState;
use crate::reduced::{
    ArticulationBase, ArticulationColliderDesc, ArticulationColliderId, ArticulationHandle,
    ArticulationJointKind, ArticulationLink,
};
use crate::state::body::{BodyDesc, BodyKind, MassProperties};
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Mat3, Quat, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationBodyMapping {
    pub link: Option<usize>,
    pub body: BodyHandle,
    /// Body-local (COM/principal axes) to the original authored link frame.
    /// The authored world pose is `body_world * link_from_body.inverse()`.
    pub link_from_body: Isometry,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArticulationJointMapping {
    pub link: usize,
    pub joint: JointHandle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationColliderMapping {
    pub source: ArticulationColliderId,
    pub link: Option<usize>,
    pub body: BodyHandle,
    pub source_shape: ColliderHandle,
    pub body_shape: ColliderHandle,
    /// Stable flattened child indices in `body_shape` belonging to this source.
    /// `None` identifies a direct, non-compound body shape.
    pub body_child_range: Option<std::ops::Range<u32>>,
    /// Original collider-local to authored link-local pose, not the new body's
    /// COM/principal frame. The baked body shape includes the frame conversion.
    pub local_pose: Isometry,
    pub material: PhysicsMaterial,
    pub self_collision: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationHandoffReport {
    pub articulation: ArticulationHandle,
    pub broken_link: usize,
    pub base: ArticulationBodyMapping,
    pub links: Vec<ArticulationBodyMapping>,
    pub joints: Vec<ArticulationJointMapping>,
    pub colliders: Vec<ArticulationColliderMapping>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedJointPolicy {
    /// Motor family has no equivalent scalar generalized coordinate contract.
    Motor {
        link: usize,
    },
    /// Independent generalized-velocity decay, armature, or non-equivalent
    /// fixed/spherical limits. Reduced damping divides the accelerated joint
    /// velocity by `1 + dt * damping`; a viscous bounded-PD effort is not an
    /// equivalent replacement.
    InertOrExtraParameters {
        link: usize,
    },
    /// Scalar motor parameters failed bounded-PD validation.
    InvalidMotor {
        link: usize,
    },
    Family {
        link: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArticulationHandoffError {
    InvalidArticulation,
    NotDegraded,
    InvalidSnapshot,
    /// The articulation's gravity is not a finite acceleration.
    UnsupportedGravity,
    UnsupportedMassProperties {
        link: Option<usize>,
    },
    CollisionExclusion(CollisionExclusionError),
    UnsupportedJointPolicy(UnsupportedJointPolicy),
    InvalidColliderGeometry {
        link: Option<usize>,
    },
    InvalidCompoundCollider(CompoundColliderError),
}

impl PhysicsWorld {
    /// Atomically converts a degraded articulation to maximum-coordinate
    /// bodies, surviving joints, and ordinary body colliders.
    ///
    /// Scalar motor feedback damping is preserved as effort. Independent
    /// reduced velocity decay on a surviving joint has no equivalent policy
    /// here and is rejected without mutating the world. Policies on the removed
    /// broken joint do not need conversion.
    pub fn handoff_articulation(
        &mut self,
        handle: ArticulationHandle,
    ) -> Result<ArticulationHandoffReport, ArticulationHandoffError> {
        let source = self
            .articulations
            .get(handle)
            .ok_or(ArticulationHandoffError::InvalidArticulation)?;
        let broken_link = source
            .broken_joint()
            .ok_or(ArticulationHandoffError::NotDegraded)?;
        let links = source.links().to_vec();
        let states = source.maximal_body_states();
        if broken_link >= links.len() || states.len() != links.len() + 1 {
            return Err(ArticulationHandoffError::InvalidSnapshot);
        }
        let attachments: Vec<_> = self
            .articulation_colliders
            .iter()
            .filter(|(_, desc)| desc.articulation == handle)
            .map(|(id, desc)| (id, desc.clone()))
            .collect();

        validate_states(&states, source.base_mode())?;
        let gravity = source.gravity();
        if !gravity.is_finite() {
            return Err(ArticulationHandoffError::UnsupportedGravity);
        }
        validate_attachments(&self.shapes, &attachments, links.len())?;
        validate_joint_policies(&links, broken_link)?;
        // Stage validated force laws before any allocation or topology change.
        let motors = links
            .iter()
            .enumerate()
            .map(|(link, model)| {
                if link == broken_link {
                    Ok(None)
                } else {
                    scalar_motor(link, model)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let frames = states
            .iter()
            .map(prepare_body_frame)
            .collect::<Result<Vec<_>, _>>()?;
        let joint_frames = links
            .iter()
            .enumerate()
            .map(|(link, model)| {
                if link == broken_link {
                    return Ok(None);
                }
                let parent = model.parent.map_or(0, |parent| parent + 1);
                let a = frames[parent].link_from_body.inverse().mul(&Isometry::new(
                    model.x_tree.translation,
                    model.x_tree.rotation.inverse(),
                ));
                let b = frames[link + 1].link_from_body.inverse();
                if !valid_frame(a) || !valid_frame(b) {
                    return Err(ArticulationHandoffError::InvalidSnapshot);
                }
                Ok(Some((a, b)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Validate all source geometry, including identity attachments. Neither the world nor
        // its append-only shape registry is cloned: cost is proportional to
        // this articulation and its expanded collision leaves.
        let attachment_poses = attachments
            .iter()
            .map(|(_, attachment)| {
                let index = attachment.link.map_or(0, |link| link + 1);
                let frame = frames[index].link_from_body;
                let pose = if frame == Isometry::IDENTITY {
                    attachment.local_pose
                } else {
                    frame.inverse().mul(&attachment.local_pose)
                };
                if !valid_frame(pose) {
                    return Err(ArticulationHandoffError::InvalidSnapshot);
                }
                Ok(pose)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut grouped = vec![Vec::new(); states.len()];
        for (source, (_, attachment)) in attachments.iter().enumerate() {
            grouped[attachment.link.map_or(0, |link| link + 1)].push(source);
        }
        let mut source_ranges = vec![None; attachments.len()];
        let mut staged_shapes = Vec::with_capacity(states.len());
        for sources in &grouped {
            let prepared = if sources.len() == 1 {
                let source = sources[0];
                Some(prepare_body_shape(
                    &self.shapes,
                    &attachments[source].1,
                    attachment_poses[source],
                )?)
            } else if sources.is_empty() {
                None
            } else {
                let children: Vec<_> = sources
                    .iter()
                    .map(|&source| CompoundChild {
                        collider: attachments[source].1.shape,
                        local_pose: attachment_poses[source],
                    })
                    .collect();
                let policies: Vec<_> = sources
                    .iter()
                    .map(|&source| attachment_policy(&attachments[source].1))
                    .collect();
                Some(PreparedBodyShape::Compound(
                    self.shapes
                        .prepare_compound_with_policies(&children, &policies)
                        .map_err(ArticulationHandoffError::InvalidCompoundCollider)?,
                ))
            };
            let compound = matches!(prepared, Some(PreparedBodyShape::Compound(_)))
                || sources.first().is_some_and(|&source| {
                    matches!(
                        self.shapes.get(attachments[source].1.shape),
                        Some(ColliderShape::Compound { .. })
                    )
                });
            if compound {
                let mut first = 0;
                for &source in sources {
                    let count = match self.shapes.get(attachments[source].1.shape) {
                        Some(ColliderShape::Compound { compound, .. }) => {
                            self.shapes
                                .compound(*compound)
                                .expect("validated compound")
                                .children()
                                .len() as u32
                        }
                        _ => 1,
                    };
                    source_ranges[source] = Some(first..first + count);
                    first += count;
                }
            }
            staged_shapes.push(prepared);
        }
        let mut body_shapes = vec![None; states.len()];
        let mut body_materials = vec![PhysicsMaterial::DEFAULT; states.len()];
        // Include collider-free members too: attaching geometry later must not
        // accidentally change which original-tree pairs may collide.
        let mut self_collision = vec![false; states.len()];
        for (_, attachment) in &attachments {
            let index = attachment.link.map_or(0, |link| link + 1);
            body_materials[index] = attachment.material;
            self_collision[index] |= attachment.self_collision;
        }
        let mut body_descs = states
            .iter()
            .enumerate()
            .map(|(index, state)| body_desc(state, None, body_materials[index], &frames[index]))
            .collect::<Result<Vec<_>, _>>()?;
        // A pose cannot encode the winding number. Validate and seed the
        // continuous hinge history before allocating anything in the world.
        // Temporary distinct handles are rebound to the new bodies below.
        let revolute_states = links
            .iter()
            .enumerate()
            .map(|(link, model)| {
                let ArticulationJointKind::Revolute { axis } = model.joint.kind else {
                    return Ok(None);
                };
                let Some((a, b)) = joint_frames[link] else {
                    return Ok(None);
                };
                let parent = model.parent.map_or(0, |parent| parent + 1);
                let mut state = RevoluteJointState::default();
                if !state.seed(
                    f64::from(source.positions()[source.dof_offsets[link]]),
                    axis,
                    JointAnchor::new(BodyHandle::new(0, 0), a.translation, a.rotation),
                    JointAnchor::new(BodyHandle::new(1, 0), b.translation, b.rotation),
                    [
                        body_descs[parent].orientation,
                        body_descs[link + 1].orientation,
                    ],
                ) {
                    return Err(ArticulationHandoffError::InvalidSnapshot);
                }
                Ok(Some(state))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.collision_exclusions()
            .validate_group_capacity()
            .map_err(ArticulationHandoffError::CollisionExclusion)?;

        // All fallible validation and shape construction is complete. From
        // this point onward the narrow mutations below cannot reject input.
        for (index, prepared) in staged_shapes.into_iter().enumerate() {
            let Some(prepared) = prepared else {
                continue;
            };
            let shape = match prepared {
                PreparedBodyShape::Existing(shape) => shape,
                PreparedBodyShape::Plane(shape) => self.shapes.insert(shape),
                PreparedBodyShape::Compound(compound) => {
                    self.shapes.insert_prepared_compound(compound)
                }
            };
            body_shapes[index] = Some(shape);
            body_descs[index].collider = Some(shape);
        }
        let mut body_handles = Vec::with_capacity(states.len());
        for desc in body_descs {
            let body = self.spawn(desc);
            // Pin even an equal value: later changes to world gravity must not
            // silently change the source articulation's gravity policy.
            assert!(self.bodies.set_gravity_override(body, Some(gravity)));
            body_handles.push(body);
        }
        // All handles are freshly allocated, distinct, live and ungrouped;
        // allocator exhaustion was checked before any world mutation.
        let members: Vec<_> = body_handles.iter().copied().zip(self_collision).collect();
        self.install_collision_exclusion_group(&members)
            .expect("validated capacity and fresh handoff bodies");

        let base = ArticulationBodyMapping {
            link: None,
            body: body_handles[0],
            link_from_body: frames[0].link_from_body,
        };
        let links_mapping = (0..links.len())
            .map(|link| ArticulationBodyMapping {
                link: Some(link),
                body: body_handles[link + 1],
                link_from_body: frames[link + 1].link_from_body,
            })
            .collect();
        let mut joint_mappings = Vec::with_capacity(links.len());
        for (link, model) in links.iter().enumerate() {
            if link == broken_link {
                continue;
            }
            let parent_body = model
                .parent
                .map_or(body_handles[0], |parent| body_handles[parent + 1]);
            let child_body = body_handles[link + 1];
            let (a, b) = joint_frames[link].expect("validated surviving joint frames");
            let mut desc = joint_desc(model, parent_body, child_body, a, b);
            desc.bounded_pd = motors[link];
            let joint = self.spawn_joint(desc);
            if let Some(mut state) = revolute_states[link] {
                state.remap_bodies(parent_body, child_body);
                self.joints
                    .get_mut(joint)
                    .expect("fresh handoff joint")
                    .revolute_state = state;
            }
            joint_mappings.push(ArticulationJointMapping { link, joint });
        }

        let collider_mappings = attachments
            .iter()
            .enumerate()
            .map(|(source, (id, desc))| {
                let index = desc.link.map_or(0, |link| link + 1);
                ArticulationColliderMapping {
                    source: *id,
                    link: desc.link,
                    body: body_handles[index],
                    source_shape: desc.shape,
                    body_shape: body_shapes[index].expect("validated attachment shape"),
                    body_child_range: source_ranges[source].clone(),
                    local_pose: desc.local_pose,
                    material: desc.material,
                    self_collision: desc.self_collision,
                }
            })
            .collect();

        if !self.remove_articulation(handle) {
            return Err(ArticulationHandoffError::InvalidArticulation);
        }
        Ok(ArticulationHandoffReport {
            articulation: handle,
            broken_link,
            base,
            links: links_mapping,
            joints: joint_mappings,
            colliders: collider_mappings,
        })
    }
}

fn validate_states(
    states: &[ArticulationBodyState],
    base_mode: ArticulationBase,
) -> Result<(), ArticulationHandoffError> {
    for (index, state) in states.iter().enumerate() {
        let is_base = index == 0;
        if state.link.is_some() != !is_base
            || !valid_frame(state.pose)
            || !state.center_of_mass.is_finite()
            || !state.linear_velocity.is_finite()
            || !state.angular_velocity.is_finite()
        {
            return Err(ArticulationHandoffError::InvalidSnapshot);
        }
        if is_base && state.fixed != (base_mode == ArticulationBase::Fixed) {
            return Err(ArticulationHandoffError::InvalidSnapshot);
        }
        if !state.fixed
            && (!state.inertia.mass.is_finite()
                || state.inertia.mass <= 0.0
                || !state.inertia.mass.recip().is_finite()
                || super::validate_inertia(&state.inertia).is_err())
        {
            return Err(ArticulationHandoffError::UnsupportedMassProperties { link: state.link });
        }
    }
    Ok(())
}

fn valid_frame(pose: Isometry) -> bool {
    pose.translation.is_finite()
        && pose.rotation.is_finite()
        && (pose.rotation.length_squared() - 1.0).abs() <= 1.0e-4
}

struct BodyFrame {
    link_from_body: Isometry,
    mass_properties: MassProperties,
}

fn prepare_body_frame(
    state: &ArticulationBodyState,
) -> Result<BodyFrame, ArticulationHandoffError> {
    if state.fixed {
        return Ok(BodyFrame {
            link_from_body: Isometry::IDENTITY,
            mass_properties: MassProperties::zero(),
        });
    }
    let error = ArticulationHandoffError::UnsupportedMassProperties { link: state.link };
    let tensor = state.inertia.inertia_com;
    // Keep authored axes when the tensor is diagonal, but apply the same
    // relative null-mode policy as the spectral path and reduced solver.
    // Otherwise an arbitrarily small frame rotation can change a locked mode
    // into an enormous inverse inertia during handoff.
    let diagonal = (0..3).all(|i| (0..3).all(|j| i == j || tensor[i][j] == 0.0));
    let (rotation, values) = if diagonal {
        (
            Quat::IDENTITY,
            [
                tensor[0][0] as f64,
                tensor[1][1] as f64,
                tensor[2][2] as f64,
            ],
        )
    } else {
        let (values, axes) = super::symmetric_spectrum(tensor, 3).ok_or(error)?;
        let mut basis = Mat3::from_cols(
            Vec3::new(axes[0][0] as f32, axes[1][0] as f32, axes[2][0] as f32),
            Vec3::new(axes[0][1] as f32, axes[1][1] as f32, axes[2][1] as f32),
            Vec3::new(axes[0][2] as f32, axes[1][2] as f32, axes[2][2] as f32),
        );
        // Eigenvectors are columns; a reflection cannot be encoded as a quat.
        if basis.determinant() < 0.0 {
            basis.z_axis = -basis.z_axis;
        }
        (Quat::from_mat3(&basis).normalize(), values)
    };
    let scale = values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if values.iter().any(|&value| value < -scale * 1.0e-6) {
        return Err(error);
    }
    let moment = |value: f64| {
        if value <= scale * 1.0e-10 {
            0.0
        } else {
            value as f32
        }
    };
    let moments = Vec3::new(moment(values[0]), moment(values[1]), moment(values[2]));
    if !moments.is_finite()
        || moments.min_element() < 0.0
        || moments
            .to_array()
            .iter()
            .any(|&value| value > 0.0 && !value.recip().is_finite())
    {
        return Err(error);
    }
    let link_from_body = Isometry::new(state.inertia.center_of_mass, rotation);
    if !valid_frame(link_from_body) {
        return Err(error);
    }
    Ok(BodyFrame {
        link_from_body,
        mass_properties: MassProperties {
            inv_mass: state.inertia.mass.recip(),
            inv_inertia: Vec3::new(
                inverse_or_zero(moments.x),
                inverse_or_zero(moments.y),
                inverse_or_zero(moments.z),
            ),
        },
    })
}

fn validate_attachments(
    shapes: &crate::collider::ShapeRegistry,
    attachments: &[(ArticulationColliderId, ArticulationColliderDesc)],
    link_count: usize,
) -> Result<(), ArticulationHandoffError> {
    for (_, attachment) in attachments {
        if attachment.link.is_some_and(|link| link >= link_count)
            || shapes.get(attachment.shape).is_none()
        {
            return Err(ArticulationHandoffError::InvalidSnapshot);
        }
        if let Some(ColliderShape::Plane { normal, offset }) = shapes.get(attachment.shape) {
            if !normal.is_finite()
                || (normal.length_squared() - 1.0).abs() > 1.0e-4
                || !offset.is_finite()
            {
                return Err(ArticulationHandoffError::InvalidColliderGeometry {
                    link: attachment.link,
                });
            }
        }
        if !valid_frame(attachment.local_pose)
            || !attachment.material.friction.is_finite()
            || attachment.material.friction < 0.0
            || !attachment.material.restitution.is_finite()
            || !(0.0..=1.0).contains(&attachment.material.restitution)
        {
            return Err(ArticulationHandoffError::InvalidColliderGeometry {
                link: attachment.link,
            });
        }
    }
    Ok(())
}

enum PreparedBodyShape {
    Existing(ColliderHandle),
    Plane(ColliderShape),
    Compound(crate::collider::compound::PreparedCompound),
}

fn prepare_body_shape(
    shapes: &crate::collider::ShapeRegistry,
    attachment: &ArticulationColliderDesc,
    body_local_pose: Isometry,
) -> Result<PreparedBodyShape, ArticulationHandoffError> {
    if let Some(&ColliderShape::Plane { normal, offset }) = shapes.get(attachment.shape) {
        if body_local_pose == Isometry::IDENTITY {
            return Ok(PreparedBodyShape::Existing(attachment.shape));
        }
        // An unbounded plane cannot be a compound leaf. Bake its equation
        // directly: n' = R n, d' = d + n' dot t. Accumulate in f64 so finite
        // translations with cancelling terms do not overflow midway.
        let normal = body_local_pose.rotation * normal;
        let translation = body_local_pose.translation;
        let offset = offset as f64
            + normal.x as f64 * translation.x as f64
            + normal.y as f64 * translation.y as f64
            + normal.z as f64 * translation.z as f64;
        let offset = offset as f32;
        if !normal.is_finite()
            || (normal.length_squared() - 1.0).abs() > 1.0e-4
            || !offset.is_finite()
        {
            return Err(ArticulationHandoffError::InvalidColliderGeometry {
                link: attachment.link,
            });
        }
        return Ok(PreparedBodyShape::Plane(ColliderShape::Plane {
            normal,
            offset,
        }));
    }
    // Preserve the direct shape fast path, unless nested leaf overrides need
    // rebinding to the source attachment's authoritative reduced policy.
    let nested_policy = match shapes.get(attachment.shape) {
        Some(ColliderShape::Compound { compound, .. }) => {
            shapes.compound(*compound).is_some_and(|data| {
                data.policies()
                    .iter()
                    .any(|policy| *policy != CompoundLeafPolicy::default())
            })
        }
        _ => false,
    };
    let children = [CompoundChild {
        collider: attachment.shape,
        local_pose: body_local_pose,
    }];
    let prepared = if nested_policy {
        shapes.prepare_compound_with_policies(&children, &[attachment_policy(attachment)])
    } else {
        // Keep the original one-source contract: the body's live material and
        // membership remain authoritative, including after user edits.
        shapes.prepare_compound(&children)
    }
    .map_err(ArticulationHandoffError::InvalidCompoundCollider)?;
    if body_local_pose == Isometry::IDENTITY && !nested_policy {
        Ok(PreparedBodyShape::Existing(attachment.shape))
    } else {
        Ok(PreparedBodyShape::Compound(prepared))
    }
}

fn attachment_policy(attachment: &ArticulationColliderDesc) -> CompoundLeafPolicy {
    CompoundLeafPolicy {
        material: Some(attachment.material),
        self_collision: Some(attachment.self_collision),
    }
}

fn body_desc(
    state: &ArticulationBodyState,
    collider: Option<ColliderHandle>,
    material: PhysicsMaterial,
    frame: &BodyFrame,
) -> Result<BodyDesc, ArticulationHandoffError> {
    let fixed = state.fixed;
    if fixed {
        return Ok(BodyDesc {
            kind: BodyKind::Static,
            position: state.pose.translation,
            orientation: state.pose.rotation,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass_properties: MassProperties::zero(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            collider,
            material,
            is_sensor: false,
            ccd: false,
        });
    }
    let pose = Isometry::new(
        state.center_of_mass,
        state.pose.rotation * frame.link_from_body.rotation,
    );
    if !valid_frame(pose) {
        return Err(ArticulationHandoffError::InvalidSnapshot);
    }
    Ok(BodyDesc {
        kind: BodyKind::Dynamic,
        position: pose.translation,
        orientation: pose.rotation,
        linear_velocity: state.linear_velocity,
        angular_velocity: state.angular_velocity,
        mass_properties: frame.mass_properties,
        linear_damping: 0.0,
        angular_damping: 0.0,
        collider,
        material,
        is_sensor: false,
        ccd: false,
    })
}

fn inverse_or_zero(value: f32) -> f32 {
    if value > 0.0 {
        value.recip()
    } else {
        0.0
    }
}

fn validate_joint_policies(
    links: &[ArticulationLink],
    broken_link: usize,
) -> Result<(), ArticulationHandoffError> {
    for (link, model) in links.iter().enumerate() {
        // Only surviving joints are converted to maximum coordinates.
        if link == broken_link {
            continue;
        }
        let joint = &model.joint;
        if joint.motor.is_some()
            && !matches!(
                joint.kind,
                ArticulationJointKind::Revolute { .. } | ArticulationJointKind::Prismatic { .. }
            )
        {
            return Err(ArticulationHandoffError::UnsupportedJointPolicy(
                UnsupportedJointPolicy::Motor { link },
            ));
        }
        // Reduced independent damping is an implicit, mass-independent decay
        // of generalized velocity, not the physical effort -damping * qdot.
        // Reject it before allocating anything rather than change its law.
        if joint.damping != 0.0
            || joint.armature != 0.0
            || (matches!(
                joint.kind,
                ArticulationJointKind::Fixed | ArticulationJointKind::Spherical
            ) && joint.limits.enabled)
        {
            return Err(ArticulationHandoffError::UnsupportedJointPolicy(
                UnsupportedJointPolicy::InertOrExtraParameters { link },
            ));
        }
    }
    Ok(())
}

fn scalar_motor(
    link: usize,
    model: &ArticulationLink,
) -> Result<Option<BoundedPd>, ArticulationHandoffError> {
    let Some(motor) = model.joint.motor else {
        return Ok(None);
    };
    // Only the active scalar x component participates, exactly as in the
    // reduced effort law. Motor feedback damping is a physical effort gain;
    // independent reduced velocity decay has already been rejected above.
    BoundedPd::new(
        f64::from(motor.target_position.x),
        f64::from(motor.target_velocity.x),
        f64::from(motor.stiffness),
        f64::from(motor.damping),
        f64::from(motor.max_force.x),
    )
    .map(Some)
    .map_err(|_| {
        ArticulationHandoffError::UnsupportedJointPolicy(UnsupportedJointPolicy::InvalidMotor {
            link,
        })
    })
}

fn joint_desc(
    link: &ArticulationLink,
    parent_body: BodyHandle,
    child_body: BodyHandle,
    parent_anchor: Isometry,
    child_anchor: Isometry,
) -> JointDesc {
    // `x_tree` maps the parent frame into the zero-coordinate child frame.
    // Keeping that frame on A and the live child frame on B preserves the
    // reduced coordinate itself: slider separation and hinge twist remain q.
    let anchor_a = JointAnchor::new(
        parent_body,
        parent_anchor.translation,
        parent_anchor.rotation,
    );
    let anchor_b = JointAnchor::new(child_body, child_anchor.translation, child_anchor.rotation);
    let compliance = 0.0;
    let kind = match link.joint.kind {
        ArticulationJointKind::Fixed => JointKind::Fixed(crate::joint::FixedJoint { compliance }),
        ArticulationJointKind::Spherical => JointKind::Spherical(SphericalJoint { compliance }),
        ArticulationJointKind::Revolute { axis } => {
            let mut joint =
                RevoluteJoint::new(axis.normalize_or_zero()).with_compliance(compliance);
            if link.joint.limits.enabled {
                joint.limit = Some(AngleLimit::new(
                    link.joint.limits.lower.x,
                    link.joint.limits.upper.x,
                ));
            }
            JointKind::Revolute(joint)
        }
        ArticulationJointKind::Prismatic { axis } => {
            let mut joint =
                PrismaticJoint::new(axis.normalize_or_zero()).with_compliance(compliance);
            if link.joint.limits.enabled {
                joint.limit = Some(LinearLimit::new(
                    link.joint.limits.lower.x,
                    link.joint.limits.upper.x,
                ));
            }
            JointKind::Prismatic(joint)
        }
    };
    JointDesc::new(anchor_a, anchor_b, kind)
}
