//! Dependency-complete handoff contracts: real geometry and source identity,
//! policy at the contacted leaf, exact filtering, queries, CCD and rollback.

use glam::{Mat3, Quat, Vec3};
use prism_physics_core::collide::generate_contacts_in;
use prism_physics_core::collider::compound::{CompoundLeafPolicy, MAX_COMPOUND_CHILDREN};
use prism_physics_core::collider::{
    ColliderHandle, ColliderShape, CompoundChild, CompoundColliderError, PhysicsMaterial,
    ShapeRegistry,
};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::pipeline::detect_contacts;
use prism_physics_core::query::QueryFilter;
use prism_physics_core::reduced::articulation::handoff::ArticulationHandoffError;
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationColliderDesc, ArticulationHandle,
    ArticulationJoint, SpatialInertia, SpatialMotion, SpatialTransform,
};
use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};
use prism_physics_core::solver::xpbd::ContactConstraint;
use prism_physics_core::solver::{Solver, XpbdSolver};
use prism_physics_core::state::body::BodyDesc;
use prism_physics_core::PhysicsWorld;
use prism_physics_geometry::Ray;

fn inertia() -> SpatialInertia {
    SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE)
}

fn source(
    world: &mut PhysicsWorld,
    base: ArticulationBase,
    mass: SpatialInertia,
    pose: Isometry,
    link_offset: Vec3,
) -> ArticulationHandle {
    let mut builder = ArticulationBuilder::new(base, mass)
        .with_gravity(Vec3::ZERO)
        .with_base_pose(pose);
    builder
        .add_link(
            None,
            SpatialTransform::new(Quat::IDENTITY, link_offset),
            mass,
            ArticulationJoint::fixed(),
        )
        .unwrap();
    world.spawn_articulation(builder.build().unwrap())
}

fn attach(
    world: &mut PhysicsWorld,
    owner: ArticulationHandle,
    link: Option<usize>,
    shape: ColliderHandle,
    pose: Isometry,
    material: PhysicsMaterial,
    self_collision: bool,
) -> prism_physics_core::reduced::ArticulationColliderId {
    let mut desc = ArticulationColliderDesc::new(owner, link, shape);
    desc.local_pose = pose;
    desc.material = material;
    desc.self_collision = self_collision;
    world.attach_articulation_collider(desc).unwrap()
}

fn degrade(world: &mut PhysicsWorld, owner: ArticulationHandle) {
    assert!(world
        .articulations
        .get_mut(owner)
        .unwrap()
        .break_joint(0)
        .is_some());
}

fn close_vec(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 3.0e-5, "{a:?} != {b:?}");
}

fn close_pose(a: Isometry, b: Isometry) {
    close_vec(a.translation, b.translation);
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        close_vec(a.rotation * axis, b.rotation * axis);
    }
}

#[test]
fn nonzero_com_principal_axes_rotations_and_repeated_nested_sources_keep_world_geometry() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let axes = Mat3::from_quat(Quat::from_rotation_y(0.7));
        let tensor = axes * Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)) * axes.transpose();
        let mass = SpatialInertia::new(
            2.0,
            Vec3::new(0.2, -0.1, 0.3),
            tensor.transpose().to_cols_array_2d(),
        );
        let owner = source(
            &mut world,
            base,
            mass,
            Isometry::new(Vec3::new(3.0, -2.0, 1.0), Quat::from_rotation_x(0.4)),
            Vec3::Y,
        );
        if base == ArticulationBase::Floating {
            assert!(world
                .articulations
                .get_mut(owner)
                .unwrap()
                .set_base_velocity(SpatialMotion::new(
                    Vec3::new(0.2, -0.3, 0.4),
                    Vec3::new(1.0, 2.0, -1.0)
                )));
        }
        let primitive = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.2, 0.4, 0.6),
        });
        let inner = [
            CompoundChild {
                collider: primitive,
                local_pose: Isometry::new(Vec3::X, Quat::from_rotation_z(0.5)),
            },
            CompoundChild {
                collider: primitive,
                local_pose: Isometry::new(-Vec3::X, Quat::from_rotation_y(-0.2)),
            },
        ];
        // Deliberately conflicting nested metadata: reduced attachment policy
        // is authoritative and must cover every descendant of that source.
        let nested = world
            .shapes
            .insert_compound_with_policies(
                &inner,
                &[CompoundLeafPolicy {
                    material: Some(PhysicsMaterial::new(9.0, 1.0)),
                    self_collision: Some(true),
                }; 2],
            )
            .unwrap();
        let mut original = Vec::new();
        for link in [None, Some(0)] {
            for index in 0..2 {
                let pose = Isometry::new(
                    Vec3::new(0.3, index as f32 * 2.0, -0.2),
                    Quat::from_rotation_x(0.3 + index as f32),
                );
                let material = PhysicsMaterial::new(0.2 + index as f32, index as f32 * 0.4);
                let flag = index == 1;
                let id = attach(&mut world, owner, link, nested, pose, material, flag);
                original.push((id, link, pose, material, flag));
            }
        }
        degrade(&mut world, owner);
        let states = world
            .articulations
            .get(owner)
            .unwrap()
            .maximal_body_states();
        let report = world.handoff_articulation(owner).unwrap();
        assert!(report.joints.is_empty());
        assert!(world.articulations.get(owner).is_none());
        assert_eq!(report.colliders.len(), original.len());
        let leaves = world.collider_leaves(&QueryFilter::ALL);
        assert_eq!(leaves.len(), 8);
        for ((id, link, pose, material, flag), mapping) in original.iter().zip(&report.colliders) {
            assert_eq!(mapping.source, *id);
            assert_eq!(mapping.source_shape, nested);
            assert_eq!(mapping.local_pose, *pose);
            assert_eq!(mapping.material, *material);
            assert_eq!(mapping.self_collision, *flag);
            assert!(world.articulation_colliders.get(*id).is_none());
            let state = &states[link.map_or(0, |i| i + 1)];
            let range = mapping.body_child_range.clone().unwrap();
            assert_eq!(range.len(), 2);
            for (index, child) in inner.iter().enumerate() {
                let leaf = leaves
                    .iter()
                    .find(|leaf| {
                        leaf.body == mapping.body
                            && leaf.child_index == Some(range.start + index as u32)
                    })
                    .unwrap();
                close_pose(leaf.pose, state.pose.mul(pose).mul(&child.local_pose));
                assert_eq!(leaf.collider, primitive);
                assert_eq!(leaf.material, *material);
                assert_eq!(leaf.self_collision, *flag);
            }
        }
        assert_eq!(report.colliders[0].body_child_range, Some(0..2));
        assert_eq!(report.colliders[1].body_child_range, Some(2..4));
        assert_eq!(report.colliders[2].body_child_range, Some(0..2));
        for (mapping, state) in core::iter::once(&report.base)
            .chain(&report.links)
            .zip(&states)
        {
            let pose = Isometry::new(
                world.bodies.position(mapping.body).unwrap(),
                world.bodies.orientation(mapping.body).unwrap(),
            );
            close_pose(pose.mul(&mapping.link_from_body.inverse()), state.pose);
            if !state.fixed {
                close_vec(pose.translation, state.center_of_mass);
                close_vec(
                    world.bodies.linear_velocity(mapping.body).unwrap(),
                    state.linear_velocity,
                );
                close_vec(
                    world.bodies.angular_velocity(mapping.body).unwrap(),
                    state.angular_velocity,
                );
                assert!(
                    (world.bodies.mass_properties(mapping.body).unwrap().inv_mass - 0.5).abs()
                        < 1e-6
                );
            }
        }
    }
}

#[test]
fn per_leaf_self_collision_and_materials_reach_rigid_constraints_and_external_contacts() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let owner = source(
        &mut world,
        ArticulationBase::Fixed,
        inertia(),
        Isometry::IDENTITY,
        Vec3::Y * 0.7,
    );
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let materials = [
        PhysicsMaterial::new(0.1, 0.2),
        PhysicsMaterial::new(0.9, 0.8),
    ];
    for (link, flags) in [(None, [false, true]), (Some(0), [true, true])] {
        for i in 0..2 {
            attach(
                &mut world,
                owner,
                link,
                sphere,
                Isometry::from_translation(Vec3::X * (i as f32 * 4.0 - 2.0)),
                materials[i],
                flags[i],
            );
        }
    }
    degrade(&mut world, owner);
    let report = world.handoff_articulation(owner).unwrap();
    let a = report.base.body;
    let b = report.links[0].body;
    let contacts = detect_contacts(&world);
    assert_eq!(
        contacts.len(),
        1,
        "only the mutually enabled leaf pair may collide"
    );
    assert_eq!(contacts[0].child_indices, [Some(1), Some(1)]);
    assert_eq!(contacts[0].leaf_materials, [Some(materials[1]); 2]);
    let plane_material = PhysicsMaterial::new(0.4, 0.1);
    let plane = world.shapes.insert(ColliderShape::Plane {
        normal: Vec3::Y,
        offset: 0.4,
    });
    let external = world.spawn(
        BodyDesc::static_at(Vec3::ZERO)
            .with_collider(plane)
            .with_material(plane_material),
    );
    let contacts = detect_contacts(&world);
    let ground: Vec<_> = contacts
        .into_iter()
        .filter(|m| {
            (m.body_a == b && m.body_b == external) || (m.body_b == b && m.body_a == external)
        })
        .collect();
    assert_eq!(ground.len(), 2);
    let constraints = ContactConstraint::build(&world.bodies.solver_view_mut(), &ground);
    assert_eq!(constraints.len(), 2);
    for (m, c) in ground.iter().zip(&constraints) {
        let side = usize::from(m.body_b == b);
        let expected = materials[m.child_indices[side].unwrap() as usize].combine(plane_material);
        assert_eq!(c.friction, expected.friction);
        assert_eq!(c.restitution, expected.restitution);
    }
    assert!(!world
        .collision_exclusions()
        .allows_leaf_pair(a, a, Some(true), Some(true)));
    assert!(world
        .collision_exclusions()
        .allows_leaf_pair(a, external, Some(false), None));
}

#[test]
fn explicit_leaf_opt_in_is_not_lost_to_coarse_body_opt_out() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let shape = world
        .shapes
        .insert_compound_with_policies(
            &[CompoundChild {
                collider: sphere,
                local_pose: Isometry::IDENTITY,
            }],
            &[CompoundLeafPolicy {
                material: None,
                self_collision: Some(true),
            }],
        )
        .unwrap();
    let a = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_collider(shape));
    let b = world.spawn(BodyDesc::dynamic_at(Vec3::Y * 0.7).with_collider(shape));
    world
        .install_collision_exclusion_group(&[(a, false), (b, false)])
        .unwrap();
    assert!(!world.collision_pair_allowed(a, b));
    let contacts = detect_contacts(&world);
    assert_eq!(contacts.len(), 1);
    assert_eq!(contacts[0].leaf_self_collision, [Some(true); 2]);
}

#[test]
fn nested_overrides_flatten_and_contact_side_metadata_survives_reversal() {
    let mut shapes = ShapeRegistry::new();
    let sphere = shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let child = CompoundChild {
        collider: sphere,
        local_pose: Isometry::IDENTITY,
    };
    let inner_policy = CompoundLeafPolicy {
        material: Some(PhysicsMaterial::new(0.2, 0.3)),
        self_collision: Some(false),
    };
    let inner = shapes
        .insert_compound_with_policies(&[child], &[inner_policy])
        .unwrap();
    let outer_policy = CompoundLeafPolicy {
        material: Some(PhysicsMaterial::new(0.8, 0.9)),
        self_collision: None,
    };
    let outer = shapes
        .insert_compound_with_policies(
            &[CompoundChild {
                collider: inner,
                local_pose: Isometry::IDENTITY,
            }],
            &[outer_policy],
        )
        .unwrap();
    let expected = CompoundLeafPolicy {
        material: outer_policy.material,
        self_collision: inner_policy.self_collision,
    };
    let ColliderShape::Compound { compound, .. } = *shapes.get(outer).unwrap() else {
        panic!()
    };
    assert_eq!(shapes.compound(compound).unwrap().policies(), &[expected]);
    for reverse in [false, true] {
        let (a, b) = if reverse {
            (sphere, outer)
        } else {
            (outer, sphere)
        };
        let contacts = generate_contacts_in(
            shapes.get(a).unwrap(),
            &Isometry::IDENTITY,
            shapes.get(b).unwrap(),
            &Isometry::from_translation(Vec3::Y * 0.7),
            &shapes,
        );
        assert_eq!(contacts.len(), 1);
        let side = usize::from(reverse);
        assert_eq!(contacts[0].child_indices[side], Some(0));
        assert_eq!(contacts[0].leaf_materials[side], expected.material);
        assert_eq!(
            contacts[0].leaf_self_collision[side],
            expected.self_collision
        );
        assert_eq!(contacts[0].leaf_materials[1 - side], None);
    }
}

#[test]
fn leaf_queries_keep_duplicate_source_identity_and_do_not_fill_union_gaps() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let owner = source(
        &mut world,
        ArticulationBase::Fixed,
        inertia(),
        Isometry::IDENTITY,
        Vec3::Y,
    );
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    for i in 0..2 {
        attach(
            &mut world,
            owner,
            None,
            sphere,
            Isometry::from_translation(Vec3::X * (i as f32 * 4.0 - 2.0)),
            PhysicsMaterial::new(i as f32, i as f32),
            i != 0,
        );
    }
    degrade(&mut world, owner);
    let report = world.handoff_articulation(owner).unwrap();
    let ray = Ray::new(-Vec3::X * 5.0, Vec3::X);
    let hits = world.raycast_leaves(&ray, &QueryFilter::ALL);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].leaf.child_index, Some(0));
    assert_eq!(hits[1].leaf.child_index, Some(1));
    assert!(hits[0].time_of_impact < hits[1].time_of_impact);
    assert_eq!(world.raycast_all(&ray, &QueryFilter::ALL).len(), 1);
    assert_eq!(
        world.spherecast_leaves(&ray, 0.1, &QueryFilter::ALL).len(),
        2
    );
    let probe = ColliderShape::Sphere { radius: 0.1 };
    assert_eq!(
        world
            .shapecast_leaves(
                &probe,
                Isometry::from_translation(-Vec3::X * 5.0),
                Vec3::X * 10.0,
                &QueryFilter::ALL
            )
            .len(),
        2
    );
    assert!(world
        .overlap_shape_leaves(&probe, Isometry::IDENTITY, &QueryFilter::ALL)
        .is_empty());
    let overlaps = world.overlap_shape_leaves(
        &probe,
        Isometry::from_translation(Vec3::X * 2.0),
        &QueryFilter::ALL,
    );
    assert_eq!(overlaps.len(), 1);
    assert_eq!(overlaps[0].child_index, Some(1));
    assert_eq!(overlaps[0].material, report.colliders[1].material);
    assert!(overlaps[0].self_collision);
    let projections = world.project_point_leaves(Vec3::ZERO, &QueryFilter::ALL);
    assert_eq!(projections.len(), 2);
    assert!(projections
        .iter()
        .all(|p| !p.is_inside && (p.distance - 1.5).abs() < 1e-5));
    assert!(world
        .collider_leaves(&QueryFilter::excluding(report.base.body))
        .is_empty());
}

#[test]
fn single_collider_keeps_direct_handles_and_live_body_policy_inheritance() {
    for transformed in [false, true] {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let owner = source(
            &mut world,
            ArticulationBase::Floating,
            inertia(),
            Isometry::IDENTITY,
            Vec3::Y,
        );
        let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let pose = if transformed {
            Isometry::new(Vec3::X, Quat::from_rotation_z(0.6))
        } else {
            Isometry::IDENTITY
        };
        attach(
            &mut world,
            owner,
            None,
            sphere,
            pose,
            PhysicsMaterial::DEFAULT,
            true,
        );
        degrade(&mut world, owner);
        let report = world.handoff_articulation(owner).unwrap();
        if !transformed {
            assert_eq!(report.colliders[0].body_shape, sphere);
            assert_eq!(report.colliders[0].body_child_range, None);
        } else {
            assert_eq!(report.colliders[0].body_child_range, Some(0..1));
        }
        let updated = PhysicsMaterial::new(0.7, 0.6);
        assert!(world.bodies.set_material(report.base.body, updated));
        let leaves = world.collider_leaves(&QueryFilter::ALL);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].material, updated);
        close_pose(leaves[0].pose, pose);
    }
}

#[test]
fn single_nested_policy_is_rebound_to_authoritative_attachment_policy() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let owner = source(
        &mut world,
        ArticulationBase::Floating,
        inertia(),
        Isometry::IDENTITY,
        Vec3::Y,
    );
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let shape = world
        .shapes
        .insert_compound_with_policies(
            &[CompoundChild {
                collider: sphere,
                local_pose: Isometry::IDENTITY,
            }],
            &[CompoundLeafPolicy {
                material: Some(PhysicsMaterial::new(9.0, 1.0)),
                self_collision: Some(true),
            }],
        )
        .unwrap();
    attach(
        &mut world,
        owner,
        None,
        shape,
        Isometry::IDENTITY,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    degrade(&mut world, owner);
    let report = world.handoff_articulation(owner).unwrap();
    assert_ne!(report.colliders[0].body_shape, shape);
    let leaf = world.collider_leaves(&QueryFilter::ALL)[0];
    assert_eq!(leaf.material, PhysicsMaterial::FRICTIONLESS);
    assert!(!leaf.self_collision);
}

#[test]
fn late_invalid_geometry_and_unbounded_union_reject_without_any_mutation() {
    for invalid in [
        ColliderShape::Sphere { radius: -0.5 },
        ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        },
    ] {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let owner = source(
            &mut world,
            ArticulationBase::Floating,
            inertia(),
            Isometry::IDENTITY,
            Vec3::Y,
        );
        let valid = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let invalid_handle = world.shapes.insert(invalid);
        // An earlier body's union is fully staged before the late bad source.
        for link in [None, Some(0)] {
            attach(
                &mut world,
                owner,
                link,
                valid,
                Isometry::IDENTITY,
                PhysicsMaterial::DEFAULT,
                false,
            );
            attach(
                &mut world,
                owner,
                link,
                if link.is_some() {
                    invalid_handle
                } else {
                    valid
                },
                Isometry::from_translation(Vec3::X),
                PhysicsMaterial::DEFAULT,
                true,
            );
        }
        degrade(&mut world, owner);
        let before = format!("{world:?}");
        let mut control = world.clone();
        let error = world.handoff_articulation(owner).unwrap_err();
        let expected = if matches!(invalid, ColliderShape::Plane { .. }) {
            CompoundColliderError::UnboundedChild
        } else {
            CompoundColliderError::InvalidGeometry
        };
        assert_eq!(
            error,
            ArticulationHandoffError::InvalidCompoundCollider(expected)
        );
        assert_eq!(format!("{world:?}"), before);
        assert_eq!(
            world.shapes.insert(ColliderShape::Sphere { radius: 0.1 }),
            control.shapes.insert(ColliderShape::Sphere { radius: 0.1 })
        );
        assert_eq!(
            world.spawn(BodyDesc::dynamic_at(Vec3::ZERO)),
            control.spawn(BodyDesc::dynamic_at(Vec3::ZERO))
        );
        assert_eq!(format!("{world:?}"), format!("{control:?}"));
    }
}

#[test]
fn identity_invalid_primitive_and_forged_compound_cache_are_not_fast_path_bypasses() {
    for forged in [false, true] {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let owner = source(
            &mut world,
            ArticulationBase::Floating,
            inertia(),
            Isometry::IDENTITY,
            Vec3::Y,
        );
        let shape = if forged {
            let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
            let compound = world
                .shapes
                .insert_compound(&[CompoundChild {
                    collider: sphere,
                    local_pose: Isometry::IDENTITY,
                }])
                .unwrap();
            let mut cached = *world.shapes.get(compound).unwrap();
            if let ColliderShape::Compound {
                ref mut local_aabb_min,
                ..
            } = cached
            {
                *local_aabb_min -= Vec3::ONE;
            }
            world.shapes.insert(cached)
        } else {
            world.shapes.insert(ColliderShape::Sphere { radius: -1.0 })
        };
        attach(
            &mut world,
            owner,
            None,
            shape,
            Isometry::IDENTITY,
            PhysicsMaterial::DEFAULT,
            false,
        );
        degrade(&mut world, owner);
        let before = format!("{world:?}");
        assert_eq!(
            world.handoff_articulation(owner),
            Err(ArticulationHandoffError::InvalidCompoundCollider(
                CompoundColliderError::InvalidGeometry
            ))
        );
        assert_eq!(format!("{world:?}"), before);
    }
}

#[test]
fn compound_metadata_validation_and_expanded_leaf_budget_are_atomic() {
    let mut shapes = ShapeRegistry::new();
    let sphere = shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let child = CompoundChild {
        collider: sphere,
        local_pose: Isometry::IDENTITY,
    };
    for policies in [
        vec![],
        vec![CompoundLeafPolicy {
            material: Some(PhysicsMaterial {
                friction: f32::NAN,
                restitution: 0.0,
            }),
            self_collision: None,
        }],
        vec![CompoundLeafPolicy {
            material: Some(PhysicsMaterial {
                friction: 0.0,
                restitution: 2.0,
            }),
            self_collision: None,
        }],
    ] {
        let before = format!("{shapes:?}");
        assert_eq!(
            shapes.insert_compound_with_policies(&[child], &policies),
            Err(CompoundColliderError::InvalidPolicy)
        );
        assert_eq!(format!("{shapes:?}"), before);
    }
    let full = shapes
        .insert_compound(&vec![child; MAX_COMPOUND_CHILDREN])
        .unwrap();
    let before = format!("{shapes:?}");
    assert_eq!(
        shapes.insert_compound(&[
            CompoundChild {
                collider: full,
                local_pose: Isometry::IDENTITY
            },
            child
        ]),
        Err(CompoundColliderError::TooManyChildren)
    );
    assert_eq!(format!("{shapes:?}"), before);
    let bad_pose = CompoundChild {
        local_pose: Isometry::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        ..child
    };
    assert_eq!(
        shapes.insert_compound(&[bad_pose]),
        Err(CompoundColliderError::InvalidGeometry)
    );
    assert_eq!(format!("{shapes:?}"), before);
}

#[test]
fn ccd_applies_both_leaf_policies_even_when_coarse_body_policy_says_no() {
    for mover_flag in [false, true] {
        for target_flag in [false, true] {
            for external in [false, true] {
                let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
                let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.25 });
                let child = CompoundChild {
                    collider: sphere,
                    local_pose: Isometry::IDENTITY,
                };
                let mover_shape = world
                    .shapes
                    .insert_compound_with_policies(
                        &[child],
                        &[CompoundLeafPolicy {
                            material: None,
                            self_collision: Some(mover_flag),
                        }],
                    )
                    .unwrap();
                let target_shape = world
                    .shapes
                    .insert_compound_with_policies(
                        &[child],
                        &[CompoundLeafPolicy {
                            material: None,
                            self_collision: Some(target_flag),
                        }],
                    )
                    .unwrap();
                let mover = world.spawn(
                    BodyDesc::dynamic_at(-Vec3::X * 2.0)
                        .with_collider(mover_shape)
                        .with_ccd(true),
                );
                let target =
                    world.spawn(BodyDesc::static_at(Vec3::ZERO).with_collider(target_shape));
                if !external {
                    world
                        .install_collision_exclusion_group(&[(mover, false), (target, false)])
                        .unwrap();
                }
                world.bodies.set_prev_position(mover, -Vec3::X * 2.0);
                world.bodies.set_position(mover, Vec3::X * 2.0);
                prism_physics_core::ccd::resolve_ccd(&mut world, 1.0 / 60.0);
                let x = world.bodies.position(mover).unwrap().x;
                if external || (mover_flag && target_flag) {
                    assert!(
                        x < -0.45 && x > -0.6,
                        "flags {mover_flag}/{target_flag}, external {external}: {x}"
                    );
                } else {
                    assert_eq!(x, 2.0);
                }
            }
        }
    }
}

fn mixed_contact_velocity(active_restitution: f32, body_restitution: f32) -> Vec3 {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    let owner = source(
        &mut world,
        ArticulationBase::Fixed,
        inertia(),
        Isometry::IDENTITY,
        Vec3::Y * 10.0,
    );
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    attach(
        &mut world,
        owner,
        None,
        sphere,
        Isometry::IDENTITY,
        PhysicsMaterial::new(0.0, active_restitution),
        false,
    );
    attach(
        &mut world,
        owner,
        None,
        sphere,
        Isometry::from_translation(Vec3::X * 10.0),
        PhysicsMaterial::new(0.0, 0.0),
        true,
    );
    degrade(&mut world, owner);
    let report = world.handoff_articulation(owner).unwrap();
    world.bodies.set_material(
        report.base.body,
        PhysicsMaterial::new(0.0, body_restitution),
    );
    let tree = source(
        &mut world,
        ArticulationBase::Floating,
        inertia(),
        Isometry::from_translation(Vec3::Y * 0.95),
        Vec3::ZERO,
    );
    assert!(world
        .articulations
        .get_mut(tree)
        .unwrap()
        .set_base_velocity(SpatialMotion::new(Vec3::ZERO, -Vec3::Y * 2.0)));
    attach(
        &mut world,
        tree,
        Some(0),
        sphere,
        Isometry::IDENTITY,
        PhysicsMaterial::FRICTIONLESS,
        false,
    );
    XpbdSolver::default().step(&mut world, 1.0 / 120.0, 1);
    world
        .articulations
        .get(tree)
        .unwrap()
        .base_velocity()
        .linear
}

#[test]
fn mixed_articulation_solver_uses_the_contacted_leaf_not_body_fallback() {
    let bouncing_wrong_fallback = mixed_contact_velocity(1.0, 0.0);
    let bouncing_same_fallback = mixed_contact_velocity(1.0, 1.0);
    let inelastic = mixed_contact_velocity(0.0, 0.0);
    close_vec(bouncing_wrong_fallback, bouncing_same_fallback);
    assert!(
        bouncing_wrong_fallback.y > inelastic.y + 0.5,
        "bounce {bouncing_wrong_fallback:?}, inelastic {inelastic:?}"
    );
}

#[test]
fn full_snapshot_replays_handoff_policy_and_lifecycle_without_stale_leaf_identity() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let owner = source(
        &mut world,
        ArticulationBase::Floating,
        inertia(),
        Isometry::IDENTITY,
        Vec3::Y * 4.0,
    );
    let sphere = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    for i in 0..2 {
        attach(
            &mut world,
            owner,
            None,
            sphere,
            Isometry::from_translation(Vec3::X * i as f32 * 2.0),
            PhysicsMaterial::new(i as f32, 0.5 * i as f32),
            i == 1,
        );
    }
    degrade(&mut world, owner);
    let before = FullStateSnapshot::capture(7, &world);
    let report = world.handoff_articulation(owner).unwrap();
    let mut replay = before.fork_world();
    assert_eq!(replay.handoff_articulation(owner).unwrap(), report);
    assert_eq!(hash_world(&world), hash_world(&replay));
    let after = FullStateSnapshot::capture(8, &world);
    let mut solver = XpbdSolver::default();
    let mut replay_solver = XpbdSolver::default();
    for _ in 0..4 {
        solver.step(&mut world, 1.0 / 120.0, 1);
        replay_solver.step(&mut replay, 1.0 / 120.0, 1);
        assert_eq!(hash_world(&world), hash_world(&replay));
        assert_eq!(
            world.collider_leaves(&QueryFilter::ALL),
            replay.collider_leaves(&QueryFilter::ALL)
        );
    }
    after.restore_into(&mut world);
    assert_eq!(hash_world(&world), after.state_hash());
    let stable = format!("{world:?}");
    assert_eq!(
        world.handoff_articulation(owner),
        Err(ArticulationHandoffError::InvalidArticulation)
    );
    assert_eq!(format!("{world:?}"), stable);
    let old = report.base.body;
    assert!(world.bodies.set_collider(old, Some(sphere)));
    let leaves = world.collider_leaves(&QueryFilter::ALL);
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].child_index, None);
    assert!(world.remove_body(old));
    assert!(world.collision_exclusions().membership(old).is_none());
    let replacement = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_collider(sphere));
    assert_ne!(old, replacement);
    assert!(world
        .collision_exclusions()
        .membership(replacement)
        .is_none());
    assert!(world
        .collider_leaves(&QueryFilter::ALL)
        .iter()
        .all(|leaf| leaf.body != old));
    after.restore_into(&mut world);
    assert_eq!(world.collider_leaves(&QueryFilter::ALL).len(), 2);
    assert_eq!(hash_world(&world), after.state_hash());
}

#[cfg(feature = "serialize")]
#[test]
fn leaf_policy_and_query_identity_have_serde_contracts() {
    fn serializable<T: serde::Serialize + for<'de> serde::Deserialize<'de>>() {}
    serializable::<CompoundLeafPolicy>();
    serializable::<prism_physics_core::query::ColliderLeaf>();
}
