//! Focused M14/M15 module-level integration contracts.
use glam::{Quat, Vec3};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::articulation::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationJoint, ArticulationStorage,
    LinkContactImpulse, SpatialInertia, SpatialMotion, SpatialTransform,
};
use prism_physics_core::soft::topology::{
    TearableMesh, TopologyTearUpdate, TopologyTearingConfig, TopologyUpdateConsumer,
    TopologyUpdateError,
};

fn inertia(mass: f32) -> SpatialInertia {
    SpatialInertia::diagonal(mass, Vec3::new(0.2, -0.1, 0.0), Vec3::splat(mass * 0.5))
}

fn tree(base: ArticulationBase) -> Articulation {
    let mut builder = ArticulationBuilder::new(base, inertia(3.0)).with_base_pose(Isometry::new(
        Vec3::new(2.0, -1.0, 0.5),
        Quat::from_rotation_y(0.3),
    ));
    for (parent, joint) in [
        (None, ArticulationJoint::revolute(Vec3::Z)),
        (Some(0), ArticulationJoint::prismatic(Vec3::X)),
        (Some(0), ArticulationJoint::spherical()),
        (Some(1), ArticulationJoint::revolute(Vec3::Y)),
        (Some(3), ArticulationJoint::fixed()),
    ] {
        builder
            .add_link(
                parent,
                SpatialTransform::new(Quat::from_rotation_x(0.1), Vec3::new(0.4, 0.3, 0.2)),
                inertia(1.5),
                joint,
            )
            .unwrap();
    }
    let mut tree = builder.build().unwrap();
    for index in 0..tree.dof_count() {
        assert!(tree.set_position(index, 0.04 * (index + 1) as f32));
        assert!(tree.set_velocity(index, -0.07 * (index + 1) as f32));
    }
    if base == ArticulationBase::Floating {
        assert!(tree.set_base_velocity(SpatialMotion::new(
            Vec3::new(0.2, 0.3, -0.1),
            Vec3::new(0.4, -0.2, 0.5),
        )));
    }
    tree
}

fn close_vec(a: Vec3, b: Vec3) {
    assert!((a - b).length() < 2.0e-5, "{a:?} != {b:?}");
}

#[test]
fn world_contacts_match_effective_mass_and_local_response() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        let mut tree = tree(base);
        let pose = tree.forward_kinematics()[3];
        let point = pose.transform_point(Vec3::new(0.7, -0.2, 0.3));
        let direction = pose.transform_vector(Vec3::Y);
        let impulse = tree
            .contact_impulse_from_world(3, point, direction)
            .unwrap();
        close_vec(impulse.point, Vec3::new(0.7, -0.2, 0.3));
        close_vec(impulse.impulse, Vec3::Y);
        let mass = tree.contact_inverse_mass(3, point, direction).unwrap();
        assert!(mass > 0.0);
        let before = tree.contact_point_velocity(3, point).unwrap();
        let _ = tree.apply_contact_impulses(&[impulse]);
        let after = tree.contact_point_velocity(3, point).unwrap();
        assert!(((after - before).dot(direction) - mass).abs() < 2.0e-5);
    }
}

#[test]
fn shuffled_contact_batches_are_bit_identical_and_ignore_invalid_entries() {
    let mut a = tree(ArticulationBase::Floating);
    let mut b = a.clone();
    let mut contacts: Vec<_> = [1.0e6, 0.03, -1.0e6, 0.4, -0.2]
        .into_iter()
        .map(|magnitude| LinkContactImpulse {
            link: 3,
            point: Vec3::new(0.3, 0.2, -0.1),
            impulse: Vec3::Y * magnitude,
        })
        .collect();
    let response = a.apply_contact_impulses(&contacts);
    contacts.reverse();
    contacts.extend([
        LinkContactImpulse {
            link: usize::MAX,
            point: Vec3::ZERO,
            impulse: Vec3::X,
        },
        LinkContactImpulse {
            link: 0,
            point: Vec3::splat(f32::NAN),
            impulse: Vec3::X,
        },
        LinkContactImpulse {
            link: 0,
            point: Vec3::ZERO,
            impulse: Vec3::splat(f32::INFINITY),
        },
    ]);
    assert_eq!(response, b.apply_contact_impulses(&contacts));
    assert_eq!(a.velocities(), b.velocities());
    assert_eq!(a.base_velocity(), b.base_velocity());
}

#[test]
fn invalid_world_contacts_and_stale_handles_are_rejected() {
    let mut storage = ArticulationStorage::new();
    let handle = storage.insert(tree(ArticulationBase::Fixed));
    assert!(storage.queue_world_contact_impulse(handle, 1, Vec3::ZERO, Vec3::Y));
    assert!(!storage.queue_world_contact_impulse(handle, 999, Vec3::ZERO, Vec3::Y));
    assert!(!storage.queue_world_contact_impulse(handle, 1, Vec3::NAN, Vec3::Y));
    assert!(storage
        .get(handle)
        .unwrap()
        .contact_inverse_mass(0, Vec3::ZERO, Vec3::ZERO)
        .is_none());
    storage.get_mut(handle).unwrap().break_joint(1).unwrap();
    assert!(!storage.queue_world_contact_impulse(handle, 1, Vec3::ZERO, Vec3::Y));
    assert!(storage.remove(handle));
    assert!(!storage.queue_world_contact_impulse(handle, 1, Vec3::ZERO, Vec3::Y));
}

#[test]
fn break_rebuild_preserves_poses_velocities_and_surviving_joints() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for broken in 0..5 {
            let mut original = tree(base);
            assert!(original.rebuild_after_break().is_none());
            let poses = original.forward_kinematics();
            let velocities = original.link_velocities();
            original.break_joint(broken).unwrap();
            assert!(original.break_joint(broken).is_none());
            let mut rebuilt = original.rebuild_after_break().unwrap();
            assert_eq!(rebuilt.detached_base_link, broken);
            assert_eq!(rebuilt.detached.base_pose(), poses[broken]);
            assert_eq!(rebuilt.detached.base_velocity(), velocities[broken]);
            assert_eq!(rebuilt.detached.base_mode(), ArticulationBase::Floating);
            let mut covered = vec![broken];
            for (component, mapping) in [
                (&rebuilt.retained, &rebuilt.retained_links),
                (&rebuilt.detached, &rebuilt.detached_links),
            ] {
                assert!(component.broken_joint().is_none());
                let rebuilt_poses = component.forward_kinematics();
                let rebuilt_velocities = component.link_velocities();
                for (new, &old) in mapping.iter().enumerate() {
                    covered.push(old);
                    close_vec(rebuilt_poses[new].translation, poses[old].translation);
                    assert!(rebuilt_poses[new].rotation.dot(poses[old].rotation).abs() > 0.99999);
                    close_vec(rebuilt_velocities[new].linear, velocities[old].linear);
                    close_vec(rebuilt_velocities[new].angular, velocities[old].angular);
                    assert_eq!(component.links()[new].joint, original.links()[old].joint);
                }
            }
            covered.sort_unstable();
            assert_eq!(covered, (0..original.link_count()).collect::<Vec<_>>());
            assert!(!rebuilt.retained.step(0.01).degraded);
            assert!(!rebuilt.detached.step(0.01).degraded);
            assert!(rebuilt.detached.base_pose().translation.is_finite());
            let frozen = original.positions().to_vec();
            assert!(original.step(0.01).degraded);
            assert_eq!(original.positions(), frozen);
            let response = original.generalized_contact_impulse_response(&[LinkContactImpulse {
                link: broken,
                point: Vec3::X,
                impulse: Vec3::Y,
            }]);
            assert!(response
                .generalized_velocity_change
                .iter()
                .all(|&v| v == 0.0));
        }
    }
}

#[test]
fn maximal_handoff_preserves_com_and_world_velocity() {
    let tree = tree(ArticulationBase::Floating);
    let states = tree.maximal_body_states();
    assert_eq!(states.len(), tree.link_count() + 1);
    assert_eq!(states[0].link, None);
    let poses = tree.forward_kinematics();
    let velocities = tree.link_velocities();
    for (link, state) in states[1..].iter().enumerate() {
        assert_eq!(state.link, Some(link));
        assert!(!state.fixed);
        close_vec(
            state.center_of_mass,
            poses[link].transform_point(tree.links()[link].inertia.center_of_mass),
        );
        close_vec(
            state.linear_velocity,
            poses[link].rotation * velocities[link].linear
                + state
                    .angular_velocity
                    .cross(state.center_of_mass - poses[link].translation),
        );
    }
}

fn mesh() -> TearableMesh {
    let positions = vec![
        Vec3::ZERO,
        Vec3::X,
        Vec3::X * 2.0,
        Vec3::Y,
        Vec3::X + Vec3::Y,
        Vec3::X * 2.0 + Vec3::Y,
        Vec3::Y * 2.0,
        Vec3::X + Vec3::Y * 2.0,
        (Vec3::X + Vec3::Y) * 2.0,
    ];
    TearableMesh::new(
        positions.clone(),
        positions,
        vec![1.0; 9],
        vec![
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ],
    )
    .unwrap()
}

fn tear(mesh: &mut TearableMesh) -> TopologyTearUpdate {
    let events = mesh
        .puncture(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::Z,
            10.0,
            TopologyTearingConfig {
                max_front_steps: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!events.is_empty());
    mesh.tear_update(events)
}

#[test]
fn topology_consumption_replaces_render_collision_and_ccd_atomically() {
    let mut mesh = mesh();
    let mut consumer = TopologyUpdateConsumer::from_mesh(&mesh);
    let initial_positions = mesh.positions.clone();
    let update = tear(&mut mesh);
    assert!(update.render.vertices.len() > initial_positions.len());
    consumer.apply(update.clone()).unwrap();
    assert_eq!(consumer.topology_version(), 1);
    assert_eq!(consumer.render(), &update.render);
    assert_eq!(consumer.collision(), &update.collision);
    assert_eq!(
        consumer.collision().ccd_revision,
        consumer.topology_version()
    );
    assert_eq!(consumer.last_events(), update.events);
    for (vertex, &previous) in consumer
        .render()
        .vertices
        .iter()
        .zip(consumer.previous_positions())
    {
        close_vec(previous, initial_positions[vertex.root as usize]);
    }
    for (triangle, bounds) in consumer
        .render()
        .triangles
        .iter()
        .zip(consumer.swept_triangle_bounds())
    {
        for &vertex in triangle {
            for point in [
                consumer.previous_positions()[vertex as usize],
                consumer.render().vertices[vertex as usize].position,
            ] {
                assert!(point.cmpge(bounds.min).all() && point.cmple(bounds.max).all());
            }
        }
    }
}

#[test]
fn topology_duplicate_stale_and_missing_revisions_leave_state_unchanged() {
    let mut mesh = mesh();
    let initial = mesh.tear_update(Vec::new());
    let mut consumer = TopologyUpdateConsumer::new(initial.clone()).unwrap();
    let first = tear(&mut mesh);
    consumer.apply(first.clone()).unwrap();
    let saved = consumer.clone();
    assert_eq!(
        consumer.apply(first),
        Err(TopologyUpdateError::StaleRevision)
    );
    assert_eq!(
        consumer.apply(initial),
        Err(TopologyUpdateError::StaleRevision)
    );
    let _second = tear(&mut mesh);
    let third = tear(&mut mesh);
    assert_eq!(consumer.apply(third), Err(TopologyUpdateError::RevisionGap));
    assert_eq!(consumer, saved);
}

#[test]
fn inconsistent_topology_payloads_are_rejected_before_any_cache_write() {
    let mut mesh = mesh();
    let initial = TopologyUpdateConsumer::from_mesh(&mesh);
    let update = tear(&mut mesh);
    let mut corruptions = Vec::new();
    let mut bad = update.clone();
    bad.collision.ccd_revision += 1;
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.render.triangles[0][0] = u32::MAX;
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.render.vertices[0].position = Vec3::NAN;
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.collision.triangle_bounds[0].min -= Vec3::ONE;
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.collision.self_collision_exclusions.pop();
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.render.seam_edges.clear();
    corruptions.push(bad);
    let mut bad = update.clone();
    bad.events[0].vertex_count += 1;
    corruptions.push(bad);
    let mut bad = update;
    bad.render.vertices[0].root = u32::MAX;
    corruptions.push(bad);
    for bad in corruptions {
        let mut consumer = initial.clone();
        assert!(consumer.apply(bad).is_err());
        assert_eq!(consumer, initial);
    }
}

#[test]
fn repeated_tears_and_consumption_are_deterministic() {
    let mut a = mesh();
    let mut b = a.clone();
    let mut ca = TopologyUpdateConsumer::from_mesh(&a);
    let mut cb = TopologyUpdateConsumer::from_mesh(&b);
    for version in 1..=4 {
        let ua = tear(&mut a);
        let ub = tear(&mut b);
        assert_eq!(ua, ub);
        ca.apply(ua).unwrap();
        cb.apply(ub).unwrap();
        assert_eq!(ca, cb);
        assert_eq!(ca.topology_version(), version);
    }
}
