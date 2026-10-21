//! Automatic multi-proxy break transactions share one allocator and commit no
//! policy, topology or wrench changes until every split has been validated.

use glam::Vec3;
use prism_physics_core::dynamics::assembly::{
    AssemblyError, BoundaryLoadError, BreakLimit, JointLoad, SolverJointLoad,
};
use prism_physics_core::dynamics::force::ForceError;
use prism_physics_core::joint::DistanceJoint;
use prism_physics_core::snapshot::world_hash::hash_world;
use prism_physics_core::world::AssemblyWorldError;
use prism_physics_core::{BodyDesc, BodyHandle, JointDesc, JointHandle, PhysicsWorld};

fn merged_pair(
    world: &mut PhysicsWorld,
    offset: f32,
) -> ([BodyHandle; 2], BodyHandle, JointHandle) {
    let members = [-4.0, 4.0].map(|x| world.spawn(BodyDesc::dynamic_at(Vec3::new(x, offset, 0.0))));
    let desc = JointDesc::distance(
        members[0],
        Vec3::ZERO,
        members[1],
        Vec3::ZERO,
        DistanceJoint::rigid(8.0),
    );
    let joint = world.joints.insert(desc);
    world
        .connect_assembly(
            members[0],
            members[1],
            Some(joint),
            desc.kind,
            Some(BreakLimit::new(10.0, 10.0).with_fatigue(0.5)),
        )
        .unwrap();
    let proxy = world.merge_assembly(&members).unwrap();
    (members, proxy, joint)
}

fn load(joint: JointHandle) -> SolverJointLoad {
    SolverJointLoad {
        joint,
        load: JointLoad {
            linear_impulse: Vec3::Y * 2.0,
            angular_impulse: Vec3::ZERO,
            substep_dt: 0.1,
        },
    }
}

fn sorted_proxies(pairs: &[([BodyHandle; 2], BodyHandle, JointHandle); 2]) -> [BodyHandle; 2] {
    let mut proxies = pairs.map(|(_, proxy, _)| proxy);
    proxies.sort_by_key(|body| (body.index(), body.generation()));
    proxies
}

#[test]
fn later_proxy_graph_failure_preserves_every_split_and_policy_until_retry() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let pairs = [merged_pair(&mut world, 0.0), merged_pair(&mut world, 20.0)];
    let proxies = sorted_proxies(&pairs);
    for proxy in proxies {
        let center = world.bodies.position(proxy).unwrap();
        assert!(world.apply_force(proxy, Vec3::new(2.0, 4.0, 6.0), center));
        assert!(world.apply_torque(proxy, Vec3::new(1.0, 3.0, 5.0)));
    }
    assert!(world.assemblies.remove_body(proxies[1]));
    let loads = pairs.map(|(_, _, joint)| load(joint));
    let before = format!("{world:?}");
    let hash = hash_world(&world).unwrap();
    for _ in 0..3 {
        assert_eq!(
            world.submit_joint_loads(0.1, &loads),
            Err(BoundaryLoadError::AssemblySplit {
                proxy: proxies[1],
                error: AssemblyWorldError::Graph(AssemblyError::ProxyConflict),
            })
        );
        assert_eq!(
            format!("{world:?}"),
            before,
            "fatigue, graph, forces and allocator"
        );
        assert_eq!(hash_world(&world).unwrap(), hash);
    }
    assert!(world.drain_assembly_split_reports().is_empty());
    world.assemblies.add_body(proxies[1]);
    let mut reverse = world.clone();
    let events = world.submit_joint_loads(0.1, &loads).unwrap();
    let reversed_events = reverse
        .submit_joint_loads(0.1, &[loads[1], loads[0]])
        .unwrap();
    assert_eq!(events, reversed_events);
    assert_eq!(hash_world(&world).unwrap(), hash_world(&reverse).unwrap());
    assert_eq!(events.len(), 2);
    let reports = world.drain_assembly_split_reports();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports.iter().map(|r| r.proxy).collect::<Vec<_>>(), proxies);
    let mut restored = Vec::new();
    for report in &reports {
        assert!(!world.bodies.contains(report.proxy));
        assert!(world.external_forces.pending(report.proxy).is_none());
        let originals = pairs.iter().find(|(_, p, _)| *p == report.proxy).unwrap().0;
        assert_eq!(
            report
                .members
                .iter()
                .map(|m| m.original)
                .collect::<Vec<_>>(),
            originals
        );
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        let proxy_center = report
            .members
            .iter()
            .map(|m| world.bodies.position(m.current).unwrap())
            .sum::<Vec3>()
            * 0.5;
        for member in &report.members {
            assert!(world.bodies.contains(member.current));
            assert!(!world.bodies.contains(member.original));
            let pending = world.external_forces.pending(member.current).unwrap();
            force += pending.force;
            torque += pending.torque
                + (world.bodies.position(member.current).unwrap() - proxy_center)
                    .cross(pending.force);
            restored.push(member.current);
        }
        assert!((force - Vec3::new(2.0, 4.0, 6.0)).length() < 1.0e-5);
        assert!((torque - Vec3::new(1.0, 3.0, 5.0)).length() < 1.0e-5);
    }
    restored.sort_by_key(|body| (body.index(), body.generation()));
    restored.dedup();
    assert_eq!(
        restored.len(),
        4,
        "one batch must not reuse the same predicted slots"
    );
    assert_eq!(world.bodies.len(), 4);
    assert!(world.bodies.columns_consistent());
    for (_, _, joint) in pairs {
        let joint = world.joints.get(joint).unwrap();
        assert!(!joint.enabled);
        assert!(restored.contains(&joint.anchor_a.body));
        assert!(restored.contains(&joint.anchor_b.body));
    }
    assert_eq!(world.assemblies.connections().count(), 0);
    assert!(world.submit_joint_loads(0.1, &loads).unwrap().is_empty());
    assert!(world.drain_assembly_split_reports().is_empty());
}

#[test]
fn later_proxy_wrench_overflow_rejects_entire_boundary_and_remains_retryable() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let pairs = [merged_pair(&mut world, 0.0), merged_pair(&mut world, 20.0)];
    let proxies = sorted_proxies(&pairs);
    let center = world.bodies.position(proxies[1]).unwrap();
    assert!(world.apply_force(proxies[1], Vec3::Y * (f32::MAX * 0.75), center));
    let loads = pairs.map(|(_, _, joint)| load(joint));
    let before = format!("{world:?}");
    for _ in 0..2 {
        assert_eq!(
            world.submit_joint_loads(0.1, &loads),
            Err(BoundaryLoadError::AssemblySplit {
                proxy: proxies[1],
                error: AssemblyWorldError::Force(ForceError::NonFiniteInput),
            })
        );
        assert_eq!(format!("{world:?}"), before);
    }
    world.external_forces.clear();
    assert_eq!(world.submit_joint_loads(0.1, &loads).unwrap().len(), 2);
    assert_eq!(world.drain_assembly_split_reports().len(), 2);
}
