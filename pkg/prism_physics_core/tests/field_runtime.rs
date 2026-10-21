use glam::Vec3;
use prism_physics_core::dynamics::field::{
    CpuFieldSampler, Field, FieldBatchError, FieldBatchSampler, FieldBody, FieldEffect,
    FieldFalloff, FieldId, FieldRegistry, FieldSample, FieldShape, FieldTargetMask,
};
use prism_physics_core::snapshot::world_hash::hash_world;
use prism_physics_core::solver::xpbd::FieldStepError;
use prism_physics_core::{BodyDesc, BodyKind, PhysicsWorld, Solver, WorldConfig, XpbdSolver};

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(WorldConfig {
        gravity: Vec3::ZERO,
        ..WorldConfig::default()
    });
    for index in 0..3 {
        world.spawn(
            BodyDesc::dynamic_at(Vec3::X * index as f32)
                .with_linear_velocity(Vec3::new(3.0, 0.5, -1.0))
                .with_damping(0.0, 0.0),
        );
    }
    world
        .fields
        .insert(Field::constant(FieldId(1), Vec3::new(2.0, 1.0, 0.0)));
    world.fields.insert(Field {
        id: FieldId(2),
        shape: FieldShape::Global,
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::Drag { coefficient: 0.75 },
        target_mask: FieldTargetMask::ALL,
    });
    world
}

struct Provider<F>(F);

impl<F> FieldBatchSampler for Provider<F>
where
    F: FnMut(&FieldRegistry, &[FieldBody]) -> Result<Vec<FieldSample>, &'static str>,
{
    type Error = &'static str;

    fn sample_batch(
        &mut self,
        fields: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, Self::Error> {
        (self.0)(fields, bodies)
    }
}

#[test]
fn cpu_provider_matches_default_xpbd_with_velocity_dependent_fields_each_substep() {
    let mut reference = world();
    let mut batched = reference.clone();
    let mut solver = XpbdSolver::default();
    let mut calls = 0;
    let mut sampler = Provider(|registry: &FieldRegistry, bodies: &[FieldBody]| {
        calls += 1;
        CpuFieldSampler
            .sample_batch(registry, bodies)
            .map_err(|never| match never {})
    });
    for _ in 0..20 {
        solver.step(&mut reference, 1.0 / 60.0, 4);
        solver
            .step_with_field_sampler(&mut batched, 1.0 / 60.0, 4, &mut sampler)
            .unwrap();
        assert_eq!(hash_world(&reference), hash_world(&batched));
    }
    assert_eq!(calls, 80);
}

#[test]
fn rejected_batches_preserve_all_pending_forces_and_sleep_state() {
    for failure in 0..4 {
        let mut world = world();
        let first = world.bodies.handle_at_slot(0).unwrap();
        world.apply_force(first, Vec3::X, Vec3::ZERO);
        world.bodies.solver_view_mut().sleeping[0] = true;
        let before = hash_world(&world).unwrap();
        let mut sampler = Provider(move |_: &FieldRegistry, bodies: &[FieldBody]| {
            if failure == 0 {
                return Err("device lost");
            }
            let count = bodies.len() - usize::from(failure == 1);
            let mut samples = vec![
                FieldSample {
                    force: Vec3::ONE,
                    torque: Vec3::ZERO
                };
                count
            ];
            if failure == 2 {
                samples[1].force.y = f32::NAN;
            } else if failure == 3 {
                samples[1].torque.z = f32::INFINITY;
            }
            Ok(samples)
        });
        let error = world.sample_fields_with(&mut sampler).unwrap_err();
        match failure {
            0 => assert_eq!(error, FieldBatchError::Provider("device lost")),
            1 => assert_eq!(
                error,
                FieldBatchError::SampleCount {
                    expected: 3,
                    actual: 2
                }
            ),
            _ => assert_eq!(error, FieldBatchError::NonFiniteSample { index: 1 }),
        }
        assert_eq!(hash_world(&world).unwrap(), before);
        assert_eq!(world.bodies.is_sleeping(first), Some(true));
    }
}

#[test]
fn accumulation_overflow_rejects_the_entire_batch_before_waking_any_body() {
    let mut world = world();
    let last = world.bodies.handle_at_slot(2).unwrap();
    world.apply_torque(last, Vec3::splat(f32::MAX));
    world.bodies.solver_view_mut().sleeping[0] = true;
    let before = hash_world(&world).unwrap();
    let mut sampler = Provider(|_: &FieldRegistry, bodies: &[FieldBody]| {
        Ok(vec![
            FieldSample {
                force: Vec3::ONE,
                torque: Vec3::splat(f32::MAX)
            };
            bodies.len()
        ])
    });
    assert_eq!(
        world.sample_fields_with(&mut sampler),
        Err(FieldBatchError::AccumulationOverflow { index: 2 })
    );
    assert_eq!(hash_world(&world).unwrap(), before);
}

#[test]
fn slot_reuse_and_nondynamic_bodies_cannot_shift_force_destinations() {
    let mut world = world();
    let old = world.bodies.handle_at_slot(1).unwrap();
    assert!(world.bodies.remove(old));
    let replacement = world.spawn(BodyDesc::dynamic_at(Vec3::splat(42.0)));
    assert_eq!(replacement.index(), old.index());
    assert_ne!(replacement.generation(), old.generation());
    let static_body = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let kinematic = world.spawn(BodyDesc {
        kind: BodyKind::Kinematic,
        ..BodyDesc::default()
    });
    world.bodies.solver_view_mut().sleeping[replacement.index() as usize] = true;
    let mut sampler = Provider(|_: &FieldRegistry, bodies: &[FieldBody]| {
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[1].position, Vec3::splat(42.0));
        assert!(bodies
            .iter()
            .all(|body| body.target_mask == FieldTargetMask::ALL));
        Ok((0..bodies.len())
            .map(|index| FieldSample {
                force: Vec3::X * (index + 1) as f32,
                torque: Vec3::ZERO,
            })
            .collect())
    });
    assert_eq!(world.sample_fields_with(&mut sampler), Ok(3));
    assert_eq!(
        world.external_forces.pending(replacement).unwrap().force,
        Vec3::X * 2.0
    );
    assert!(world.external_forces.pending(old).is_none());
    assert!(world.external_forces.pending(static_body).is_none());
    assert!(world.external_forces.pending(kinematic).is_none());
    assert_eq!(world.bodies.is_sleeping(replacement), Some(false));
}

#[test]
fn zero_wrenches_do_not_wake_sleepers_or_allocate_force_entries() {
    let mut world = world();
    world.fields = FieldRegistry::new();
    world.bodies.solver_view_mut().sleeping[0] = true;
    let before = hash_world(&world).unwrap();
    assert_eq!(world.sample_fields_with(&mut CpuFieldSampler), Ok(0));
    assert_eq!(hash_world(&world).unwrap(), before);
}

#[test]
fn invalid_duration_does_not_call_provider_or_change_world() {
    let mut world = world();
    let before = hash_world(&world).unwrap();
    let mut sampler = Provider(
        |_: &FieldRegistry, _: &[FieldBody]| -> Result<Vec<FieldSample>, &'static str> {
            panic!("invalid duration must not dispatch a provider")
        },
    );
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(world.prepare_external_substep_with(dt, &mut sampler), Ok(0));
        assert_eq!(
            XpbdSolver::default().step_with_field_sampler(&mut world, dt, 3, &mut sampler),
            Err(FieldStepError::InvalidTimeStep)
        );
        assert_eq!(hash_world(&world).unwrap(), before);
    }
}

#[test]
fn later_failure_reports_committed_substeps_and_preserves_the_failed_boundary() {
    let mut world = world();
    let mut reference = world.clone();
    let mut solver = XpbdSolver::default();
    solver.step(&mut reference, 0.01, 1);
    let mut calls = 0;
    let mut sampler = Provider(|fields: &FieldRegistry, bodies: &[FieldBody]| {
        calls += 1;
        if calls == 2 {
            return Err("mapping failed");
        }
        CpuFieldSampler
            .sample_batch(fields, bodies)
            .map_err(|never| match never {})
    });
    assert_eq!(
        solver.step_with_field_sampler(&mut world, 0.03, 3, &mut sampler),
        Err(FieldStepError::Sampling {
            completed_substeps: 1,
            source: FieldBatchError::Provider("mapping failed"),
        })
    );
    assert_eq!(calls, 2);
    assert_eq!(hash_world(&world), hash_world(&reference));
}

#[test]
fn large_ascending_batches_accumulate_against_existing_out_of_order_forces() {
    let mut world = PhysicsWorld::new(WorldConfig::default());
    let handles: Vec<_> = (0..10_000)
        .map(|_| world.spawn(BodyDesc::dynamic_at(Vec3::ZERO)))
        .collect();
    for &body in handles.iter().rev().step_by(17) {
        assert!(world.apply_torque(body, Vec3::Y));
    }
    world.fields.insert(Field::constant(FieldId(1), Vec3::X));
    assert_eq!(
        world.sample_fields_with(&mut CpuFieldSampler),
        Ok(handles.len())
    );
    for &body in &handles {
        assert_eq!(world.external_forces.pending(body).unwrap().force, Vec3::X);
    }
    for &body in handles.iter().rev().step_by(17) {
        assert_eq!(world.external_forces.pending(body).unwrap().torque, Vec3::Y);
    }
}
