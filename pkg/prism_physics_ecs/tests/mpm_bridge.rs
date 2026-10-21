use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_physics_core::mpm::{Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver};
use prism_physics_ecs::{
    MpmAuthority, MpmBackendPolicy, MpmBinding, MpmDiagnostics, MpmFallbackReason, MpmIssue,
    MpmParticleSnapshot, MpmRuntimeResource, PhysicsBindings, PrismPhysicsPlugin, RigidBody, Vec3,
};
use prism_transform::Transform;

fn solver() -> MpmSolver {
    let mut particles = MaterialPoints::new();
    particles.spawn(Vec3::new(0.9, 1.2, 0.9), Vec3::ZERO, 1.0, 1.0e-3);
    MpmSolver::new(
        particles,
        Grid::new(16, 16, 16, 0.1, Vec3::ZERO),
        MpmMaterial::default(),
        MpmConfig::new(1.0e-3, Vec3::new(0.0, -9.81, 0.0)),
    )
}

fn app() -> App {
    app_with_timestep(Duration::from_millis(1))
}

fn app_with_timestep(timestep: Duration) -> App {
    let mut app = App::new();
    app.set_fixed_timestep(timestep)
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(timestep))
        .add_plugins(PrismPhysicsPlugin);
    app
}

#[test]
fn mpm_runs_with_rigid_body_and_publishes_versioned_snapshot() {
    let mut app = app();
    let mpm_entity = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    let rigid_entity = app
        .world_mut()
        .spawn((RigidBody::dynamic(), Transform::IDENTITY));

    assert!(app.world().get_resource::<MpmRuntimeResource>().is_some());
    assert!(app.world().get_resource::<MpmDiagnostics>().is_some());
    app.update();

    let snapshot = app
        .world()
        .get::<MpmParticleSnapshot>(mpm_entity)
        .expect("MPM writeback snapshot");
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.tick, 1);
    assert_eq!(snapshot.positions.len(), 1);
    assert!(snapshot.positions[0].is_finite());
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(rigid_entity)
        .is_some());

    let first_y = snapshot.positions[0].y;
    app.update();
    let second = app.world().get::<MpmParticleSnapshot>(mpm_entity).unwrap();
    assert_eq!(second.version, 2);
    assert_eq!(second.tick, 2);
    assert!(second.positions[0].y < first_y);
    assert_eq!(
        app.world()
            .resource::<MpmRuntimeResource>()
            .info(mpm_entity)
            .unwrap()
            .authority,
        MpmAuthority::Cpu
    );
    let diagnostics = app.world().resource::<MpmDiagnostics>();
    assert_eq!(diagnostics.tick, 2);
    assert_eq!(diagnostics.steps, 1);
    assert!(diagnostics.issues.is_empty());
}

#[test]
fn mpm_binding_removal_and_despawn_release_runtime_and_snapshot() {
    let mut app = app();
    let removed = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    let despawned = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.update();

    assert!(app.world_mut().remove::<MpmBinding>(removed));
    assert!(app.world_mut().despawn(despawned));
    app.update();

    let runtimes = app.world().resource::<MpmRuntimeResource>();
    assert!(!runtimes.contains(removed));
    assert!(!runtimes.contains(despawned));
    assert!(app.world().get::<MpmParticleSnapshot>(removed).is_none());
}

#[test]
fn invalid_authoring_is_rejected_and_can_be_repaired() {
    let mut app = app();
    let entity = app
        .world_mut()
        .spawn((MpmBinding::cpu(solver()).with_step_budget(0),));
    app.update();

    assert!(!app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(entity));
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|diagnostic| diagnostic.entity == Some(entity)
            && diagnostic.issue == MpmIssue::InvalidAuthoring));

    *app.world_mut().get_mut::<MpmBinding>(entity).unwrap() = MpmBinding::cpu(solver());
    app.update();

    assert!(app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(entity));
    assert_eq!(
        app.world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .version,
        1
    );
}

#[test]
fn fixed_step_mismatch_does_not_publish_or_advance() {
    let mut app = app_with_timestep(Duration::from_millis(2));
    let entity = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.update();

    let info = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert_eq!(info.fixed_tick, 0);
    assert_eq!(info.snapshot_version, 0);
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|diagnostic| diagnostic.entity == Some(entity)
            && diagnostic.issue == MpmIssue::FixedStepMismatch));

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .solver
        .config
        .dt = 2.0e-3;
    app.update();

    let snapshot = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.tick, 2);
}

#[test]
fn solver_backend_and_budget_edits_rebuild_the_runtime() {
    let mut app = app();
    let entity = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.update();
    app.update();
    assert_eq!(
        app.world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .version,
        2
    );

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .solver
        .config
        .gravity = Vec3::ZERO;
    app.update();
    let solver_edit = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert_eq!(solver_edit.fixed_tick, 1);
    assert_eq!(solver_edit.snapshot_version, 1);

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .max_steps_per_advance = 2;
    app.update();
    let budget_edit = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert_eq!(budget_edit.fixed_tick, 1);
    assert_eq!(budget_edit.snapshot_version, 1);

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .backend = MpmBackendPolicy::PreferGpu;
    app.update();
    let backend_edit = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert_eq!(backend_edit.fixed_tick, 1);
    assert_eq!(backend_edit.snapshot_version, 1);
}

#[test]
fn diagnostics_separate_fixed_time_from_aggregate_runtime_work() {
    let mut app = app();
    app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.update();

    let diagnostics = app.world().resource::<MpmDiagnostics>();
    assert_eq!(diagnostics.steps, 2);
    assert_eq!(diagnostics.simulated_dt, 1.0e-3);
}

#[test]
fn preferred_gpu_reports_actual_authority_or_explicit_fallback() {
    let mut app = app();
    let entity = app.world_mut().spawn((MpmBinding::prefer_gpu(solver()),));
    app.update();

    let info = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .expect("preferred-GPU runtime");
    match info.authority {
        MpmAuthority::Gpu => {
            assert_eq!(info.fallback, None);
            assert!(!app
                .world()
                .resource::<MpmDiagnostics>()
                .issues
                .iter()
                .any(|diagnostic| matches!(diagnostic.issue, MpmIssue::BackendFallback(_))));
        }
        MpmAuthority::Cpu => {
            assert_eq!(info.fallback, Some(MpmFallbackReason::GpuUnavailable));
            assert!(app
                .world()
                .resource::<MpmDiagnostics>()
                .issues
                .iter()
                .any(|diagnostic| diagnostic.entity == Some(entity)
                    && diagnostic.issue
                        == MpmIssue::BackendFallback(MpmFallbackReason::GpuUnavailable)));
        }
    }
}
