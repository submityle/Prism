use std::sync::Arc;
use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_physics_core::mpm::{Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver};
use prism_physics_ecs::{
    MpmAuthority, MpmBackendPolicy, MpmBinding, MpmDiagnostics, MpmFallbackReason, MpmIssue,
    MpmParticleSnapshot, MpmRuntimeResource, MpmSurfaceConfig, MpmSurfaceSnapshot, PhysicsBindings,
    PrismPhysicsPlugin, RigidBody, Vec3,
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
fn mpm_surface_reconstruction_tracks_source_version_and_clears_when_disabled() {
    let mut app = app();
    let entity = app
        .world_mut()
        .spawn((
            MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig {
                radius: 0.2,
                dx: 0.05,
                iso: 0.1,
                padding_cells: 1,
                max_field_nodes: 100_000,
                max_cells: 20_000,
                ..MpmSurfaceConfig::default()
            }),
        ));

    app.update();
    let particles = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    let surface = app.world().get::<MpmSurfaceSnapshot>(entity).unwrap();
    assert_eq!(surface.version, particles.version);
    assert_eq!(surface.tick, particles.tick);
    assert!(!surface.mesh.is_empty());
    assert!(surface.mesh.indices_are_valid());
    assert_eq!(surface.mesh.normals.len(), surface.mesh.positions.len());

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = None;
    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
}

#[test]
fn mpm_surface_budget_rejects_unbounded_field_before_allocation() {
    let mut app = app();
    let entity = app
        .world_mut()
        .spawn((
            MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig {
                max_field_nodes: 8,
                max_cells: 1,
                ..MpmSurfaceConfig::default()
            }),
        ));

    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|diagnostic| diagnostic.entity == Some(entity)
            && diagnostic.issue == MpmIssue::SurfaceBudgetExceeded));
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
fn every_surface_work_output_budget_clears_stale_mesh_and_recovers_without_advancing_particles() {
    for (name, rejected) in [
        (
            "sampling",
            MpmSurfaceConfig {
                max_sample_visits: 1,
                ..MpmSurfaceConfig::default()
            },
        ),
        (
            "vertices",
            MpmSurfaceConfig {
                max_surface_vertices: 1,
                ..MpmSurfaceConfig::default()
            },
        ),
        (
            "triangles",
            MpmSurfaceConfig {
                max_surface_triangles: 1,
                ..MpmSurfaceConfig::default()
            },
        ),
    ] {
        let mut app = app();
        let entity =
            app.world_mut()
                .spawn((MpmBinding::cpu(solver())
                    .with_surface_reconstruction(MpmSurfaceConfig::default()),));
        app.update();
        let original = app
            .world()
            .get::<MpmSurfaceSnapshot>(entity)
            .unwrap()
            .clone();
        let particle_version = app
            .world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .version;
        app.set_fixed_timestep(Duration::from_millis(2))
            .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(2)));
        app.world_mut()
            .get_mut::<MpmBinding>(entity)
            .unwrap()
            .surface = Some(rejected);
        app.update();
        assert!(
            app.world().get::<MpmSurfaceSnapshot>(entity).is_none(),
            "{name}"
        );
        assert_eq!(
            app.world()
                .get::<MpmParticleSnapshot>(entity)
                .unwrap()
                .version,
            particle_version
        );
        assert!(
            app.world()
                .resource::<MpmDiagnostics>()
                .issues
                .iter()
                .any(|diagnostic| diagnostic.entity == Some(entity)
                    && diagnostic.issue == MpmIssue::SurfaceBudgetExceeded),
            "{name}"
        );
        app.world_mut()
            .get_mut::<MpmBinding>(entity)
            .unwrap()
            .surface = Some(MpmSurfaceConfig::default());
        app.update();
        let restored = app.world().get::<MpmSurfaceSnapshot>(entity).unwrap();
        assert_eq!(restored.version, original.version, "{name}");
        assert_eq!(restored.epoch, original.epoch, "{name}");
        assert!(restored.revision > original.revision, "{name}");
        assert_eq!(restored.mesh, original.mesh, "{name}");
    }
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

#[test]
fn snapshot_storage_is_shared_and_runtime_epochs_survive_removal() {
    let mut app = app();
    let entity = app.world_mut().spawn((MpmBinding::cpu(solver()),));
    app.update();
    let first = app
        .world()
        .get::<MpmParticleSnapshot>(entity)
        .unwrap()
        .clone();
    let inspected = app
        .world()
        .resource::<MpmRuntimeResource>()
        .snapshot(entity)
        .unwrap();
    assert_ne!(first.epoch, 0);
    assert!(Arc::ptr_eq(&first.positions, &inspected.positions));
    assert!(Arc::ptr_eq(&first.velocities, &inspected.velocities));
    assert!(Arc::ptr_eq(&first.affine, &inspected.affine));
    assert!(Arc::ptr_eq(&first.deformation, &inspected.deformation));
    assert!(Arc::ptr_eq(&first.plastic_det, &inspected.plastic_det));
    app.update();
    assert_eq!(
        app.world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .epoch,
        first.epoch
    );
    app.world_mut()
        .resource_mut::<MpmRuntimeResource>()
        .remove(entity);
    app.update();
    let recreated = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert!(recreated.epoch > first.epoch);
    assert_eq!(recreated.version, 1);
    assert!(first.positions[0].is_finite());
}

#[test]
fn rebuilding_a_runtime_clears_old_generation_even_when_new_step_is_rejected() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
    ));
    app.update();
    let epoch = app
        .world()
        .get::<MpmParticleSnapshot>(entity)
        .unwrap()
        .epoch;
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_some());
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .solver
        .config
        .dt = 0.002;
    app.update();
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    let runtime = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert!(runtime.epoch > epoch);
    assert_eq!(runtime.fixed_tick, 0);
    assert_eq!(runtime.snapshot_version, 0);
}

#[test]
fn surface_config_edits_publish_or_clear_without_a_successful_simulation_step() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
    ));
    app.update();
    let first = app
        .world()
        .get::<MpmSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    assert!(Arc::ptr_eq(
        &first.mesh,
        &app.world().get::<MpmSurfaceSnapshot>(entity).unwrap().mesh
    ));
    // Only the engine clock changes. The live solver cannot advance at this dt.
    app.set_fixed_timestep(Duration::from_millis(2))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(2)));
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface
        .as_mut()
        .unwrap()
        .radius = 0.15;
    app.update();
    let changed = app.world().get::<MpmSurfaceSnapshot>(entity).unwrap();
    assert_eq!(changed.epoch, first.epoch);
    assert_eq!(changed.version, first.version);
    assert_eq!(changed.tick, first.tick);
    assert!(changed.revision > first.revision);
    assert!(!Arc::ptr_eq(&changed.mesh, &first.mesh));
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = None;
    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = Some(MpmSurfaceConfig {
        dx: f32::NAN,
        ..MpmSurfaceConfig::default()
    });
    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| d.issue == MpmIssue::InvalidSurfaceConfig));
    assert_eq!(
        app.world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .version,
        first.version
    );
}

#[test]
fn surface_budget_edits_clear_the_old_mesh_even_if_fixed_step_is_rejected() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
    ));
    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_some());
    app.set_fixed_timestep(Duration::from_millis(2))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(2)));
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = Some(MpmSurfaceConfig {
        max_field_nodes: 8,
        max_cells: 1,
        ..MpmSurfaceConfig::default()
    });
    app.update();
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| d.issue == MpmIssue::SurfaceBudgetExceeded));
}

#[test]
fn remove_readd_equal_authoring_starts_a_new_runtime_lifetime() {
    let mut app = app();
    let binding =
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default());
    let entity = app.world_mut().spawn((binding.clone(),));
    app.update();
    app.update();
    let previous = app
        .world()
        .get::<MpmParticleSnapshot>(entity)
        .unwrap()
        .clone();
    assert_eq!(previous.version, 2);
    app.world_mut().remove::<MpmBinding>(entity);
    app.world_mut().insert(entity, (binding,));
    app.update();
    let current = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert!(current.epoch > previous.epoch);
    assert_eq!(current.version, 1);
    assert!(current.positions[0].y > previous.positions[0].y);
    let surface = app.world().get::<MpmSurfaceSnapshot>(entity).unwrap();
    assert_eq!(surface.epoch, current.epoch);
    assert_eq!(surface.revision, 1);
}

#[test]
fn exhausted_fixed_tick_retires_outputs_without_wrapping_or_stepping() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
    ));
    app.update();
    app.world_mut().resource_mut::<MpmDiagnostics>().tick = u64::MAX;
    for _ in 0..2 {
        app.update();
        assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
        assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
        assert!(app.world().resource::<MpmRuntimeResource>().is_empty());
        let diagnostics = app.world().resource::<MpmDiagnostics>();
        assert_eq!(diagnostics.tick, u64::MAX);
        assert_eq!(diagnostics.steps, 0);
        assert!(diagnostics
            .issues
            .iter()
            .any(|d| d.issue == MpmIssue::SequenceExhausted));
    }
}
