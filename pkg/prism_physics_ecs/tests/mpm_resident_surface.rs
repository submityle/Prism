//! Resident publication lifecycle tests; GPU buffers are never mapped here.
use std::sync::OnceLock;
use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::{Entity, LifecycleEvent, Resource, World};
use prism_physics_core::mpm::{Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver};
use prism_physics_ecs::{
    MpmAuthority, MpmBackendPolicy, MpmBinding, MpmDiagnostics, MpmGpuHost, MpmIssue,
    MpmParticleSnapshot, MpmResidentSurfaceConfig, MpmResidentSurfaceSnapshot, MpmRuntimeResource,
    MpmSurfaceConfig, MpmSurfaceSnapshot, PrismPhysicsPlugin, Vec3,
};
use prism_physics_gpu::mpm::gpu::surface::{ResidentSurfaceConfig, ResidentSurfaceError};
use prism_physics_gpu::GpuContext;

fn context() -> Option<GpuContext> {
    static HOST: OnceLock<Option<GpuContext>> = OnceLock::new();
    let result = HOST.get_or_init(GpuContext::try_headless).clone();
    assert!(
        result.is_some() || std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set but no GPU available"
    );
    result
}

fn solver() -> MpmSolver {
    let mut particles = MaterialPoints::new();
    particles.spawn(Vec3::ONE, Vec3::new(0.1, 0.2, 0.05), 1.0, 0.001);
    MpmSolver::new(
        particles,
        Grid::new(16, 16, 16, 0.1, Vec3::ZERO),
        MpmMaterial::default(),
        MpmConfig::new(0.001, Vec3::new(0.0, -9.81, 0.0)),
    )
}

fn config() -> MpmResidentSurfaceConfig {
    MpmResidentSurfaceConfig::resident_only(ResidentSurfaceConfig {
        origin: Vec3::splat(0.8),
        nodes: [9; 3],
        dx: 0.05,
        radius: 0.16,
        iso: 0.13,
        max_field_nodes: 729,
        max_cells: 512,
        max_sample_visits: 729,
        max_vertices: 512 * 36,
        max_live_publications: 4,
    })
}

fn binding() -> MpmBinding {
    MpmBinding::with_backend(solver(), MpmBackendPolicy::RequireGpu, 1)
}

fn app(host: Option<&GpuContext>) -> App {
    let mut app = App::new();
    app.set_fixed_timestep(Duration::from_millis(1))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(1)))
        .add_plugins(PrismPhysicsPlugin);
    if let Some(host) = host {
        app.world_mut().insert_resource(MpmGpuHost(host.clone()));
    }
    app
}

// Reject the solver's fixed dt, without changing authoring or rebuilding its
// runtime. Publication/configuration retries must preserve the source tick.
fn reject_steps(app: &mut App) {
    app.set_fixed_timestep(Duration::from_millis(2))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(2)));
}

#[test]
fn resident_only_uses_host_and_versions_without_creating_portable_snapshots() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((binding(), config()));
    app.update();
    let first = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    assert!(first.surface.is_for_context(&ctx));
    assert_eq!((first.version, first.tick, first.revision), (1, 1, 1));
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    let runtimes = app.world().resource::<MpmRuntimeResource>();
    assert!(runtimes.snapshot(entity).is_none());
    let info = runtimes.info(entity).unwrap();
    assert_eq!(info.authority, MpmAuthority::Gpu);
    assert_eq!((info.fixed_tick, info.snapshot_version), (1, 1));

    reject_steps(&mut app);
    app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(entity)
        .unwrap()
        .surface
        .iso = 0.2;
    app.update();
    let changed = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap();
    assert_eq!(
        (changed.epoch, changed.version, changed.tick),
        (first.epoch, 1, 1)
    );
    assert_eq!((changed.revision, changed.surface.sequence()), (2, 2));
    assert_eq!(changed.surface.config().iso, 0.2);
    assert_eq!(first.surface.config().iso, 0.13);
    assert_eq!(app.world().resource::<MpmDiagnostics>().steps, 0);
}

#[test]
fn held_lease_capacity_retries_without_advancing_or_relabeling_source_tick() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let mut limited = config();
    limited.surface.max_live_publications = 1;
    let entity = app.world_mut().spawn((binding(), limited));
    app.update();
    let pin = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| {
            d.entity == Some(entity)
                && d.issue == MpmIssue::ResidentSurface(ResidentSurfaceError::InFlightLimit)
        }));
    reject_steps(&mut app);
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    assert!(pin.surface.is_for_context(&ctx));
    drop(pin);
    app.update();
    let retried = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap();
    assert_eq!((retried.version, retried.tick, retried.revision), (2, 2, 2));
    assert_eq!(retried.surface.sequence(), 2);
    assert_eq!(
        app.world()
            .resource::<MpmRuntimeResource>()
            .info(entity)
            .unwrap()
            .fixed_tick,
        2
    );
    assert_eq!(app.world().resource::<MpmDiagnostics>().steps, 0);
}

#[test]
fn invalid_and_budget_configs_remove_stale_publications_and_recover_same_state() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((binding(), config()));
    app.update();
    let first = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    reject_steps(&mut app);
    for (rejected, error) in [
        (
            ResidentSurfaceConfig {
                dx: 0.0,
                ..config().surface
            },
            ResidentSurfaceError::InvalidConfig,
        ),
        (
            ResidentSurfaceConfig {
                max_field_nodes: 728,
                ..config().surface
            },
            ResidentSurfaceError::BudgetExceeded,
        ),
        (
            ResidentSurfaceConfig {
                max_cells: 511,
                ..config().surface
            },
            ResidentSurfaceError::BudgetExceeded,
        ),
        (
            ResidentSurfaceConfig {
                max_sample_visits: 728,
                ..config().surface
            },
            ResidentSurfaceError::BudgetExceeded,
        ),
        (
            ResidentSurfaceConfig {
                max_vertices: u32::MAX,
                ..config().surface
            },
            ResidentSurfaceError::DeviceLimit,
        ),
    ] {
        app.world_mut()
            .get_mut::<MpmResidentSurfaceConfig>(entity)
            .unwrap()
            .surface = rejected;
        app.update();
        assert!(app
            .world()
            .get::<MpmResidentSurfaceSnapshot>(entity)
            .is_none());
        assert!(app
            .world()
            .resource::<MpmDiagnostics>()
            .issues
            .iter()
            .any(|d| { d.entity == Some(entity) && d.issue == MpmIssue::ResidentSurface(error) }));
        *app.world_mut()
            .get_mut::<MpmResidentSurfaceConfig>(entity)
            .unwrap() = config();
        app.update();
        let recovered = app
            .world()
            .get::<MpmResidentSurfaceSnapshot>(entity)
            .unwrap();
        assert_eq!(
            (recovered.epoch, recovered.version, recovered.tick),
            (first.epoch, 1, 1)
        );
        assert!(recovered.revision > first.revision);
    }
}

#[test]
fn one_entity_budget_failure_does_not_block_other_publications_or_advance_recovery() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let mut rejected = config();
    rejected.surface.max_sample_visits = 0;
    let failed = app.world_mut().spawn((binding(), rejected));
    let healthy = app.world_mut().spawn((binding(), config()));
    app.update();

    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(failed)
        .is_none());
    let published = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(healthy)
        .unwrap()
        .clone();
    assert_eq!(
        (published.version, published.tick, published.revision),
        (1, 1, 1)
    );
    assert!(published.surface.is_for_context(&ctx));
    let failed_epoch = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(failed)
        .unwrap()
        .epoch;
    let diagnostics = app.world().resource::<MpmDiagnostics>();
    assert_eq!(diagnostics.steps, 2);
    assert!(diagnostics.issues.iter().any(|d| {
        d.entity == Some(failed)
            && d.issue == MpmIssue::ResidentSurface(ResidentSurfaceError::BudgetExceeded)
    }));
    assert!(!diagnostics.issues.iter().any(|d| d.entity == Some(healthy)));

    reject_steps(&mut app);
    *app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(failed)
        .unwrap() = config();
    app.update();

    let recovered = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(failed)
        .unwrap();
    assert_eq!(
        (
            recovered.epoch,
            recovered.version,
            recovered.tick,
            recovered.revision
        ),
        (failed_epoch, 1, 1, 1)
    );
    assert_eq!(recovered.surface.sequence(), 1);
    let unchanged = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(healthy)
        .unwrap();
    assert_eq!(
        (
            unchanged.epoch,
            unchanged.version,
            unchanged.tick,
            unchanged.revision
        ),
        (published.epoch, 1, 1, 1)
    );
    assert_eq!(unchanged.surface.sequence(), published.surface.sequence());
    for entity in [failed, healthy] {
        let runtimes = app.world().resource::<MpmRuntimeResource>();
        let info = runtimes.info(entity).unwrap();
        assert_eq!((info.fixed_tick, info.snapshot_version), (1, 1));
        assert!(runtimes.snapshot(entity).is_none());
        assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
        assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    }
    assert_eq!(app.world().resource::<MpmDiagnostics>().steps, 0);
    assert!(!app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| matches!(d.issue, MpmIssue::ResidentSurface(_))));
}

#[test]
fn readback_opt_in_cpu_surface_override_and_config_removal_preserve_source_version() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((binding(), config()));
    app.update();
    reject_steps(&mut app);
    app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(entity)
        .unwrap()
        .readback_particles = true;
    app.update();
    let portable = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert_eq!((portable.version, portable.tick), (1, 1));
    assert!(app
        .world()
        .resource::<MpmRuntimeResource>()
        .snapshot(entity)
        .is_some());
    app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(entity)
        .unwrap()
        .readback_particles = false;
    app.update();
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = Some(MpmSurfaceConfig::default());
    app.update();
    let cpu_surface = app.world().get::<MpmSurfaceSnapshot>(entity).unwrap();
    assert_eq!((cpu_surface.version, cpu_surface.tick), (1, 1));
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_some());
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = None;
    app.update();
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());

    assert!(app.world_mut().remove::<MpmResidentSurfaceConfig>(entity));
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    let portable = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert_eq!((portable.version, portable.tick), (1, 1));
}

#[test]
fn host_replacement_and_binding_lifecycle_invalidate_epochs_but_not_consumer_leases() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((binding(), config()));
    app.update();
    let pin = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    reject_steps(&mut app);
    app.world_mut().insert_resource(MpmGpuHost(ctx.clone()));
    app.update();
    assert_eq!(
        app.world()
            .resource::<MpmRuntimeResource>()
            .info(entity)
            .unwrap()
            .epoch,
        pin.epoch
    );

    let replacement = GpuContext::from_wgpu(
        ctx.instance().clone(),
        ctx.adapter().clone(),
        ctx.device().clone(),
        ctx.queue().clone(),
    );
    app.world_mut()
        .insert_resource(MpmGpuHost(replacement.clone()));
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    let new_epoch = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap()
        .epoch;
    assert!(new_epoch > pin.epoch);
    assert!(!pin.surface.is_for_context(&replacement));
    assert!(pin.surface.is_for_context(&ctx));

    app.set_fixed_timestep(Duration::from_millis(1))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(1)));
    app.update();
    let fresh = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap();
    assert_eq!((fresh.epoch, fresh.version), (new_epoch, 1));
    assert!(fresh.surface.is_for_context(&replacement));
    assert!(app.world_mut().remove::<MpmBinding>(entity));
    app.world_mut().insert(entity, (binding(),));
    app.update();
    assert!(
        app.world()
            .get::<MpmResidentSurfaceSnapshot>(entity)
            .unwrap()
            .epoch
            > new_epoch
    );
    assert!(app.world_mut().remove::<MpmBinding>(entity));
    app.update();
    assert!(!app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(entity));
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());

    let removed = app.world_mut().spawn((binding(), config()));
    app.update();
    assert!(app.world_mut().despawn(removed));
    app.update();
    assert!(!app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(removed));
    assert!(pin.surface.is_for_context(&ctx));
}

#[test]
fn invalid_binding_clears_native_output_and_recovery_starts_a_new_epoch() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((binding(), config()));
    app.update();
    let pin = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();

    reject_steps(&mut app);
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .max_steps_per_advance = 0;
    app.update();
    assert!(!app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(entity));
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
    assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| d.entity == Some(entity) && d.issue == MpmIssue::InvalidAuthoring));

    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .max_steps_per_advance = 1;
    app.update();
    let info = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    let new_epoch = info.epoch;
    assert!(new_epoch > pin.epoch);
    assert_eq!((info.fixed_tick, info.snapshot_version), (0, 0));
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());

    app.set_fixed_timestep(Duration::from_millis(1))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(1)));
    app.update();
    let recovered = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap();
    assert_eq!(
        (recovered.epoch, recovered.version, recovered.revision),
        (new_epoch, 1, 1)
    );
    assert_eq!(recovered.surface.sequence(), 1);
    assert_eq!(
        (pin.version, pin.revision, pin.surface.sequence()),
        (1, 1, 1)
    );
    assert!(pin.surface.is_for_context(&ctx));
    assert!(recovered.surface.is_for_context(&ctx));
    assert!(app.world().get::<MpmParticleSnapshot>(entity).is_none());
}

#[test]
fn cpu_backend_rejects_native_publication_but_keeps_portable_output() {
    let mut app = app(None);
    let entity = app.world_mut().spawn((MpmBinding::cpu(solver()), config()));
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    let portable = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
    assert_eq!((portable.version, portable.tick), (1, 1));
    assert!(app
        .world()
        .resource::<MpmDiagnostics>()
        .issues
        .iter()
        .any(|d| {
            d.entity == Some(entity)
                && d.issue == MpmIssue::ResidentSurface(ResidentSurfaceError::BackendUnavailable)
        }));
}

#[test]
fn publication_observers_see_committed_resources_and_cannot_leave_replaced_authoring_stale() {
    #[derive(Resource, Default)]
    struct Replaced(bool);
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    app.init_resource::<Replaced>();
    let mut cfg = config();
    cfg.readback_particles = true;
    let entity = app.world_mut().spawn((binding(), cfg));
    app.world_mut()
        .observe::<MpmParticleSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
            assert!(ctx.world.contains_resource::<MpmRuntimeResource>());
            assert!(ctx.world.contains_resource::<MpmDiagnostics>());
            if ctx.entity == entity && !ctx.world.resource::<Replaced>().0 {
                ctx.world.resource_mut::<Replaced>().0 = true;
                assert!(ctx.world.remove::<MpmResidentSurfaceConfig>(entity));
                assert!(ctx.world.insert(entity, (cfg,)));
            }
        });
    app.update();
    assert!(app.world().resource::<Replaced>().0);
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    app.world_mut()
        .observe::<MpmResidentSurfaceSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
            assert!(ctx.world.contains_resource::<MpmRuntimeResource>());
            assert!(ctx.world.contains_resource::<MpmDiagnostics>());
            if ctx.entity == entity {
                assert!(ctx.world.remove::<MpmResidentSurfaceConfig>(entity));
            }
        });
    app.update();
    assert!(app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .is_none());
    assert!(app
        .world()
        .get::<MpmResidentSurfaceConfig>(entity)
        .is_none());
    app.update();
    assert_eq!(
        app.world()
            .get::<MpmParticleSnapshot>(entity)
            .unwrap()
            .version,
        3
    );
}

#[derive(Resource, Default)]
struct EditedInObserver(bool);

#[derive(Clone, Copy, Debug)]
enum PublicationTrigger {
    Particles,
    PortableSurface,
    ResidentSurface,
}

#[derive(Clone, Copy, Debug)]
enum InPlaceEdit {
    Solver,
    StepBudget,
    CpuBackend,
    InvalidStepBudget,
    SurfaceConfig,
    NoChange,
}

fn edit_in_observer(world: &mut World, entity: Entity, edit: InPlaceEdit) {
    if world.resource::<EditedInObserver>().0 {
        return;
    }
    world.resource_mut::<EditedInObserver>().0 = true;
    // Deliberately do not remove/reinsert: no lifecycle authoring observer is
    // fired by these writes. Publication must inspect the authored identity.
    let mut authored = world.get_mut::<MpmBinding>(entity).unwrap();
    match edit {
        InPlaceEdit::Solver => authored.solver.config.gravity.x = 0.75,
        InPlaceEdit::StepBudget => authored.max_steps_per_advance = 2,
        InPlaceEdit::CpuBackend => authored.backend = MpmBackendPolicy::Cpu,
        InPlaceEdit::InvalidStepBudget => authored.max_steps_per_advance = 0,
        InPlaceEdit::SurfaceConfig => authored.surface.as_mut().unwrap().iso = 0.2,
        InPlaceEdit::NoChange => {
            let budget = authored.max_steps_per_advance;
            authored.max_steps_per_advance = budget;
        }
    }
}

fn observe_in_place_edit(
    app: &mut App,
    entity: Entity,
    trigger: PublicationTrigger,
    edit: InPlaceEdit,
) {
    app.init_resource::<EditedInObserver>();
    match trigger {
        PublicationTrigger::Particles => {
            app.world_mut()
                .observe::<MpmParticleSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
                    if ctx.entity == entity {
                        edit_in_observer(ctx.world, entity, edit);
                    }
                });
        }
        PublicationTrigger::PortableSurface => {
            app.world_mut()
                .observe::<MpmSurfaceSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
                    if ctx.entity == entity {
                        edit_in_observer(ctx.world, entity, edit);
                    }
                });
        }
        PublicationTrigger::ResidentSurface => {
            app.world_mut().observe::<MpmResidentSurfaceSnapshot, _>(
                LifecycleEvent::Insert,
                move |ctx| {
                    if ctx.entity == entity {
                        edit_in_observer(ctx.world, entity, edit);
                    }
                },
            );
        }
    }
}

fn assert_no_publications(world: &World, entity: Entity) {
    assert!(world.get::<MpmParticleSnapshot>(entity).is_none());
    assert!(world.get::<MpmSurfaceSnapshot>(entity).is_none());
    assert!(world.get::<MpmResidentSurfaceSnapshot>(entity).is_none());
    assert!(world
        .resource::<MpmRuntimeResource>()
        .snapshot(entity)
        .is_none());
}

fn assert_portable_identity(world: &World, entity: Entity) -> (u64, u64, u64) {
    let particles = world.get::<MpmParticleSnapshot>(entity).unwrap();
    let surface = world.get::<MpmSurfaceSnapshot>(entity).unwrap();
    let cached = world
        .resource::<MpmRuntimeResource>()
        .snapshot(entity)
        .unwrap();
    let identity = (particles.epoch, particles.version, particles.tick);
    assert_eq!((surface.epoch, surface.version, surface.tick), identity);
    assert_eq!((cached.epoch, cached.version, cached.tick), identity);
    assert_eq!(
        world
            .resource::<MpmRuntimeResource>()
            .info(entity)
            .unwrap()
            .epoch,
        identity.0
    );
    identity
}

#[test]
fn portable_observer_in_place_solver_edit_retires_cached_epoch_without_a_gpu() {
    for trigger in [
        PublicationTrigger::Particles,
        PublicationTrigger::PortableSurface,
    ] {
        let mut app = app(None);
        let entity =
            app.world_mut()
                .spawn((MpmBinding::cpu(solver())
                    .with_surface_reconstruction(MpmSurfaceConfig::default()),));
        app.update();
        let old = assert_portable_identity(app.world(), entity);
        observe_in_place_edit(&mut app, entity, trigger, InPlaceEdit::Solver);
        app.update();
        assert!(app.world().resource::<EditedInObserver>().0);
        assert_no_publications(app.world(), entity);
        assert!(!app
            .world()
            .resource::<MpmRuntimeResource>()
            .contains(entity));
        app.update();
        let fresh = assert_portable_identity(app.world(), entity);
        assert!(fresh.0 > old.0);
        assert_eq!(fresh.1, 1);
        assert!(fresh.2 > old.2);
    }
}

#[test]
fn publication_observers_in_place_runtime_edits_never_mix_native_and_portable_epochs() {
    let Some(ctx) = context() else {
        return;
    };
    for (trigger, edit) in [
        (PublicationTrigger::Particles, InPlaceEdit::Solver),
        (PublicationTrigger::PortableSurface, InPlaceEdit::StepBudget),
        (PublicationTrigger::ResidentSurface, InPlaceEdit::CpuBackend),
        (
            PublicationTrigger::ResidentSurface,
            InPlaceEdit::InvalidStepBudget,
        ),
    ] {
        let mut app = app(Some(&ctx));
        let mut cfg = config();
        cfg.readback_particles = true;
        let entity = app.world_mut().spawn((
            binding().with_surface_reconstruction(MpmSurfaceConfig::default()),
            cfg,
        ));
        app.update();
        let old = assert_portable_identity(app.world(), entity);
        let pin = app
            .world()
            .get::<MpmResidentSurfaceSnapshot>(entity)
            .unwrap()
            .clone();
        assert_eq!((pin.epoch, pin.version, pin.tick), old);
        observe_in_place_edit(&mut app, entity, trigger, edit);
        app.update();
        assert!(
            app.world().resource::<EditedInObserver>().0,
            "{trigger:?}/{edit:?}"
        );
        assert_no_publications(app.world(), entity);
        assert!(!app
            .world()
            .resource::<MpmRuntimeResource>()
            .contains(entity));
        if matches!(edit, InPlaceEdit::InvalidStepBudget) {
            app.update();
            assert_no_publications(app.world(), entity);
            assert!(app
                .world()
                .resource::<MpmDiagnostics>()
                .issues
                .iter()
                .any(|d| { d.entity == Some(entity) && d.issue == MpmIssue::InvalidAuthoring }));
            app.world_mut()
                .get_mut::<MpmBinding>(entity)
                .unwrap()
                .max_steps_per_advance = 1;
        }
        app.update();
        let fresh = assert_portable_identity(app.world(), entity);
        assert!(fresh.0 > old.0);
        assert_eq!(fresh.1, 1);
        assert!(fresh.2 > old.2);
        if matches!(edit, InPlaceEdit::CpuBackend) {
            assert!(app
                .world()
                .get::<MpmResidentSurfaceSnapshot>(entity)
                .is_none());
            assert_eq!(
                app.world()
                    .resource::<MpmRuntimeResource>()
                    .info(entity)
                    .unwrap()
                    .authority,
                MpmAuthority::Cpu
            );
        } else {
            let native = app
                .world()
                .get::<MpmResidentSurfaceSnapshot>(entity)
                .unwrap();
            assert_eq!((native.epoch, native.version, native.tick), fresh);
            assert_eq!(native.surface.sequence(), 1);
        }
        // Invalidating ECS ownership must not revoke a consumer's old lease.
        assert!(pin.surface.is_for_context(&ctx));
        assert_eq!((pin.epoch, pin.version, pin.tick), old);
        assert_eq!(pin.surface.sequence(), 1);
    }
}

#[test]
fn resident_observer_in_place_surface_edit_retries_same_source_without_rebuilding() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let entity = app.world_mut().spawn((
        binding().with_surface_reconstruction(MpmSurfaceConfig::default()),
        config(),
    ));
    app.update();
    let pin = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap()
        .clone();
    observe_in_place_edit(
        &mut app,
        entity,
        PublicationTrigger::ResidentSurface,
        InPlaceEdit::SurfaceConfig,
    );
    app.update();
    assert!(app.world().resource::<EditedInObserver>().0);
    assert_no_publications(app.world(), entity);
    let info = app
        .world()
        .resource::<MpmRuntimeResource>()
        .info(entity)
        .unwrap();
    assert_eq!(
        (info.epoch, info.snapshot_version, info.fixed_tick),
        (pin.epoch, 2, 2)
    );
    reject_steps(&mut app);
    app.update();
    let portable = assert_portable_identity(app.world(), entity);
    assert_eq!(portable, (pin.epoch, 2, 2));
    let native = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(entity)
        .unwrap();
    assert_eq!((native.epoch, native.version, native.tick), portable);
    assert!(native.revision > pin.revision);
    assert_eq!(app.world().resource::<MpmDiagnostics>().steps, 0);
    assert_eq!(
        app.world()
            .get::<MpmBinding>(entity)
            .unwrap()
            .surface
            .unwrap()
            .iso,
        0.2
    );
    assert_eq!(pin.surface.sequence(), 1);
}

#[test]
fn later_resident_observer_in_place_edit_invalidates_an_earlier_entity_only() {
    let Some(ctx) = context() else {
        return;
    };
    let mut app = app(Some(&ctx));
    let a = app.world_mut().spawn((
        binding().with_surface_reconstruction(MpmSurfaceConfig::default()),
        config(),
    ));
    let b = app.world_mut().spawn((
        binding().with_surface_reconstruction(MpmSurfaceConfig::default()),
        config(),
    ));
    let (earlier, later) = if a < b { (a, b) } else { (b, a) };
    app.update();
    let old_earlier = assert_portable_identity(app.world(), earlier);
    let old_later = assert_portable_identity(app.world(), later);
    app.init_resource::<EditedInObserver>();
    app.world_mut()
        .observe::<MpmResidentSurfaceSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
            if ctx.entity == later && !ctx.world.resource::<EditedInObserver>().0 {
                // Earlier has already passed its own local publication postcheck.
                let native = ctx
                    .world
                    .get::<MpmResidentSurfaceSnapshot>(earlier)
                    .unwrap();
                assert_eq!(native.version, 2);
                edit_in_observer(ctx.world, earlier, InPlaceEdit::Solver);
            }
        });
    app.update();
    assert!(app.world().resource::<EditedInObserver>().0);
    assert_no_publications(app.world(), earlier);
    assert!(!app
        .world()
        .resource::<MpmRuntimeResource>()
        .contains(earlier));
    let stable = assert_portable_identity(app.world(), later);
    assert_eq!((stable.0, stable.1), (old_later.0, 2));
    let native = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(later)
        .unwrap();
    assert_eq!((native.epoch, native.version, native.tick), stable);
    app.update();
    let fresh = assert_portable_identity(app.world(), earlier);
    assert!(fresh.0 > old_earlier.0);
    assert_eq!(fresh.1, 1);
    let native = app
        .world()
        .get::<MpmResidentSurfaceSnapshot>(earlier)
        .unwrap();
    assert_eq!((native.epoch, native.version, native.tick), fresh);
    assert_eq!(assert_portable_identity(app.world(), later).0, old_later.0);
}

#[test]
fn observer_in_place_equal_value_touch_does_not_retire_the_runtime() {
    let mut app = app(None);
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
    ));
    app.update();
    let old = assert_portable_identity(app.world(), entity);
    observe_in_place_edit(
        &mut app,
        entity,
        PublicationTrigger::Particles,
        InPlaceEdit::NoChange,
    );
    app.update();
    assert!(app.world().resource::<EditedInObserver>().0);
    let current = assert_portable_identity(app.world(), entity);
    assert_eq!((current.0, current.1), (old.0, 2));
    app.update();
    assert_eq!(assert_portable_identity(app.world(), entity).0, old.0);
}

#[test]
fn invalid_nan_surface_configs_do_not_invalidate_portable_source_identity() {
    let mut app = app(None);
    let entity = app.world_mut().spawn((
        MpmBinding::cpu(solver()).with_surface_reconstruction(MpmSurfaceConfig::default()),
        config(),
    ));
    app.update();
    let old = assert_portable_identity(app.world(), entity);
    reject_steps(&mut app);
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface
        .as_mut()
        .unwrap()
        .dx = f32::NAN;
    app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(entity)
        .unwrap()
        .surface
        .dx = f32::NAN;
    for _ in 0..2 {
        app.update();
        let particles = app.world().get::<MpmParticleSnapshot>(entity).unwrap();
        assert_eq!((particles.epoch, particles.version, particles.tick), old);
        let cached = app
            .world()
            .resource::<MpmRuntimeResource>()
            .snapshot(entity)
            .unwrap();
        assert_eq!((cached.epoch, cached.version, cached.tick), old);
        assert!(app.world().get::<MpmSurfaceSnapshot>(entity).is_none());
        assert!(app
            .world()
            .get::<MpmResidentSurfaceSnapshot>(entity)
            .is_none());
        assert_eq!(app.world().resource::<MpmDiagnostics>().steps, 0);
    }
    app.world_mut()
        .get_mut::<MpmBinding>(entity)
        .unwrap()
        .surface = Some(MpmSurfaceConfig::default());
    *app.world_mut()
        .get_mut::<MpmResidentSurfaceConfig>(entity)
        .unwrap() = config();
    app.update();
    assert_eq!(assert_portable_identity(app.world(), entity), old);
}
