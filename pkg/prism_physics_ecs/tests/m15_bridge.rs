//! End-to-end coverage for the M15 ECS topology bridge.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use glam::Vec3;
use prism_app::{App, TimeUpdateStrategy};
use prism_physics_core::soft::body::SoftBody;
use prism_physics_core::soft::topology::{
    TearableMesh, TopologyTearingConfig, TopologyUpdateConsumer,
};
use prism_physics_core::soft::{ParticleHandle, SoftBodyHandle, SoftSolverConfig};
use prism_physics_core::world::WorldTopologyTearUpdate;
use prism_physics_ecs::{
    M15TopologyAdapter, M15TopologyAdapterError, M15TopologyBridge, M15TopologyBridgeError,
    M15TopologyBridgePlugin, PhysicsBindings, PhysicsWorldResource, PrismPhysicsPlugin,
    SoftBodyBinding,
};

#[derive(Default)]
struct AdapterLog {
    registrations: Vec<(SoftBodyHandle, u64)>,
    updates: Vec<WorldTopologyTearUpdate>,
    states: Vec<(SoftBodyHandle, TopologyUpdateConsumer)>,
    removals: Vec<SoftBodyHandle>,
}

struct RecordingAdapter {
    log: Arc<Mutex<AdapterLog>>,
    reject_apply: bool,
}

impl M15TopologyAdapter for RecordingAdapter {
    fn register(
        &mut self,
        body: SoftBodyHandle,
        state: &TopologyUpdateConsumer,
    ) -> Result<(), M15TopologyAdapterError> {
        self.log
            .lock()
            .unwrap()
            .registrations
            .push((body, state.topology_version()));
        Ok(())
    }

    fn apply(
        &mut self,
        update: &WorldTopologyTearUpdate,
        state: &TopologyUpdateConsumer,
    ) -> Result<(), M15TopologyAdapterError> {
        if self.reject_apply {
            return Err(M15TopologyAdapterError::Rejected);
        }
        let mut log = self.log.lock().unwrap();
        log.updates.push(update.clone());
        log.states.push((update.body, state.clone()));
        Ok(())
    }

    fn remove(&mut self, body: SoftBodyHandle) {
        self.log.lock().unwrap().removals.push(body);
    }
}

fn app() -> App {
    physics_app(true)
}

fn physics_app(bridge: bool) -> App {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins(PrismPhysicsPlugin);
    if bridge {
        app.add_plugins(M15TopologyBridgePlugin);
    }
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .gravity = Vec3::ZERO;
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 1;
    app
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

fn tearing_body() -> SoftBody {
    let mut body = SoftBody::new(SoftSolverConfig {
        gravity: Vec3::ZERO,
        damping: 0.0,
        substeps: 1,
        iterations: 0,
        ..SoftSolverConfig::default()
    });
    let topology = mesh();
    for position in topology.positions.iter().copied() {
        body.spawn(position, 1.0);
    }
    body.attach_topology(topology, 0.0, 0.0).unwrap();
    body.topology_config = TopologyTearingConfig {
        break_stress: 0.1,
        max_front_steps: 1,
        ..TopologyTearingConfig::default()
    };
    body.particles
        .set_position(ParticleHandle::from_index(3), Vec3::new(3.0, 0.0, 3.0));
    body
}

fn soft_body(app: &App, entity: prism_ecs::prelude::Entity) -> SoftBodyHandle {
    app.world()
        .resource::<PhysicsBindings>()
        .soft_body(entity)
        .unwrap()
}

#[test]
fn fixed_step_routes_one_atomic_render_collision_and_ccd_revision() {
    let log = Arc::new(Mutex::new(AdapterLog::default()));
    let mut app = app();
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&log),
            reject_apply: false,
        });

    app.update();

    let body = soft_body(&app, entity);
    let log = log.lock().unwrap();
    assert_eq!(log.registrations, vec![(body, 0)]);
    assert_eq!(log.updates.len(), 1);
    assert_eq!(log.states.len(), 1);
    let update = &log.updates[0];
    let state = &log.states[0].1;
    assert_eq!(update.body, body);
    assert_eq!(update.update.topology_version, 1);
    assert_eq!(update.update.render.topology_version, 1);
    assert_eq!(update.update.collision.topology_version, 1);
    assert_eq!(update.update.collision.ccd_revision, 1);
    assert_eq!(
        update.update.render.triangles,
        update.update.collision.triangles
    );
    assert!(!update.update.events.is_empty());
    assert_eq!(state.topology_version(), 1);
    assert_eq!(state.render(), &update.update.render);
    assert_eq!(state.collision(), &update.update.collision);
    assert_eq!(
        state.swept_triangle_bounds().len(),
        state.collision().triangles.len()
    );
    drop(log);

    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.revision(body), Some(1));
    assert_eq!(bridge.pending_len(), 0);
    assert_eq!(bridge.diagnostics().applied_updates, 1);
    assert_eq!(bridge.diagnostics().registered_bodies, 1);
    assert_eq!(bridge.diagnostics().last_error, None);
}

#[test]
fn adapter_rejection_keeps_revision_queued_for_ordered_retry() {
    let rejected_log = Arc::new(Mutex::new(AdapterLog::default()));
    let mut app = app();
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&rejected_log),
            reject_apply: true,
        });

    app.update();

    let body = soft_body(&app, entity);
    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.revision(body), Some(0));
    assert_eq!(bridge.pending_len(), 1);
    assert_eq!(bridge.diagnostics().applied_updates, 0);
    assert_eq!(
        bridge.diagnostics().last_error,
        Some(M15TopologyBridgeError::AdapterRejected(body))
    );
    assert!(rejected_log.lock().unwrap().updates.is_empty());

    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .soft_bodies
        .get_mut(body)
        .unwrap()
        .topology_config
        .break_stress = f32::MAX;
    let accepted_log = Arc::new(Mutex::new(AdapterLog::default()));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&accepted_log),
            reject_apply: false,
        });

    app.update();

    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.revision(body), Some(1));
    assert_eq!(bridge.pending_len(), 0);
    assert_eq!(bridge.diagnostics().applied_updates, 1);
    assert_eq!(bridge.diagnostics().last_error, None);
    let accepted_log = accepted_log.lock().unwrap();
    assert_eq!(accepted_log.registrations, vec![(body, 0)]);
    assert_eq!(accepted_log.updates.len(), 1);
    assert_eq!(accepted_log.updates[0].update.topology_version, 1);
}

#[test]
fn retiring_soft_body_unregisters_consumer_and_downstream_state() {
    let log = Arc::new(Mutex::new(AdapterLog::default()));
    let mut app = app();
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&log),
            reject_apply: false,
        });
    app.update();
    let body = soft_body(&app, entity);

    assert!(app.world_mut().remove::<SoftBodyBinding>(entity));
    app.update();

    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.consumer(body), None);
    assert_eq!(bridge.pending_len(), 0);
    assert_eq!(bridge.diagnostics().registered_bodies, 0);
    assert_eq!(log.lock().unwrap().removals, vec![body]);
}

#[test]
fn late_plugin_without_history_registers_the_current_physics_revision() {
    let mut app = physics_app(false);
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.update();
    let body = soft_body(&app, entity);
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert_eq!(physics.drain_topology_updates().len(), 1);
        let live = physics.soft_bodies.get_mut(body).unwrap();
        assert_eq!(live.topology_version(), Some(1));
        live.topology_config.max_front_steps = 0;
    }
    app.add_plugins(M15TopologyBridgePlugin);
    let log = Arc::new(Mutex::new(AdapterLog::default()));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&log),
            reject_apply: false,
        });
    app.update();
    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.revision(body), Some(1));
    assert_eq!(bridge.pending_len(), 0);
    assert_eq!(bridge.diagnostics().last_error, None);
    let log = log.lock().unwrap();
    assert_eq!(log.registrations, vec![(body, 1)]);
    assert!(log.updates.is_empty());
}

#[test]
fn late_plugin_bootstraps_missing_history_then_applies_retained_updates_in_order() {
    let mut app = physics_app(false);
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.update();
    let body = soft_body(&app, entity);
    let mut retained = Vec::new();
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.drain_topology_updates();
        let live = physics.soft_bodies.get_mut(body).unwrap();
        for expected in [2, 3] {
            let events = live
                .puncture_topology(Vec3::ONE, Vec3::Z, 100.0)
                .expect("remaining interior cloth edge");
            assert!(!events.is_empty());
            let update = live.topology_update.take().unwrap();
            assert_eq!(update.topology_version, expected);
            retained.push(WorldTopologyTearUpdate { body, update });
        }
        live.topology_config.max_front_steps = 0;
        physics.topology_updates = retained.clone();
    }
    app.add_plugins(M15TopologyBridgePlugin);
    let log = Arc::new(Mutex::new(AdapterLog::default()));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&log),
            reject_apply: false,
        });
    app.update();
    let bridge = app.world().resource::<M15TopologyBridge>();
    assert_eq!(bridge.revision(body), Some(3));
    assert_eq!(bridge.pending_len(), 0);
    assert_eq!(bridge.diagnostics().last_error, None);
    assert_eq!(bridge.diagnostics().applied_updates, 1);
    let log = log.lock().unwrap();
    assert_eq!(log.registrations, vec![(body, 2)]);
    assert_eq!(log.updates, vec![retained[1].clone()]);
}

#[test]
fn stable_stream_reuses_consumer_arrays_and_never_reregisters() {
    let log = Arc::new(Mutex::new(AdapterLog::default()));
    let mut app = app();
    let entity = app.world_mut().spawn((SoftBodyBinding(tearing_body()),));
    app.world_mut()
        .resource_mut::<M15TopologyBridge>()
        .set_adapter(RecordingAdapter {
            log: Arc::clone(&log),
            reject_apply: false,
        });
    app.update();
    let body = soft_body(&app, entity);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .soft_bodies
        .get_mut(body)
        .unwrap()
        .topology_config
        .max_front_steps = 0;
    let arrays = {
        let state = app
            .world()
            .resource::<M15TopologyBridge>()
            .consumer(body)
            .unwrap();
        (
            state.render().vertices.as_ptr(),
            state.render().triangles.as_ptr(),
            state.collision().triangle_bounds.as_ptr(),
            state.previous_positions().as_ptr(),
        )
    };
    for _ in 0..8 {
        app.update();
        let state = app
            .world()
            .resource::<M15TopologyBridge>()
            .consumer(body)
            .unwrap();
        assert_eq!(state.topology_version(), 1);
        assert_eq!(
            arrays,
            (
                state.render().vertices.as_ptr(),
                state.render().triangles.as_ptr(),
                state.collision().triangle_bounds.as_ptr(),
                state.previous_positions().as_ptr(),
            )
        );
    }
    let log = log.lock().unwrap();
    assert_eq!(log.registrations, vec![(body, 0)]);
    assert_eq!(log.updates.len(), 1);
}
