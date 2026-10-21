//! M15 production bridge for versioned soft-body topology replacements.
//!
//! The physics core remains the only topology producer. This module binds its
//! complete render/collision/CCD payloads to ECS-managed soft-body lifetimes
//! and an engine-independent downstream adapter. The adapter is called only
//! after the core consumer has validated a complete next revision; rejection
//! leaves the original payload queued for an ordered retry.

use std::collections::{HashMap, HashSet};

use prism_app::{App, FixedPostUpdate, Plugin};
use prism_ecs::prelude::*;
use prism_physics_core::soft::topology::{
    TearableMesh, TopologyUpdateConsumer, TopologyUpdateError,
};
use prism_physics_core::soft::SoftBodyHandle;
use prism_physics_core::world::WorldTopologyTearUpdate;

use crate::{PhysicsBindings, PhysicsWorldResource, SoftBodyBinding};

/// Failure reported by a render/collision adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M15TopologyAdapterError {
    /// The adapter could not commit the complete topology state.
    Rejected,
}

/// Callable downstream contract for one ECS-managed topology stream.
///
/// `register` receives the complete initial state. `apply` receives both the
/// producer payload and the already validated candidate state, including the
/// consumer's conservative swept triangle bounds. Implementations must commit
/// render, collision, and CCD data as one all-or-nothing operation and return
/// [`M15TopologyAdapterError::Rejected`] without advancing their revision when
/// that is not possible.
pub trait M15TopologyAdapter: Send + Sync + 'static {
    /// Registers a live body and its current complete topology state.
    fn register(
        &mut self,
        body: SoftBodyHandle,
        state: &TopologyUpdateConsumer,
    ) -> Result<(), M15TopologyAdapterError>;

    /// Atomically accepts one validated next revision.
    fn apply(
        &mut self,
        update: &WorldTopologyTearUpdate,
        state: &TopologyUpdateConsumer,
    ) -> Result<(), M15TopologyAdapterError>;

    /// Removes all downstream state owned by a retired soft body.
    fn remove(&mut self, body: SoftBodyHandle);
}

/// Why the M15 bridge could not advance a queued topology revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M15TopologyBridgeError {
    /// Updates are waiting, but no downstream adapter is installed.
    MissingAdapter,
    /// The producer referenced a body without a registered topology stream.
    UnregisteredBody(SoftBodyHandle),
    /// The core atomic consumer rejected the producer payload.
    InvalidUpdate {
        /// Soft body that produced the invalid payload.
        body: SoftBodyHandle,
        /// Core revision or payload consistency failure.
        source: TopologyUpdateError,
    },
    /// The downstream adapter rejected registration or a complete update.
    AdapterRejected(SoftBodyHandle),
}

/// Observable state for production monitoring and retry diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct M15TopologyDiagnostics {
    /// Number of complete revisions accepted by the downstream adapter.
    pub applied_updates: u64,
    /// Number of updates currently retained for ordered retry.
    pub pending_updates: usize,
    /// Number of live topology streams tracked by the bridge.
    pub registered_bodies: usize,
    /// Most recent routing failure, cleared after a successful flush.
    pub last_error: Option<M15TopologyBridgeError>,
}

/// ECS resource owning M15 topology revision state and retry queues.
#[derive(Resource, Default)]
pub struct M15TopologyBridge {
    consumers: HashMap<SoftBodyHandle, TopologyUpdateConsumer>,
    pending: Vec<WorldTopologyTearUpdate>,
    adapter: Option<Box<dyn M15TopologyAdapter>>,
    adapter_bodies: HashSet<SoftBodyHandle>,
    diagnostics: M15TopologyDiagnostics,
}

impl M15TopologyBridge {
    /// Installs a downstream adapter. Existing live bodies are registered on
    /// the next fixed post-update before queued revisions are retried.
    pub fn set_adapter<A>(&mut self, adapter: A)
    where
        A: M15TopologyAdapter,
    {
        self.detach_current_adapter();
        self.adapter = Some(Box::new(adapter));
        self.diagnostics.last_error = None;
    }

    /// Removes and returns the current adapter without dropping queued data.
    pub fn take_adapter(&mut self) -> Option<Box<dyn M15TopologyAdapter>> {
        self.adapter_bodies.clear();
        self.adapter.take()
    }

    /// Returns the last locally committed state for `body`.
    #[must_use]
    pub fn consumer(&self, body: SoftBodyHandle) -> Option<&TopologyUpdateConsumer> {
        self.consumers.get(&body)
    }

    /// Returns the last locally committed topology revision for `body`.
    #[must_use]
    pub fn revision(&self, body: SoftBodyHandle) -> Option<u64> {
        self.consumer(body)
            .map(TopologyUpdateConsumer::topology_version)
    }

    /// Returns the number of payloads retained for retry.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Returns production counters and the most recent routing error.
    #[must_use]
    pub const fn diagnostics(&self) -> &M15TopologyDiagnostics {
        &self.diagnostics
    }

    fn detach_current_adapter(&mut self) {
        let mut bodies: Vec<_> = self.adapter_bodies.drain().collect();
        bodies.sort_by_key(|body| handle_key(*body));
        if let Some(adapter) = self.adapter.as_mut() {
            for body in bodies {
                adapter.remove(body);
            }
        }
        self.adapter = None;
    }

    fn synchronize_live_bodies(&mut self, live: Vec<(SoftBodyHandle, TearableMesh)>) {
        let live_handles: HashSet<_> = live.iter().map(|(body, _)| *body).collect();
        let mut retired: Vec<_> = self
            .consumers
            .keys()
            .copied()
            .filter(|body| !live_handles.contains(body))
            .collect();
        retired.sort_by_key(|body| handle_key(*body));
        for body in retired {
            self.consumers.remove(&body);
            self.pending.retain(|update| update.body != body);
            if self.adapter_bodies.remove(&body) {
                if let Some(adapter) = self.adapter.as_mut() {
                    adapter.remove(body);
                }
            }
        }

        for (body, mesh) in live {
            self.consumers
                .entry(body)
                .or_insert_with(|| TopologyUpdateConsumer::from_mesh(&mesh));
        }
        self.refresh_diagnostics();
    }

    fn enqueue(&mut self, updates: Vec<WorldTopologyTearUpdate>) {
        self.pending.extend(
            updates
                .into_iter()
                .filter(|update| self.consumers.contains_key(&update.body)),
        );
        self.refresh_diagnostics();
    }

    fn flush(&mut self) {
        if self.adapter.is_none() {
            self.diagnostics.last_error =
                (!self.pending.is_empty()).then_some(M15TopologyBridgeError::MissingAdapter);
            self.refresh_diagnostics();
            return;
        }
        if !self.register_adapter_bodies() {
            self.refresh_diagnostics();
            return;
        }
        if self.pending.is_empty() {
            self.diagnostics.last_error = None;
            self.refresh_diagnostics();
            return;
        }

        let queued = std::mem::take(&mut self.pending);
        let mut iter = queued.into_iter();
        while let Some(world_update) = iter.next() {
            let body = world_update.body;
            let Some(current) = self.consumers.get(&body) else {
                self.diagnostics.last_error = Some(M15TopologyBridgeError::UnregisteredBody(body));
                self.pending.push(world_update);
                self.pending.extend(iter);
                self.refresh_diagnostics();
                return;
            };

            if is_registered_snapshot(current, &world_update) {
                continue;
            }

            let mut candidate = current.clone();
            if let Err(source) = candidate.apply(world_update.update.clone()) {
                self.diagnostics.last_error =
                    Some(M15TopologyBridgeError::InvalidUpdate { body, source });
                self.pending.push(world_update);
                self.pending.extend(iter);
                self.refresh_diagnostics();
                return;
            }

            let accepted = self
                .adapter
                .as_mut()
                .expect("adapter presence checked before routing")
                .apply(&world_update, &candidate)
                .is_ok();
            if !accepted {
                self.diagnostics.last_error = Some(M15TopologyBridgeError::AdapterRejected(body));
                self.pending.push(world_update);
                self.pending.extend(iter);
                self.refresh_diagnostics();
                return;
            }

            self.consumers.insert(body, candidate);
            self.diagnostics.applied_updates = self.diagnostics.applied_updates.saturating_add(1);
        }

        self.diagnostics.last_error = None;
        self.refresh_diagnostics();
    }

    fn register_adapter_bodies(&mut self) -> bool {
        let mut bodies: Vec<_> = self.consumers.keys().copied().collect();
        bodies.sort_by_key(|body| handle_key(*body));
        for body in bodies {
            if self.adapter_bodies.contains(&body) {
                continue;
            }
            let state = self
                .consumers
                .get(&body)
                .expect("body was collected from the consumer registry");
            let accepted = self
                .adapter
                .as_mut()
                .expect("adapter presence checked before registration")
                .register(body, state)
                .is_ok();
            if !accepted {
                self.diagnostics.last_error = Some(M15TopologyBridgeError::AdapterRejected(body));
                return false;
            }
            self.adapter_bodies.insert(body);
        }
        true
    }

    fn refresh_diagnostics(&mut self) {
        self.diagnostics.pending_updates = self.pending.len();
        self.diagnostics.registered_bodies = self.consumers.len();
    }
}

/// Installs the M15 fixed-post-update route. Add it together with
/// [`crate::PrismPhysicsPlugin`].
#[derive(Default)]
pub struct M15TopologyBridgePlugin;

impl Plugin for M15TopologyBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<M15TopologyBridge>()
            .add_systems(FixedPostUpdate, route_topology_updates);
    }
}

fn route_topology_updates(world: &mut World) {
    if world.get_resource::<PhysicsBindings>().is_none()
        || world.get_resource::<PhysicsWorldResource>().is_none()
    {
        return;
    }

    let query = world.query::<(Entity, &SoftBodyBinding)>();
    let mut authored: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, binding)| {
            binding
                .0
                .topology
                .as_ref()
                .map(|topology| (entity, topology.clone()))
        })
        .collect();
    authored.sort_by_key(|(entity, _)| *entity);

    let mut live = Vec::with_capacity(authored.len());
    {
        let bindings = world.resource::<PhysicsBindings>();
        for (entity, topology) in authored {
            if let Some(body) = bindings.soft_body(entity) {
                live.push((body, topology));
            }
        }
    }
    live.sort_by_key(|(body, _)| handle_key(*body));

    let mut bridge = world
        .remove_resource::<M15TopologyBridge>()
        .expect("M15TopologyBridgePlugin initializes its resource");
    bridge.synchronize_live_bodies(live);

    let updates = {
        let mut physics = world
            .remove_resource::<PhysicsWorldResource>()
            .expect("physics resource presence checked before routing");
        let updates = physics.drain_topology_updates();
        world.insert_resource(physics);
        updates
    };
    bridge.enqueue(updates);
    bridge.flush();
    world.insert_resource(bridge);
}

fn is_registered_snapshot(
    current: &TopologyUpdateConsumer,
    update: &WorldTopologyTearUpdate,
) -> bool {
    update.update.topology_version == current.topology_version()
        && update.update.events.is_empty()
        && &update.update.render == current.render()
        && &update.update.collision == current.collision()
}

const fn handle_key(body: SoftBodyHandle) -> (u32, u32) {
    (body.index(), body.generation())
}
