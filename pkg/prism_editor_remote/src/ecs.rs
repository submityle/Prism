//! Runtime-side synchronization of a real [`World`].
//!
//! Explicitly registered components form the editable projection. The host lends
//! its world exclusively at a schedule barrier; protocol reads inspect actual
//! typed columns, and writes prepare every concrete value before changing any
//! column. Structural commands are deliberately rejected: arbitrary ECS hooks
//! cannot be rolled back by a transport transaction. Host structural changes
//! are observed in snapshots and retained deltas. See the dedicated ECS adapter
//! document for the exact value mapping and frame/restore fence.

use crate::{
    Capability, Change, ChangeSet, CommitResult, ComponentId, EndpointCapabilities,
    EndpointDescriptor, EntityId, EntityRecord, ErrorCode, Revision, SyncError, SyncStore,
    TargetId, Value, WorldSnapshot,
};
use core::{any::type_name, cell::RefCell, mem};
use prism_ecs::{
    component::{Component, StorageType},
    entity::Entity,
    world::World,
};
use prism_reflect::{
    DynamicArray, DynamicEnum, DynamicList, DynamicMap, DynamicStruct, DynamicTupleStruct,
    DynamicVariant, FromReflect, GetTypeRegistration, Reflect, ReflectRef, StableTypeId, TypeInfo,
    TypeRegistry, VariantKind, VariantType,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Launch/handshake adapter for a host serving [`EcsSyncStore`] over the
/// existing bounded runtime transport. The host owns its executable, scene
/// loading and schedule; no demo-specific arguments or components are injected.
#[derive(Clone, Debug)]
pub struct PrismEcsRuntimeAdapter {
    target: TargetId,
}

impl PrismEcsRuntimeAdapter {
    /// Requires this target, the Prism ECS endpoint, and all ECS sync
    /// capabilities during initialization, then probes an actual snapshot.
    #[must_use]
    pub const fn new(target: TargetId) -> Self {
        Self { target }
    }
}

const CAPABILITIES: [&str; 6] = [
    "snapshotRead",
    "incrementalRead",
    "componentWrite",
    "propertyWrite",
    "revisionChecks",
    "deterministicCodec",
];

fn json_object(
    entries: impl IntoIterator<Item = (&'static str, crate::JsonValue)>,
) -> crate::JsonValue {
    crate::JsonValue::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
}

impl crate::EngineRuntimeAdapter for PrismEcsRuntimeAdapter {
    fn engine_id(&self) -> &str {
        "prism-ecs"
    }

    fn prepare_launch(
        &self,
        requested: &crate::RuntimeLaunchSpec,
    ) -> Result<crate::RuntimeLaunchSpec, crate::RuntimeError> {
        if requested.program.as_os_str().is_empty() {
            return Err(crate::RuntimeError::Adapter(
                "Prism ECS host executable is empty".into(),
            ));
        }
        Ok(requested.clone())
    }

    fn initialize(
        &self,
        session: &mut dyn crate::RuntimeSession,
        timeout: Duration,
    ) -> Result<(), crate::RuntimeError> {
        use crate::{JsonNumber, JsonValue, RuntimeError, RuntimeOperation};
        let malformed = || {
            RuntimeError::Adapter(
                "incompatible Prism ECS initialization or snapshot response".into(),
            )
        };
        let start = Instant::now();
        let mut call = |id, method, params| {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Err(RuntimeError::Timeout {
                    operation: RuntimeOperation::Start,
                    timeout,
                });
            }
            let exchange = session.request(crate::request(id, method, Some(params)), remaining)?;
            if let Some(error) = exchange.response.error {
                return Err(RuntimeError::Adapter(format!(
                    "Prism ECS {method}: {} ({})",
                    error.message, error.code
                )));
            }
            exchange.response.result.ok_or_else(malformed)
        };
        let number = |n| JsonValue::Number(JsonNumber::Unsigned(n));
        let result = call(
            1,
            "prism/initialize",
            json_object([
                (
                    "protocol",
                    json_object([
                        (
                            "major",
                            number(u64::from(crate::ProtocolVersion::CURRENT.major)),
                        ),
                        (
                            "minor",
                            number(u64::from(crate::ProtocolVersion::CURRENT.minor)),
                        ),
                    ]),
                ),
                (
                    "capabilities",
                    JsonValue::Array(
                        CAPABILITIES
                            .into_iter()
                            .map(|s| JsonValue::String(s.into()))
                            .collect(),
                    ),
                ),
            ]),
        )?;
        let JsonValue::Object(result) = result else {
            return Err(malformed());
        };
        let Some(JsonValue::Object(protocol)) = result.get("protocol") else {
            return Err(malformed());
        };
        let Some(JsonValue::Object(endpoint)) = result.get("endpoint") else {
            return Err(malformed());
        };
        let Some(JsonValue::Array(selected)) = result.get("capabilities") else {
            return Err(malformed());
        };
        let Some(JsonValue::Array(targets)) = endpoint.get("targets") else {
            return Err(malformed());
        };
        if protocol.get("major") != Some(&number(u64::from(crate::ProtocolVersion::CURRENT.major)))
            || endpoint.get("name") != Some(&JsonValue::String("prism-ecs".into()))
            || !targets.contains(&JsonValue::String(self.target.raw().to_string()))
            || CAPABILITIES
                .iter()
                .any(|s| !selected.contains(&JsonValue::String((*s).into())))
        {
            return Err(malformed());
        }
        let result = call(
            2,
            "prism/world/readSnapshot",
            json_object([("target", JsonValue::String(self.target.raw().to_string()))]),
        )?;
        let JsonValue::Object(snapshot) = result else {
            return Err(malformed());
        };
        if snapshot.get("target") != Some(&JsonValue::String(self.target.raw().to_string()))
            || !matches!(
                snapshot.get("revision"),
                Some(JsonValue::Number(JsonNumber::Unsigned(_)))
            )
            || !matches!(snapshot.get("entities"), Some(JsonValue::Object(_)))
        {
            return Err(malformed());
        }
        Ok(())
    }
}

/// Bounds for projection, requests, and retained history. All must be nonzero.
#[derive(Clone, Copy, Debug)]
pub struct EcsSyncLimits {
    /// Maximum live entities, including entities without exposed components.
    pub max_entities: usize,
    /// Maximum total value nodes in one projection or request.
    pub max_value_nodes: usize,
    /// Maximum aggregate string/byte payload in one projection or request.
    pub max_value_bytes: usize,
    /// Maximum nesting depth; the root has depth zero.
    pub max_depth: usize,
    /// Maximum commands in one write or aggregated incremental response.
    pub max_changes: usize,
    /// Maximum revision batches retained. Each is also bounded above.
    pub history_batches: usize,
}

impl Default for EcsSyncLimits {
    fn default() -> Self {
        Self {
            max_entities: 16_384,
            max_value_nodes: 262_144,
            max_value_bytes: 8 * 1024 * 1024,
            max_depth: 64,
            max_changes: 65_536,
            history_batches: 16,
        }
    }
}

struct Binding {
    stable: StableTypeId,
    writable: bool,
    prepare: fn(&dyn Reflect) -> Result<Box<dyn Reflect>, SyncError>,
    replace: fn(&mut World, Entity, Box<dyn Reflect>) -> Box<dyn Reflect>,
}

struct State {
    snapshot: WorldSnapshot,
    history: VecDeque<ChangeSet>,
    dirty: bool,
    entity_namespace: u64,
    archetype_generation: u64,
}

/// An exclusive, schedule-barrier adapter over actual Prism ECS storage.
///
/// The borrow prevents systems from racing a batch. [`Self::world_mut`] is the
/// host's frame boundary and invalidates all previously issued write revisions,
/// even for an ABA edit or restore whose projected values are unchanged. A new
/// adapter starts a new session: use a fresh target ID on reconnect/recreation.
pub struct EcsSyncStore<'w> {
    world: &'w mut World,
    world_id: usize,
    descriptor: EndpointDescriptor,
    limits: EcsSyncLimits,
    bindings: BTreeMap<ComponentId, Binding>,
    types: TypeRegistry,
    state: RefCell<State>,
}

impl<'w> EcsSyncStore<'w> {
    /// Borrows the engine world; no copy or substitute store is created.
    pub fn new(
        world: &'w mut World,
        target: TargetId,
        limits: EcsSyncLimits,
    ) -> Result<Self, SyncError> {
        if [
            limits.max_entities,
            limits.max_value_nodes,
            limits.max_value_bytes,
            limits.max_depth,
            limits.max_changes,
            limits.history_batches,
        ]
        .contains(&0)
        {
            return Err(invalid("ECS limits must all be nonzero"));
        }
        let mut descriptor = EndpointDescriptor::new(
            "prism-ecs",
            EndpointCapabilities::from_capabilities([
                Capability::SnapshotRead,
                Capability::IncrementalRead,
                Capability::ComponentWrite,
                Capability::PropertyWrite,
                Capability::RevisionChecks,
                Capability::DeterministicCodec,
            ]),
        );
        descriptor.add_target(target);
        let archetype_generation = world.archetype_generation();
        let entity_namespace = fresh_namespace()?;
        Ok(Self {
            world_id: world.id().get(),
            world,
            descriptor,
            limits,
            bindings: BTreeMap::new(),
            types: TypeRegistry::new(),
            state: RefCell::new(State {
                snapshot: WorldSnapshot::empty(target),
                history: VecDeque::new(),
                dirty: false,
                entity_namespace,
                archetype_generation,
            }),
        })
    }

    /// Exposes `T` under its exact Rust type path and installs the real ECS
    /// reflection bridge. Shared components are readable but never writable.
    ///
    /// `FromReflect` must faithfully reconstruct the exposed component; ignored
    /// fields follow that implementation's defaults. Clone, conversion and Drop
    /// implementations are trusted engine code and must not panic or have
    /// external side effects. Remote data never chooses a Rust type dynamically.
    pub fn register_component<T>(&mut self) -> ComponentId
    where
        T: Component + Reflect + FromReflect + GetTypeRegistration + Clone,
    {
        fn prepare<T: FromReflect>(value: &dyn Reflect) -> Result<Box<dyn Reflect>, SyncError> {
            T::from_reflect(value)
                .map(|v| Box::new(v) as Box<dyn Reflect>)
                .ok_or_else(|| invalid(format!("cannot reconstruct {}", type_name::<T>())))
        }
        fn replace<T: Component + Reflect>(
            world: &mut World,
            entity: Entity,
            value: Box<dyn Reflect>,
        ) -> Box<dyn Reflect> {
            let mut typed = value
                .into_any()
                .downcast::<T>()
                .expect("prepared exact component type");
            let slot = world
                .get_mut::<T>(entity)
                .expect("preflighted live mutable component")
                .into_inner();
            mem::swap(slot, typed.as_mut());
            typed
        }
        let id = ComponentId::from_static(type_name::<T>());
        let stable = self.world.register_reflect_component::<T>();
        self.types.register::<T>();
        self.bindings.insert(
            id.clone(),
            Binding {
                stable,
                writable: T::STORAGE != StorageType::Shared,
                prepare: prepare::<T>,
                replace: replace::<T>,
            },
        );
        self.state.get_mut().dirty = true;
        id
    }

    /// Registers a nested composite shape, e.g. `Vec<Options>` and `Options`.
    /// Scalar leaf types need no registration. Does not expose a component.
    pub fn register_value_type<T: GetTypeRegistration>(&mut self) {
        self.types.register::<T>();
    }

    /// Borrows the actual engine for host systems, spawn/despawn, and restore.
    /// Every successful call fences previous revisions, including no-op edits.
    /// Remains available after a failed scan so the host can repair its world;
    /// multiple unobserved host phases coalesce into one revision.
    /// Replacing the entire World requires a new adapter and fresh target ID.
    pub fn world_mut(&mut self) -> Result<&mut World, SyncError> {
        if self.world.id().get() != self.world_id {
            return Err(invalid(
                "bound World was replaced; create a fresh target/session",
            ));
        }
        next_revision(self.state.get_mut().snapshot.revision)?;
        self.state.get_mut().dirty = true;
        Ok(self.world)
    }

    /// Shared access to the same engine storage (no mutation or tick marking).
    #[must_use]
    pub fn world(&self) -> &World {
        self.world
    }

    /// Encodes session/restore identity plus the full entity index/generation.
    pub fn entity_id(&self, entity: Entity) -> Result<EntityId, SyncError> {
        if self.world.id().get() != self.world_id || !self.world.contains(entity) {
            return Err(invalid("entity is not live in the bound ECS world"));
        }
        self.synchronize()?;
        Ok(wire_entity(self.state.borrow().entity_namespace, entity))
    }

    fn entity(&self, id: EntityId) -> Result<Entity, SyncError> {
        if id.raw() >> 64 != u128::from(self.state.borrow().entity_namespace) {
            return Err(SyncError::MissingEntity { entity: id });
        }
        let bits = u64::try_from(id.raw() & u128::from(u64::MAX)).expect("masked entity bits");
        Entity::from_bits(bits)
            .filter(|e| self.world.contains(*e))
            .ok_or(SyncError::MissingEntity { entity: id })
    }

    /// Refreshes from real columns. A failed scan leaves history untouched, and
    /// every protocol operation retries the scan instead of serving stale data.
    /// Returns the current optimistic-write revision.
    pub fn synchronize(&self) -> Result<Revision, SyncError> {
        if self.world.id().get() != self.world_id {
            return Err(invalid(
                "bound World was replaced; create a fresh target/session",
            ));
        }
        if self.world.entities().has_outstanding_reservations() {
            return Err(invalid(
                "flush outstanding ECS entity reservations before synchronization",
            ));
        }
        let entity_count = self.world.entities().len() as usize;
        if entity_count > self.limits.max_entities {
            return Err(invalid("ECS entity budget exceeded"));
        }
        let mut state = self.state.borrow_mut();
        let generation = self.world.archetype_generation();
        let namespace = if generation == state.archetype_generation {
            state.entity_namespace
        } else {
            fresh_namespace()?
        };
        let mut snapshot = WorldSnapshot::empty(state.snapshot.target);
        let mut budget = Budget::new(self.limits);
        for arch in self.world.archetypes().iter() {
            for &entity in arch.table().entities() {
                if snapshot.entities.len() >= self.limits.max_entities {
                    return Err(invalid("ECS entity budget exceeded"));
                }
                let mut record = EntityRecord::new();
                for (id, binding) in &self.bindings {
                    if let Some(value) = self.world.get_reflect(entity, binding.stable) {
                        budget.bytes(id.as_str().len())?;
                        record
                            .components
                            .insert(id.clone(), encode(value, &mut budget, 0)?);
                    }
                }
                snapshot
                    .entities
                    .insert(wire_entity(namespace, entity), record);
            }
        }
        if snapshot.entities.len() != entity_count {
            return Err(invalid(
                "place flushed ECS entity reservations with spawn_at before synchronization",
            ));
        }
        let changes = diff(&state.snapshot, &snapshot);
        if changes.is_empty() && !state.dirty {
            return Ok(state.snapshot.revision);
        }
        snapshot.revision = next_revision(state.snapshot.revision)?;
        // A no-value-change host barrier is an intentional revision gap: a
        // fresh snapshot is required, rather than inventing an ECS mutation.
        if changes.is_empty() || changes.len() > self.limits.max_changes {
            state.history.clear();
        } else {
            let batch = ChangeSet::new(state.snapshot.revision, snapshot.revision, changes)?;
            state.history.push_back(batch);
            while state.history.len() > self.limits.history_batches {
                state.history.pop_front();
            }
        }
        state.snapshot = snapshot;
        state.entity_namespace = namespace;
        state.archetype_generation = generation;
        state.dirty = false;
        Ok(state.snapshot.revision)
    }

    fn target(&self, target: TargetId) -> Result<(), SyncError> {
        if self.descriptor.targets == [target] {
            Ok(())
        } else {
            Err(SyncError::UnknownTarget { target })
        }
    }
}

impl SyncStore for EcsSyncStore<'_> {
    fn descriptor(&self) -> &EndpointDescriptor {
        &self.descriptor
    }

    fn read_snapshot(&self, target: TargetId) -> Result<WorldSnapshot, SyncError> {
        self.target(target)?;
        self.synchronize()?;
        Ok(self.state.borrow().snapshot.clone())
    }

    fn read_changes(&self, target: TargetId, from: Revision) -> Result<ChangeSet, SyncError> {
        self.target(target)?;
        let current = self.synchronize()?;
        if from > current {
            return Err(SyncError::RevisionConflict {
                target,
                expected: current,
                actual: from,
            });
        }
        if from == current {
            return Ok(ChangeSet::empty(current));
        }
        let state = self.state.borrow();
        let oldest = state.history.front().map_or(current, |b| b.base_revision);
        if from < oldest {
            return Err(SyncError::RevisionUnavailable {
                target,
                requested: from,
                oldest_available: oldest,
            });
        }
        let mut changes = Vec::new();
        let mut budget = Budget::new(self.limits);
        for batch in &state.history {
            if batch.target_revision > from {
                if changes.len().saturating_add(batch.changes.len()) > self.limits.max_changes {
                    return Err(invalid(
                        "ECS incremental response budget exceeded; request a snapshot",
                    ));
                }
                for change in &batch.changes {
                    match change {
                        Change::SetComponent {
                            component, value, ..
                        } => {
                            budget.bytes(component.as_str().len())?;
                            budget.value(value, 0)?;
                        }
                        Change::RemoveComponent { component, .. } => {
                            budget.bytes(component.as_str().len())?;
                        }
                        _ => {}
                    }
                }
                changes.extend(batch.changes.iter().cloned());
            }
        }
        ChangeSet::new(from, current, changes)
    }

    fn apply_changes(
        &mut self,
        target: TargetId,
        base: Revision,
        changes: &[Change],
    ) -> Result<CommitResult, SyncError> {
        self.target(target)?;
        let current = self.synchronize()?;
        if base != current {
            return Err(SyncError::RevisionConflict {
                target,
                expected: current,
                actual: base,
            });
        }
        if changes.len() > self.limits.max_changes {
            return Err(invalid("ECS write command budget exceeded"));
        }
        let mut budget = Budget::new(self.limits);
        let mut touched = BTreeSet::new();
        for change in changes {
            let (entity, component, value) = match change {
                Change::SetComponent { entity, component, value } => (*entity, component, Some(value)),
                Change::SetProperty { entity, component, path, value } => {
                    if path.segments().len() > self.limits.max_depth { return Err(invalid("ECS property path depth exceeded")); }
                    for segment in path.segments() {
                        if let crate::PropertySegment::Field(s) | crate::PropertySegment::Key(s) = segment { budget.bytes(s.len())?; }
                    }
                    (*entity, component, Some(value))
                }
                Change::RemoveProperty { entity, component, path } => {
                    if path.segments().len() > self.limits.max_depth { return Err(invalid("ECS property path depth exceeded")); }
                    for segment in path.segments() {
                        if let crate::PropertySegment::Field(s) | crate::PropertySegment::Key(s) = segment { budget.bytes(s.len())?; }
                    }
                    (*entity, component, None)
                }
                _ => return Err(invalid("remote structural commands are unsupported; mutate the host ECS at a frame barrier")),
            };
            self.entity(entity)?;
            budget.bytes(component.as_str().len())?;
            let binding = self
                .bindings
                .get(component)
                .ok_or_else(|| invalid(format!("unregistered ECS component {component}")))?;
            if !binding.writable {
                return Err(invalid(format!(
                    "shared ECS component {component} is read-only"
                )));
            }
            if self
                .world
                .get_reflect(self.entity(entity)?, binding.stable)
                .is_none()
            {
                return Err(SyncError::MissingComponent {
                    entity,
                    component: component.clone(),
                });
            }
            if let Some(value) = value {
                budget.value(value, 0)?;
            }
            touched.insert((entity, component.clone()));
        }
        let mut staged = self.state.borrow().snapshot.clone();
        crate::model::apply_changes(&mut staged, changes)?;
        let mut prepared = Vec::new();
        for (id, component) in touched {
            let binding = &self.bindings[&component];
            let value = &staged.entities[&id].components[&component];
            let dynamic = decode(
                component.as_str(),
                value,
                &self.types,
                self.limits.max_depth,
                0,
            )?;
            let typed = (binding.prepare)(dynamic.as_ref())?;
            let mut check = Budget::new(self.limits);
            if encode(typed.as_ref(), &mut check, 0)? != *value {
                return Err(invalid(format!(
                    "{} reconstruction does not preserve the exact value",
                    component
                )));
            }
            prepared.push((self.entity(id)?, binding.replace, typed));
        }
        // Bound the final projection as well as each incoming value. No engine
        // write occurs until all conversions, limits and revisions succeed.
        let mut check = Budget::new(self.limits);
        for record in staged.entities.values() {
            for (id, value) in &record.components {
                check.bytes(id.as_str().len())?;
                check.value(value, 0)?;
            }
        }
        let accepted = diff(&self.state.borrow().snapshot, &staged);
        if accepted.is_empty() {
            return Ok(CommitResult {
                target,
                revision: current,
                changes: Vec::new(),
            });
        }
        if accepted.len() > self.limits.max_changes {
            return Err(invalid("ECS commit delta budget exceeded"));
        }
        staged.revision = next_revision(current)?;
        let batch = ChangeSet::new(current, staged.revision, accepted.clone())?;
        let mut history = self.state.borrow().history.clone();
        history.push_back(batch);
        while history.len() > self.limits.history_batches {
            history.pop_front();
        }
        let mut retired = Vec::with_capacity(prepared.len());
        for (entity, replace, typed) in prepared {
            retired.push(replace(self.world, entity, typed));
        }
        let revision = staged.revision;
        let state = self.state.get_mut();
        state.snapshot = staged;
        state.history = history;
        // Old components are dropped only after the entire batch is published.
        drop(retired);
        Ok(CommitResult {
            target,
            revision,
            changes: accepted,
        })
    }
}

fn invalid(detail: impl Into<String>) -> SyncError {
    SyncError::InvalidRequest {
        code: ErrorCode::InvalidRequest,
        detail: detail.into(),
    }
}

fn next_revision(revision: Revision) -> Result<Revision, SyncError> {
    revision
        .raw()
        .checked_add(1)
        .map(Revision::from_raw)
        .ok_or_else(|| invalid("ECS revision exhausted; create a fresh target/session"))
}

fn fresh_namespace() -> Result<u64, SyncError> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| invalid("ECS entity namespace exhausted"))
}

fn wire_entity(namespace: u64, entity: Entity) -> EntityId {
    EntityId::from_raw((u128::from(namespace) << 64) | u128::from(entity.to_bits()))
}

fn diff(old: &WorldSnapshot, new: &WorldSnapshot) -> Vec<Change> {
    let mut result = Vec::new();
    for &entity in old.entities.keys() {
        if !new.entities.contains_key(&entity) {
            result.push(Change::DeleteEntity { entity });
        }
    }
    for (&entity, record) in &new.entities {
        let previous = old.entities.get(&entity);
        if previous.is_none() {
            result.push(Change::CreateEntity { entity });
        }
        if let Some(previous) = previous {
            for component in previous.components.keys() {
                if !record.components.contains_key(component) {
                    result.push(Change::RemoveComponent {
                        entity,
                        component: component.clone(),
                    });
                }
            }
        }
        for (component, value) in &record.components {
            if previous.and_then(|p| p.components.get(component)) != Some(value) {
                result.push(Change::SetComponent {
                    entity,
                    component: component.clone(),
                    value: value.clone(),
                });
            }
        }
    }
    result
}

struct Budget {
    limits: EcsSyncLimits,
    nodes: usize,
    bytes: usize,
}
impl Budget {
    fn new(limits: EcsSyncLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            bytes: 0,
        }
    }
    fn node(&mut self, depth: usize) -> Result<(), SyncError> {
        if depth > self.limits.max_depth || self.nodes >= self.limits.max_value_nodes {
            return Err(invalid("ECS value depth/node budget exceeded"));
        }
        self.nodes += 1;
        Ok(())
    }
    fn bytes(&mut self, bytes: usize) -> Result<(), SyncError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("ECS byte budget overflow"))?;
        if self.bytes > self.limits.max_value_bytes {
            return Err(invalid("ECS value byte budget exceeded"));
        }
        Ok(())
    }
    fn value(&mut self, value: &Value, depth: usize) -> Result<(), SyncError> {
        self.node(depth)?;
        match value {
            Value::String(s) => self.bytes(s.len())?,
            Value::Bytes(b) => self.bytes(b.len())?,
            Value::Array(a) => {
                for v in a {
                    self.value(v, depth + 1)?;
                }
            }
            Value::Object(o) => {
                for (k, v) in o {
                    self.bytes(k.len())?;
                    self.value(v, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn encode(value: &dyn Reflect, budget: &mut Budget, depth: usize) -> Result<Value, SyncError> {
    budget.node(depth)?;
    macro_rules! leaf {
        ($ty:ty, $map:expr) => {
            if let Some(v) = value.downcast_ref::<$ty>() {
                return Ok(($map)(v));
            }
        };
    }
    leaf!(bool, |v: &bool| Value::Bool(*v));
    macro_rules! signed { ($($ty:ty),*) => { $(leaf!($ty, |v: &$ty| signed_value(i64::from(*v))));* }; }
    macro_rules! unsigned { ($($ty:ty),*) => { $(leaf!($ty, |v: &$ty| Value::Unsigned(u64::from(*v))));* }; }
    signed!(i8, i16, i32, i64);
    unsigned!(u8, u16, u32, u64);
    leaf!(isize, |v: &isize| signed_value(*v as i64));
    leaf!(usize, |v: &usize| Value::Unsigned(*v as u64));
    leaf!(f32, |v: &f32| Value::Unsigned(u64::from(v.to_bits())));
    leaf!(f64, |v: &f64| Value::Float(v.to_bits()));
    if let Some(v) = value.downcast_ref::<i128>() {
        budget.bytes(16)?;
        return Ok(Value::Bytes(v.to_le_bytes().to_vec()));
    }
    if let Some(v) = value.downcast_ref::<u128>() {
        budget.bytes(16)?;
        return Ok(Value::Bytes(v.to_le_bytes().to_vec()));
    }
    if let Some(v) = value.downcast_ref::<String>() {
        budget.bytes(v.len())?;
        return Ok(Value::String(v.clone()));
    }
    if let Some(v) = value.downcast_ref::<char>() {
        budget.bytes(v.len_utf8())?;
        return Ok(Value::String(v.to_string()));
    }
    let child = |v: Option<&dyn Reflect>, budget: &mut Budget, depth| {
        encode(
            v.ok_or_else(|| invalid("inconsistent ECS reflection metadata"))?,
            budget,
            depth,
        )
    };
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            let mut fields = BTreeMap::new();
            for i in 0..s.field_count() {
                let name = s
                    .name_at(i)
                    .ok_or_else(|| invalid("missing reflected field name"))?;
                budget.bytes(name.len())?;
                if fields
                    .insert(name.into(), child(s.field_at(i), budget, depth + 1)?)
                    .is_some()
                {
                    return Err(invalid("duplicate reflected field name"));
                }
            }
            Ok(Value::Object(fields))
        }
        ReflectRef::TupleStruct(s) => (0..s.field_count())
            .map(|i| child(s.field(i), budget, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        ReflectRef::List(s) => (0..s.len())
            .map(|i| child(s.get(i), budget, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        ReflectRef::Array(s) => (0..s.len())
            .map(|i| child(s.get(i), budget, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        ReflectRef::Enum(e) => {
            budget.bytes("variant".len() + "fields".len())?;
            budget.node(depth + 1)?;
            budget.bytes(e.variant_name().len())?;
            let fields = match e.variant_type() {
                VariantType::Unit => {
                    budget.node(depth + 1)?;
                    Value::Null
                }
                VariantType::Tuple => {
                    budget.node(depth + 1)?;
                    Value::Array(
                        (0..e.field_count())
                            .map(|i| child(e.field_at(i), budget, depth + 2))
                            .collect::<Result<_, _>>()?,
                    )
                }
                VariantType::Struct => {
                    budget.node(depth + 1)?;
                    let TypeInfo::Enum(info) = e.type_info() else {
                        return Err(invalid("inconsistent enum metadata"));
                    };
                    let Some(variant) = info.variant(e.variant_name()) else {
                        return Err(invalid("unknown reflected enum variant"));
                    };
                    let VariantKind::Struct(descriptors) = variant.kind() else {
                        return Err(invalid("inconsistent enum variant shape"));
                    };
                    let mut fields = BTreeMap::new();
                    for descriptor in descriptors {
                        budget.bytes(descriptor.name().len())?;
                        fields.insert(
                            descriptor.name().into(),
                            child(e.field(descriptor.name()), budget, depth + 2)?,
                        );
                    }
                    Value::Object(fields)
                }
                _ => return Err(invalid("unsupported enum variant kind")),
            };
            Value::object([
                ("variant", Value::String(e.variant_name().into())),
                ("fields", fields),
            ])
        }
        ReflectRef::Map(m) => {
            let TypeInfo::Map(info) = m.type_info() else {
                return Err(invalid("inconsistent map metadata"));
            };
            if info.key_type_name() != type_name::<String>() {
                return Err(invalid("only String-keyed ECS maps are supported"));
            }
            let mut entries = BTreeMap::new();
            for (key, value) in m.iter_reflect() {
                let key = key
                    .downcast_ref::<String>()
                    .ok_or_else(|| invalid("map key is not an exact String"))?;
                if matches!(key.as_str(), "$prism.bytes" | "$prism.floatBits") {
                    return Err(invalid(
                        "ECS map key collides with reserved JSON-RPC value tags",
                    ));
                }
                budget.bytes(key.len())?;
                if entries
                    .insert(key.clone(), encode(value, budget, depth + 1)?)
                    .is_some()
                {
                    return Err(invalid("duplicate reflected map key"));
                }
            }
            Ok(Value::Object(entries))
        }
        _ => Err(invalid(format!(
            "unsupported ECS reflection type {} (sets and opaque leaves are not mapped)",
            value.type_name()
        ))),
    }
}

fn object(value: &Value) -> Result<&BTreeMap<String, Value>, SyncError> {
    match value {
        Value::Object(o) => Ok(o),
        _ => Err(invalid("expected an exact object shape")),
    }
}

// JSON has no signed-positive tag. Canonicalize by sign and let the registered
// Rust schema supply signedness/width; the binary protocol uses the same form.
fn signed_value(value: i64) -> Value {
    match u64::try_from(value) {
        Ok(value) => Value::Unsigned(value),
        Err(_) => Value::Signed(value),
    }
}
fn array(value: &Value) -> Result<&[Value], SyncError> {
    match value {
        Value::Array(a) => Ok(a),
        _ => Err(invalid("expected an exact array shape")),
    }
}

fn decode(
    name: &str,
    value: &Value,
    types: &TypeRegistry,
    max_depth: usize,
    depth: usize,
) -> Result<Box<dyn Reflect>, SyncError> {
    if depth > max_depth {
        return Err(invalid("ECS decode depth exceeded"));
    }
    macro_rules! leaf {
        ($ty:ty, $pattern:pat => $expr:expr) => {
            if name == type_name::<$ty>() {
                return match value {
                    $pattern => Ok(Box::new($expr) as Box<dyn Reflect>),
                    _ => Err(invalid(format!("incorrect wire value kind for {name}"))),
                };
            }
        };
    }
    macro_rules! integer { ($variant:ident, $($ty:ty),*) => { $(
        leaf!($ty, Value::$variant(v) => <$ty>::try_from(*v).map_err(|_| invalid(format!("integer outside {name} range")))?);
    )* }; }
    macro_rules! signed_integer { ($($ty:ty),*) => { $(
        if name == type_name::<$ty>() {
            let v = match value {
                Value::Signed(v) if *v < 0 => *v,
                Value::Unsigned(v) => i64::try_from(*v).map_err(|_| invalid(format!("integer outside {name} range")))?,
                _ => return Err(invalid(format!("{name} requires a canonical sign-based integer"))),
            };
            return Ok(Box::new(<$ty>::try_from(v).map_err(|_| invalid(format!("integer outside {name} range")))?));
        }
    )* }; }
    signed_integer!(i8, i16, i32, i64, isize);
    integer!(Unsigned, u8, u16, u32, u64, usize);
    leaf!(bool, Value::Bool(v) => *v);
    leaf!(String, Value::String(v) => v.clone());
    leaf!(char, Value::String(v) => {
        let mut chars = v.chars();
        let c = chars.next().ok_or_else(|| invalid("char needs one Unicode scalar"))?;
        if chars.next().is_some() { return Err(invalid("char needs exactly one Unicode scalar")); } c
    });
    leaf!(f32, Value::Unsigned(v) => f32::from_bits(u32::try_from(*v).map_err(|_| invalid("f32 requires exact 32-bit bits"))?));
    leaf!(f64, Value::Float(v) => f64::from_bits(*v));
    leaf!(i128, Value::Bytes(v) => i128::from_le_bytes(v.as_slice().try_into().map_err(|_| invalid("i128 needs exactly 16 LE bytes"))?));
    leaf!(u128, Value::Bytes(v) => u128::from_le_bytes(v.as_slice().try_into().map_err(|_| invalid("u128 needs exactly 16 LE bytes"))?));
    let info = types
        .get_with_name(name)
        .ok_or_else(|| {
            invalid(format!(
                "unregistered nested ECS type {name}; call register_value_type"
            ))
        })?
        .type_info();
    let child = |name: &str, value: &Value| decode(name, value, types, max_depth, depth + 1);
    match info {
        TypeInfo::Struct(s) => {
            let fields = object(value)?;
            if fields.len() != s.field_count() {
                return Err(invalid(format!(
                    "{} requires all and only its declared fields",
                    s.type_name()
                )));
            }
            let mut result = DynamicStruct::new();
            result.set_represented_type_name(s.type_name());
            for f in s.fields() {
                result.insert_boxed(
                    f.name(),
                    child(
                        f.type_name(),
                        fields
                            .get(f.name())
                            .ok_or_else(|| invalid(format!("missing field {}", f.name())))?,
                    )?,
                );
            }
            Ok(Box::new(result))
        }
        TypeInfo::TupleStruct(s) => {
            let fields = array(value)?;
            if fields.len() != s.field_count() {
                return Err(invalid("tuple field count mismatch"));
            }
            let mut result = DynamicTupleStruct::new();
            result.set_represented_type_name(s.type_name());
            for (f, v) in s.fields().iter().zip(fields) {
                result.insert_boxed(child(f.type_name(), v)?);
            }
            Ok(Box::new(result))
        }
        TypeInfo::List(s) => {
            let mut result = DynamicList::new();
            result.set_represented_type_name(s.type_name());
            for v in array(value)? {
                result.push_boxed(child(s.item_type_name(), v)?);
            }
            Ok(Box::new(result))
        }
        TypeInfo::Array(s) => {
            let values = array(value)?;
            if values.len() != s.capacity() {
                return Err(invalid("fixed ECS array length mismatch"));
            }
            let mut result = DynamicArray::new();
            result.set_represented_type_name(s.type_name());
            for v in values {
                result.push_boxed(child(s.item_type_name(), v)?);
            }
            Ok(Box::new(result))
        }
        TypeInfo::Enum(e) => {
            let fields = object(value)?;
            if fields.len() != 2 {
                return Err(invalid("enum requires only variant and fields"));
            }
            let Some(Value::String(variant)) = fields.get("variant") else {
                return Err(invalid("enum needs variant string"));
            };
            let fields = fields
                .get("fields")
                .ok_or_else(|| invalid("enum needs fields payload"))?;
            let v = e
                .variant(variant)
                .ok_or_else(|| invalid(format!("unknown {} variant {variant}", e.type_name())))?;
            let payload = match v.kind() {
                VariantKind::Unit => {
                    if *fields != Value::Null {
                        return Err(invalid("unit enum needs null payload"));
                    }
                    DynamicVariant::Unit
                }
                VariantKind::Tuple(descriptors) => {
                    let fields = array(fields)?;
                    if fields.len() != descriptors.len() {
                        return Err(invalid("enum tuple arity mismatch"));
                    }
                    DynamicVariant::Tuple(
                        descriptors
                            .iter()
                            .zip(fields)
                            .map(|(f, value)| {
                                decode(f.type_name(), value, types, max_depth, depth + 2)
                            })
                            .collect::<Result<_, _>>()?,
                    )
                }
                VariantKind::Struct(descriptors) => {
                    let fields = object(fields)?;
                    if fields.len() != descriptors.len() {
                        return Err(invalid("enum struct field count mismatch"));
                    }
                    let mut values = Vec::new();
                    for f in descriptors {
                        values.push((
                            f.name(),
                            decode(
                                f.type_name(),
                                fields
                                    .get(f.name())
                                    .ok_or_else(|| invalid("missing enum field"))?,
                                types,
                                max_depth,
                                depth + 2,
                            )?,
                        ));
                    }
                    DynamicVariant::Struct(values)
                }
                _ => return Err(invalid("unsupported enum kind")),
            };
            let mut result = DynamicEnum::new(v.index(), v.name(), payload);
            result.set_represented_type_name(e.type_name());
            Ok(Box::new(result))
        }
        TypeInfo::Map(m) => {
            if m.key_type_name() != type_name::<String>() {
                return Err(invalid("only String-keyed ECS maps are supported"));
            }
            let mut result = DynamicMap::new();
            result.set_represented_type_name(m.type_name());
            for (key, value) in object(value)? {
                result.insert_boxed(Box::new(key.clone()), child(m.value_type_name(), value)?);
            }
            Ok(Box::new(result))
        }
        _ => Err(invalid(format!("unsupported ECS type {name}"))),
    }
}
