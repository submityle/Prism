//! ECS inspection/editing sessions backed by the existing UI tools.
//!
//! Read models consume `WorldReport` and `prism_reflect::inspect`; `element`
//! renders them into ordinary Loom elements that existing query, snapshot and
//! replay tools understand. Successful edits record real World checkpoints and
//! the corresponding UI frame. Undo/redo restore whole-world checkpoints, not
//! just the selected field: record checkpoints explicitly when simulation or
//! other editors mutate the World outside this session. Snapshot-registered
//! resources follow the ECS snapshot contract; external side effects and
//! hooks/observers are not replayed by navigation. User clone/hook panics retain
//! the underlying ECS panic semantics, rather than promising transactional I/O.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use prism_ecs::{
    diagnostics::{inspector::WorldReport, profiler::FlameGraph},
    entity::Entity,
    world::World,
};
use prism_reflect::{
    diff, inspect, reflect_path_mut, ApplyError, DiffError, InspectorNode, ParsedPath, Patch,
    Reflect, StableTypeId, TypeRegistry,
};
use prism_ui::Element;
use prism_ui_timetravel::{EcsHistoryError, EcsTimeline, Frame};

/// An owned property tree for a reflected component on the selected entity.
#[derive(Debug, Clone)]
pub struct EcsComponentInspection {
    /// Stable type id accepted by the session's editing methods.
    pub type_id: StableTypeId,
    /// Real reflected field values, types and optional metadata hints.
    pub properties: InspectorNode,
}

/// Owned panel data; it can outlive the World borrow used to capture it.
#[derive(Debug, Clone)]
pub struct EcsInspection {
    /// Archetype, chunk occupancy and change-tick diagnostics.
    pub world: WorldReport,
    /// The selected entity, if still alive at capture time.
    pub selected: Option<Entity>,
    /// Reflected components sorted by stable type id.
    pub components: Vec<EcsComponentInspection>,
    /// The most recently supplied instrumented schedule profile.
    pub profile: Option<FlameGraph>,
}

impl EcsInspection {
    /// Build a real UI view, consumable by `Query`, devtools and UI time travel.
    /// Property values are text; hosts route their editing widgets to the
    /// session API using stable type ids and reflection paths.
    pub fn element(&self) -> Element {
        let mut root = Element::box_()
            .class("ecs-inspector")
            .child(Element::text(format!(
                "entities={} archetypes={} components={} tick={:?}",
                self.world.entity_count,
                self.world.archetype_count,
                self.world.component_count,
                self.world.change_tick,
            )));
        for archetype in &self.world.archetypes {
            root = root.child(
                Element::text(format!(
                    "archetype {:?}: rows={} capacity={} chunks={}",
                    archetype.id,
                    archetype.occupancy.live_rows,
                    archetype.occupancy.capacity(),
                    archetype.occupancy.chunk_count,
                ))
                .class("ecs-archetype"),
            );
        }
        if let Some(entity) = self.selected {
            root = root.child(Element::text(format!("selected {entity:?}")));
        }
        for component in &self.components {
            root = root.child(property_element(&component.properties).class("ecs-component"));
        }
        if let Some(profile) = &self.profile {
            let mut view = Element::box_().class("ecs-profile");
            for (label, duration) in profile.hotspots() {
                view = view.child(Element::text(format!(
                    "{label}: {} ns self",
                    duration.as_nanos()
                )));
            }
            root = root.child(view);
        }
        root
    }
}

fn property_element(node: &InspectorNode) -> Element {
    let mut view = Element::box_()
        .class("ecs-property")
        .child(Element::text(format!(
            "{}: {}{}",
            node.label,
            node.type_name,
            node.value
                .as_ref()
                .map_or_else(String::new, |value| format!(" = {value}")),
        )));
    for child in &node.children {
        view = view.child(property_element(child));
    }
    view
}

/// Failures are reported before editing when registration, selection or draft
/// validation is insufficient. Failed validation does not truncate redo.
#[derive(Debug)]
pub enum EcsInspectorError {
    /// Snapshot/World identity or history navigation failure.
    History(EcsHistoryError),
    /// No entity has been selected.
    NoSelection,
    /// The selected entity has been despawned.
    DeadEntity(Entity),
    /// The component type is not registered for dynamic ECS access.
    UnregisteredType(StableTypeId),
    /// The selected entity does not carry this component.
    MissingComponent(StableTypeId),
    /// The field path is malformed or does not resolve on the component.
    InvalidPath(String),
    /// The reflected value cannot reconstruct the requested component type.
    InvalidValue,
    /// A field value was rejected while editing an owned reflection draft.
    Apply(ApplyError),
    /// A patch was rejected while editing an owned reflection draft.
    Patch(DiffError),
}

impl fmt::Display for EcsInspectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::History(error) => error.fmt(f),
            Self::NoSelection => f.write_str("no ECS entity selected"),
            Self::DeadEntity(entity) => write!(f, "selected entity {entity:?} is no longer alive"),
            Self::UnregisteredType(id) => write!(f, "unregistered reflected component {id:?}"),
            Self::MissingComponent(id) => write!(f, "selected entity has no component {id:?}"),
            Self::InvalidPath(path) => write!(f, "invalid component field path: {path}"),
            Self::InvalidValue => f.write_str("reflected value cannot reconstruct the component"),
            Self::Apply(error) => write!(f, "field edit rejected: {error:?}"),
            Self::Patch(error) => write!(f, "component patch rejected: {error:?}"),
        }
    }
}

impl std::error::Error for EcsInspectorError {}

impl From<EcsHistoryError> for EcsInspectorError {
    fn from(error: EcsHistoryError) -> Self {
        Self::History(error)
    }
}

/// Reusable editor session over a caller-owned World, without a runtime borrow.
///
/// The optional `ecs` feature enables this adapter and its std snapshot backend.
/// Reflection metadata is supplied through `registry_mut`; those display hints
/// are not an authorization policy for programmatic edits.
pub struct EcsInspectorSession {
    history: EcsTimeline,
    selected: Option<Entity>,
    registry: TypeRegistry,
    profile: Option<FlameGraph>,
}

impl EcsInspectorSession {
    /// Capture the initial World/UI checkpoint. All resident components need
    /// snapshot clone glue; only reflection-registered components are editable.
    pub fn new(world: &World, capacity: usize) -> Result<Self, EcsInspectorError> {
        let mut session = Self {
            history: EcsTimeline::new(world, capacity),
            selected: None,
            registry: TypeRegistry::new(),
            profile: None,
        };
        session.checkpoint(world, "Initial")?;
        Ok(session)
    }

    /// Select a live entity, or clear selection. Rejected selections leave the
    /// previous selection intact, including when a different World is supplied.
    pub fn select(
        &mut self,
        world: &World,
        entity: Option<Entity>,
    ) -> Result<(), EcsInspectorError> {
        self.history.validate_world(world)?;
        if let Some(entity) = entity
            && !world.contains(entity)
        {
            return Err(EcsInspectorError::DeadEntity(entity));
        }
        self.selected = entity;
        Ok(())
    }

    /// Registry used by the existing reflection inspector for property hints.
    pub fn registry_mut(&mut self) -> &mut TypeRegistry {
        &mut self.registry
    }

    /// Accept the actual output of a host's instrumented schedule execution.
    pub fn set_profile(&mut self, profile: FlameGraph) {
        self.profile = Some(profile);
    }

    /// Inspect live diagnostics and selected values without marking components
    /// changed. A despawned selection produces an empty property panel.
    pub fn inspect(&self, world: &World) -> Result<EcsInspection, EcsInspectorError> {
        self.history.validate_world(world)?;
        let selected = self.selected.filter(|&entity| world.contains(entity));
        let components = selected.map_or_else(Vec::new, |entity| {
            world
                .reflect_entity(entity)
                .into_iter()
                .map(|(type_id, value)| EcsComponentInspection {
                    type_id,
                    properties: inspect(value, &self.registry),
                })
                .collect()
        });
        Ok(EcsInspection {
            world: WorldReport::capture(world),
            selected,
            components,
            profile: self.profile.clone(),
        })
    }

    /// Read a live component through its stable type id, without a typed query.
    pub fn read<'w>(
        &self,
        world: &'w World,
        type_id: StableTypeId,
    ) -> Result<&'w dyn Reflect, EcsInspectorError> {
        let entity = self.target(world, type_id)?;
        world
            .get_reflect(entity, type_id)
            .ok_or(EcsInspectorError::MissingComponent(type_id))
    }

    /// Insert or replace a reflected component and record the resulting World
    /// and UI. Equal values are true no-ops and preserve the redo branch.
    pub fn set_component(
        &mut self,
        world: &mut World,
        type_id: StableTypeId,
        value: &dyn Reflect,
        label: impl Into<String>,
    ) -> Result<bool, EcsInspectorError> {
        let entity = self.target(world, type_id)?;
        if StableTypeId::of_path(value.type_name()) != type_id {
            return Err(EcsInspectorError::InvalidValue);
        }
        if let Some(live) = world.get_reflect(entity, type_id)
            && diff(live, value).is_unchanged()
        {
            return Ok(false);
        }
        // A newly inserted component must also be undoable. Checking the whole
        // current world alone would miss missing glue on an absent component.
        let component = world
            .reflect_component_id(type_id)
            .expect("target checked registration");
        if world
            .components()
            .info(component)
            .and_then(prism_ecs::component::ComponentInfo::clone_fn)
            .is_none()
        {
            return Err(EcsHistoryError::MissingCloneGlue(alloc::vec![component]).into());
        }
        self.commit(world, label.into(), |world| {
            world.insert_reflect(entity, value)
        })
    }

    /// Edit one reflection path on an owned draft, then reconstruct the typed
    /// component through the normal ECS insertion/change-detection path.
    pub fn edit_field(
        &mut self,
        world: &mut World,
        type_id: StableTypeId,
        path: &str,
        value: &dyn Reflect,
        label: impl Into<String>,
    ) -> Result<bool, EcsInspectorError> {
        let mut draft = self.read(world, type_id)?.reflect_clone();
        let parsed = ParsedPath::parse(path)
            .map_err(|error| EcsInspectorError::InvalidPath(error.to_string()))?;
        let field = reflect_path_mut(&mut *draft, &parsed)
            .ok_or_else(|| EcsInspectorError::InvalidPath(path.to_string()))?;
        field.apply(value).map_err(EcsInspectorError::Apply)?;
        self.set_component(world, type_id, &*draft, label)
    }

    /// Validate an entire patch on a draft before touching live storage. A patch
    /// with a valid first field and invalid later field cannot partially apply.
    pub fn apply_patch(
        &mut self,
        world: &mut World,
        type_id: StableTypeId,
        patch: &Patch,
        label: impl Into<String>,
    ) -> Result<bool, EcsInspectorError> {
        let mut draft = self.read(world, type_id)?.reflect_clone();
        patch.apply(&mut *draft).map_err(EcsInspectorError::Patch)?;
        self.set_component(world, type_id, &*draft, label)
    }

    /// Remove a reflected component, retaining its real value for undo.
    pub fn remove_component(
        &mut self,
        world: &mut World,
        type_id: StableTypeId,
        label: impl Into<String>,
    ) -> Result<bool, EcsInspectorError> {
        let entity = self.target(world, type_id)?;
        if world.get_reflect(entity, type_id).is_none() {
            return Ok(false);
        }
        self.commit(world, label.into(), |world| {
            world.remove_reflect(entity, type_id)
        })
    }

    /// Record a host-driven simulation/editor checkpoint and its current panel.
    pub fn checkpoint(
        &mut self,
        world: &World,
        label: impl Into<String>,
    ) -> Result<usize, EcsInspectorError> {
        let label = label.into();
        let panel = self.inspect(world)?.element();
        Ok(self.history.record_with_ui(
            world,
            label.clone(),
            Some(Frame::capture(label, &panel)),
        )?)
    }

    /// Restore an earlier whole-world checkpoint, including its exact ticks.
    pub fn undo(&mut self, world: &mut World) -> Result<bool, EcsInspectorError> {
        let moved = self.history.undo(world)?.is_some();
        if moved {
            self.after_seek(world);
        }
        Ok(moved)
    }

    /// Restore a later whole-world checkpoint.
    pub fn redo(&mut self, world: &mut World) -> Result<bool, EcsInspectorError> {
        let moved = self.history.redo(world)?.is_some();
        if moved {
            self.after_seek(world);
        }
        Ok(moved)
    }

    /// Restore a retained checkpoint. Hosts can read its paired UI frame from
    /// `history().current()` or rebuild a panel with `inspect`.
    pub fn seek(&mut self, world: &mut World, index: usize) -> Result<(), EcsInspectorError> {
        self.history.seek(world, index)?;
        self.after_seek(world);
        Ok(())
    }

    /// Read-only access to labels, UI frames, snapshots and cursor state.
    pub fn history(&self) -> &EcsTimeline {
        &self.history
    }

    fn target(&self, world: &World, type_id: StableTypeId) -> Result<Entity, EcsInspectorError> {
        self.history.validate_world(world)?;
        let entity = self.selected.ok_or(EcsInspectorError::NoSelection)?;
        if !world.contains(entity) {
            return Err(EcsInspectorError::DeadEntity(entity));
        }
        if world.reflect_component_id(type_id).is_none() {
            return Err(EcsInspectorError::UnregisteredType(type_id));
        }
        Ok(entity)
    }

    fn commit(
        &mut self,
        world: &mut World,
        label: String,
        apply: impl FnOnce(&mut World) -> bool,
    ) -> Result<bool, EcsInspectorError> {
        let before = world
            .try_snapshot()
            .map_err(EcsHistoryError::MissingCloneGlue)?;
        let identity = world.id();
        if !apply(world) {
            return Err(EcsInspectorError::InvalidValue);
        }
        if let Err(error) = self.checkpoint(world, label) {
            // Hooks can change resident component sets. A failed post-edit
            // capture rolls back if this is still the compatible source World.
            if world.id() == identity
                && world.components().schema_generation() == before.schema_generation()
            {
                world.restore(&before);
            }
            return Err(error);
        }
        Ok(true)
    }

    fn after_seek(&mut self, world: &World) {
        self.selected = self.selected.filter(|&entity| world.contains(entity));
        self.profile = None;
    }
}
