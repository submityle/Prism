//! Explicit bridge between validated editor documents and reflected ECS models.
//!
//! The editor owns stable SelectionItem identities and authoring metadata;
//! ECS owns live generational entities. This bridge records their mapping,
//! projects Outliner rows into ChildOf, and transfers only model roots whose
//! component types the host registered with World reflection. It embeds no
//! project-specific gameplay types.

use alloc::{
    collections::BTreeMap,
    collections::VecDeque,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use prism_ecs::{entity::Entity, hierarchy::ChildOf, world::World};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_reflect::{StableTypeId, TypeRegistry};

use crate::{ControllerError, ControllerLimits, EditorController};

/// Failure while projecting or capturing one editor document through ECS.
#[derive(Debug)]
pub enum EditorEcsDocumentError {
    /// An edit gesture is active, so the model is not a stable snapshot.
    Busy,
    /// The destination World has no reflected component registration for a model type.
    UnregisteredComponent(String),
    /// An object is attached to more than one Outliner row.
    DuplicateObjectRow(SelectionItem),
    /// An Outliner row refers to an object absent from the controller.
    UnknownOutlinerObject(SelectionItem),
    /// The controller and bridge no longer have the same identity structure.
    DocumentShapeChanged,
    /// No ECS entity is mapped to an object present in the controller.
    MissingObjectMapping(SelectionItem),
    /// No ECS entity is mapped to an Outliner row present in the controller.
    MissingOutlinerMapping(OutlinerId),
    /// An Outliner entity has a `ChildOf` target outside this document bridge.
    ForeignOutlinerParent(Entity),
    /// A mapped generational entity has been despawned or replaced.
    StaleEntity(Entity),
    /// A mapped entity no longer carries the reflected model component.
    MissingComponent {
        /// Stable editor identity of the model.
        item: SelectionItem,
        /// Concrete reflected type path.
        type_path: String,
    },
    /// ECS rejected a reflected component insertion after preflight.
    InsertRejected {
        /// Stable editor identity of the model.
        item: SelectionItem,
        /// Concrete reflected type path.
        type_path: String,
    },
    /// A fresh Outliner edge unexpectedly replaced an existing parent.
    HierarchyConflict {
        /// Child Outliner row.
        child: OutlinerId,
        /// Parent Outliner row.
        parent: OutlinerId,
    },
    /// The ECS `ChildOf` graph contains a cycle and cannot form an Outliner tree.
    HierarchyCycle,
    /// The reconstructed controller rejected a model or Outliner snapshot.
    Controller(ControllerError),
}

impl fmt::Display for EditorEcsDocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => f.write_str("cannot bridge an active editor document edit"),
            Self::UnregisteredComponent(path) => {
                write!(f, "ECS reflected component is not registered: {path}")
            }
            Self::DuplicateObjectRow(item) => {
                write!(f, "editor object has multiple Outliner rows: {item:?}")
            }
            Self::UnknownOutlinerObject(item) => {
                write!(
                    f,
                    "Outliner row refers to an unknown editor object: {item:?}"
                )
            }
            Self::DocumentShapeChanged => {
                f.write_str("editor document identities changed since ECS materialization")
            }
            Self::MissingObjectMapping(item) => {
                write!(f, "ECS bridge has no entity for editor object: {item:?}")
            }
            Self::MissingOutlinerMapping(id) => {
                write!(f, "ECS bridge has no entity for Outliner row {id:?}")
            }
            Self::ForeignOutlinerParent(entity) => {
                write!(
                    f,
                    "Outliner entity has a parent outside this document: {entity:?}"
                )
            }
            Self::StaleEntity(entity) => write!(f, "ECS bridge entity is stale: {entity:?}"),
            Self::MissingComponent { item, type_path } => {
                write!(f, "ECS entity for {item:?} has no component {type_path}")
            }
            Self::InsertRejected { item, type_path } => {
                write!(f, "ECS rejected component {type_path} for {item:?}")
            }
            Self::HierarchyConflict { child, parent } => {
                write!(f, "ECS rejected Outliner edge {child:?} -> {parent:?}")
            }
            Self::HierarchyCycle => f.write_str("ECS Outliner hierarchy contains a cycle"),
            Self::Controller(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for EditorEcsDocumentError {}

impl From<ControllerError> for EditorEcsDocumentError {
    fn from(value: ControllerError) -> Self {
        Self::Controller(value)
    }
}

/// Owns one document's stable identity mapping into a live ECS World.
///
/// It creates one entity per Outliner row, including grouping rows. Objects
/// absent from the Outliner receive detached entities. An object may bind to at
/// most one row. Capturing returns a new clean controller suitable for
/// DocumentCodec encoding; it never mutates the source controller or its
/// undo/redo history.
pub struct EditorEcsDocumentBridge {
    objects: BTreeMap<SelectionItem, Entity>,
    rows: BTreeMap<OutlinerId, Entity>,
    row_ids: BTreeMap<Entity, OutlinerId>,
    owned: Vec<Entity>,
}

impl EditorEcsDocumentBridge {
    /// Materializes every model and Outliner row into World.
    ///
    /// Component registrations and identity bindings are checked before the
    /// first entity is allocated. If insertion or hierarchy linking fails, all
    /// entities allocated by this call are despawned. Lifecycle hooks and
    /// observer side effects triggered during that cleanup cannot be undone.
    pub fn materialize(
        world: &mut World,
        controller: &EditorController,
    ) -> Result<Self, EditorEcsDocumentError> {
        if controller.has_active_edit() {
            return Err(EditorEcsDocumentError::Busy);
        }

        let mut models = Vec::with_capacity(controller.document_objects().len());
        let mut model_types = BTreeMap::new();
        for (item, root, _) in controller.document_objects() {
            let path = root.type_name().to_string();
            let stable = StableTypeId::of_path(&path);
            let Some(component) = world.reflect_component_id(stable) else {
                return Err(EditorEcsDocumentError::UnregisteredComponent(path));
            };
            if world.reflect_stable_id(component) != Some(stable) {
                return Err(EditorEcsDocumentError::UnregisteredComponent(path));
            }
            model_types.insert(item.clone(), (path, stable));
            models.push((item.clone(), root.reflect_clone()));
        }

        let mut object_rows = BTreeMap::new();
        for row in controller.document_outliner_nodes() {
            if let Some(item) = row.selection() {
                if !model_types.contains_key(item) {
                    return Err(EditorEcsDocumentError::UnknownOutlinerObject(item.clone()));
                }
                if object_rows.insert(item.clone(), row.id()).is_some() {
                    return Err(EditorEcsDocumentError::DuplicateObjectRow(item.clone()));
                }
            }
        }

        let mut bridge = Self {
            objects: BTreeMap::new(),
            rows: BTreeMap::new(),
            row_ids: BTreeMap::new(),
            owned: Vec::with_capacity(controller.document_outliner_nodes().len() + models.len()),
        };

        for row in controller.document_outliner_nodes() {
            let entity = world.spawn(());
            bridge.owned.push(entity);
            bridge.rows.insert(row.id(), entity);
            bridge.row_ids.insert(entity, row.id());
            if let Some(item) = row.selection() {
                bridge.objects.insert(item.clone(), entity);
            }
        }
        for (item, _) in &models {
            if !bridge.objects.contains_key(item) {
                let entity = world.spawn(());
                bridge.owned.push(entity);
                bridge.objects.insert(item.clone(), entity);
            }
        }

        for (item, root) in models {
            let entity = bridge.objects[&item];
            if !world.insert_reflect(entity, &*root) {
                let type_path = model_types[&item].0.clone();
                bridge.rollback(world);
                return Err(EditorEcsDocumentError::InsertRejected { item, type_path });
            }
        }

        if controller
            .document_outliner_nodes()
            .iter()
            .any(|row| row.parent().is_some())
        {
            world.register_child_of();
            for row in controller.document_outliner_nodes() {
                let Some(parent_id) = row.parent() else {
                    continue;
                };
                let child = bridge.rows[&row.id()];
                let parent = bridge.rows[&parent_id];
                if world.add_relation::<ChildOf>(child, parent).is_some() {
                    bridge.rollback(world);
                    return Err(EditorEcsDocumentError::HierarchyConflict {
                        child: row.id(),
                        parent: parent_id,
                    });
                }
            }
        }
        Ok(bridge)
    }

    /// Returns the live entity associated with an editor object.
    #[must_use]
    pub fn entity_for_object(&self, item: &SelectionItem) -> Option<Entity> {
        self.objects.get(item).copied()
    }

    /// Returns the live entity associated with an Outliner row.
    #[must_use]
    pub fn entity_for_outliner(&self, id: OutlinerId) -> Option<Entity> {
        self.rows.get(&id).copied()
    }

    /// Number of bridge-owned entities, including grouping rows.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.owned.len()
    }

    /// Captures live ECS models and the source Outliner into a new controller.
    ///
    /// The source controller supplies validation metadata and type registries;
    /// the returned controller can be encoded and decoded through the normal
    /// PRISMDOC codec. Missing or stale mappings fail without mutating source.
    pub fn capture_controller(
        &self,
        world: &World,
        source: &EditorController,
        limits: ControllerLimits,
    ) -> Result<EditorController, EditorEcsDocumentError> {
        if source.has_active_edit() {
            return Err(EditorEcsDocumentError::Busy);
        }
        let object_ids_match = source
            .document_objects()
            .all(|(item, _, _)| self.objects.contains_key(item));
        let row_ids_match = source
            .document_outliner_nodes()
            .iter()
            .all(|row| self.rows.contains_key(&row.id()));
        if source.document_objects().len() != self.objects.len()
            || source.document_outliner_nodes().len() != self.rows.len()
            || !object_ids_match
            || !row_ids_match
        {
            return Err(EditorEcsDocumentError::DocumentShapeChanged);
        }

        let mut captured = EditorController::with_limits(limits);
        for (item, original, registry) in source.document_objects() {
            let entity = self
                .objects
                .get(item)
                .copied()
                .ok_or_else(|| EditorEcsDocumentError::MissingObjectMapping(item.clone()))?;
            if !world.contains(entity) {
                return Err(EditorEcsDocumentError::StaleEntity(entity));
            }
            let type_path = original.type_name().to_string();
            let type_id = StableTypeId::of_path(&type_path);
            let value = world.get_reflect(entity, type_id).ok_or_else(|| {
                EditorEcsDocumentError::MissingComponent {
                    item: item.clone(),
                    type_path,
                }
            })?;
            captured.register_object(
                item.clone(),
                value.reflect_clone(),
                copy_type_registry(registry),
            )?;
        }
        let mut roots = VecDeque::new();
        let mut children = BTreeMap::<OutlinerId, Vec<OutlinerNode>>::new();
        for row in source.document_outliner_nodes() {
            let entity = self
                .rows
                .get(&row.id())
                .copied()
                .ok_or(EditorEcsDocumentError::MissingOutlinerMapping(row.id()))?;
            if !world.contains(entity) {
                return Err(EditorEcsDocumentError::StaleEntity(entity));
            }
            let parent = world
                .relation_targets::<ChildOf>(entity)
                .first()
                .map(|parent| {
                    self.row_ids
                        .get(parent)
                        .copied()
                        .ok_or(EditorEcsDocumentError::ForeignOutlinerParent(*parent))
                })
                .transpose()?;
            let node = copy_outliner_node(row, parent);
            if let Some(parent) = parent {
                children.entry(parent).or_default().push(node);
            } else {
                roots.push_back(node);
            }
        }
        let mut registered = 0;
        while let Some(node) = roots.pop_front() {
            let id = node.id();
            captured.register_outliner_node(node)?;
            registered += 1;
            if let Some(mut descendants) = children.remove(&id) {
                roots.extend(descendants.drain(..));
            }
        }
        if registered != self.rows.len() {
            return Err(EditorEcsDocumentError::HierarchyCycle);
        }
        Ok(captured)
    }

    /// Despawns all still-live bridge entities in reverse creation order.
    /// Returns the number of bridge-owned entities that were live on entry,
    /// including entities removed by a relation's cascade policy.
    pub fn despawn(self, world: &mut World) -> usize {
        let live_owned = self
            .owned
            .iter()
            .filter(|entity| world.contains(**entity))
            .count();
        for entity in self.owned.into_iter().rev() {
            world.despawn(entity);
        }
        live_owned
    }

    fn rollback(&mut self, world: &mut World) {
        for entity in self.owned.drain(..).rev() {
            world.despawn(entity);
        }
        self.objects.clear();
        self.rows.clear();
        self.row_ids.clear();
    }
}

fn copy_outliner_node(source: &OutlinerNode, parent: Option<OutlinerId>) -> OutlinerNode {
    let mut node = OutlinerNode::new(source.id(), parent, source.label());
    if let Some(item) = source.selection() {
        node.set_selection(item.clone());
    }
    node.set_expanded(source.is_expanded());
    node.set_visible(source.is_visible());
    node
}

fn copy_type_registry(source: &TypeRegistry) -> TypeRegistry {
    let mut copy = TypeRegistry::new();
    for registration in source.iter() {
        if source
            .get(registration.type_id())
            .is_some_and(|registered| registered.type_name() == registration.type_name())
        {
            copy.add_registration(registration.clone());
        } else {
            copy.register_runtime(registration.clone());
        }
    }
    copy
}
