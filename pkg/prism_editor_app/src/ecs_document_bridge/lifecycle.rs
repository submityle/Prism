//! Explicit authoring-structure publication without rebuilding surviving models.

use super::*;
use alloc::collections::BTreeSet;
use prism_ecs::relation::CleanupPolicy;

/// Work performed by one successful structural synchronization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditorEcsStructureSync {
    /// Newly allocated entities, including newly authored grouping rows.
    pub spawned: usize,
    /// Previously owned live entities removed from the document and World.
    pub despawned: usize,
    /// Surviving or new entities whose `ChildOf` target changed.
    pub reparented: usize,
}

impl EditorEcsDocumentBridge {
    /// Publishes committed create/delete/undo/redo and Outliner edits to ECS.
    ///
    /// Object identity takes precedence over row identity: a surviving authored
    /// object keeps its generational entity, even when moved to another row or
    /// detached from the Outliner. Its model and transient components are not
    /// overwritten. New objects alone are initialized from the controller.
    /// Existing property edits use the model-edit path, not this structural API.
    /// Recreated objects after undo receive fresh generational entities.
    ///
    /// Registration, duplicate bindings, stale retained entities, unexpected
    /// reflected components, relation policy and deletion boundaries are checked
    /// before allocating or modifying entities. New model insertions are staged
    /// before existing hierarchy or ownership changes. Failed insertion removes
    /// the newly allocated entities and leaves the mapping unchanged. As with
    /// materialization, host hook/observer side effects and panics are not a
    /// transactional World rollback.
    /// Every reflected insertion and removal is followed by a World identity
    /// check. A replacement returns `WorldMismatch` without using more old
    /// handles; the host must discard this bridge, not retry it on the new World.
    /// Hook effects on the original World are not reverted, including effects
    /// occurring after deletion preflight or during staged-entity cleanup.
    ///
    /// Work follows document identities and incoming edges of removed entities;
    /// it never scans the World's entities, type registry or relation graph.
    /// A no-op does not write model components or touch relation edges.
    pub fn synchronize_structure(
        &mut self,
        world: &mut World,
        source: &EditorController,
    ) -> Result<EditorEcsStructureSync, EditorEcsDocumentError> {
        if world.id() != self.world_id {
            return Err(EditorEcsDocumentError::WorldMismatch);
        }
        if source.has_active_edit() {
            return Err(EditorEcsDocumentError::Busy);
        }
        validate_child_of_policy(world)?;

        let mut objects = BTreeMap::new();
        let mut retained = BTreeSet::new();
        for (item, root, _) in source.document_objects() {
            let stable = StableTypeId::of_path(root.type_name());
            if world
                .reflect_component_id(stable)
                .and_then(|id| world.reflect_stable_id(id))
                != Some(stable)
            {
                return Err(EditorEcsDocumentError::UnregisteredComponent(
                    root.type_name().to_string(),
                ));
            }
            let entity = self.objects.get(item).copied();
            if let Some(entity) = entity {
                validate_model(world, entity, item, root.type_name())?;
                retained.insert(entity);
            }
            objects.insert(item.clone(), entity);
        }

        let mut bound = BTreeSet::new();
        let old_objects: BTreeSet<_> = self.objects.values().copied().collect();
        // Rows referring to newly created objects are resolved after staging.
        let mut rows = BTreeMap::new();
        for row in source.document_outliner_nodes() {
            let entity = if let Some(item) = row.selection() {
                let entity = objects
                    .get(item)
                    .ok_or_else(|| EditorEcsDocumentError::UnknownOutlinerObject(item.clone()))?;
                if !bound.insert(item.clone()) {
                    return Err(EditorEcsDocumentError::DuplicateObjectRow(item.clone()));
                }
                *entity
            } else {
                // A former object row must not alias an object that survived
                // elsewhere. All object reservations were collected above.
                self.rows
                    .get(&row.id())
                    .copied()
                    .filter(|entity| !old_objects.contains(entity))
            };
            if let Some(entity) = entity {
                if !world.contains(entity) {
                    return Err(EditorEcsDocumentError::StaleEntity(entity));
                }
                retained.insert(entity);
            }
            rows.insert(row.id(), entity);
        }

        let removed: BTreeSet<_> = self
            .owned
            .iter()
            .copied()
            .filter(|entity| !retained.contains(entity) && world.contains(*entity))
            .collect();
        validate_deletions(world, &retained, &removed)?;

        let mut staged = Self {
            world_id: self.world_id,
            objects: BTreeMap::new(),
            rows: BTreeMap::new(),
            row_ids: BTreeMap::new(),
            owned: Vec::new(),
        };
        for (item, root, _) in source.document_objects() {
            if objects[item].is_none() {
                let entity = world.spawn(());
                staged.owned.push(entity);
                let inserted = world.insert_reflect(entity, root);
                // A hook can replace World. Stop before another allocation or
                // any attempt to clean up colliding handles in the new World.
                if world.id() != self.world_id {
                    return Err(EditorEcsDocumentError::WorldMismatch);
                }
                if !inserted {
                    staged.rollback(world);
                    return Err(EditorEcsDocumentError::InsertRejected {
                        item: item.clone(),
                        type_path: root.type_name().to_string(),
                    });
                }
                objects.insert(item.clone(), Some(entity));
            }
        }
        // Hooks may change relation policies or add incoming edges after the
        // first preflight. Recheck before changing any existing hierarchy.
        if let Err(error) = validate_deletions(world, &retained, &removed) {
            staged.rollback(world);
            return Err(error);
        }
        for (item, root, _) in source.document_objects() {
            let entity = objects[item].expect("all object slots were staged");
            if let Err(error) = validate_model(world, entity, item, root.type_name()) {
                staged.rollback(world);
                return Err(error);
            }
        }
        for entity in rows.values().flatten() {
            if !world.contains(*entity) {
                staged.rollback(world);
                return Err(EditorEcsDocumentError::StaleEntity(*entity));
            }
        }
        for row in source.document_outliner_nodes() {
            let entity = match row.selection() {
                Some(item) => objects[item].expect("all object slots were staged"),
                None => rows[&row.id()].unwrap_or_else(|| {
                    let entity = world.spawn(());
                    staged.owned.push(entity);
                    entity
                }),
            };
            rows.insert(row.id(), Some(entity));
        }

        let next_rows: BTreeMap<_, _> = rows
            .into_iter()
            .map(|(id, entity)| (id, entity.expect("all row slots were staged")))
            .collect();
        let next_objects: BTreeMap<_, _> = objects
            .into_iter()
            .map(|(item, entity)| (item, entity.expect("all object slots were staged")))
            .collect();
        let mut desired: BTreeMap<Entity, Option<Entity>> = next_objects
            .values()
            .map(|&entity| (entity, None))
            .collect();
        for row in source.document_outliner_nodes() {
            desired.insert(
                next_rows[&row.id()],
                row.parent().map(|parent| next_rows[&parent]),
            );
        }
        let changed: Vec<_> = desired
            .iter()
            .filter(|(entity, parent)| {
                world.relation_targets::<ChildOf>(**entity) != parent.as_slice()
            })
            .map(|(&entity, &parent)| (entity, parent))
            .collect();
        // Cut old edges first. This avoids transient cycles on hierarchy
        // inversions, and prevents target-delete cascades into retained rows.
        for entity in changed
            .iter()
            .map(|&(entity, _)| entity)
            .chain(removed.iter().copied())
        {
            let parents = world.relation_targets::<ChildOf>(entity).to_vec();
            for parent in parents {
                world.remove_relation::<ChildOf>(entity, parent);
            }
        }
        if changed.iter().any(|(_, parent)| parent.is_some()) {
            world.register_child_of();
        }
        for &(entity, parent) in &changed {
            if let Some(parent) = parent {
                world.add_relation::<ChildOf>(entity, parent);
            }
        }
        let result = EditorEcsStructureSync {
            spawned: staged.owned.len(),
            despawned: removed.len(),
            reparented: changed.len(),
        };
        // Preserve creation order for retained allocations. New allocations
        // are appended, so teardown remains deterministic and reverse-ordered.
        self.owned.retain(|entity| retained.contains(entity));
        self.owned.append(&mut staged.owned);
        self.objects = next_objects;
        self.rows = next_rows;
        self.row_ids = self
            .rows
            .iter()
            .map(|(&id, &entity)| (entity, id))
            .collect();
        for entity in removed {
            world.despawn(entity);
            if world.id() != self.world_id {
                return Err(EditorEcsDocumentError::WorldMismatch);
            }
        }
        Ok(result)
    }
}

pub(super) fn validate_model(
    world: &World,
    entity: Entity,
    item: &SelectionItem,
    type_path: &str,
) -> Result<(), EditorEcsDocumentError> {
    if !world.contains(entity) {
        return Err(EditorEcsDocumentError::StaleEntity(entity));
    }
    let stable = StableTypeId::of_path(type_path);
    if world.get_reflect(entity, stable).is_none() {
        return Err(EditorEcsDocumentError::MissingComponent {
            item: item.clone(),
            type_path: type_path.to_string(),
        });
    }
    let unexpected: Vec<_> = entity_component_ids(world, entity)
        .ok_or(EditorEcsDocumentError::StaleEntity(entity))?
        .into_iter()
        .filter_map(|id| {
            world.reflect_stable_id(id).filter(|&id| id != stable)?;
            Some(world.components().info(id)?.name().to_string())
        })
        .collect();
    if !unexpected.is_empty() {
        return Err(EditorEcsDocumentError::UnexpectedReflectedComponents {
            item: item.clone(),
            type_paths: unexpected,
        });
    }
    Ok(())
}

pub(super) fn validate_child_of_policy(world: &World) -> Result<(), EditorEcsDocumentError> {
    if world
        .components()
        .id_of::<ChildOf>()
        .and_then(|id| world.relations().kind(id))
        .is_some_and(|kind| kind != &ChildOf::KIND)
    {
        return Err(EditorEcsDocumentError::IncompatibleChildOfRelation);
    }
    Ok(())
}

fn validate_deletions(
    world: &World,
    retained: &BTreeSet<Entity>,
    removed: &BTreeSet<Entity>,
) -> Result<(), EditorEcsDocumentError> {
    validate_child_of_policy(world)?;
    let child_of = world.components().id_of::<ChildOf>();
    for &target in removed {
        for (relation, holder) in world.relations().index().incoming_edges(target) {
            // Owned ChildOf sources are detached or reparented before deletion.
            if Some(relation) == child_of
                && (retained.contains(&holder) || removed.contains(&holder))
            {
                continue;
            }
            let policy = world
                .relations()
                .kind(relation)
                .map(|kind| kind.on_delete_target)
                .unwrap_or_default();
            if policy == CleanupPolicy::Panic
                || (policy == CleanupPolicy::Delete && !removed.contains(&holder))
            {
                return Err(EditorEcsDocumentError::DeletionConflict {
                    target,
                    source: holder,
                });
            }
        }
    }
    Ok(())
}
