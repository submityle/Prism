//! Override-layer composition (design §7.3, Prefab Variant).
//!
//! A nested variant is not a copy of its parent: variant `B` "inherits" `A`
//! purely as *a set of overrides over `A`*. Instantiating a variant therefore
//! replays a stack of sparse layers bottom-up — base template, then each variant
//! layer, then the instance layer:
//!
//! ```text
//! A  (base)   ← B variant overrides   ← C variant overrides   ← instance overrides
//! ```
//!
//! Composition applies the documented conflict rules:
//!
//! - a property override on an upper layer **replaces** the same key on a lower
//!   layer (last-write-wins);
//! - structural ops concatenate in layer order (lower first);
//! - an entity removed by *any* layer makes every property override still
//!   targeting it an **orphan**, which is dropped and reported (design §7.3
//!   `OrphanOverride`) — this is what keeps an upper removal from being
//!   resurrected by a lower override.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::identity::LocalEntityId;
use crate::overrides::{ComponentTypeId, OverrideSet, StructuralOp};
use crate::path::FieldPath;

/// The `(target, component, path)` key of an override dropped as an orphan
/// during composition (design §7.3).
pub type OrphanKey = (LocalEntityId, ComponentTypeId, FieldPath);

/// A structural conflict found while composing otherwise independently valid
/// layers.
///
/// A layer is allowed to use the same local id or component operation as a
/// different layer because it cannot see the other layer.  Once layers are
/// replayed as one override set, however, repeating the same structural
/// operation is invalid.  The layer indices are zero-based and refer to the
/// input passed to [`compose_layers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverrideCompositionError {
    /// The same entity was added more than once.
    DuplicateAddEntity {
        /// The reserved local id repeated by the additions.
        local: LocalEntityId,
        /// The layer containing the first addition.
        first_layer: usize,
        /// The layer containing the repeated addition.
        layer: usize,
    },
    /// The same entity was removed more than once.
    DuplicateRemoveEntity {
        /// The local id repeated by the removals.
        local: LocalEntityId,
        /// The layer containing the first removal.
        first_layer: usize,
        /// The layer containing the repeated removal.
        layer: usize,
    },
    /// The same component was added to the same entity more than once.
    DuplicateAddComponent {
        /// The target entity.
        target: LocalEntityId,
        /// The component repeated on the target.
        component: ComponentTypeId,
        /// The layer containing the first addition.
        first_layer: usize,
        /// The layer containing the repeated addition.
        layer: usize,
    },
    /// The same component was removed from the same entity more than once.
    DuplicateRemoveComponent {
        /// The target entity.
        target: LocalEntityId,
        /// The component repeated on the target.
        component: ComponentTypeId,
        /// The layer containing the first removal.
        first_layer: usize,
        /// The layer containing the repeated removal.
        layer: usize,
    },
    /// An entity was added under a parent removed by an earlier layer.
    AddedEntityParentRemoved {
        /// The entity that could not be added.
        local: LocalEntityId,
        /// The parent removed before this operation.
        parent: LocalEntityId,
        /// The layer containing the invalid addition.
        layer: usize,
    },
    /// A component operation targeted an entity removed by an earlier layer.
    ComponentTargetRemoved {
        /// The entity that is no longer alive.
        target: LocalEntityId,
        /// The component operation's component.
        component: ComponentTypeId,
        /// The layer containing the invalid operation.
        layer: usize,
    },
    /// Removing an entity would leave an added child with a dangling parent.
    RemoveEntityWouldOrphanChild {
        /// The parent that cannot be removed.
        parent: LocalEntityId,
        /// The live added child that refers to `parent`.
        child: LocalEntityId,
        /// The layer containing the invalid removal.
        layer: usize,
    },
}

/// The result of composing an override-layer stack (design §7.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverrideComposition {
    /// The single effective override set after replaying every layer.
    pub effective: OverrideSet,
    /// Property-override keys dropped because their target was removed by some
    /// layer or because their component was removed by the final structural
    /// state.
    pub orphans: Vec<OrphanKey>,
    /// Structural conflicts found while replaying the layers.
    ///
    /// For a deterministic best-effort result, the first occurrence of a
    /// duplicate structural operation is retained in [`Self::effective`] and
    /// later duplicates are omitted.  Callers that require a valid authored
    /// composition must treat a non-empty list as an error rather than
    /// accepting the best-effort result.
    pub errors: Vec<OverrideCompositionError>,
}

/// Composes override layers bottom-up: `layers[0]` is the base-most layer and
/// the last element is the instance layer (design §7.3).
///
/// Later layers win on a property key; structural ops keep layer order;
/// duplicate structural ops are reported and omitted after their first
/// occurrence; orphan property overrides are dropped and reported.
#[must_use]
pub fn compose_layers(layers: &[OverrideSet]) -> OverrideComposition {
    let mut effective = OverrideSet::new();
    let mut errors = Vec::new();
    let mut added_entities = BTreeMap::new();
    let mut removed_entities = BTreeMap::new();
    let mut added_components = BTreeMap::new();
    let mut removed_components = BTreeMap::new();
    let mut entity_state = BTreeMap::new();
    let mut added_entity_parents = BTreeMap::new();

    for (layer_index, layer) in layers.iter().enumerate() {
        for property in layer.properties() {
            effective.set_property(property.clone());
        }
        for op in layer.structural() {
            let duplicate = match op {
                StructuralOp::AddEntity { local, .. } => {
                    if let Some(&first_layer) = added_entities.get(local) {
                        errors.push(OverrideCompositionError::DuplicateAddEntity {
                            local: *local,
                            first_layer,
                            layer: layer_index,
                        });
                        true
                    } else {
                        added_entities.insert(*local, layer_index);
                        false
                    }
                }
                StructuralOp::RemoveEntity { local } => {
                    if let Some(&first_layer) = removed_entities.get(local) {
                        errors.push(OverrideCompositionError::DuplicateRemoveEntity {
                            local: *local,
                            first_layer,
                            layer: layer_index,
                        });
                        true
                    } else if let Some((&child, _)) =
                        added_entity_parents.iter().find(|(child, parent)| {
                            **parent == *local && entity_state.get(*child) != Some(&false)
                        })
                    {
                        errors.push(OverrideCompositionError::RemoveEntityWouldOrphanChild {
                            parent: *local,
                            child,
                            layer: layer_index,
                        });
                        true
                    } else {
                        removed_entities.insert(*local, layer_index);
                        entity_state.insert(*local, false);
                        false
                    }
                }
                StructuralOp::AddComponent {
                    target, component, ..
                } => {
                    let key = (*target, *component);
                    if let Some(&first_layer) = added_components.get(&key) {
                        errors.push(OverrideCompositionError::DuplicateAddComponent {
                            target: *target,
                            component: *component,
                            first_layer,
                            layer: layer_index,
                        });
                        true
                    } else if entity_state.get(target) == Some(&false) {
                        errors.push(OverrideCompositionError::ComponentTargetRemoved {
                            target: *target,
                            component: *component,
                            layer: layer_index,
                        });
                        true
                    } else {
                        removed_components.remove(&key);
                        added_components.insert(key, layer_index);
                        false
                    }
                }
                StructuralOp::RemoveComponent { target, component } => {
                    let key = (*target, *component);
                    if let Some(&first_layer) = removed_components.get(&key) {
                        errors.push(OverrideCompositionError::DuplicateRemoveComponent {
                            target: *target,
                            component: *component,
                            first_layer,
                            layer: layer_index,
                        });
                        true
                    } else if entity_state.get(target) == Some(&false) {
                        errors.push(OverrideCompositionError::ComponentTargetRemoved {
                            target: *target,
                            component: *component,
                            layer: layer_index,
                        });
                        true
                    } else {
                        added_components.remove(&key);
                        removed_components.insert(key, layer_index);
                        false
                    }
                }
            };
            if duplicate {
                continue;
            }
            if let StructuralOp::AddEntity { local, parent, .. } = op {
                if entity_state.get(parent) == Some(&false) {
                    errors.push(OverrideCompositionError::AddedEntityParentRemoved {
                        local: *local,
                        parent: *parent,
                        layer: layer_index,
                    });
                    continue;
                }
                entity_state.insert(*local, true);
                added_entity_parents.insert(*local, *parent);
            }
            effective.push_structural(op.clone());
        }
    }
    let orphans = effective.drop_orphans();
    OverrideComposition {
        effective,
        orphans,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::{compose_layers, OverrideCompositionError};
    use crate::identity::LocalEntityId;
    use crate::overrides::{
        ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride, StructuralOp,
    };
    use crate::path::FieldPath;
    use alloc::vec;

    fn prop(target: u32, component: u64, path: &str, value: f32) -> PropertyOverride {
        PropertyOverride::new(
            LocalEntityId::new(target),
            ComponentTypeId::from_raw(component),
            FieldPath::parse(path).unwrap(),
            OverrideValue::F32(value),
        )
    }

    #[test]
    fn upper_layer_property_replaces_lower() {
        let mut base = OverrideSet::new();
        base.set_property(prop(0, 1, "scale.x", 1.0));
        base.set_property(prop(0, 1, "scale.y", 1.0));

        let mut variant = OverrideSet::new();
        variant.set_property(prop(0, 1, "scale.x", 3.0));

        let composed = compose_layers(&[base, variant]);
        assert!(composed.orphans.is_empty());
        assert_eq!(composed.effective.property_count(), 2);
        let x = FieldPath::parse("scale.x").unwrap();
        assert_eq!(
            composed
                .effective
                .property(LocalEntityId::new(0), ComponentTypeId::from_raw(1), &x)
                .unwrap()
                .value,
            OverrideValue::F32(3.0),
            "the upper layer's value wins"
        );
    }

    #[test]
    fn removal_orphans_lower_property_and_reports_it() {
        let mut base = OverrideSet::new();
        base.set_property(prop(1, 2, "health", 50.0));

        let mut instance = OverrideSet::new();
        instance.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(1),
        });

        let composed = compose_layers(&[base, instance]);
        assert_eq!(composed.orphans.len(), 1);
        assert_eq!(composed.orphans[0].0, LocalEntityId::new(1));
        assert_eq!(composed.effective.property_count(), 0);
        assert_eq!(composed.effective.structural_count(), 1);
    }

    #[test]
    fn three_layers_compose_in_order() {
        let mut a = OverrideSet::new();
        a.set_property(prop(0, 1, "v", 1.0));
        let mut b = OverrideSet::new();
        b.set_property(prop(0, 1, "w", 2.0));
        let mut c = OverrideSet::new();
        c.set_property(prop(0, 1, "v", 9.0));

        let composed = compose_layers(&[a, b, c]);
        assert_eq!(composed.effective.property_count(), 2);
        let v = FieldPath::parse("v").unwrap();
        assert_eq!(
            composed
                .effective
                .property(LocalEntityId::new(0), ComponentTypeId::from_raw(1), &v)
                .unwrap()
                .value,
            OverrideValue::F32(9.0)
        );
    }

    #[test]
    fn empty_stack_composes_to_empty() {
        let composed = compose_layers(&[]);
        assert!(composed.effective.is_empty());
        assert!(composed.orphans.is_empty());
        assert!(composed.errors.is_empty());
    }

    #[test]
    fn duplicate_remove_entity_across_layers_is_reported_and_not_replayed() {
        let mut lower = OverrideSet::new();
        lower.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        let mut upper = OverrideSet::new();
        upper.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        assert!(lower.validate_for_template(1).is_ok());
        assert!(upper.validate_for_template(1).is_ok());

        let composed = compose_layers(&[lower, upper]);
        assert_eq!(composed.effective.structural_count(), 1);
        assert_eq!(
            composed.errors,
            vec![OverrideCompositionError::DuplicateRemoveEntity {
                local: LocalEntityId::new(0),
                first_layer: 0,
                layer: 1,
            }]
        );
    }

    #[test]
    fn duplicate_add_entity_across_layers_is_reported_and_not_replayed() {
        let add = || StructuralOp::AddEntity {
            local: LocalEntityId::reserved(0),
            parent: LocalEntityId::new(0),
            archetype: Default::default(),
            data: vec![],
        };
        let mut lower = OverrideSet::new();
        lower.push_structural(add());
        let mut upper = OverrideSet::new();
        upper.push_structural(add());
        assert!(lower.validate_for_template(1).is_ok());
        assert!(upper.validate_for_template(1).is_ok());

        let composed = compose_layers(&[lower, upper]);
        assert_eq!(composed.effective.structural_count(), 1);
        assert_eq!(
            composed.errors,
            vec![OverrideCompositionError::DuplicateAddEntity {
                local: LocalEntityId::reserved(0),
                first_layer: 0,
                layer: 1,
            }]
        );
    }

    #[test]
    fn inverse_component_operations_across_layers_are_preserved() {
        let component = ComponentTypeId::from_raw(7);
        let mut add = OverrideSet::new();
        add.push_structural(StructuralOp::AddComponent {
            target: LocalEntityId::new(0),
            component,
            data: vec![1],
        });
        let mut remove = OverrideSet::new();
        remove.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component,
        });
        let mut add_again = OverrideSet::new();
        add_again.push_structural(StructuralOp::AddComponent {
            target: LocalEntityId::new(0),
            component,
            data: vec![2],
        });

        let composed = compose_layers(&[add, remove, add_again]);
        assert!(composed.errors.is_empty());
        assert_eq!(composed.effective.structural_count(), 3);
        assert!(matches!(
            &composed.effective.structural()[2],
            StructuralOp::AddComponent { data, .. } if data == &vec![2]
        ));
    }

    #[test]
    fn duplicate_component_removal_across_layers_is_reported() {
        let component = ComponentTypeId::from_raw(7);
        let mut lower = OverrideSet::new();
        lower.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component,
        });
        let mut upper = OverrideSet::new();
        upper.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component,
        });
        assert!(lower.validate_for_template(1).is_ok());
        assert!(upper.validate_for_template(1).is_ok());

        let composed = compose_layers(&[lower, upper]);
        assert_eq!(composed.effective.structural_count(), 1);
        assert_eq!(
            composed.errors,
            vec![OverrideCompositionError::DuplicateRemoveComponent {
                target: LocalEntityId::new(0),
                component,
                first_layer: 0,
                layer: 1,
            }]
        );
    }

    #[test]
    fn remove_component_drops_property_override_and_reports_orphan() {
        let component = ComponentTypeId::from_raw(7);
        let mut lower = OverrideSet::new();
        lower.set_property(prop(0, 7, "health", 100.0));
        let mut upper = OverrideSet::new();
        upper.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component,
        });

        let composed = compose_layers(&[lower, upper]);
        assert!(composed.errors.is_empty());
        assert_eq!(composed.effective.property_count(), 0);
        assert_eq!(composed.effective.structural_count(), 1);
        assert_eq!(
            composed.orphans,
            vec![(
                LocalEntityId::new(0),
                component,
                FieldPath::parse("health").unwrap()
            )]
        );
    }
}
