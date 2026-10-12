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

use alloc::vec::Vec;

use crate::identity::LocalEntityId;
use crate::overrides::{ComponentTypeId, OverrideSet};
use crate::path::FieldPath;

/// The `(target, component, path)` key of an override dropped as an orphan
/// during composition (design §7.3).
pub type OrphanKey = (LocalEntityId, ComponentTypeId, FieldPath);

/// The result of composing an override-layer stack (design §7.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverrideComposition {
    /// The single effective override set after replaying every layer.
    pub effective: OverrideSet,
    /// Property-override keys dropped because their target was removed by some
    /// layer.
    pub orphans: Vec<OrphanKey>,
}

/// Composes override layers bottom-up: `layers[0]` is the base-most layer and
/// the last element is the instance layer (design §7.3).
///
/// Later layers win on a property key; structural ops keep layer order; orphan
/// property overrides are dropped and reported.
#[must_use]
pub fn compose_layers(layers: &[OverrideSet]) -> OverrideComposition {
    let mut effective = OverrideSet::new();
    for layer in layers {
        for property in layer.properties() {
            effective.set_property(property.clone());
        }
        for op in layer.structural() {
            effective.push_structural(op.clone());
        }
    }
    let orphans = effective.drop_orphans();
    OverrideComposition {
        effective,
        orphans,
    }
}

#[cfg(test)]
mod tests {
    use super::compose_layers;
    use crate::identity::LocalEntityId;
    use crate::overrides::{
        ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride, StructuralOp,
    };
    use crate::path::FieldPath;

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
    }
}
