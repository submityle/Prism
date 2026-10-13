//! Exact-type property override application for reflected live components.

use alloc::string::String;

use crate::overrides::{OverrideValue, PropertyOverride};
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_reflect::{reflect_path_mut, ParsedPath, Reflect, StableTypeId};

/// Failure while applying one property override to a live entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PropertyApplyError {
    /// The entity or reflected component is not present in the world.
    MissingComponent,
    /// The scene field path could not be represented by the reflect path parser.
    InvalidPath,
    /// Aggregate bytes and other opaque values require a typed materializer.
    UnsupportedValue,
    /// The override scalar has a different concrete type from the reflected slot.
    TypeMismatch,
    /// The reflected path rejected the write.
    WriteRejected,
}

/// Applies one exact-type scalar/string property override to a live reflected component.
///
/// The component is marked changed through `World::get_reflect_mut`. Numeric
/// coercion, schema metadata clamping, and aggregate `Bytes` values are not
/// silently guessed; callers receive a structured error instead.
pub fn apply_property_override(
    world: &mut World,
    entity: Entity,
    property: &PropertyOverride,
) -> Result<(), PropertyApplyError> {
    let type_id = StableTypeId::from_raw(property.component.raw());
    let component = world
        .get_reflect_mut(entity, type_id)
        .ok_or(PropertyApplyError::MissingComponent)?;
    let path = reflect_path_text(&property.path);
    let parsed = ParsedPath::parse(&path).map_err(|_| PropertyApplyError::InvalidPath)?;
    let slot = reflect_path_mut(component, &parsed).ok_or(PropertyApplyError::InvalidPath)?;

    match &property.value {
        OverrideValue::Bool(value) => apply(slot, value),
        OverrideValue::I64(value) => apply(slot, value),
        OverrideValue::U64(value) => apply(slot, value),
        OverrideValue::F32(value) => apply(slot, value),
        OverrideValue::F64(value) => apply(slot, value),
        OverrideValue::Str(value) => apply(slot, value),
        OverrideValue::Bytes(_) => Err(PropertyApplyError::UnsupportedValue),
    }
}

fn apply(target: &mut dyn Reflect, value: &dyn Reflect) -> Result<(), PropertyApplyError> {
    target
        .apply(value)
        .map_err(|_| PropertyApplyError::TypeMismatch)
}

fn reflect_path_text(path: &crate::path::FieldPath) -> String {
    let mut text = String::new();
    for (index, segment) in path.segments().iter().enumerate() {
        match segment {
            crate::path::FieldSegment::Name(name) => {
                if index != 0 {
                    text.push('.');
                }
                text.push_str(name);
            }
            crate::path::FieldSegment::Index(value) => {
                text.push('#');
                text.push_str(&value.to_string());
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overrides::ComponentTypeId;
    use crate::path::FieldPath;
    use alloc::string::String;
    use alloc::vec;
    use prism_ecs::component::Component;
    use prism_reflect::Reflect;

    #[derive(Clone, Debug, Reflect, PartialEq)]
    struct Settings {
        speed: u64,
        enabled: bool,
        label: String,
    }
    impl Component for Settings {}

    fn property(path: &str, value: OverrideValue) -> PropertyOverride {
        PropertyOverride::new(
            crate::identity::LocalEntityId::new(0),
            ComponentTypeId::of::<Settings>(),
            FieldPath::parse(path).unwrap(),
            value,
        )
    }

    #[test]
    fn applies_exact_scalar_and_string_overrides() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let entity = world.spawn(Settings {
            speed: 2,
            enabled: false,
            label: String::from("old"),
        });
        apply_property_override(
            &mut world,
            entity,
            &property("speed", OverrideValue::U64(9)),
        )
        .unwrap();
        apply_property_override(
            &mut world,
            entity,
            &property("label", OverrideValue::Str(String::from("new"))),
        )
        .unwrap();
        let settings = world.get::<Settings>(entity).unwrap();
        assert_eq!(settings.speed, 9);
        assert_eq!(settings.label, "new");
    }

    #[test]
    fn rejects_mismatch_opaque_values_and_missing_components() {
        let mut world = World::new();
        world.register_reflect_component::<Settings>();
        let entity = world.spawn(Settings {
            speed: 2,
            enabled: false,
            label: String::from("old"),
        });
        assert_eq!(
            apply_property_override(
                &mut world,
                entity,
                &property("speed", OverrideValue::I64(9))
            ),
            Err(PropertyApplyError::TypeMismatch)
        );
        assert_eq!(
            apply_property_override(
                &mut world,
                entity,
                &property("speed", OverrideValue::Bytes(vec![9]))
            ),
            Err(PropertyApplyError::UnsupportedValue)
        );
        let other = world.spawn(());
        assert_eq!(
            apply_property_override(&mut world, other, &property("speed", OverrideValue::U64(9))),
            Err(PropertyApplyError::MissingComponent)
        );
    }
}
