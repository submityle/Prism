//! Reflected configuration: schema validation + range clamping (design §25.3).
//!
//! The design (§25.3) delegates configuration / cvar rules — range clamping,
//! defaults, renames, and rejecting invalid input — to the reflection layer
//! rather than re-deriving them inside `prism_app`. This module is that seam: it
//! applies [`prism_reflect`]'s [`clamp`] (repair out-of-range numeric fields,
//! reporting each) and [`validate`] (reject structurally invalid input) to a
//! reflected configuration value against the type's schema + metadata.
//!
//! Invalid input is *reported* as [`ValidationError`], never panicked on, which
//! is the safety boundary the design asks for (§24.8). Defaults / renames ride
//! on the same metadata ([`FieldMetadata::default_value`](prism_reflect::schema::FieldMetadata))
//! and migration chain and are layered on top of this primitive.

use prism_reflect::Reflect;
use prism_reflect::schema::{Clamped, TypeMetadata, TypeSchema, ValidationError, clamp, validate};

/// Build the [`TypeSchema`] describing `value`'s reflected type at `version`.
///
/// The static [`TypeInfo`](prism_reflect::TypeInfo) is read from the value
/// itself ([`Reflect::type_info`]), so callers do not need the `Typed` derive.
#[must_use]
pub fn schema_for(value: &dyn Reflect, name: &'static str, version: u32) -> TypeSchema {
    TypeSchema::new(name, version, value.type_info())
}

/// Clamp every range-constrained field of `value` into its declared range, then
/// validate the result against `schema` and `metadata`.
///
/// Returns the fields that were clamped (empty when none were), or every
/// [`ValidationError`] found. Out-of-range numbers are *repaired* (clamped);
/// structural problems — a missing required field, a range-constrained field
/// that is not numeric, or a non-struct value — are *rejected*. Never panics.
///
/// # Errors
/// Every [`ValidationError`] collected by `validate` (never an empty `Vec`).
pub fn clamp_and_validate(
    value: &mut dyn Reflect,
    schema: &TypeSchema,
    metadata: &TypeMetadata,
) -> Result<Vec<Clamped>, Vec<ValidationError>> {
    let clamped = clamp(value, metadata);
    validate(&*value, schema, Some(metadata))?;
    Ok(clamped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_reflect::schema::FieldMetadata;

    #[derive(prism_reflect::Reflect, Default)]
    struct Render {
        exposure: f32,
        bounces: i32,
    }

    fn metadata() -> TypeMetadata {
        TypeMetadata::new()
            .with_field(FieldMetadata::new("exposure").with_range(0.0, 2.0))
            .with_field(FieldMetadata::new("bounces").with_range(0.0, 8.0))
    }

    #[test]
    fn clamps_out_of_range_fields_and_reports_them() {
        let mut render = Render {
            exposure: 5.0,
            bounces: -3,
        };
        let schema = schema_for(&render, "Render", 1);
        let clamped = clamp_and_validate(&mut render, &schema, &metadata()).expect("valid");
        assert_eq!(clamped.len(), 2);
        assert_eq!(render.exposure, 2.0); // clamped up to the declared max
        assert_eq!(render.bounces, 0);
    }

    #[test]
    fn in_range_values_are_untouched() {
        let mut render = Render {
            exposure: 1.0,
            bounces: 3,
        };
        let schema = schema_for(&render, "Render", 1);
        let clamped = clamp_and_validate(&mut render, &schema, &metadata()).expect("valid");
        assert!(clamped.is_empty());
        assert_eq!(render.exposure, 1.0);
        assert_eq!(render.bounces, 3);
    }

    #[test]
    fn rejects_a_missing_required_field() {
        let mut render = Render::default();
        let schema = schema_for(&render, "Render", 1).with_required_fields(&["absent"]);
        let errors = clamp_and_validate(&mut render, &schema, &metadata()).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredField { field } if field.as_str() == "absent"
        )));
    }

    #[test]
    fn rejects_a_non_struct_value() {
        let mut value = 7u32;
        let schema = schema_for(&value, "u32", 1);
        let errors = clamp_and_validate(&mut value, &schema, &TypeMetadata::new()).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::NotAStruct))
        );
    }
}
