//! Field-level network replication built on reflection diff/patch.
//!
//! A replication plan selects which named fields of a struct participate in
//! replication; [`replicated_diff`] then computes a minimal [`Patch`] covering
//! only those fields, and [`apply_replicated`] applies such a patch to a
//! target. This is the reflection backing for design §24.5's field-level
//! network delta (`prism_replication`): the engine serializes only the fields
//! that both changed and opted into replication.
//!
//! Opt-in/opt-out intent is read from the owning type's
//! [`TypeMetadata`](crate::TypeMetadata) using two custom attribute keys that
//! mirror the design's `#[reflect(replicate)]` / `#[reflect(no_replicate)]`
//! annotations: [`REPLICATE_KEY`] and [`NO_REPLICATE_KEY`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::diff::DiffError;
use crate::schema::{AttributeValue, FieldMetadata};
use crate::{diff, Patch, Reflect, ReflectMut, ReflectRef, TypeInfo, TypeMetadata};

/// Custom field-metadata key marking a field as opted into replication.
pub const REPLICATE_KEY: &str = "replicate";
/// Custom field-metadata key marking a field as excluded from replication.
pub const NO_REPLICATE_KEY: &str = "no_replicate";

/// How a [`ReplicationPlan`] is derived from field metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationPolicy {
    /// Replicate only fields explicitly flagged with [`REPLICATE_KEY`] = true.
    OptIn,
    /// Replicate every field except those flagged with [`NO_REPLICATE_KEY`] =
    /// true.
    OptOut,
}

/// An ordered allowlist of struct field names that participate in replication.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplicationPlan {
    fields: Vec<String>,
}

fn flag_is_set(meta: &FieldMetadata, key: &str) -> bool {
    matches!(meta.custom(key), Some(AttributeValue::Bool(true)))
}

impl ReplicationPlan {
    /// Build a plan from an explicit list of field names.
    #[must_use]
    pub fn new(fields: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            fields: fields.into_iter().map(Into::into).collect(),
        }
    }

    /// Derive a plan from a struct [`TypeInfo`] and its [`TypeMetadata`] under
    /// `policy`.
    ///
    /// Under [`ReplicationPolicy::OptIn`] a field is included only when its
    /// metadata sets [`REPLICATE_KEY`] to `true`. Under
    /// [`ReplicationPolicy::OptOut`] every named field is included unless its
    /// metadata sets [`NO_REPLICATE_KEY`] to `true`. Field order follows the
    /// struct's declared field order. Non-struct type info yields an empty
    /// plan.
    #[must_use]
    pub fn from_type(
        info: &TypeInfo,
        meta: Option<&TypeMetadata>,
        policy: ReplicationPolicy,
    ) -> Self {
        let TypeInfo::Struct(struct_info) = info else {
            return Self::default();
        };
        let mut fields = Vec::new();
        for field in struct_info.fields() {
            let name = field.name();
            let field_meta = meta.and_then(|m| m.field(name));
            let include = match policy {
                ReplicationPolicy::OptIn => {
                    field_meta.is_some_and(|m| flag_is_set(m, REPLICATE_KEY))
                }
                ReplicationPolicy::OptOut => {
                    !field_meta.is_some_and(|m| flag_is_set(m, NO_REPLICATE_KEY))
                }
            };
            if include {
                fields.push(name.to_string());
            }
        }
        Self { fields }
    }

    /// The replicated field names, in order.
    #[must_use]
    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Whether the plan selects no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Compute a minimal [`Patch`] over only the fields selected by `plan`.
///
/// Both `old` and `new` must be the same named-field struct. Each selected
/// field is diffed independently; unchanged fields are dropped so the result
/// carries only the delta. If nothing selected changed, the result is
/// [`Patch::Unchanged`].
///
/// # Errors
/// Returns [`ReplicationError::NotAStruct`] if either value is not a struct, or
/// [`ReplicationError::MissingField`] if a planned field is absent from either
/// value.
pub fn replicated_diff(
    old: &dyn Reflect,
    new: &dyn Reflect,
    plan: &ReplicationPlan,
) -> Result<Patch, ReplicationError> {
    let (ReflectRef::Struct(old_s), ReflectRef::Struct(new_s)) =
        (old.reflect_ref(), new.reflect_ref())
    else {
        return Err(ReplicationError::NotAStruct);
    };
    let mut changed = Vec::new();
    for name in &plan.fields {
        let old_field = old_s
            .field(name)
            .ok_or_else(|| ReplicationError::MissingField(name.clone()))?;
        let new_field = new_s
            .field(name)
            .ok_or_else(|| ReplicationError::MissingField(name.clone()))?;
        let patch = diff(old_field, new_field);
        if !patch.is_unchanged() {
            changed.push((name.clone(), patch));
        }
    }
    if changed.is_empty() {
        Ok(Patch::Unchanged)
    } else {
        Ok(Patch::Struct(changed))
    }
}

/// Apply a replication patch produced by [`replicated_diff`] onto `target`.
///
/// # Errors
/// Returns a [`DiffError`] if the patch does not structurally match `target`.
pub fn apply_replicated(target: &mut dyn Reflect, patch: &Patch) -> Result<(), DiffError> {
    // Replication patches only ever touch named struct fields; guard the shape
    // so a mismatched target fails loudly rather than silently.
    if let Patch::Struct(_) = patch
        && !matches!(target.reflect_mut(), ReflectMut::Struct(_))
    {
        return Err(DiffError::KindMismatch);
    }
    patch.apply(target)
}

/// An error produced while planning or computing a replication delta.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplicationError {
    /// A replicated diff was requested between values that are not structs.
    NotAStruct,
    /// A planned field name was not found on one of the values.
    MissingField(String),
}

impl fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplicationError::NotAStruct => f.write_str("replicated diff requires struct values"),
            ReplicationError::MissingField(name) => {
                write!(f, "replicated field `{name}` is missing")
            }
        }
    }
}

impl core::error::Error for ReplicationError {}
