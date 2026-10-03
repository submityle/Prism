//! Structural reflection diff and patch merge (design §24.4, §22 — milestone **M5**).
//!
//! Two reflected values of the same shape are compared field-by-field,
//! element-by-element, into a minimal [`Patch`] that records only what differs.
//! Applying that patch back onto a clone of the first value reproduces the
//! second (within the limits documented below). This powers hot-reload (patch
//! one field without rebuilding the object), editor undo/redo, prefab
//! overrides, and reflection-driven network deltas (design §24.4/§24.5).
//!
//! # Round-trip boundary
//! [`Patch::apply`] is built strictly on the existing
//! [`Reflect::apply`](crate::Reflect::apply) capability and the kind traits'
//! mutators, so it inherits their boundaries honestly:
//!
//! - **Lists** only grow or modify in place; a shorter target becomes a whole
//!   [`Patch::Replace`]. Appended elements are pushed via
//!   [`List::push`](crate::List::push), which requires the element's
//!   `reflect_clone` to be acceptable to the concrete list (true for leaf
//!   element types; a derived-struct element clones to a `Dynamic*` value that
//!   a concrete `Vec` rejects — exactly as [`Reflect::apply`](crate::Reflect::apply)
//!   already behaves).
//! - **Maps/sets** only modify existing entries or insert/add new ones; key or
//!   element removal makes the whole value a [`Patch::Replace`]. Inserted
//!   values follow the same leaf-vs-`Dynamic` rule as lists.
//! - **Enums** diff field-wise only when both values share the active variant;
//!   a variant switch is a [`Patch::Replace`], which round-trips onto a
//!   [`DynamicEnum`](crate::DynamicEnum) target (concrete enums reject a variant
//!   switch, matching [`Reflect::apply`](crate::Reflect::apply)).
//!
//! Within those bounds the identity holds: for any `b` reachable from `a` by
//! `apply`, `let mut m = a.reflect_clone(); diff(&a, &b).apply(&mut *m)` leaves
//! `m` equal to `b`.
//!
//! ```
//! use prism_reflect::{Reflect, diff};
//!
//! #[derive(Reflect, Clone, PartialEq, Debug)]
//! struct Point {
//!     x: i32,
//!     y: i32,
//! }
//!
//! let a = Point { x: 1, y: 2 };
//! let b = Point { x: 1, y: 9 };
//!
//! let patch = diff(&a, &b);
//! let mut merged = a.clone();
//! patch.apply(&mut merged).expect("apply patch");
//! assert_eq!(merged, b);
//! ```

use crate::apply::ApplyError;
use crate::dynamic::reflect_values_equal;
use crate::reflect::{Reflect, ReflectMut, ReflectRef};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// A minimal, structural description of the difference between two reflected
/// values, produced by [`diff`] and consumed by [`Patch::apply`]/[`merge`].
///
/// Nested kinds carry only the entries that changed; an unchanged value (or an
/// unchanged nested field) is represented by [`Patch::Unchanged`] or simply
/// omitted from its parent's change list.
#[non_exhaustive]
pub enum Patch {
    /// The two values compared equal; applying this is a no-op.
    Unchanged,
    /// The values are incompatible or differ wholesale; replace the target by
    /// applying this boxed clone of the new value.
    Replace(Box<dyn Reflect>),
    /// A named-field struct: `(field name, field patch)` for each changed field.
    Struct(Vec<(String, Patch)>),
    /// A tuple struct: `(field index, field patch)` for each changed field.
    TupleStruct(Vec<(usize, Patch)>),
    /// An enum whose active variant is unchanged: `(field index, field patch)`
    /// for each changed field of that variant.
    Enum(Vec<(usize, Patch)>),
    /// A list that grew or changed in place: in-place `(index, patch)` edits for
    /// shared indices plus boxed clones appended past the original length.
    List {
        /// Per-index patches for elements shared by both lengths.
        modified: Vec<(usize, Patch)>,
        /// Boxed clones of the elements appended beyond the original length.
        appended: Vec<Box<dyn Reflect>>,
    },
    /// A fixed-length array: `(index, patch)` for each changed element.
    Array(Vec<(usize, Patch)>),
    /// A map that kept all original keys: per-key patches for changed values
    /// plus boxed `(key, value)` clones for newly inserted entries.
    Map {
        /// Patches for values whose key was present in both maps.
        modified: Vec<(Box<dyn Reflect>, Patch)>,
        /// Boxed `(key, value)` clones for keys present only in the new map.
        inserted: Vec<(Box<dyn Reflect>, Box<dyn Reflect>)>,
    },
    /// A set that kept all original elements: boxed clones of the added members.
    Set(Vec<Box<dyn Reflect>>),
}

impl fmt::Debug for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Patch::Unchanged => f.write_str("Unchanged"),
            Patch::Replace(value) => {
                f.debug_tuple("Replace").field(&value.type_name()).finish()
            }
            Patch::Struct(fields) => f.debug_tuple("Struct").field(fields).finish(),
            Patch::TupleStruct(fields) => f.debug_tuple("TupleStruct").field(fields).finish(),
            Patch::Enum(fields) => f.debug_tuple("Enum").field(fields).finish(),
            Patch::List { modified, appended } => f
                .debug_struct("List")
                .field("modified", modified)
                .field("appended", &appended.len())
                .finish(),
            Patch::Array(elements) => f.debug_tuple("Array").field(elements).finish(),
            Patch::Map { modified, inserted } => f
                .debug_struct("Map")
                .field("modified", &modified.len())
                .field("inserted", &inserted.len())
                .finish(),
            Patch::Set(added) => f.debug_struct("Set").field("added", &added.len()).finish(),
        }
    }
}

impl Patch {
    /// Whether this patch represents no change at all.
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        matches!(self, Patch::Unchanged)
    }

    /// Apply this patch onto `target`, mutating it toward the diffed value.
    ///
    /// # Errors
    /// Returns a [`DiffError`] when `target`'s kind does not match the patch,
    /// an expected field/element/key is missing, a container rejects an
    /// appended or inserted element, or the underlying
    /// [`Reflect::apply`](crate::Reflect::apply) for a [`Patch::Replace`] fails.
    pub fn apply(&self, target: &mut dyn Reflect) -> Result<(), DiffError> {
        match self {
            Patch::Unchanged => Ok(()),
            Patch::Replace(value) => target.apply(&**value).map_err(DiffError::Apply),
            Patch::Struct(fields) => {
                let ReflectMut::Struct(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (name, patch) in fields {
                    let field = target
                        .field_mut(name)
                        .ok_or_else(|| DiffError::MissingField { name: name.clone() })?;
                    patch.apply(field)?;
                }
                Ok(())
            }
            Patch::TupleStruct(fields) => {
                let ReflectMut::TupleStruct(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (index, patch) in fields {
                    let field = target
                        .field_mut(*index)
                        .ok_or(DiffError::MissingElement { index: *index })?;
                    patch.apply(field)?;
                }
                Ok(())
            }
            Patch::Enum(fields) => {
                let ReflectMut::Enum(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (index, patch) in fields {
                    let field = target
                        .field_at_mut(*index)
                        .ok_or(DiffError::MissingElement { index: *index })?;
                    patch.apply(field)?;
                }
                Ok(())
            }
            Patch::List { modified, appended } => {
                let ReflectMut::List(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (index, patch) in modified {
                    let element = target
                        .get_mut(*index)
                        .ok_or(DiffError::MissingElement { index: *index })?;
                    patch.apply(element)?;
                }
                for value in appended {
                    target
                        .push(value.reflect_clone())
                        .map_err(|_| DiffError::PushRejected)?;
                }
                Ok(())
            }
            Patch::Array(elements) => {
                let ReflectMut::Array(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (index, patch) in elements {
                    let element = target
                        .get_mut(*index)
                        .ok_or(DiffError::MissingElement { index: *index })?;
                    patch.apply(element)?;
                }
                Ok(())
            }
            Patch::Map { modified, inserted } => {
                let ReflectMut::Map(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for (key, patch) in modified {
                    let value = target.get_mut(&**key).ok_or(DiffError::MissingKey)?;
                    patch.apply(value)?;
                }
                for (key, value) in inserted {
                    target
                        .insert(key.reflect_clone(), value.reflect_clone())
                        .map_err(|_| DiffError::InsertRejected)?;
                }
                Ok(())
            }
            Patch::Set(added) => {
                let ReflectMut::Set(target) = target.reflect_mut() else {
                    return Err(DiffError::KindMismatch);
                };
                for value in added {
                    target
                        .insert(value.reflect_clone())
                        .map_err(|_| DiffError::InsertRejected)?;
                }
                Ok(())
            }
        }
    }
}

/// Compute the minimal [`Patch`] turning `a` into `b`.
///
/// Values of the same reflected kind are compared recursively; mismatched kinds
/// (or shapes outside the [round-trip boundary](self#round-trip-boundary))
/// collapse to a whole-value [`Patch::Replace`]. `diff(a, a)` is always
/// [`Patch::Unchanged`].
#[must_use]
pub fn diff(a: &dyn Reflect, b: &dyn Reflect) -> Patch {
    match (a.reflect_ref(), b.reflect_ref()) {
        (ReflectRef::Struct(a), ReflectRef::Struct(b)) => {
            if a.field_count() != b.field_count() {
                return replace(b.as_reflect());
            }
            let mut fields = Vec::new();
            for index in 0..a.field_count() {
                let Some(name) = a.name_at(index) else {
                    return replace(b.as_reflect());
                };
                let (Some(field_a), Some(field_b)) = (a.field_at(index), b.field(name)) else {
                    return replace(b.as_reflect());
                };
                let child = diff(field_a, field_b);
                if !child.is_unchanged() {
                    fields.push((name.to_string(), child));
                }
            }
            collapse(fields, Patch::Struct)
        }
        (ReflectRef::TupleStruct(a), ReflectRef::TupleStruct(b)) => {
            if a.field_count() != b.field_count() {
                return replace(b.as_reflect());
            }
            let mut fields = Vec::new();
            for index in 0..a.field_count() {
                let (Some(field_a), Some(field_b)) = (a.field(index), b.field(index)) else {
                    return replace(b.as_reflect());
                };
                let child = diff(field_a, field_b);
                if !child.is_unchanged() {
                    fields.push((index, child));
                }
            }
            collapse(fields, Patch::TupleStruct)
        }
        (ReflectRef::Enum(a), ReflectRef::Enum(b)) => {
            if a.variant_index() != b.variant_index()
                || a.variant_name() != b.variant_name()
                || a.field_count() != b.field_count()
            {
                return replace(b.as_reflect());
            }
            let mut fields = Vec::new();
            for index in 0..a.field_count() {
                let (Some(field_a), Some(field_b)) = (a.field_at(index), b.field_at(index)) else {
                    return replace(b.as_reflect());
                };
                let child = diff(field_a, field_b);
                if !child.is_unchanged() {
                    fields.push((index, child));
                }
            }
            collapse(fields, Patch::Enum)
        }
        (ReflectRef::List(a), ReflectRef::List(b)) => {
            if b.len() < a.len() {
                return replace(b.as_reflect());
            }
            let mut modified = Vec::new();
            for index in 0..a.len() {
                let (Some(element_a), Some(element_b)) = (a.get(index), b.get(index)) else {
                    return replace(b.as_reflect());
                };
                let child = diff(element_a, element_b);
                if !child.is_unchanged() {
                    modified.push((index, child));
                }
            }
            let mut appended = Vec::new();
            for index in a.len()..b.len() {
                let Some(element) = b.get(index) else {
                    return replace(b.as_reflect());
                };
                appended.push(element.reflect_clone());
            }
            if modified.is_empty() && appended.is_empty() {
                Patch::Unchanged
            } else {
                Patch::List { modified, appended }
            }
        }
        (ReflectRef::Array(a), ReflectRef::Array(b)) => {
            if a.len() != b.len() {
                return replace(b.as_reflect());
            }
            let mut elements = Vec::new();
            for index in 0..a.len() {
                let (Some(element_a), Some(element_b)) = (a.get(index), b.get(index)) else {
                    return replace(b.as_reflect());
                };
                let child = diff(element_a, element_b);
                if !child.is_unchanged() {
                    elements.push((index, child));
                }
            }
            collapse(elements, Patch::Array)
        }
        (ReflectRef::Map(a), ReflectRef::Map(b)) => {
            let mut modified = Vec::new();
            for (key, value_a) in a.iter_reflect() {
                let Some(value_b) = b.get(key) else {
                    return replace(b.as_reflect());
                };
                let child = diff(value_a, value_b);
                if !child.is_unchanged() {
                    modified.push((key.reflect_clone(), child));
                }
            }
            let mut inserted = Vec::new();
            for (key, value_b) in b.iter_reflect() {
                if a.get(key).is_none() {
                    inserted.push((key.reflect_clone(), value_b.reflect_clone()));
                }
            }
            if modified.is_empty() && inserted.is_empty() {
                Patch::Unchanged
            } else {
                Patch::Map { modified, inserted }
            }
        }
        (ReflectRef::Set(a), ReflectRef::Set(b)) => {
            for value in a.iter_reflect() {
                if !b.contains(value) {
                    return replace(b.as_reflect());
                }
            }
            let mut added = Vec::new();
            for value in b.iter_reflect() {
                if !a.contains(value) {
                    added.push(value.reflect_clone());
                }
            }
            if added.is_empty() {
                Patch::Unchanged
            } else {
                Patch::Set(added)
            }
        }
        (ReflectRef::Value(a), ReflectRef::Value(b)) => {
            if reflect_values_equal(a, b) {
                Patch::Unchanged
            } else {
                replace(b)
            }
        }
        _ => replace(b),
    }
}

/// Apply `patch` onto `target` (a free-function alias for [`Patch::apply`]).
///
/// # Errors
/// Propagates any [`DiffError`] produced by [`Patch::apply`].
pub fn merge(target: &mut dyn Reflect, patch: &Patch) -> Result<(), DiffError> {
    patch.apply(target)
}

/// Build a whole-value [`Patch::Replace`] from a reflected value.
fn replace(value: &dyn Reflect) -> Patch {
    Patch::Replace(value.reflect_clone())
}

/// Collapse an index/name-keyed change list into `Unchanged` when empty, else
/// wrap it with `wrap`.
fn collapse<K>(changes: Vec<(K, Patch)>, wrap: fn(Vec<(K, Patch)>) -> Patch) -> Patch {
    if changes.is_empty() {
        Patch::Unchanged
    } else {
        wrap(changes)
    }
}

/// An error produced while applying a [`Patch`].
#[derive(Debug)]
#[non_exhaustive]
pub enum DiffError {
    /// The underlying [`Reflect::apply`](crate::Reflect::apply) for a
    /// [`Patch::Replace`] failed.
    Apply(ApplyError),
    /// The target's reflected kind did not match the patch's kind.
    KindMismatch,
    /// A struct field named by the patch was absent on the target.
    MissingField {
        /// The missing field's name.
        name: String,
    },
    /// An indexed element named by the patch was absent on the target.
    MissingElement {
        /// The missing element's index.
        index: usize,
    },
    /// A map key named by the patch was absent on the target.
    MissingKey,
    /// The target list rejected an appended element (type incompatible).
    PushRejected,
    /// The target map or set rejected an inserted element (type incompatible).
    InsertRejected,
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::Apply(error) => write!(f, "patch replace failed: {error}"),
            DiffError::KindMismatch => f.write_str("patch kind does not match the target value"),
            DiffError::MissingField { name } => {
                write!(f, "patched field `{name}` is absent on the target")
            }
            DiffError::MissingElement { index } => {
                write!(f, "patched element {index} is absent on the target")
            }
            DiffError::MissingKey => f.write_str("patched map key is absent on the target"),
            DiffError::PushRejected => {
                f.write_str("target list rejected an appended element (incompatible type)")
            }
            DiffError::InsertRejected => {
                f.write_str("target map/set rejected an inserted element (incompatible type)")
            }
        }
    }
}

impl std::error::Error for DiffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DiffError::Apply(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ApplyError> for DiffError {
    fn from(error: ApplyError) -> Self {
        DiffError::Apply(error)
    }
}
