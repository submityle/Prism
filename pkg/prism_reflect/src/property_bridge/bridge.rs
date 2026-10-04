//! The type-safe get/set property bridge (design §24.6).

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::any::type_name;
use core::fmt;

use crate::path::ParsePathError;
use crate::schema::{clamp, AttributeValue, TypeMetadata};
use crate::{reflect_path, reflect_path_mut, Access, ApplyError, ParsedPath, Reflect, TypeRegistry};

/// A reflection-backed property surface for scripts and editors.
///
/// The bridge borrows a [`TypeRegistry`] so writes can consult a field's
/// [`TypeMetadata`] (read-only / range intent); every accessor operates on
/// whatever root value the caller supplies, so one bridge serves many values of
/// many types.
#[derive(Clone, Copy)]
pub struct PropertyBridge<'a> {
    registry: &'a TypeRegistry,
}

impl<'a> PropertyBridge<'a> {
    /// Create a bridge that resolves metadata through `registry`.
    #[must_use]
    pub fn new(registry: &'a TypeRegistry) -> Self {
        Self { registry }
    }

    /// The registry this bridge consults for field metadata.
    #[must_use]
    pub fn registry(&self) -> &'a TypeRegistry {
        self.registry
    }

    /// Read the property at `path` within `root`.
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`] if `path` does not parse, or
    /// [`PropertyError::NoSuchProperty`] if it does not resolve.
    pub fn get<'r>(
        &self,
        root: &'r dyn Reflect,
        path: &str,
    ) -> Result<&'r dyn Reflect, PropertyError> {
        let parsed = ParsedPath::parse(path).map_err(PropertyError::Path)?;
        reflect_path(root, &parsed).ok_or_else(|| PropertyError::NoSuchProperty(path.to_string()))
    }

    /// Mutably borrow the property at `path` within `root`.
    ///
    /// This is the raw mutable slot and does **not** enforce read-only or range
    /// metadata; prefer [`set`](Self::set)/[`set_scalar`](Self::set_scalar) for
    /// metadata-aware writes.
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`] or [`PropertyError::NoSuchProperty`] as
    /// [`get`](Self::get) does.
    pub fn get_mut<'r>(
        &self,
        root: &'r mut dyn Reflect,
        path: &str,
    ) -> Result<&'r mut dyn Reflect, PropertyError> {
        let parsed = ParsedPath::parse(path).map_err(PropertyError::Path)?;
        reflect_path_mut(root, &parsed)
            .ok_or_else(|| PropertyError::NoSuchProperty(path.to_string()))
    }

    /// Read the property at `path` and downcast it to the concrete type `T`.
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`]/[`PropertyError::NoSuchProperty`] if the
    /// path is invalid or unresolved, or [`PropertyError::TypeMismatch`] if the
    /// property is not a `T`.
    pub fn get_as<'r, T: Reflect>(
        &self,
        root: &'r dyn Reflect,
        path: &str,
    ) -> Result<&'r T, PropertyError> {
        let value = self.get(root, path)?;
        value.downcast_ref::<T>().ok_or_else(|| PropertyError::TypeMismatch {
            path: path.to_string(),
            expected: type_name::<T>().to_string(),
            found: value.type_name().to_string(),
        })
    }

    /// Write `value` into the property at `path`, applying it onto the existing
    /// value in place.
    ///
    /// The write is metadata-aware: a field flagged
    /// [`readonly`](crate::schema::FieldMetadata::is_readonly) in its parent's
    /// [`TypeMetadata`] is rejected, and a range-constrained field is clamped
    /// into its bounds after assignment.
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`]/[`PropertyError::NoSuchProperty`] for an
    /// invalid or unresolved path, [`PropertyError::ReadOnly`] if the target
    /// field is read-only, or [`PropertyError::Apply`] if `value` is
    /// incompatible with the target slot.
    pub fn set(
        &self,
        root: &mut dyn Reflect,
        path: &str,
        value: &dyn Reflect,
    ) -> Result<(), PropertyError> {
        let parsed = ParsedPath::parse(path).map_err(PropertyError::Path)?;
        self.reject_readonly(root, &parsed, path)?;
        let slot = reflect_path_mut(root, &parsed)
            .ok_or_else(|| PropertyError::NoSuchProperty(path.to_string()))?;
        slot.apply(value).map_err(PropertyError::Apply)?;
        self.clamp_terminal(root, &parsed);
        Ok(())
    }

    /// Read the leaf property at `path` as a dynamic [`AttributeValue`] scalar.
    ///
    /// Integer leaves widen to [`AttributeValue::Int`] when they fit an `i64`
    /// and otherwise to [`AttributeValue::Float`]; floating-point leaves widen
    /// to [`AttributeValue::Float`]; booleans and strings map directly. This is
    /// the read half of the dynamic scalar exchange a scripting runtime uses.
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`]/[`PropertyError::NoSuchProperty`] for an
    /// invalid or unresolved path, or [`PropertyError::NotScalar`] if the
    /// property is not a supported scalar leaf.
    pub fn read_scalar(
        &self,
        root: &dyn Reflect,
        path: &str,
    ) -> Result<AttributeValue, PropertyError> {
        let value = self.get(root, path)?;
        scalar_of(value).ok_or_else(|| PropertyError::NotScalar {
            path: path.to_string(),
            found: value.type_name().to_string(),
        })
    }

    /// Write a dynamic [`AttributeValue`] scalar into the leaf property at
    /// `path`, coercing it to the target leaf's concrete type.
    ///
    /// An [`Int`](AttributeValue::Int) coerces into any integer leaf that can
    /// represent it losslessly (bounds-checked) or into a floating-point leaf;
    /// a [`Float`](AttributeValue::Float) coerces into a floating-point leaf; a
    /// [`Bool`](AttributeValue::Bool) into a `bool`; a
    /// [`Text`](AttributeValue::Text) into a [`String`]. Narrowing that would
    /// lose data (an out-of-range integer, or a float into an integer leaf) is
    /// rejected rather than silently truncated. Read-only and range metadata
    /// are honoured exactly as in [`set`](Self::set).
    ///
    /// # Errors
    /// Returns [`PropertyError::Path`]/[`PropertyError::NoSuchProperty`],
    /// [`PropertyError::ReadOnly`], or [`PropertyError::TypeMismatch`] if the
    /// scalar cannot be represented by the target leaf.
    pub fn set_scalar(
        &self,
        root: &mut dyn Reflect,
        path: &str,
        value: &AttributeValue,
    ) -> Result<(), PropertyError> {
        let parsed = ParsedPath::parse(path).map_err(PropertyError::Path)?;
        self.reject_readonly(root, &parsed, path)?;
        let slot = reflect_path_mut(root, &parsed)
            .ok_or_else(|| PropertyError::NoSuchProperty(path.to_string()))?;
        assign_scalar(slot, value).map_err(|expected| PropertyError::TypeMismatch {
            path: path.to_string(),
            expected,
            found: scalar_kind(value).to_string(),
        })?;
        self.clamp_terminal(root, &parsed);
        Ok(())
    }

    /// Enumerate the top-level editable properties of `root` (design §24.6).
    ///
    /// Delegates to [`properties`](super::properties); hidden fields are
    /// omitted and each descriptor carries the field's resolved metadata.
    #[must_use]
    pub fn properties(&self, root: &dyn Reflect) -> Vec<super::PropertyDescriptor> {
        super::properties(root, self.registry)
    }

    /// Resolve the parent-relative metadata for a path's terminal named field.
    fn terminal_metadata(
        &self,
        root: &dyn Reflect,
        parsed: &ParsedPath,
    ) -> Option<crate::schema::FieldMetadata> {
        let (prefix, Access::Field(name)) = parsed.split_terminal()? else {
            return None;
        };
        let parent = reflect_path(root, &prefix)?;
        let meta = self.type_metadata(parent)?;
        meta.field(name).cloned()
    }

    /// Reject a write to a read-only terminal field.
    fn reject_readonly(
        &self,
        root: &dyn Reflect,
        parsed: &ParsedPath,
        path: &str,
    ) -> Result<(), PropertyError> {
        if self
            .terminal_metadata(root, parsed)
            .is_some_and(|m| m.is_readonly())
        {
            return Err(PropertyError::ReadOnly(path.to_string()));
        }
        Ok(())
    }

    /// Clamp a path's terminal field into its metadata range, if any.
    fn clamp_terminal(&self, root: &mut dyn Reflect, parsed: &ParsedPath) {
        let Some(field_meta) = self.terminal_metadata(root, parsed) else {
            return;
        };
        if field_meta.range().is_none() {
            return;
        }
        let Some((prefix, _terminal)) = parsed.split_terminal() else {
            return;
        };
        let Some(parent) = reflect_path_mut(root, &prefix) else {
            return;
        };
        // A single-field metadata view clamps exactly the field just written,
        // leaving the parent's other fields untouched.
        let single = TypeMetadata::new().with_field(field_meta);
        let _ = clamp(parent, &single);
    }

    fn type_metadata(&self, value: &dyn Reflect) -> Option<&'a TypeMetadata> {
        self.registry
            .get_with_name(value.type_name())
            .and_then(|reg| reg.data::<TypeMetadata>())
    }
}

impl fmt::Debug for PropertyBridge<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PropertyBridge").finish_non_exhaustive()
    }
}

/// Render a leaf reflected value as a dynamic [`AttributeValue`] scalar, or
/// `None` if it is not a supported scalar leaf.
fn scalar_of(value: &dyn Reflect) -> Option<AttributeValue> {
    let any = value.as_any();
    if let Some(v) = any.downcast_ref::<bool>() {
        return Some(AttributeValue::Bool(*v));
    }
    if let Some(v) = any.downcast_ref::<String>() {
        return Some(AttributeValue::Text(v.clone()));
    }
    if let Some(v) = any.downcast_ref::<f32>() {
        return Some(AttributeValue::Float(f64::from(*v)));
    }
    if let Some(v) = any.downcast_ref::<f64>() {
        return Some(AttributeValue::Float(*v));
    }

    macro_rules! read_int {
        ($($ty:ty),* $(,)?) => {$(
            if let Some(v) = any.downcast_ref::<$ty>() {
                return Some(match i64::try_from(*v) {
                    Ok(i) => AttributeValue::Int(i),
                    // Values beyond i64 (large u64/u128/i128) widen to Float so
                    // the scalar stays representable rather than failing.
                    Err(_) => AttributeValue::Float(*v as f64),
                });
            }
        )*};
    }
    read_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
    None
}

/// Coerce a dynamic scalar into the concrete leaf `slot`, returning the
/// expected-type name (for a [`PropertyError::TypeMismatch`]) on failure.
fn assign_scalar(slot: &mut dyn Reflect, value: &AttributeValue) -> Result<(), String> {
    let any = slot.as_any_mut();
    match value {
        AttributeValue::Bool(b) => {
            if let Some(target) = any.downcast_mut::<bool>() {
                *target = *b;
                return Ok(());
            }
            Err("bool".to_string())
        }
        AttributeValue::Text(text) => {
            if let Some(target) = any.downcast_mut::<String>() {
                target.clone_from(text);
                return Ok(());
            }
            Err("String".to_string())
        }
        AttributeValue::Float(f) => {
            if let Some(target) = any.downcast_mut::<f64>() {
                *target = *f;
                return Ok(());
            }
            if let Some(target) = any.downcast_mut::<f32>() {
                *target = *f as f32;
                return Ok(());
            }
            Err("f32 or f64".to_string())
        }
        AttributeValue::Int(i) => assign_int(any, *i),
    }
}

/// Coerce a signed integer scalar into any integer or float leaf, bounds-checked.
fn assign_int(any: &mut dyn core::any::Any, i: i64) -> Result<(), String> {
    macro_rules! write_int {
        ($($ty:ty),* $(,)?) => {$(
            if let Some(target) = any.downcast_mut::<$ty>() {
                return match <$ty>::try_from(i) {
                    Ok(v) => {
                        *target = v;
                        Ok(())
                    }
                    Err(_) => Err(concat!("an in-range ", stringify!($ty)).to_string()),
                };
            }
        )*};
    }
    write_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

    if let Some(target) = any.downcast_mut::<f64>() {
        *target = i as f64;
        return Ok(());
    }
    if let Some(target) = any.downcast_mut::<f32>() {
        *target = i as f32;
        return Ok(());
    }
    Err("an integer or floating-point leaf".to_string())
}

/// The human-readable kind name of a scalar, for error messages.
fn scalar_kind(value: &AttributeValue) -> &'static str {
    match value {
        AttributeValue::Bool(_) => "bool scalar",
        AttributeValue::Int(_) => "integer scalar",
        AttributeValue::Float(_) => "float scalar",
        AttributeValue::Text(_) => "text scalar",
    }
}

/// An error produced by a [`PropertyBridge`] access.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PropertyError {
    /// The property path string failed to parse.
    Path(ParsePathError),
    /// The parsed path did not resolve to a property in the root.
    NoSuchProperty(String),
    /// A write targeted a field flagged read-only by its metadata.
    ReadOnly(String),
    /// A typed read or scalar write found an incompatible concrete type.
    TypeMismatch {
        /// The property path.
        path: String,
        /// The type (or scalar leaf) the operation expected.
        expected: String,
        /// The concrete type (or scalar kind) actually found.
        found: String,
    },
    /// A scalar read targeted a property that is not a supported scalar leaf.
    NotScalar {
        /// The property path.
        path: String,
        /// The concrete type found instead of a scalar leaf.
        found: String,
    },
    /// Writing the supplied value into the target slot failed.
    Apply(ApplyError),
}

impl fmt::Display for PropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyError::Path(err) => write!(f, "invalid property path: {}", err.message()),
            PropertyError::NoSuchProperty(path) => {
                write!(f, "property `{path}` did not resolve")
            }
            PropertyError::ReadOnly(path) => write!(f, "property `{path}` is read-only"),
            PropertyError::TypeMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "property `{path}` expected {expected} but found {found}"
            ),
            PropertyError::NotScalar { path, found } => {
                write!(f, "property `{path}` is not a scalar leaf (found {found})")
            }
            PropertyError::Apply(err) => write!(f, "property assignment failed: {err}"),
        }
    }
}

impl core::error::Error for PropertyError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            PropertyError::Path(err) => Some(err),
            PropertyError::Apply(err) => Some(err),
            _ => None,
        }
    }
}
