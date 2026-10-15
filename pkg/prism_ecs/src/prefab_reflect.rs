//! Field-level prefab inheritance (design §16.3), available with `reflect`.
//!
//! [`World::get_inherited_reflect`] resolves a fresh reflected value from the
//! current nearest-first ancestry and applies only explicitly tracked fields.
//! No concrete component or inherited defaults are cached on the instance.
//! Parent edits therefore become visible on the next resolved read. Ordinary
//! `get`, queries and `get_inherited::<T>` keep their component-level semantics.
//!
//! Precedence is breadth first, with direct parents in insertion order. The
//! nearest concrete component supplies the base; field overrides on that entity
//! and the preceding ancestors are layered over it, nearer fields winning.
//! A removal marker acts as an absent base, hiding more distant components.
//! Field overrides remain dormant while their base is absent, and resume if it
//! returns. An invalidated path or field type returns an error instead of
//! silently discarding an override. All reachable ancestry must be live and
//! acyclic, even when a local value would otherwise shadow it.
//!
//! Paths use [`ParsedPath`] syntax. An override owns the entire selected field
//! or subtree. Overlapping paths on one entity are rejected; callers can clear
//! the enclosing override before overriding its children. Setting a field to
//! its inherited value still explicitly overrides it. `Patch::Unchanged`, in
//! contrast, records nothing. Components are keyed by [`StableTypeId`], never
//! by world-local component IDs. [`PrefabOverrides::components`] exposes those
//! records for scene tooling. With `serialize`, `save_reflect_archive` persists
//! these policies and the ordered `IsA` ancestry separately from concrete
//! components, including overrides whose base is currently absent.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{self, Write};

use prism_reflect::{
    reflect_path, Access, DiffError, DynamicArray, DynamicEnum, DynamicList, DynamicMap,
    DynamicStruct, DynamicTupleStruct, DynamicVariant, ParsePathError, ParsedPath, Patch, Reflect,
    ReflectRef, StableTypeId, TypeInfo, VariantKind, VariantType,
};

use super::PrefabError;
use crate::component::{Component, StorageType};
use crate::entity::Entity;
use crate::world::World;

/// Failure to resolve or edit a reflected prefab component.
#[derive(Debug)]
pub enum PrefabReflectError {
    /// An entity, ancestor or inheritance edge is invalid.
    Ancestry(PrefabError),
    /// The world's reflection bridge does not know this stable type ID.
    UnregisteredType(StableTypeId),
    /// A field edit requires a currently resolved component.
    MissingComponent(StableTypeId),
    /// A textual field path could not be parsed.
    InvalidPath(ParsePathError),
    /// A field override cannot select the entire component.
    EmptyPath,
    /// A field or element is absent in the current inherited shape.
    MissingField(String),
    /// Two paths on the same entity select overlapping subtrees.
    OverlappingField(String),
    /// A field value has a different reflected type from its inherited field.
    FieldTypeMismatch {
        /// The type of the inherited field.
        expected: &'static str,
        /// The type of the override value.
        actual: &'static str,
    },
    /// A value names the expected type but has an incompatible reflected shape.
    InvalidFieldShape(&'static str),
    /// The reflection library rejected the requested field patch.
    Patch(DiffError),
}

impl fmt::Display for PrefabReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ancestry(error) => error.fmt(f),
            Self::UnregisteredType(id) => write!(f, "unregistered prefab component {id:?}"),
            Self::MissingComponent(id) => write!(f, "prefab component {id:?} is absent"),
            Self::InvalidPath(error) => error.fmt(f),
            Self::EmptyPath => f.write_str("a prefab override must select a field"),
            Self::MissingField(path) => write!(f, "prefab field {path:?} is absent"),
            Self::OverlappingField(path) => write!(f, "prefab field {path:?} overlaps an override"),
            Self::FieldTypeMismatch { expected, actual } => {
                write!(f, "prefab field expects {expected}, received {actual}")
            }
            Self::InvalidFieldShape(name) => write!(f, "invalid reflected prefab shape for {name}"),
            Self::Patch(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for PrefabReflectError {}

impl From<PrefabError> for PrefabReflectError {
    fn from(error: PrefabError) -> Self {
        Self::Ancestry(error)
    }
}

/// One explicitly overridden field, containing no inherited sibling values.
///
/// Equality compares normalized paths and reflected values, independently of
/// textual spelling and shared allocation identity. Floating point leaves use
/// exact bits (including NaN payloads and signed zero). Opaque leaf types with
/// no reflected comparison support conservatively compare unequal, even to a
/// clone; this may mark an unchanged snapshot dirty but never hides a change.
#[derive(Clone, Debug)]
pub struct PrefabFieldOverride {
    path: ParsedPath,
    spelling: String,
    // Immutable sharing makes metadata snapshot-cloneable without requiring
    // Patch itself to implement Clone or copying unrelated component values.
    patch: Arc<Patch>,
}

// Ownership compares normalized paths, not their UI spelling or Arc identity.
// Leaves use reflection's exact primitive comparison (including float bits).
// Opaque leaves with no comparison support conservatively compare unequal.
impl PartialEq for PrefabFieldOverride {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && match (self.patch(), other.patch()) {
                (Patch::Replace(a), Patch::Replace(b)) => prefab_values_equal(&**a, &**b),
                _ => false, // Stored overrides are always normalized Replace patches.
            }
    }
}

fn prefab_values_equal(a: &dyn Reflect, b: &dyn Reflect) -> bool {
    if a.type_name() != b.type_name() {
        return false;
    }
    match (a.reflect_ref(), b.reflect_ref()) {
        (ReflectRef::Struct(a), ReflectRef::Struct(b)) => {
            a.field_count() == b.field_count()
                && (0..a.field_count()).all(|i| {
                    a.name_at(i)
                        .and_then(|name| a.field(name).zip(b.field(name)))
                        .is_some_and(|(a, b)| prefab_values_equal(a, b))
                })
        }
        (ReflectRef::TupleStruct(a), ReflectRef::TupleStruct(b)) => {
            a.field_count() == b.field_count()
                && (0..a.field_count()).all(|i| {
                    a.field(i)
                        .zip(b.field(i))
                        .is_some_and(|(a, b)| prefab_values_equal(a, b))
                })
        }
        (ReflectRef::Enum(a), ReflectRef::Enum(b)) => {
            a.variant_name() == b.variant_name()
                && a.variant_index() == b.variant_index()
                && a.variant_type() == b.variant_type()
                && a.field_count() == b.field_count()
                && if a.variant_type() == VariantType::Struct {
                    // Dynamic enums preserve names but may use a different
                    // insertion order from a concrete declaration.
                    enum_field_names(a).is_some_and(|names| {
                        names.len() == a.field_count()
                            && names.into_iter().all(|name| {
                                a.field(name)
                                    .zip(b.field(name))
                                    .is_some_and(|(a, b)| prefab_values_equal(a, b))
                            })
                    })
                } else {
                    (0..a.field_count()).all(|i| {
                        a.field_at(i)
                            .zip(b.field_at(i))
                            .is_some_and(|(a, b)| prefab_values_equal(a, b))
                    })
                }
        }
        (ReflectRef::List(a), ReflectRef::List(b)) => {
            a.len() == b.len()
                && (0..a.len()).all(|i| {
                    a.get(i)
                        .zip(b.get(i))
                        .is_some_and(|(a, b)| prefab_values_equal(a, b))
                })
        }
        (ReflectRef::Array(a), ReflectRef::Array(b)) => {
            a.len() == b.len()
                && (0..a.len()).all(|i| {
                    a.get(i)
                        .zip(b.get(i))
                        .is_some_and(|(a, b)| prefab_values_equal(a, b))
                })
        }
        (ReflectRef::Map(a), ReflectRef::Map(b)) => {
            let a: Vec<_> = a.iter_reflect().collect();
            let b: Vec<_> = b.iter_reflect().collect();
            unordered_equal(&a, &b, |(ak, av), (bk, bv)| {
                prefab_values_equal(*ak, *bk) && prefab_values_equal(*av, *bv)
            })
        }
        (ReflectRef::Set(a), ReflectRef::Set(b)) => {
            let a: Vec<_> = a.iter_reflect().collect();
            let b: Vec<_> = b.iter_reflect().collect();
            unordered_equal(&a, &b, |a, b| prefab_values_equal(*a, *b))
        }
        (ReflectRef::Value(a), ReflectRef::Value(b)) => {
            (a.is::<()>() && b.is::<()>()) || prism_reflect::diff(a, b).is_unchanged()
        }
        _ => false,
    }
}

fn unordered_equal<T>(a: &[T], b: &[T], equal: impl Fn(&T, &T) -> bool) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut matched = alloc::vec![false; b.len()];
    a.iter().all(|value| {
        let Some(index) = b
            .iter()
            .enumerate()
            .position(|(index, other)| !matched[index] && equal(value, other))
        else {
            return false;
        };
        matched[index] = true;
        true
    })
}

fn enum_field_names(value: &dyn prism_reflect::Enum) -> Option<Vec<&'static str>> {
    if let Some(dynamic) = value.as_any().downcast_ref::<DynamicEnum>() {
        if let DynamicVariant::Struct(fields) = dynamic.variant() {
            return Some(fields.iter().map(|(name, _)| *name).collect());
        }
        return None;
    }
    let TypeInfo::Enum(info) = value.type_info() else {
        return None;
    };
    let VariantKind::Struct(fields) = info.variant(value.variant_name())?.kind() else {
        return None;
    };
    Some(fields.iter().map(|field| field.name()).collect())
}

impl PrefabFieldOverride {
    /// The parsed field path. Aliases such as `hp` and `.hp` compare equally.
    pub fn path(&self) -> &ParsedPath {
        &self.path
    }

    /// The original textual path, suitable for replay through the edit API.
    pub fn path_str(&self) -> &str {
        &self.spelling
    }

    /// The field-only replacement, suitable for replay through the edit API.
    pub fn patch(&self) -> &Patch {
        &self.patch
    }

    #[cfg(feature = "serialize")]
    pub(crate) fn archive_path(&self) -> String {
        path_spelling(self.path.segments())
    }

    #[cfg(feature = "serialize")]
    pub(crate) fn from_archive(
        canonical: &str,
        spelling: &str,
        value: Box<dyn Reflect>,
    ) -> Result<Self, PrefabReflectError> {
        let path = parse_field(canonical)?;
        parse_field(spelling)?;
        Ok(Self {
            path,
            spelling: spelling.into(),
            patch: Arc::new(Patch::Replace(value)),
        })
    }
}

/// An entity's explicit policy for one reflected component.
#[derive(Clone, Debug)]
pub enum PrefabComponentOverride {
    /// Overrides for individual fields, in edit order.
    Fields(Vec<PrefabFieldOverride>),
    /// Hide this component, including values supplied by more distant ancestors.
    Removed,
}

impl PartialEq for PrefabComponentOverride {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Removed, Self::Removed) => true,
            (Self::Fields(a), Self::Fields(b)) => unordered_equal(a, b, PartialEq::eq),
            _ => false,
        }
    }
}

/// Per-entity prefab policies, stored sparsely and cleaned up on despawn.
///
/// This contains only explicit field patches and removal markers. It can be
/// registered with `register_snapshot_component::<PrefabOverrides>()` when
/// snapshotting entities with overrides. IDs are the reflection bridge's stable
/// schema IDs; local component registration order does not affect them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefabOverrides {
    components: BTreeMap<StableTypeId, PrefabComponentOverride>,
}

impl Component for PrefabOverrides {
    const STORAGE: StorageType = StorageType::SparseSet;
}

impl PrefabOverrides {
    /// The explicit override policy for `type_id`, if any.
    pub fn component(&self, type_id: StableTypeId) -> Option<&PrefabComponentOverride> {
        self.components.get(&type_id)
    }

    /// Override records sorted by stable type ID, for inspectors/scene tooling.
    pub fn components(
        &self,
    ) -> impl Iterator<Item = (StableTypeId, &PrefabComponentOverride)> + '_ {
        self.components.iter().map(|(id, value)| (*id, value))
    }

    #[cfg(feature = "serialize")]
    pub(crate) fn insert_archived(
        &mut self,
        id: StableTypeId,
        policy: PrefabComponentOverride,
    ) -> Result<(), PrefabReflectError> {
        if let PrefabComponentOverride::Fields(fields) = &policy {
            for (i, field) in fields.iter().enumerate() {
                for previous in &fields[..i] {
                    let a = previous.path.segments();
                    let b = field.path.segments();
                    if a.starts_with(b) || b.starts_with(a) {
                        return Err(PrefabReflectError::OverlappingField(field.spelling.clone()));
                    }
                }
            }
        }
        self.components.insert(id, policy);
        Ok(())
    }
}

// Both live resolution and archive validation use this ordering/masking rule.
// The caller validates the resulting shape against its concrete or static
// schema; Dynamic* values decoded from archives do not carry concrete TypeInfo.
pub(crate) fn resolve_prefab_layers<'a>(
    layers: impl IntoIterator<Item = (Option<&'a PrefabComponentOverride>, Option<&'a dyn Reflect>)>,
) -> Result<Option<(&'a dyn Reflect, Box<dyn Reflect>)>, PrefabReflectError> {
    let mut fields: Vec<&PrefabFieldOverride> = Vec::new();
    for (policy, base) in layers {
        match policy {
            Some(PrefabComponentOverride::Removed) => return Ok(None),
            Some(PrefabComponentOverride::Fields(overrides)) => {
                for field in overrides {
                    if !fields
                        .iter()
                        .any(|nearer| field.path.segments().starts_with(nearer.path.segments()))
                    {
                        fields.push(field);
                    }
                }
            }
            None => {}
        }
        if let Some(base) = base {
            let mut value = base.reflect_clone();
            for field in fields.into_iter().rev() {
                value = replace_field(&*value, field.path.segments(), field)?;
            }
            return Ok(Some((base, value)));
        }
    }
    Ok(None)
}

fn parse_field(path: &str) -> Result<ParsedPath, PrefabReflectError> {
    let parsed = ParsedPath::parse(path).map_err(PrefabReflectError::InvalidPath)?;
    if parsed.is_empty() {
        return Err(PrefabReflectError::EmptyPath);
    }
    Ok(parsed)
}

impl World {
    fn check_prefab_reflect_type(&self, type_id: StableTypeId) -> Result<(), PrefabReflectError> {
        if self.reflect_component_id(type_id).is_none() {
            return Err(PrefabReflectError::UnregisteredType(type_id));
        }
        Ok(())
    }

    /// Resolve a component with current inherited defaults and field overrides.
    ///
    /// `Ok(None)` means no concrete base survives inheritance/removal. The
    /// returned owned dynamic value can be reconstructed with `FromReflect`.
    /// Resolution is read-only and does not mark concrete components changed.
    pub fn get_inherited_reflect(
        &self,
        entity: Entity,
        type_id: StableTypeId,
    ) -> Result<Option<Box<dyn Reflect>>, PrefabReflectError> {
        self.check_prefab_reflect_type(type_id)?;
        self.validate_prefab_ancestry(entity)?;
        let layers = core::iter::once(entity)
            .chain(self.prefab_chain(entity))
            .map(|current| {
                (
                    self.get::<PrefabOverrides>(current)
                        .and_then(|o| o.component(type_id)),
                    self.get_reflect(current, type_id),
                )
            });
        let Some((base, value)) = resolve_prefab_layers(layers)? else {
            return Ok(None);
        };
        check_field_shape(base, &*value)?;
        Ok(Some(value))
    }

    /// Explicitly replace one inherited field/subtree with `value`.
    ///
    /// The same value as the parent still pins this field. Equal paths replace
    /// the existing record; enclosing/descendant conflicts are rejected before
    /// mutation. This never inserts the concrete component on the instance.
    pub fn override_prefab_field(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
        path: &str,
        value: &dyn Reflect,
    ) -> Result<(), PrefabReflectError> {
        self.apply_prefab_field_patch(
            entity,
            type_id,
            path,
            &Patch::Replace(value.reflect_clone()),
        )
    }

    /// Apply a reflection patch to the selected field and pin its resulting value.
    ///
    /// A structural patch edits only that selected field, which is the unit of
    /// ownership. To inherit sibling subfields, select a deeper path. Patches
    /// are applied to a scratch value and errors leave all live state untouched.
    /// `Replace` replaces exactly, including shorter containers and enum variant
    /// changes; other patch kinds follow `prism_reflect::Patch::apply` semantics.
    pub fn apply_prefab_field_patch(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
        path: &str,
        patch: &Patch,
    ) -> Result<(), PrefabReflectError> {
        let parsed = parse_field(path)?;
        let resolved = self
            .get_inherited_reflect(entity, type_id)?
            .ok_or(PrefabReflectError::MissingComponent(type_id))?;
        let parsed = canonical_field(&*resolved, &parsed)?;
        let selected = reflect_path(&*resolved, &parsed)
            .ok_or_else(|| PrefabReflectError::MissingField(path.to_string()))?;
        if patch.is_unchanged() {
            return Ok(());
        }
        let value = match patch {
            Patch::Replace(value) => {
                check_field_type(selected, &**value)?;
                value.reflect_clone()
            }
            _ => {
                let mut value = selected.reflect_clone();
                patch
                    .apply(&mut *value)
                    .map_err(PrefabReflectError::Patch)?;
                value
            }
        };
        if let Some(PrefabComponentOverride::Fields(fields)) = self
            .get::<PrefabOverrides>(entity)
            .and_then(|overrides| overrides.component(type_id))
        {
            for field in fields {
                let a = field.path.segments();
                let b = parsed.segments();
                if a != b && (a.starts_with(b) || b.starts_with(a)) {
                    return Err(PrefabReflectError::OverlappingField(path.to_string()));
                }
            }
        }
        let field = PrefabFieldOverride {
            path: parsed,
            spelling: path.to_string(),
            patch: Arc::new(Patch::Replace(value)),
        };
        // Validate the final field replacement before making metadata live.
        let candidate = replace_field(&*resolved, field.path.segments(), &field)?;
        // DynamicList accepts arbitrary appended types. Validate against the
        // concrete provider, whose schema still describes its element type.
        for provider in core::iter::once(entity).chain(self.prefab_chain(entity)) {
            if let Some(base) = self.get_reflect(provider, type_id) {
                check_field_shape(base, &*candidate)?;
                break;
            }
        }
        if !self.has::<PrefabOverrides>(entity) {
            self.insert(entity, PrefabOverrides::default());
        }
        let mut overrides = self
            .get_mut::<PrefabOverrides>(entity)
            .expect("inserted metadata");
        let policy = overrides
            .components
            .entry(type_id)
            .or_insert_with(|| PrefabComponentOverride::Fields(Vec::new()));
        let PrefabComponentOverride::Fields(fields) = policy else {
            // A removal marker cannot have resolved a base above.
            unreachable!("removed component has no resolved field")
        };
        if let Some(existing) = fields
            .iter_mut()
            .find(|existing| existing.path == field.path)
        {
            *existing = field;
        } else {
            fields.push(field);
        }
        Ok(())
    }

    /// Clear one exact field override, exposing its current inherited value.
    ///
    /// Returns whether a record existed. Clearing does not require successful
    /// resolution, so it can repair paths after an ancestor changes shape or
    /// disappears. Named enum fields accept positional aliases while their
    /// variant is available, or the original spelling after invalidation.
    pub fn clear_prefab_field_override(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
        path: &str,
    ) -> Result<bool, PrefabReflectError> {
        let parsed = parse_field(path)?;
        self.check_prefab_reflect_type(type_id)?;
        if !self.contains(entity) {
            return Err(PrefabError::MissingEntity(entity).into());
        }
        let canonical = self
            .get_inherited_reflect(entity, type_id)
            .ok()
            .flatten()
            .and_then(|value| canonical_field(&*value, &parsed).ok());
        let index = self
            .get::<PrefabOverrides>(entity)
            .and_then(|overrides| overrides.component(type_id))
            .and_then(|policy| match policy {
                PrefabComponentOverride::Fields(fields) => fields.iter().position(|f| {
                    f.path == parsed
                        || canonical.as_ref() == Some(&f.path)
                        || ParsedPath::parse(&f.spelling).ok().as_ref() == Some(&parsed)
                }),
                PrefabComponentOverride::Removed => None,
            });
        let Some(index) = index else {
            return Ok(false);
        };
        {
            let mut overrides = self
                .get_mut::<PrefabOverrides>(entity)
                .expect("metadata exists");
            if let Some(PrefabComponentOverride::Fields(fields)) =
                overrides.components.get_mut(&type_id)
            {
                fields.remove(index);
                if fields.is_empty() {
                    overrides.components.remove(&type_id);
                }
            }
        }
        self.prune_prefab_overrides(entity);
        Ok(true)
    }

    /// Remove a component locally and block its inherited values for descendants.
    ///
    /// Existing field records for this type are discarded. The removal marker
    /// also blocks future additions on parents until `reset_prefab_component`.
    pub fn remove_prefab_component(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
    ) -> Result<(), PrefabReflectError> {
        self.check_prefab_reflect_type(type_id)?;
        self.validate_prefab_ancestry(entity)?;
        self.remove_reflect(entity, type_id);
        if !self.has::<PrefabOverrides>(entity) {
            self.insert(entity, PrefabOverrides::default());
        }
        self.get_mut::<PrefabOverrides>(entity)
            .expect("inserted metadata")
            .components
            .insert(type_id, PrefabComponentOverride::Removed);
        Ok(())
    }

    /// Remove local concrete state, field overrides and removal markers for a
    /// component, returning to inheritance (or absence if no ancestor supplies it).
    ///
    /// Returns whether any local state was removed. Invalid ancestor paths do
    /// not prevent resetting the entity's own state.
    pub fn reset_prefab_component(
        &mut self,
        entity: Entity,
        type_id: StableTypeId,
    ) -> Result<bool, PrefabReflectError> {
        self.check_prefab_reflect_type(type_id)?;
        if !self.contains(entity) {
            return Err(PrefabError::MissingEntity(entity).into());
        }
        let mut changed = self.remove_reflect(entity, type_id);
        if self
            .get::<PrefabOverrides>(entity)
            .is_some_and(|o| o.components.contains_key(&type_id))
        {
            self.get_mut::<PrefabOverrides>(entity)
                .expect("metadata exists")
                .components
                .remove(&type_id);
            changed = true;
        }
        self.prune_prefab_overrides(entity);
        Ok(changed)
    }

    fn prune_prefab_overrides(&mut self, entity: Entity) {
        if self
            .get::<PrefabOverrides>(entity)
            .is_some_and(|o| o.components.is_empty())
        {
            self.remove::<PrefabOverrides>(entity);
        }
    }
}

fn check_field_type(
    expected: &dyn Reflect,
    actual: &dyn Reflect,
) -> Result<(), PrefabReflectError> {
    check_type_name(expected.type_name(), actual)
}

fn check_type_name(expected: &'static str, actual: &dyn Reflect) -> Result<(), PrefabReflectError> {
    if expected != actual.type_name() {
        return Err(PrefabReflectError::FieldTypeMismatch {
            expected,
            actual: actual.type_name(),
        });
    }
    Ok(())
}

// Normalize aliases before comparing ownership paths, so `mode.x` and `mode#0`
// cannot own the same named enum field twice.
fn canonical_field(
    root: &dyn Reflect,
    parsed: &ParsedPath,
) -> Result<ParsedPath, PrefabReflectError> {
    let mut current = root;
    let mut steps = Vec::new();
    let missing = || PrefabReflectError::MissingField(alloc::format!("{parsed:?}"));
    for step in parsed.segments() {
        let mut normalized = step.clone();
        current = match (current.reflect_ref(), step) {
            (ReflectRef::Struct(value), Access::Field(name)) => value.field(name),
            (ReflectRef::Enum(value), Access::Field(name)) => value.field(name),
            (ReflectRef::TupleStruct(value), Access::TupleIndex(index)) => value.field(*index),
            (ReflectRef::Enum(value), Access::TupleIndex(index)) => {
                if let Some(dynamic) = current.downcast_ref::<DynamicEnum>() {
                    if let DynamicVariant::Struct(fields) = dynamic.variant() {
                        let (name, _) = fields.get(*index).ok_or_else(missing)?;
                        normalized = Access::Field((*name).into());
                    }
                }
                value.field_at(*index)
            }
            (ReflectRef::List(value), Access::ListIndex(index)) => value.get(*index),
            (ReflectRef::Array(value), Access::ListIndex(index)) => value.get(*index),
            (ReflectRef::Map(value), Access::Key(key)) => value.get(key),
            _ => None,
        }
        .ok_or_else(missing)?;
        steps.push(normalized);
    }
    parse_field(&path_spelling(&steps))
}

fn path_spelling(steps: &[Access]) -> String {
    let mut text = String::new();
    for step in steps {
        match step {
            Access::Field(name) => {
                let _ = write!(text, ".{name}");
            }
            Access::TupleIndex(index) => {
                let _ = write!(text, "#{index}");
            }
            Access::ListIndex(index) => {
                let _ = write!(text, "[{index}]");
            }
            Access::Key(key) => {
                let quote = if key.contains('\'') { '"' } else { '\'' };
                let _ = write!(text, "[{quote}{key}{quote}]");
            }
        }
    }
    text
}

// Validate dynamic patch output against the concrete provider's reflected
// schema. In particular, dynamic containers themselves do not enforce T.
fn check_field_shape(
    expected: &dyn Reflect,
    actual: &dyn Reflect,
) -> Result<(), PrefabReflectError> {
    check_field_type(expected, actual)?;
    let invalid = || PrefabReflectError::InvalidFieldShape(expected.type_name());
    match (expected.reflect_ref(), actual.reflect_ref()) {
        (ReflectRef::Struct(a), ReflectRef::Struct(b)) => {
            if a.field_count() != b.field_count() {
                return Err(invalid());
            }
            for i in 0..a.field_count() {
                let name = a.name_at(i).ok_or_else(invalid)?;
                check_field_shape(
                    a.field_at(i).ok_or_else(invalid)?,
                    b.field(name).ok_or_else(invalid)?,
                )?;
            }
        }
        (ReflectRef::TupleStruct(a), ReflectRef::TupleStruct(b)) => {
            if a.field_count() != b.field_count() {
                return Err(invalid());
            }
            for i in 0..a.field_count() {
                check_field_shape(
                    a.field(i).ok_or_else(invalid)?,
                    b.field(i).ok_or_else(invalid)?,
                )?;
            }
        }
        (ReflectRef::List(a), ReflectRef::List(b)) => {
            let TypeInfo::List(info) = expected.type_info() else {
                return Err(invalid());
            };
            for item in b.iter_reflect() {
                check_type_name(info.item_type_name(), item)?;
                if let Some(sample) = a.get(0) {
                    check_field_shape(sample, item)?;
                }
            }
        }
        (ReflectRef::Array(a), ReflectRef::Array(b)) => {
            if a.len() != b.len() {
                return Err(invalid());
            }
            for i in 0..a.len() {
                check_field_shape(a.get(i).ok_or_else(invalid)?, b.get(i).ok_or_else(invalid)?)?;
            }
        }
        (ReflectRef::Map(a), ReflectRef::Map(b)) => {
            let TypeInfo::Map(info) = expected.type_info() else {
                return Err(invalid());
            };
            for (key, item) in b.iter_reflect() {
                check_type_name(info.key_type_name(), key)?;
                check_type_name(info.value_type_name(), item)?;
                if let Some((sample_key, sample)) = a.iter_reflect().next() {
                    check_field_shape(sample_key, key)?;
                    check_field_shape(a.get(key).unwrap_or(sample), item)?;
                }
            }
        }
        (ReflectRef::Set(a), ReflectRef::Set(b)) => {
            let TypeInfo::Set(info) = expected.type_info() else {
                return Err(invalid());
            };
            for item in b.iter_reflect() {
                check_type_name(info.value_type_name(), item)?;
                if let Some(sample) = a.iter_reflect().next() {
                    check_field_shape(sample, item)?;
                }
            }
        }
        (ReflectRef::Enum(a), ReflectRef::Enum(b)) => {
            let TypeInfo::Enum(info) = expected.type_info() else {
                return Err(invalid());
            };
            let variant = info.variant(b.variant_name()).ok_or_else(invalid)?;
            if variant.index() != b.variant_index() || variant.field_count() != b.field_count() {
                return Err(invalid());
            }
            match variant.kind() {
                VariantKind::Unit if b.variant_type() == VariantType::Unit => {}
                VariantKind::Tuple(fields) if b.variant_type() == VariantType::Tuple => {
                    for field in fields {
                        check_type_name(
                            field.type_name(),
                            b.field_at(field.index()).ok_or_else(invalid)?,
                        )?;
                    }
                }
                VariantKind::Struct(fields) if b.variant_type() == VariantType::Struct => {
                    for field in fields {
                        check_type_name(
                            field.type_name(),
                            b.field(field.name()).ok_or_else(invalid)?,
                        )?;
                    }
                }
                _ => return Err(invalid()),
            }
            if a.variant_index() == b.variant_index() {
                for i in 0..a.field_count() {
                    check_field_shape(
                        a.field_at(i).ok_or_else(invalid)?,
                        b.field_at(i).ok_or_else(invalid)?,
                    )?;
                }
            }
        }
        (ReflectRef::Value(_), ReflectRef::Value(_)) => {}
        _ => return Err(invalid()),
    }
    Ok(())
}

// Reflect::apply merges collections rather than replacing them. Rebuild only
// the path's containers so a field replacement can shrink a list/map or switch
// enum variants exactly. These are ephemeral resolved values, never stored
// component snapshots masquerading as field overrides.
fn replace_field(
    root: &dyn Reflect,
    path: &[Access],
    field: &PrefabFieldOverride,
) -> Result<Box<dyn Reflect>, PrefabReflectError> {
    let missing = || PrefabReflectError::MissingField(field.spelling.clone());
    let Some((head, rest)) = path.split_first() else {
        let Patch::Replace(value) = &*field.patch else {
            unreachable!("normalized field patch")
        };
        check_field_type(root, &**value)?;
        return Ok(value.reflect_clone());
    };
    match (root.reflect_ref(), head) {
        (ReflectRef::Struct(value), Access::Field(name)) => {
            value.field(name).ok_or_else(missing)?;
            let mut out = DynamicStruct::new();
            out.set_represented_type_name(root.type_name());
            for index in 0..value.field_count() {
                let key = value.name_at(index).ok_or_else(missing)?;
                let child = value.field_at(index).ok_or_else(missing)?;
                out.insert_boxed(
                    key,
                    if key == name {
                        replace_field(child, rest, field)?
                    } else {
                        child.reflect_clone()
                    },
                );
            }
            Ok(Box::new(out))
        }
        (ReflectRef::TupleStruct(value), Access::TupleIndex(index)) => {
            value.field(*index).ok_or_else(missing)?;
            let mut out = DynamicTupleStruct::new();
            out.set_represented_type_name(root.type_name());
            for i in 0..value.field_count() {
                let child = value.field(i).ok_or_else(missing)?;
                out.insert_boxed(if i == *index {
                    replace_field(child, rest, field)?
                } else {
                    child.reflect_clone()
                });
            }
            Ok(Box::new(out))
        }
        (ReflectRef::List(value), Access::ListIndex(index)) => {
            value.get(*index).ok_or_else(missing)?;
            let mut out = DynamicList::new();
            out.set_represented_type_name(root.type_name());
            for i in 0..value.len() {
                let child = value.get(i).ok_or_else(missing)?;
                out.push_boxed(if i == *index {
                    replace_field(child, rest, field)?
                } else {
                    child.reflect_clone()
                });
            }
            Ok(Box::new(out))
        }
        (ReflectRef::Array(value), Access::ListIndex(index)) => {
            value.get(*index).ok_or_else(missing)?;
            let mut out = DynamicArray::new();
            out.set_represented_type_name(root.type_name());
            for i in 0..value.len() {
                let child = value.get(i).ok_or_else(missing)?;
                out.push_boxed(if i == *index {
                    replace_field(child, rest, field)?
                } else {
                    child.reflect_clone()
                });
            }
            Ok(Box::new(out))
        }
        (ReflectRef::Map(value), Access::Key(key)) => {
            value.get(key).ok_or_else(missing)?;
            let mut out = DynamicMap::new();
            out.set_represented_type_name(root.type_name());
            for (k, child) in value.iter_reflect() {
                let updated = if k.downcast_ref::<String>() == Some(key) {
                    replace_field(child, rest, field)?
                } else {
                    child.reflect_clone()
                };
                out.insert_boxed(k.reflect_clone(), updated);
            }
            Ok(Box::new(out))
        }
        (ReflectRef::Enum(_), Access::Field(_) | Access::TupleIndex(_)) => {
            // Derived enum clones retain their named variant payload, including
            // names omitted from DynamicEnum's static TypeInfo.
            let cloned = root.reflect_clone();
            let Some(dynamic) = cloned.downcast_ref::<DynamicEnum>() else {
                return Err(missing());
            };
            let mut variant = dynamic.clone_variant();
            let selected = match (&mut variant, head) {
                (DynamicVariant::Struct(fields), Access::Field(name)) => fields
                    .iter_mut()
                    .find(|(key, _)| *key == name)
                    .map(|(_, v)| v),
                (DynamicVariant::Struct(fields), Access::TupleIndex(index)) => {
                    fields.get_mut(*index).map(|(_, v)| v)
                }
                (DynamicVariant::Tuple(fields), Access::TupleIndex(index)) => {
                    fields.get_mut(*index)
                }
                _ => None,
            }
            .ok_or_else(missing)?;
            *selected = replace_field(&**selected, rest, field)?;
            let ReflectRef::Enum(value) = root.reflect_ref() else {
                unreachable!()
            };
            let mut out = DynamicEnum::new(value.variant_index(), value.variant_name(), variant);
            out.set_represented_type_name(root.type_name());
            Ok(Box::new(out))
        }
        _ => Err(missing()),
    }
}

#[cfg(test)]
mod equality_tests {
    use super::*;

    fn field(value: &dyn Reflect) -> PrefabFieldOverride {
        PrefabFieldOverride {
            path: ParsedPath::parse("value").unwrap(),
            spelling: "value".into(),
            patch: Arc::new(Patch::Replace(value.reflect_clone())),
        }
    }

    #[test]
    fn primitive_equality_is_bit_exact_and_unknown_opaque_is_conservative() {
        #[derive(Clone)]
        struct Opaque;
        prism_reflect::impl_reflect_value!(Opaque);
        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(field(&nan), field(&nan));
        assert_ne!(field(&nan), field(&f32::from_bits(0x7fc0_1235)));
        assert_ne!(field(&0.0f32), field(&-0.0f32));
        let nan64 = f64::from_bits(0x7ff8_0000_0000_1234);
        assert_eq!(field(&nan64), field(&nan64));
        assert_ne!(field(&0.0f64), field(&-0.0f64));
        assert_eq!(field(&()), field(&()));
        let opaque = field(&Opaque);
        assert_ne!(
            opaque,
            opaque.clone(),
            "shared Arc identity cannot hide an unsupported leaf"
        );
    }

    #[test]
    fn nested_structs_enums_maps_and_sets_compare_values_not_insertion_order() {
        fn value(reverse: bool, number: i32) -> Box<dyn Reflect> {
            let mut map = DynamicMap::new();
            let mut set = prism_reflect::DynamicSet::new();
            let entries = if reverse {
                [("b", 2), ("a", number)]
            } else {
                [("a", number), ("b", 2)]
            };
            for (key, number) in entries {
                map.insert_value(key.to_string(), number);
                set.push_value(key.to_string());
            }
            let payload = if reverse {
                alloc::vec![
                    ("set", Box::new(set) as Box<dyn Reflect>),
                    ("map", Box::new(map) as Box<dyn Reflect>)
                ]
            } else {
                alloc::vec![
                    ("map", Box::new(map) as Box<dyn Reflect>),
                    ("set", Box::new(set) as Box<dyn Reflect>)
                ]
            };
            let mut out = DynamicStruct::new();
            out.insert(
                "enum",
                DynamicEnum::new(0, "Named", DynamicVariant::Struct(payload)),
            );
            Box::new(out)
        }
        let a = field(&*value(false, 1));
        let b = field(&*value(true, 1));
        assert_eq!(a, b);
        assert_eq!(b, a);
        assert_ne!(a, field(&*value(true, 9)));
        assert_ne!(field(&alloc::vec![1i32, 2]), field(&alloc::vec![2i32, 1]));
    }

    #[test]
    fn unordered_matching_never_reuses_an_equal_element() {
        assert!(!unordered_equal(&[1, 1], &[1, 2], PartialEq::eq));
        assert!(unordered_equal(&[1, 2], &[2, 1], PartialEq::eq));
    }
}
