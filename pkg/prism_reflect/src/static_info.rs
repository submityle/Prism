//! Compile-time static type descriptors — design §24.1.
//!
//! The milestone-era [`TypeInfo`](crate::TypeInfo) is cached behind a
//! `OnceLock` and built from a heap-allocated `Vec` on first access. That is
//! cheap but not *free*: the first touch runs code and allocates. §24.1 asks
//! for a strictly stronger guarantee — a **`const`-constructible**, zero-
//! allocation, zero-lazy-init descriptor that is baked into the binary's
//! read-only data and carries **no runtime registration cost at all**.
//!
//! [`StaticTypeInfo`] is that descriptor. It is produced by
//! `#[derive(StaticTyped)]` (see `prism_reflect_macros`) entirely from
//! `const` inputs:
//!
//! - field names are string literals,
//! - field byte offsets come from [`core::mem::offset_of!`] (a `const`),
//! - type names / [`StableTypeId`]s are stored as `fn` pointers
//!   ([`core::any::type_name`] is not yet a stable `const fn`), so the whole
//!   descriptor still evaluates in a `const` context with no heap use.
//!
//! Because [`StaticTyped::STATIC_TYPE_INFO`] is an associated `const`, the
//! descriptor lives in the binary's static data and
//! [`StaticTyped::static_type_info`] hands out a `&'static` to it with no
//! `OnceLock`, no `Mutex`, and no allocation. A registry built on top of this
//! (design §24.1's "`StableTypeId` → `&'static TypeInfo`" map) therefore only
//! *records a pointer*, it never *constructs* data.

use crate::StableTypeId;

/// The reflected shape captured by a [`StaticTypeInfo`].
///
/// Only named-field structs are modelled here; that is the shape §24.1/§24.2
/// target (the dotted field paths like `"transform.translation.x"` in the
/// design all walk named structs). Other kinds keep using the lazy
/// [`TypeInfo`](crate::TypeInfo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StaticKind {
    /// A struct with named fields.
    Struct,
}

/// A single named field inside a [`StaticTypeInfo`].
///
/// Every member is `const`-evaluable, so a `&'static [StaticField]` can be
/// baked directly into the binary with no runtime construction.
#[derive(Clone, Copy)]
pub struct StaticField {
    name: &'static str,
    offset: usize,
    type_name_fn: fn() -> &'static str,
    type_id_fn: fn() -> StableTypeId,
}

impl StaticField {
    /// Build a field descriptor from its name, byte offset, and type-name /
    /// [`StableTypeId`] accessors.
    ///
    /// The `type_name_fn`/`type_id_fn` are almost always
    /// `core::any::type_name::<F>` and `StableTypeId::of_type::<F>` supplied by
    /// the derive macro; they are stored as `fn` pointers so this constructor
    /// stays `const` (neither function is a stable `const fn`).
    #[must_use]
    pub const fn new(
        name: &'static str,
        offset: usize,
        type_name_fn: fn() -> &'static str,
        type_id_fn: fn() -> StableTypeId,
    ) -> Self {
        Self {
            name,
            offset,
            type_name_fn,
            type_id_fn,
        }
    }

    /// The field name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The field's byte offset within its owning struct, from
    /// [`core::mem::offset_of!`].
    ///
    /// This is descriptive metadata (layout inspection, debugging, matching the
    /// design's "`offset_of` 常量"); the safe accessors in
    /// [`accessor`](crate::accessor) navigate by generated reference, never by
    /// raw pointer arithmetic (the workspace is `#![forbid(unsafe_code)]`).
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// The field's fully-qualified type name ([`core::any::type_name`]).
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        (self.type_name_fn)()
    }

    /// The field's deterministic [`StableTypeId`].
    #[must_use]
    pub fn stable_type_id(&self) -> StableTypeId {
        (self.type_id_fn)()
    }
}

impl core::fmt::Debug for StaticField {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StaticField")
            .field("name", &self.name)
            .field("offset", &self.offset)
            .field("type_name", &self.type_name())
            .field("stable_type_id", &self.stable_type_id())
            .finish()
    }
}

impl PartialEq for StaticField {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.offset == other.offset
            && self.type_name() == other.type_name()
    }
}

impl Eq for StaticField {}

/// A `const`-constructible, zero-registration-cost static type descriptor
/// (design §24.1).
///
/// Unlike [`TypeInfo`](crate::TypeInfo) this performs **no** lazy
/// initialisation and **no** allocation: the whole value is a compile-time
/// constant living in the binary's read-only data.
#[derive(Clone, Copy)]
pub struct StaticTypeInfo {
    kind: StaticKind,
    type_name_fn: fn() -> &'static str,
    type_id_fn: fn() -> StableTypeId,
    fields: &'static [StaticField],
}

impl StaticTypeInfo {
    /// Build a named-struct descriptor.
    ///
    /// `type_name_fn`/`type_id_fn` are stored as `fn` pointers so the whole
    /// constructor is `const`; the derive supplies
    /// `core::any::type_name::<T>` and `StableTypeId::of_type::<T>`.
    #[must_use]
    pub const fn new_struct(
        type_name_fn: fn() -> &'static str,
        type_id_fn: fn() -> StableTypeId,
        fields: &'static [StaticField],
    ) -> Self {
        Self {
            kind: StaticKind::Struct,
            type_name_fn,
            type_id_fn,
            fields,
        }
    }

    /// The reflected shape (currently always [`StaticKind::Struct`]).
    #[must_use]
    pub const fn kind(&self) -> StaticKind {
        self.kind
    }

    /// The fully-qualified type name ([`core::any::type_name`]).
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        (self.type_name_fn)()
    }

    /// The type's deterministic [`StableTypeId`].
    #[must_use]
    pub fn stable_type_id(&self) -> StableTypeId {
        (self.type_id_fn)()
    }

    /// The ordered field descriptors (declaration order).
    #[must_use]
    pub const fn fields(&self) -> &'static [StaticField] {
        self.fields
    }

    /// Number of fields.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Look up a field descriptor by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&'static StaticField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// The index of a field by name, or `None` if absent.
    ///
    /// Resolving a name to an index once and then indexing [`fields`] is the
    /// cheap, allocation-free way to turn a repeated name lookup into an O(1)
    /// slice index (used by the §24.2 access plan).
    ///
    /// [`fields`]: StaticTypeInfo::fields
    #[must_use]
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }
}

impl core::fmt::Debug for StaticTypeInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StaticTypeInfo")
            .field("kind", &self.kind)
            .field("type_name", &self.type_name())
            .field("stable_type_id", &self.stable_type_id())
            .field("fields", &self.fields)
            .finish()
    }
}

impl PartialEq for StaticTypeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.type_name() == other.type_name()
            && self.fields == other.fields
    }
}

impl Eq for StaticTypeInfo {}

/// A type that exposes a `const`-constructed [`StaticTypeInfo`] (design §24.1).
///
/// Derived with `#[derive(StaticTyped)]`. The associated `const` guarantees the
/// descriptor is materialised at compile time; [`static_type_info`] hands out a
/// `&'static` to it with no `OnceLock`, no lock, and no allocation.
///
/// [`static_type_info`]: StaticTyped::static_type_info
pub trait StaticTyped {
    /// The compile-time type descriptor for `Self`.
    const STATIC_TYPE_INFO: StaticTypeInfo;

    /// Borrow the `const` descriptor as a `&'static`.
    ///
    /// `&Self::STATIC_TYPE_INFO` is a reference to an associated `const` with no
    /// interior mutability, so it is const-promoted to `'static` — the pointer
    /// is handed out directly with zero runtime cost.
    #[must_use]
    fn static_type_info() -> &'static StaticTypeInfo {
        &Self::STATIC_TYPE_INFO
    }
}
