//! Runtime-defined composite types (design §17, §22 — milestone **M5**).
//!
//! Scripts and data-driven tools sometimes need reflected types that do not
//! exist as concrete Rust types — for example an ECS component whose layout is
//! read from a data file at startup (the flecs "runtime meta type" shape).
//! [`StructTypeBuilder`] and [`EnumTypeBuilder`] assemble a [`TypeInfo`] at
//! runtime and register it by name into a [`TypeRegistry`](crate::TypeRegistry)
//! via [`register_runtime`](crate::TypeRegistry::register_runtime), after which
//! it drives serialization/deserialization exactly like a derived type.
//!
//! A runtime [`TypeInfo`] must outlive every value described by it, so the
//! builders promote their owned data to `&'static` with [`Box::leak`]. This is
//! safe (no `unsafe`) and intended: runtime type definitions are created once
//! at load time and live for the rest of the program.
//!
//! ```
//! use prism_reflect::{DynamicStruct, StructTypeBuilder, TypeInfo, TypeRegistry};
//!
//! let mut registry = TypeRegistry::new();
//! let info = StructTypeBuilder::new("game::Health")
//!     .with_field("current", "i32")
//!     .with_field("max", "i32")
//!     .register(&mut registry);
//!
//! assert!(matches!(info, TypeInfo::Struct(_)));
//! assert!(registry.get_with_name("game::Health").is_some());
//!
//! let mut value = DynamicStruct::new();
//! value.set_represented_type_name("game::Health");
//! value.insert("current", 7_i32);
//! value.insert("max", 10_i32);
//! let bytes = prism_reflect::to_binary(&value).expect("encode");
//! let decoded = prism_reflect::from_binary(&bytes, &registry, info).expect("decode");
//! assert!(matches!(decoded.reflect_ref(), prism_reflect::ReflectRef::Struct(_)));
//! ```

use crate::registry::{TypeRegistration, TypeRegistry};
use crate::type_info::{
    EnumInfo, NamedField, StructInfo, TypeInfo, UnnamedField, VariantInfo, VariantKind,
};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Promote an owned string to `&'static str` for a runtime type definition.
fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

/// Promote an owned [`TypeInfo`] to `&'static` for a runtime type definition.
fn leak_info(info: TypeInfo) -> &'static TypeInfo {
    Box::leak(Box::new(info))
}

/// Builder for a runtime-defined named-field struct type.
///
/// Collects a type name and ordered named fields, then produces a leaked
/// `&'static` [`TypeInfo`] via [`build_info`](Self::build_info) or registers it
/// with [`register`](Self::register).
#[derive(Debug, Clone)]
pub struct StructTypeBuilder {
    type_name: &'static str,
    fields: Vec<NamedField>,
}

impl StructTypeBuilder {
    /// Start a struct builder with a `&'static` type name.
    #[must_use]
    pub fn new(type_name: &'static str) -> Self {
        Self {
            type_name,
            fields: Vec::new(),
        }
    }

    /// Start a struct builder with an owned type name (leaked to `&'static`).
    #[must_use]
    pub fn new_owned(type_name: String) -> Self {
        Self::new(leak_str(type_name))
    }

    /// Add a field with a `&'static` name and field type name.
    #[must_use]
    pub fn with_field(mut self, name: &'static str, type_name: &'static str) -> Self {
        self.fields.push(NamedField::new(name, type_name));
        self
    }

    /// Add a field with owned strings (leaked to `&'static`).
    #[must_use]
    pub fn with_field_owned(self, name: String, type_name: String) -> Self {
        let name = leak_str(name);
        let type_name = leak_str(type_name);
        self.with_field(name, type_name)
    }

    /// Finish the builder into a leaked `&'static` [`TypeInfo`].
    #[must_use]
    pub fn build_info(self) -> &'static TypeInfo {
        leak_info(TypeInfo::Struct(StructInfo::new(self.type_name, self.fields)))
    }

    /// Build the [`TypeInfo`] and register it by name as a runtime type.
    ///
    /// Returns the leaked `&'static` [`TypeInfo`] for use as a deserialization
    /// target.
    pub fn register(self, registry: &mut TypeRegistry) -> &'static TypeInfo {
        let type_name = self.type_name;
        let info = self.build_info();
        registry.register_runtime(TypeRegistration::runtime(type_name, info));
        info
    }
}

/// Builder for a runtime-defined enum type.
///
/// Variants are appended in order; each one is assigned the next sequential
/// index automatically.
#[derive(Debug, Clone)]
pub struct EnumTypeBuilder {
    type_name: &'static str,
    variants: Vec<VariantInfo>,
}

impl EnumTypeBuilder {
    /// Start an enum builder with a `&'static` type name.
    #[must_use]
    pub fn new(type_name: &'static str) -> Self {
        Self {
            type_name,
            variants: Vec::new(),
        }
    }

    /// Start an enum builder with an owned type name (leaked to `&'static`).
    #[must_use]
    pub fn new_owned(type_name: String) -> Self {
        Self::new(leak_str(type_name))
    }

    /// Append a unit variant (no fields).
    #[must_use]
    pub fn with_unit_variant(mut self, name: &'static str) -> Self {
        let index = self.variants.len();
        self.variants
            .push(VariantInfo::new(name, index, VariantKind::Unit));
        self
    }

    /// Append a tuple variant whose positional fields have the given type
    /// names.
    #[must_use]
    pub fn with_tuple_variant(mut self, name: &'static str, field_types: Vec<&'static str>) -> Self {
        let index = self.variants.len();
        let fields = field_types
            .into_iter()
            .enumerate()
            .map(|(position, type_name)| UnnamedField::new(position, type_name))
            .collect();
        self.variants
            .push(VariantInfo::new(name, index, VariantKind::Tuple(fields)));
        self
    }

    /// Append a struct variant whose named fields are `(name, type_name)`
    /// pairs.
    #[must_use]
    pub fn with_struct_variant(
        mut self,
        name: &'static str,
        fields: Vec<(&'static str, &'static str)>,
    ) -> Self {
        let index = self.variants.len();
        let named = fields
            .into_iter()
            .map(|(field_name, type_name)| NamedField::new(field_name, type_name))
            .collect();
        self.variants
            .push(VariantInfo::new(name, index, VariantKind::Struct(named)));
        self
    }

    /// Finish the builder into a leaked `&'static` [`TypeInfo`].
    #[must_use]
    pub fn build_info(self) -> &'static TypeInfo {
        leak_info(TypeInfo::Enum(EnumInfo::new(self.type_name, self.variants)))
    }

    /// Build the [`TypeInfo`] and register it by name as a runtime type.
    ///
    /// Returns the leaked `&'static` [`TypeInfo`] for use as a deserialization
    /// target.
    pub fn register(self, registry: &mut TypeRegistry) -> &'static TypeInfo {
        let type_name = self.type_name;
        let info = self.build_info();
        registry.register_runtime(TypeRegistration::runtime(type_name, info));
        info
    }
}
