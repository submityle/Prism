//! The runtime-constructed [`DynamicEnum`] and its [`DynamicVariant`] payload.

use crate::kinds::{Enum, VariantType};
use crate::reflect::Reflect;
use crate::type_info::{EnumInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// The payload shape of a [`DynamicEnum`]'s active variant.
#[derive(Default)]
pub enum DynamicVariant {
    /// A unit variant with no fields.
    #[default]
    Unit,
    /// A tuple variant with positional fields.
    Tuple(Vec<Box<dyn Reflect>>),
    /// A struct variant with named fields in declaration order.
    Struct(Vec<(&'static str, Box<dyn Reflect>)>),
}

/// An enum value assembled at runtime without a concrete Rust type.
///
/// A `DynamicEnum` tracks the active variant's declaration index, its
/// `&'static` name, and the field payload ([`DynamicVariant`]). Unlike a
/// concrete enum it can switch to any variant at runtime with
/// [`set_variant`](Self::set_variant), which is what lets
/// [`apply`](Reflect::apply) perform variant changes.
pub struct DynamicEnum {
    represented_type_name: Option<&'static str>,
    variant_index: usize,
    variant_name: &'static str,
    variant: DynamicVariant,
}

impl Default for DynamicEnum {
    fn default() -> Self {
        Self {
            represented_type_name: None,
            variant_index: 0,
            variant_name: "",
            variant: DynamicVariant::Unit,
        }
    }
}

impl DynamicEnum {
    /// Create a dynamic enum from an active variant index, name, and payload.
    #[must_use]
    pub fn new(variant_index: usize, variant_name: &'static str, variant: DynamicVariant) -> Self {
        Self {
            represented_type_name: None,
            variant_index,
            variant_name,
            variant,
        }
    }

    /// Record the concrete type name this dynamic value stands in for.
    pub fn set_represented_type_name(&mut self, name: &'static str) {
        self.represented_type_name = Some(name);
    }

    /// The represented concrete type name, when one was set.
    #[must_use]
    pub fn represented_type_name(&self) -> Option<&'static str> {
        self.represented_type_name
    }

    /// Switch to a different active variant, replacing index/name/payload.
    pub fn set_variant(
        &mut self,
        variant_index: usize,
        variant_name: &'static str,
        variant: DynamicVariant,
    ) {
        self.variant_index = variant_index;
        self.variant_name = variant_name;
        self.variant = variant;
    }

    /// Borrow the active variant payload.
    #[must_use]
    pub fn variant(&self) -> &DynamicVariant {
        &self.variant
    }

    /// Deep-clone the active variant payload (cloning each boxed field).
    #[must_use]
    pub fn clone_variant(&self) -> DynamicVariant {
        match &self.variant {
            DynamicVariant::Unit => DynamicVariant::Unit,
            DynamicVariant::Tuple(fields) => {
                DynamicVariant::Tuple(fields.iter().map(|f| f.reflect_clone()).collect())
            }
            DynamicVariant::Struct(fields) => DynamicVariant::Struct(
                fields
                    .iter()
                    .map(|(name, value)| (*name, value.reflect_clone()))
                    .collect(),
            ),
        }
    }
}

impl Enum for DynamicEnum {
    fn variant_name(&self) -> &'static str {
        self.variant_name
    }

    fn variant_index(&self) -> usize {
        self.variant_index
    }

    fn variant_type(&self) -> VariantType {
        match &self.variant {
            DynamicVariant::Unit => VariantType::Unit,
            DynamicVariant::Tuple(_) => VariantType::Tuple,
            DynamicVariant::Struct(_) => VariantType::Struct,
        }
    }

    fn field(&self, name: &str) -> Option<&dyn Reflect> {
        match &self.variant {
            DynamicVariant::Struct(fields) => fields
                .iter()
                .find(|(field_name, _)| *field_name == name)
                .map(|(_, value)| &**value),
            _ => None,
        }
    }

    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> {
        match &mut self.variant {
            DynamicVariant::Struct(fields) => fields
                .iter_mut()
                .find(|(field_name, _)| *field_name == name)
                .map(|(_, value)| &mut **value),
            _ => None,
        }
    }

    fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
        match &self.variant {
            DynamicVariant::Unit => None,
            DynamicVariant::Tuple(fields) => fields.get(index).map(|value| &**value),
            DynamicVariant::Struct(fields) => fields.get(index).map(|(_, value)| &**value),
        }
    }

    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        match &mut self.variant {
            DynamicVariant::Unit => None,
            DynamicVariant::Tuple(fields) => fields.get_mut(index).map(|value| &mut **value),
            DynamicVariant::Struct(fields) => {
                fields.get_mut(index).map(|(_, value)| &mut **value)
            }
        }
    }

    fn field_count(&self) -> usize {
        match &self.variant {
            DynamicVariant::Unit => 0,
            DynamicVariant::Tuple(fields) => fields.len(),
            DynamicVariant::Struct(fields) => fields.len(),
        }
    }
}

impl Reflect for DynamicEnum {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicEnum")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::Enum(EnumInfo::new("prism_reflect::DynamicEnum", Vec::new()))
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_reflect(&self) -> &dyn Reflect {
        self
    }

    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }

    fn reflect_ref(&self) -> ReflectRef<'_> {
        ReflectRef::Enum(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Enum(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicEnum::new(self.variant_index, self.variant_name, self.clone_variant());
        cloned.represented_type_name = self.represented_type_name;
        Box::new(cloned)
    }
}
