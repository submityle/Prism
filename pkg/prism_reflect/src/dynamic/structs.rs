//! The runtime-constructed [`DynamicStruct`].

use crate::reflect::{Reflect, Struct};
use crate::type_info::{StructInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// A named-field struct assembled at runtime without a concrete Rust type.
///
/// Fields are stored as boxed [`Reflect`] values keyed by `&'static` names in
/// insertion order. A `DynamicStruct` can stand in for a concrete struct it
/// *represents* (via
/// [`set_represented_type_name`](Self::set_represented_type_name)), be
/// [`apply`](Reflect::apply)-ed onto one, or be converted back with
/// [`FromReflect`](crate::FromReflect).
#[derive(Default)]
pub struct DynamicStruct {
    represented_type_name: Option<&'static str>,
    names: Vec<&'static str>,
    values: Vec<Box<dyn Reflect>>,
}

impl DynamicStruct {
    /// Create an empty dynamic struct.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Insert or replace a field by name with a boxed reflected value.
    pub fn insert_boxed(&mut self, name: &'static str, value: Box<dyn Reflect>) {
        if let Some(index) = self.names.iter().position(|existing| *existing == name) {
            self.values[index] = value;
        } else {
            self.names.push(name);
            self.values.push(value);
        }
    }

    /// Insert or replace a field by name with a concrete reflected value.
    pub fn insert<T: Reflect>(&mut self, name: &'static str, value: T) {
        self.insert_boxed(name, Box::new(value));
    }
}

impl Struct for DynamicStruct {
    fn field(&self, name: &str) -> Option<&dyn Reflect> {
        let index = self.names.iter().position(|existing| *existing == name)?;
        Some(&*self.values[index])
    }

    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> {
        let index = self.names.iter().position(|existing| *existing == name)?;
        Some(&mut *self.values[index])
    }

    fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
        self.values.get(index).map(|value| &**value)
    }

    fn name_at(&self, index: usize) -> Option<&'static str> {
        self.names.get(index).copied()
    }

    fn field_count(&self) -> usize {
        self.names.len()
    }
}

impl Reflect for DynamicStruct {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicStruct")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| TypeInfo::Struct(StructInfo::new("prism_reflect::DynamicStruct", Vec::new())))
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
        ReflectRef::Struct(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Struct(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicStruct::new();
        cloned.represented_type_name = self.represented_type_name;
        for (name, value) in self.names.iter().zip(self.values.iter()) {
            cloned.insert_boxed(name, value.reflect_clone());
        }
        Box::new(cloned)
    }
}
