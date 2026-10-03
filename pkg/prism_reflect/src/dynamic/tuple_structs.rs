//! The runtime-constructed [`DynamicTupleStruct`].

use crate::reflect::{Reflect, TupleStruct};
use crate::type_info::{TupleStructInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// A tuple struct (positional fields) assembled at runtime without a concrete
/// Rust type.
///
/// Fields are stored as boxed [`Reflect`] values in positional order. Like
/// [`DynamicStruct`](crate::DynamicStruct) it can carry the name of the
/// concrete type it represents and participates in
/// [`apply`](Reflect::apply)/[`FromReflect`](crate::FromReflect).
#[derive(Default)]
pub struct DynamicTupleStruct {
    represented_type_name: Option<&'static str>,
    values: Vec<Box<dyn Reflect>>,
}

impl DynamicTupleStruct {
    /// Create an empty dynamic tuple struct.
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

    /// Append a boxed reflected field.
    pub fn insert_boxed(&mut self, value: Box<dyn Reflect>) {
        self.values.push(value);
    }

    /// Append a concrete reflected field.
    pub fn insert<T: Reflect>(&mut self, value: T) {
        self.insert_boxed(Box::new(value));
    }
}

impl TupleStruct for DynamicTupleStruct {
    fn field(&self, index: usize) -> Option<&dyn Reflect> {
        self.values.get(index).map(|value| &**value)
    }

    fn field_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.values.get_mut(index).map(|value| &mut **value)
    }

    fn field_count(&self) -> usize {
        self.values.len()
    }
}

impl Reflect for DynamicTupleStruct {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicTupleStruct")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::TupleStruct(TupleStructInfo::new(
                "prism_reflect::DynamicTupleStruct",
                Vec::new(),
            ))
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
        ReflectRef::TupleStruct(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::TupleStruct(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicTupleStruct::new();
        cloned.represented_type_name = self.represented_type_name;
        for value in &self.values {
            cloned.insert_boxed(value.reflect_clone());
        }
        Box::new(cloned)
    }
}
