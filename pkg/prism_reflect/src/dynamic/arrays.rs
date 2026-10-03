//! The runtime-constructed [`DynamicArray`].

use crate::kinds::Array;
use crate::reflect::Reflect;
use crate::type_info::{ArrayInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// A fixed-length (once built) array assembled at runtime.
///
/// Elements are boxed [`Reflect`] values in index order. Its length is whatever
/// was pushed during construction and is treated as fixed by the
/// [`Array`](crate::Array) kind; [`apply`](Reflect::apply) patches up to the
/// shared length.
#[derive(Default)]
pub struct DynamicArray {
    represented_type_name: Option<&'static str>,
    values: Vec<Box<dyn Reflect>>,
}

impl DynamicArray {
    /// Create an empty dynamic array.
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

    /// Append a boxed reflected element while building the array.
    pub fn push_boxed(&mut self, value: Box<dyn Reflect>) {
        self.values.push(value);
    }

    /// Append a concrete reflected element while building the array.
    pub fn push_value<T: Reflect>(&mut self, value: T) {
        self.push_boxed(Box::new(value));
    }
}

impl Array for DynamicArray {
    fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.values.get(index).map(|value| &**value)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.values.get_mut(index).map(|value| &mut **value)
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_array(&self) -> &dyn Array {
        self
    }
}

impl Reflect for DynamicArray {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicArray")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::Array(ArrayInfo::new(
                "prism_reflect::DynamicArray",
                "dyn prism_reflect::Reflect",
                0,
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
        ReflectRef::Array(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Array(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicArray::new();
        cloned.represented_type_name = self.represented_type_name;
        for value in &self.values {
            cloned.push_boxed(value.reflect_clone());
        }
        Box::new(cloned)
    }
}
