//! The runtime-constructed [`DynamicList`].

use crate::kinds::List;
use crate::reflect::Reflect;
use crate::type_info::{ListInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// A growable, heterogeneously-boxed list assembled at runtime.
///
/// Elements are boxed [`Reflect`] values in index order. Unlike `Vec<T>` the
/// element type is not statically fixed, so pushes always succeed; the list is
/// the construction counterpart used by [`apply`](Reflect::apply) and
/// [`FromReflect`](crate::FromReflect) for sequence kinds.
#[derive(Default)]
pub struct DynamicList {
    represented_type_name: Option<&'static str>,
    values: Vec<Box<dyn Reflect>>,
}

impl DynamicList {
    /// Create an empty dynamic list.
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

    /// Append a boxed reflected element.
    pub fn push_boxed(&mut self, value: Box<dyn Reflect>) {
        self.values.push(value);
    }

    /// Append a concrete reflected element.
    pub fn push_value<T: Reflect>(&mut self, value: T) {
        self.push_boxed(Box::new(value));
    }
}

impl List for DynamicList {
    fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.values.get(index).map(|value| &**value)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.values.get_mut(index).map(|value| &mut **value)
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn push(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        self.values.push(value);
        Ok(())
    }

    fn as_list(&self) -> &dyn List {
        self
    }
}

impl Reflect for DynamicList {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicList")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::List(ListInfo::new(
                "prism_reflect::DynamicList",
                "dyn prism_reflect::Reflect",
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
        ReflectRef::List(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::List(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicList::new();
        cloned.represented_type_name = self.represented_type_name;
        for value in &self.values {
            cloned.push_boxed(value.reflect_clone());
        }
        Box::new(cloned)
    }
}
