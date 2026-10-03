//! The runtime-constructed [`DynamicSet`].

use crate::dynamic::reflect_values_equal;
use crate::kinds::{Set, SetIter};
use crate::reflect::Reflect;
use crate::type_info::{SetInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use alloc::boxed::Box;
use std::sync::OnceLock;
use alloc::vec::Vec;

/// A unique-value set assembled at runtime without a concrete value type.
///
/// Elements are boxed [`Reflect`] values stored in insertion order; membership
/// is decided with
/// [`reflect_values_equal`](crate::dynamic::reflect_values_equal), so elements
/// should be leaf values (strings/scalars). It is the construction counterpart
/// used by [`apply`](Reflect::apply) and [`FromReflect`](crate::FromReflect)
/// for the set kind, mirroring [`DynamicList`](crate::DynamicList) but with
/// value-dedup semantics.
#[derive(Default)]
pub struct DynamicSet {
    represented_type_name: Option<&'static str>,
    values: Vec<Box<dyn Reflect>>,
}

impl DynamicSet {
    /// Create an empty dynamic set.
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

    /// Insert a boxed element, deduplicating by structural value equality.
    ///
    /// Returns `true` when the value was newly inserted and `false` when an
    /// equal value was already present.
    pub fn push_boxed(&mut self, value: Box<dyn Reflect>) -> bool {
        if self
            .values
            .iter()
            .any(|existing| reflect_values_equal(&**existing, &*value))
        {
            return false;
        }
        self.values.push(value);
        true
    }

    /// Insert a concrete element, deduplicating by structural value equality.
    pub fn push_value<T: Reflect>(&mut self, value: T) -> bool {
        self.push_boxed(Box::new(value))
    }
}

impl Set for DynamicSet {
    fn contains(&self, value: &dyn Reflect) -> bool {
        self.values
            .iter()
            .any(|existing| reflect_values_equal(&**existing, value))
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn insert(&mut self, value: Box<dyn Reflect>) -> Result<bool, Box<dyn Reflect>> {
        Ok(self.push_boxed(value))
    }

    fn iter_reflect(&self) -> SetIter<'_> {
        SetIter::new(Box::new(self.values.iter().map(|value| &**value)))
    }
}

impl Reflect for DynamicSet {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicSet")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::Set(SetInfo::new(
                "prism_reflect::DynamicSet",
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
        ReflectRef::Set(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Set(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicSet::new();
        cloned.represented_type_name = self.represented_type_name;
        for value in &self.values {
            cloned.values.push(value.reflect_clone());
        }
        Box::new(cloned)
    }
}
