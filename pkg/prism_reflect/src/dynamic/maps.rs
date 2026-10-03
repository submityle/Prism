//! The runtime-constructed [`DynamicMap`].

use crate::dynamic::reflect_values_equal;
use crate::kinds::{Map, MapIter};
use crate::reflect::Reflect;
use crate::type_info::{MapInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use core::any::Any;
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec::Vec;

/// A boxed key/value pair in dynamic form.
type ReflectPair = Box<dyn Reflect>;

/// A key/value map assembled at runtime without concrete key/value types.
///
/// Entries are boxed `(key, value)` pairs stored in insertion order; lookups
/// compare keys with
/// [`reflect_values_equal`](crate::dynamic::reflect_values_equal), so keys
/// should be leaf values (strings/scalars). It is the construction counterpart
/// used by [`apply`](Reflect::apply) and [`FromReflect`](crate::FromReflect)
/// for the map kind.
#[derive(Default)]
pub struct DynamicMap {
    represented_type_name: Option<&'static str>,
    entries: Vec<(ReflectPair, ReflectPair)>,
}

impl DynamicMap {
    /// Create an empty dynamic map.
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

    /// Insert or replace an entry with boxed key/value reflected values.
    pub fn insert_boxed(&mut self, key: ReflectPair, value: ReflectPair) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(existing, _)| reflect_values_equal(&**existing, &*key))
        {
            self.entries[index].1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    /// Insert or replace an entry with concrete key/value reflected values.
    pub fn insert_value<K: Reflect, V: Reflect>(&mut self, key: K, value: V) {
        self.insert_boxed(Box::new(key), Box::new(value));
    }
}

impl Map for DynamicMap {
    fn get(&self, key: &dyn Reflect) -> Option<&dyn Reflect> {
        self.entries
            .iter()
            .find(|(existing, _)| reflect_values_equal(&**existing, key))
            .map(|(_, value)| &**value)
    }

    fn get_mut(&mut self, key: &dyn Reflect) -> Option<&mut dyn Reflect> {
        self.entries
            .iter_mut()
            .find(|(existing, _)| reflect_values_equal(&**existing, key))
            .map(|(_, value)| &mut **value)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn insert(
        &mut self,
        key: ReflectPair,
        value: ReflectPair,
    ) -> Result<(), (ReflectPair, ReflectPair)> {
        self.insert_boxed(key, value);
        Ok(())
    }

    fn iter_reflect(&self) -> MapIter<'_> {
        MapIter::new(Box::new(
            self.entries
                .iter()
                .map(|(key, value)| (&**key, &**value)),
        ))
    }
}

impl Reflect for DynamicMap {
    fn type_name(&self) -> &'static str {
        self.represented_type_name
            .unwrap_or("prism_reflect::DynamicMap")
    }

    fn type_info(&self) -> &'static TypeInfo {
        static CELL: OnceLock<TypeInfo> = OnceLock::new();
        CELL.get_or_init(|| {
            TypeInfo::Map(MapInfo::new(
                "prism_reflect::DynamicMap",
                "dyn prism_reflect::Reflect",
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
        ReflectRef::Map(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Map(self)
    }

    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = DynamicMap::new();
        cloned.represented_type_name = self.represented_type_name;
        for (key, value) in &self.entries {
            cloned.insert_boxed(key.reflect_clone(), value.reflect_clone());
        }
        Box::new(cloned)
    }
}
