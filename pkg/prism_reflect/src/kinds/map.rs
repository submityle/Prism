//! The `Map` kind: key/value associative collections (`HashMap`/`BTreeMap`).

use crate::reflect::Reflect;
use crate::type_info::{MapInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef, Typed};
use std::boxed::Box;
use std::collections::{BTreeMap, HashMap};

/// A boxed key/value pair in dynamic form.
type ReflectPair = Box<dyn Reflect>;

/// Reflected access to a key/value map (`HashMap`/`BTreeMap`).
pub trait Map: Reflect {
    /// Look up a value by a dynamically-typed key.
    fn get(&self, key: &dyn Reflect) -> Option<&dyn Reflect>;
    /// Mutably look up a value by a dynamically-typed key.
    fn get_mut(&mut self, key: &dyn Reflect) -> Option<&mut dyn Reflect>;
    /// Number of entries.
    fn len(&self) -> usize;
    /// Whether the map holds no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Insert a boxed key/value pair.
    ///
    /// # Errors
    /// Returns `Err((key, value))` when either box's dynamic type does not
    /// match the map's key/value types.
    fn insert(
        &mut self,
        key: ReflectPair,
        value: ReflectPair,
    ) -> Result<(), (ReflectPair, ReflectPair)>;
    /// Iterate the entries as `(&dyn Reflect, &dyn Reflect)` pairs.
    fn iter_reflect(&self) -> MapIter<'_>;
}

/// Iterator over a [`Map`]'s entries in the backing collection's order.
pub struct MapIter<'a> {
    inner: Box<dyn Iterator<Item = (&'a dyn Reflect, &'a dyn Reflect)> + 'a>,
}

impl<'a> MapIter<'a> {
    /// Wrap a concrete entry iterator.
    #[must_use]
    pub fn new(inner: Box<dyn Iterator<Item = (&'a dyn Reflect, &'a dyn Reflect)> + 'a>) -> Self {
        Self { inner }
    }
}

impl<'a> Iterator for MapIter<'a> {
    type Item = (&'a dyn Reflect, &'a dyn Reflect);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

macro_rules! impl_reflect_map {
    ($map:ident, $($key_bound:tt)+) => {
        impl<K, V> Map for $map<K, V>
        where
            K: Reflect + Typed + $($key_bound)+,
            V: Reflect + Typed,
        {
            fn get(&self, key: &dyn Reflect) -> Option<&dyn Reflect> {
                let key = key.downcast_ref::<K>()?;
                $map::get(self, key).map(|v| v as &dyn Reflect)
            }

            fn get_mut(&mut self, key: &dyn Reflect) -> Option<&mut dyn Reflect> {
                let key = key.downcast_ref::<K>()?;
                $map::get_mut(self, key).map(|v| v as &mut dyn Reflect)
            }

            fn len(&self) -> usize {
                $map::len(self)
            }

            fn insert(
                &mut self,
                key: ReflectPair,
                value: ReflectPair,
            ) -> Result<(), (ReflectPair, ReflectPair)> {
                let key = match key.downcast::<K>() {
                    Ok(key) => key,
                    Err(key) => return Err((key, value)),
                };
                let value = match value.downcast::<V>() {
                    Ok(value) => value,
                    Err(value) => return Err((key as ReflectPair, value)),
                };
                $map::insert(self, *key, *value);
                Ok(())
            }

            fn iter_reflect(&self) -> MapIter<'_> {
                MapIter::new(Box::new(
                    self.iter().map(|(k, v)| (k as &dyn Reflect, v as &dyn Reflect)),
                ))
            }
        }

        impl<K, V> Reflect for $map<K, V>
        where
            K: Reflect + Typed + $($key_bound)+,
            V: Reflect + Typed,
        {
            fn type_name(&self) -> &'static str {
                ::core::any::type_name::<Self>()
            }
            fn type_info(&self) -> &'static TypeInfo {
                <Self as Typed>::type_info()
            }
            fn as_any(&self) -> &dyn ::core::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any {
                self
            }
            fn into_any(self: Box<Self>) -> Box<dyn ::core::any::Any> {
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
                let mut cloned = crate::DynamicMap::new();
                cloned.set_represented_type_name(::core::any::type_name::<Self>());
                for (key, value) in self.iter() {
                    cloned.insert_boxed(
                        Reflect::reflect_clone(key),
                        Reflect::reflect_clone(value),
                    );
                }
                Box::new(cloned)
            }
        }

        impl<K, V> Typed for $map<K, V>
        where
            K: Reflect + Typed + $($key_bound)+,
            V: Reflect + Typed,
        {
            fn type_info() -> &'static TypeInfo {
                crate::cache::intern::<Self, _>(|| {
                    TypeInfo::Map(MapInfo::new(
                        ::core::any::type_name::<Self>(),
                        ::core::any::type_name::<K>(),
                        ::core::any::type_name::<V>(),
                    ))
                })
            }
        }

        impl<K, V> crate::registry::GetTypeRegistration for $map<K, V>
        where
            K: Reflect + Typed + $($key_bound)+,
            V: Reflect + Typed,
        {
            fn get_type_registration() -> crate::registry::TypeRegistration {
                crate::registry::TypeRegistration::of::<Self>()
            }
        }
    };
}

impl_reflect_map!(HashMap, ::core::hash::Hash + ::core::cmp::Eq);
impl_reflect_map!(BTreeMap, ::core::cmp::Ord);
