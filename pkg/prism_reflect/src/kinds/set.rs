//! The `Set` kind: unique-value collections (`HashSet`/`BTreeSet`).

use crate::reflect::Reflect;
use crate::type_info::{SetInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef, Typed};
use std::boxed::Box;
use std::collections::{BTreeSet, HashSet};

/// Reflected access to a unique-value set (`HashSet`/`BTreeSet`).
pub trait Set: Reflect {
    /// Whether the set contains a value equal to the dynamically-typed `value`.
    fn contains(&self, value: &dyn Reflect) -> bool;
    /// Number of elements.
    fn len(&self) -> usize;
    /// Whether the set holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Insert a boxed value.
    ///
    /// # Errors
    /// Returns `Err(value)` when `value`'s dynamic type is not the set's
    /// element type. On success returns `Ok(true)` when the value was newly
    /// inserted and `Ok(false)` when it was already present.
    fn insert(&mut self, value: Box<dyn Reflect>) -> Result<bool, Box<dyn Reflect>>;
    /// Iterate the elements as `&dyn Reflect` in the backing order.
    fn iter_reflect(&self) -> SetIter<'_>;
}

/// Iterator over a [`Set`]'s elements in the backing collection's order.
pub struct SetIter<'a> {
    inner: Box<dyn Iterator<Item = &'a dyn Reflect> + 'a>,
}

impl<'a> SetIter<'a> {
    /// Wrap a concrete element iterator.
    #[must_use]
    pub fn new(inner: Box<dyn Iterator<Item = &'a dyn Reflect> + 'a>) -> Self {
        Self { inner }
    }
}

impl<'a> Iterator for SetIter<'a> {
    type Item = &'a dyn Reflect;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

macro_rules! impl_reflect_set {
    ($set:ident, $($value_bound:tt)+) => {
        impl<T> Set for $set<T>
        where
            T: Reflect + Typed + $($value_bound)+,
        {
            fn contains(&self, value: &dyn Reflect) -> bool {
                match value.downcast_ref::<T>() {
                    Some(value) => $set::contains(self, value),
                    None => false,
                }
            }

            fn len(&self) -> usize {
                $set::len(self)
            }

            fn insert(&mut self, value: Box<dyn Reflect>) -> Result<bool, Box<dyn Reflect>> {
                let value = value.downcast::<T>()?;
                Ok($set::insert(self, *value))
            }

            fn iter_reflect(&self) -> SetIter<'_> {
                SetIter::new(Box::new(self.iter().map(|v| v as &dyn Reflect)))
            }
        }

        impl<T> Reflect for $set<T>
        where
            T: Reflect + Typed + $($value_bound)+,
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
                ReflectRef::Set(self)
            }
            fn reflect_mut(&mut self) -> ReflectMut<'_> {
                ReflectMut::Set(self)
            }
        }

        impl<T> Typed for $set<T>
        where
            T: Reflect + Typed + $($value_bound)+,
        {
            fn type_info() -> &'static TypeInfo {
                crate::cache::intern::<Self, _>(|| {
                    TypeInfo::Set(SetInfo::new(
                        ::core::any::type_name::<Self>(),
                        ::core::any::type_name::<T>(),
                    ))
                })
            }
        }

        impl<T> crate::registry::GetTypeRegistration for $set<T>
        where
            T: Reflect + Typed + $($value_bound)+,
        {
            fn get_type_registration() -> crate::registry::TypeRegistration {
                crate::registry::TypeRegistration::of::<Self>()
            }
        }
    };
}

impl_reflect_set!(HashSet, ::core::hash::Hash + ::core::cmp::Eq);
impl_reflect_set!(BTreeSet, ::core::cmp::Ord);
