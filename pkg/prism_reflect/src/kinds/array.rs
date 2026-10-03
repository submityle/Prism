//! The `Array` kind: fixed-length sequences `[T; N]`.

use crate::reflect::Reflect;
use crate::type_info::{ArrayInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef, Typed};
use std::boxed::Box;

/// Reflected access to a fixed-length array (`[T; N]`).
pub trait Array: Reflect {
    /// Borrow the element at `index`.
    fn get(&self, index: usize) -> Option<&dyn Reflect>;
    /// Mutably borrow the element at `index`.
    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect>;
    /// The fixed element count `N`.
    fn len(&self) -> usize;
    /// Whether the array has zero elements (`N == 0`).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Iterate the elements as `&dyn Reflect` in index order.
    fn iter_reflect(&self) -> ArrayIter<'_> {
        ArrayIter::new(self.as_array())
    }
    /// Upcast to `&dyn Array` (needed by the default iterator).
    fn as_array(&self) -> &dyn Array;
}

/// Index-order iterator over an [`Array`]'s elements.
pub struct ArrayIter<'a> {
    array: &'a dyn Array,
    index: usize,
}

impl<'a> ArrayIter<'a> {
    /// Build an iterator positioned at the first element.
    #[must_use]
    pub fn new(array: &'a dyn Array) -> Self {
        Self { array, index: 0 }
    }
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = &'a dyn Reflect;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.array.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T: Reflect + Typed, const N: usize> Array for [T; N] {
    fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.as_slice().get(index).map(|v| v as &dyn Reflect)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.as_mut_slice()
            .get_mut(index)
            .map(|v| v as &mut dyn Reflect)
    }

    fn len(&self) -> usize {
        N
    }

    fn as_array(&self) -> &dyn Array {
        self
    }
}

impl<T: Reflect + Typed, const N: usize> Reflect for [T; N] {
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
        ReflectRef::Array(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Array(self)
    }
}

impl<T: Reflect + Typed, const N: usize> Typed for [T; N] {
    fn type_info() -> &'static TypeInfo {
        crate::cache::intern::<Self, _>(|| {
            TypeInfo::Array(ArrayInfo::new(
                ::core::any::type_name::<Self>(),
                ::core::any::type_name::<T>(),
                N,
            ))
        })
    }
}

impl<T: Reflect + Typed, const N: usize> crate::registry::GetTypeRegistration for [T; N] {
    fn get_type_registration() -> crate::registry::TypeRegistration {
        crate::registry::TypeRegistration::of::<Self>()
    }
}
