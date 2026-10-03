//! The `List` kind: growable, homogeneous sequences such as `Vec<T>`.

use crate::reflect::Reflect;
use crate::type_info::{ListInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef, Typed};
use std::boxed::Box;
use std::vec::Vec;

/// Reflected access to a growable, homogeneous list (`Vec<T>`).
pub trait List: Reflect {
    /// Borrow the element at `index`.
    fn get(&self, index: usize) -> Option<&dyn Reflect>;
    /// Mutably borrow the element at `index`.
    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect>;
    /// Number of elements currently stored.
    fn len(&self) -> usize;
    /// Whether the list holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Append a boxed element.
    ///
    /// # Errors
    /// Returns `Err(value)` when `value`'s dynamic type is not the list's
    /// element type.
    fn push(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>>;
    /// Iterate the elements as `&dyn Reflect` in index order.
    fn iter_reflect(&self) -> ListIter<'_> {
        ListIter::new(self.as_list())
    }
    /// Upcast to `&dyn List` (needed by the default iterator).
    fn as_list(&self) -> &dyn List;
}

/// Index-order iterator over a [`List`]'s elements.
pub struct ListIter<'a> {
    list: &'a dyn List,
    index: usize,
}

impl<'a> ListIter<'a> {
    /// Build an iterator positioned at the first element.
    #[must_use]
    pub fn new(list: &'a dyn List) -> Self {
        Self { list, index: 0 }
    }
}

impl<'a> Iterator for ListIter<'a> {
    type Item = &'a dyn Reflect;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.list.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.list.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T: Reflect + Typed> List for Vec<T> {
    fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.as_slice().get(index).map(|v| v as &dyn Reflect)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        self.as_mut_slice()
            .get_mut(index)
            .map(|v| v as &mut dyn Reflect)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn push(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        let typed = value.downcast::<T>()?;
        Vec::push(self, *typed);
        Ok(())
    }

    fn as_list(&self) -> &dyn List {
        self
    }
}

impl<T: Reflect + Typed> Reflect for Vec<T> {
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
        ReflectRef::List(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::List(self)
    }
    fn reflect_clone(&self) -> Box<dyn Reflect> {
        let mut cloned = crate::DynamicList::new();
        cloned.set_represented_type_name(::core::any::type_name::<Self>());
        for item in self.as_slice() {
            cloned.push_boxed(Reflect::reflect_clone(item));
        }
        Box::new(cloned)
    }
}

impl<T: Reflect + Typed> Typed for Vec<T> {
    fn type_info() -> &'static TypeInfo {
        crate::cache::intern::<Self, _>(|| {
            TypeInfo::List(ListInfo::new(
                ::core::any::type_name::<Self>(),
                ::core::any::type_name::<T>(),
            ))
        })
    }
}

impl<T: Reflect + Typed> crate::registry::GetTypeRegistration for Vec<T> {
    fn get_type_registration() -> crate::registry::TypeRegistration {
        crate::registry::TypeRegistration::of::<Self>()
    }
}
