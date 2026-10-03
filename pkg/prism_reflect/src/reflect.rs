//! The core `Reflect` trait and its down-casting views.

use crate::apply::ApplyError;
use crate::kinds::{Array, Enum, List, Map, Set};
use crate::type_info::TypeInfo;
use core::any::Any;
use std::boxed::Box;

/// The universal reflection trait: a bridge from Rust's static type world to
/// the dynamic data-driven world (editor/scripting/serialization/network).
///
/// The trait exposes static `TypeInfo`, `Any`-based downcasting, owned
/// `Box<dyn Any>` recovery, and the `ReflectRef`/`ReflectMut` views used to
/// walk every reflected shape (struct/tuple-struct/enum/list/array/map/set and
/// leaf values), plus the M2 [`apply`](Reflect::apply)/[`reflect_clone`](Reflect::reflect_clone)
/// state-transfer operations. `reflect_hash`/`reflect_partial_eq` (design §8)
/// land in later milestones.
pub trait Reflect: Any + Send + Sync {
    /// The runtime type name of `self`.
    fn type_name(&self) -> &'static str;

    /// The static, `OnceLock`-cached descriptor for this type.
    fn type_info(&self) -> &'static TypeInfo;

    /// Upcast to `&dyn Any` for concrete downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Mutable upcast to `&mut dyn Any`.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Consume a boxed value and recover its `Box<dyn Any>` for owned
    /// downcasting (used by the dynamic container inserts).
    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    /// Upcast to `&dyn Reflect`.
    fn as_reflect(&self) -> &dyn Reflect;

    /// Mutable upcast to `&mut dyn Reflect`.
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect;

    /// Down-explore the value into a typed read view.
    fn reflect_ref(&self) -> ReflectRef<'_>;

    /// Down-explore the value into a typed mutable view.
    fn reflect_mut(&mut self) -> ReflectMut<'_>;

    /// Recursively copy the state of `source` into `self`.
    ///
    /// Container kinds patch by name/index/key and recurse; leaf values
    /// clone-assign. A [`DynamicEnum`](crate::DynamicEnum) target may switch to
    /// the source's active variant, while a concrete enum requires the variant
    /// to already match. The default implementation dispatches on
    /// [`reflect_mut`](Reflect::reflect_mut); leaf value types override it with
    /// a direct clone-assign.
    ///
    /// # Errors
    /// Returns an [`ApplyError`] when the two values' kinds disagree, a leaf
    /// type mismatch occurs, a concrete enum variant cannot be switched, or a
    /// container rejects an element.
    fn apply(&mut self, source: &dyn Reflect) -> Result<(), ApplyError> {
        crate::apply::apply_impl(self.as_reflect_mut(), source)
    }

    /// Produce an owned, deep clone of this value as a boxed `dyn Reflect`.
    ///
    /// Leaf and `Copy` types clone/copy directly; container and derived types
    /// rebuild an equivalent [`Dynamic*`](crate::dynamic) value whose elements
    /// are themselves `reflect_clone`d. The result can be
    /// [`apply`](Reflect::apply)-ed onto a concrete value or converted back with
    /// [`FromReflect`](crate::FromReflect).
    fn reflect_clone(&self) -> Box<dyn Reflect>;
}

impl dyn Reflect {
    /// Attempt to downcast a shared reference to a concrete type.
    #[must_use]
    pub fn downcast_ref<T: Reflect>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    /// Attempt to downcast a mutable reference to a concrete type.
    #[must_use]
    pub fn downcast_mut<T: Reflect>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }

    /// Attempt to downcast an owned boxed value to a concrete type, returning
    /// the original boxed value on a type mismatch.
    ///
    /// # Errors
    /// Returns `Err(self)` when the dynamic type is not `T`.
    pub fn downcast<T: Reflect>(self: Box<Self>) -> Result<Box<T>, Box<dyn Reflect>> {
        if self.is::<T>() {
            Ok(self
                .into_any()
                .downcast::<T>()
                .expect("type checked immediately above"))
        } else {
            Err(self)
        }
    }

    /// Whether this value is of concrete type `T`.
    #[must_use]
    pub fn is<T: Reflect>(&self) -> bool {
        self.as_any().is::<T>()
    }
}

/// A type that can report its static `TypeInfo` without an instance.
pub trait Typed {
    /// The static, `OnceLock`-cached descriptor for this type.
    fn type_info() -> &'static TypeInfo;
}

/// Reflected access to a struct with named fields.
pub trait Struct: Reflect {
    /// Borrow a field by name.
    fn field(&self, name: &str) -> Option<&dyn Reflect>;
    /// Mutably borrow a field by name.
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect>;
    /// Borrow a field by positional index.
    fn field_at(&self, index: usize) -> Option<&dyn Reflect>;
    /// The field name at a positional index.
    fn name_at(&self, index: usize) -> Option<&'static str>;
    /// Number of fields.
    fn field_count(&self) -> usize;
}

/// Reflected access to a tuple struct (positional fields).
pub trait TupleStruct: Reflect {
    /// Borrow a field by position.
    fn field(&self, index: usize) -> Option<&dyn Reflect>;
    /// Mutably borrow a field by position.
    fn field_mut(&mut self, index: usize) -> Option<&mut dyn Reflect>;
    /// Number of fields.
    fn field_count(&self) -> usize;
}

/// A read-only, kind-tagged view obtained from [`Reflect::reflect_ref`].
#[non_exhaustive]
pub enum ReflectRef<'a> {
    /// A named-field struct.
    Struct(&'a dyn Struct),
    /// A tuple struct.
    TupleStruct(&'a dyn TupleStruct),
    /// An enum value.
    Enum(&'a dyn Enum),
    /// A growable, homogeneous list (`Vec<T>`).
    List(&'a dyn List),
    /// A fixed-length array (`[T; N]`).
    Array(&'a dyn Array),
    /// A key/value map (`HashMap`/`BTreeMap`).
    Map(&'a dyn Map),
    /// A unique-value set (`HashSet`/`BTreeSet`).
    Set(&'a dyn Set),
    /// A leaf value.
    Value(&'a dyn Reflect),
}

/// A mutable, kind-tagged view obtained from [`Reflect::reflect_mut`].
#[non_exhaustive]
pub enum ReflectMut<'a> {
    /// A named-field struct.
    Struct(&'a mut dyn Struct),
    /// A tuple struct.
    TupleStruct(&'a mut dyn TupleStruct),
    /// An enum value.
    Enum(&'a mut dyn Enum),
    /// A growable, homogeneous list (`Vec<T>`).
    List(&'a mut dyn List),
    /// A fixed-length array (`[T; N]`).
    Array(&'a mut dyn Array),
    /// A key/value map (`HashMap`/`BTreeMap`).
    Map(&'a mut dyn Map),
    /// A unique-value set (`HashSet`/`BTreeSet`).
    Set(&'a mut dyn Set),
    /// A leaf value.
    Value(&'a mut dyn Reflect),
}
