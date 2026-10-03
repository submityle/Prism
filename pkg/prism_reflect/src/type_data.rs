//! Per-type side-channel data stored in the [`TypeRegistry`].
//!
//! [`TypeData`] is a cloneable, type-erased payload attached to a
//! [`crate::registry::TypeRegistration`]. It lets subsystems (serialization,
//! default construction, component bridges, ...) hang behaviour off a reflected
//! type without the kernel knowing about them. [`ReflectDefault`] is the first
//! concrete payload and proves the registry can default-construct a registered
//! type purely through reflection.

use crate::reflect::Reflect;
use core::any::Any;
use std::boxed::Box;

/// A cloneable, type-erased payload attached to a registered type.
pub trait TypeData: Any + Send + Sync {
    /// Clone this payload into a new boxed trait object.
    fn clone_type_data(&self) -> Box<dyn TypeData>;
    /// Upcast to `&dyn Any` for concrete downcasting.
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn TypeData> {
    fn clone(&self) -> Self {
        self.clone_type_data()
    }
}

/// Type data holding a reflection-driven default constructor for a type.
#[derive(Clone)]
pub struct ReflectDefault {
    default: fn() -> Box<dyn Reflect>,
}

impl ReflectDefault {
    /// Build the default-constructor payload for `T`.
    #[must_use]
    pub fn new<T: Reflect + Default>() -> Self {
        Self {
            default: || Box::new(T::default()),
        }
    }

    /// Construct a fresh default value as a boxed `dyn Reflect`.
    #[must_use]
    pub fn default_value(&self) -> Box<dyn Reflect> {
        (self.default)()
    }
}

impl TypeData for ReflectDefault {
    fn clone_type_data(&self) -> Box<dyn TypeData> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
