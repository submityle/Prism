//! The [`FunctionRegistry`]: a name-keyed catalogue of reflected functions.
//!
//! Mirroring the [`TypeRegistry`](crate::TypeRegistry) for types, this registry
//! stores [`DynamicFunction`]s by name so a script or editor can invoke them
//! dynamically through [`call`](FunctionRegistry::call) (design §12 — the
//! script-bridge/editor-action substrate).

use crate::func::args::ArgList;
use crate::func::call::{DynamicFunction, FunctionError, IntoFunction};
use crate::reflect::Reflect;
use alloc::boxed::Box;
use alloc::string::String;
use std::collections::HashMap;

/// A runtime catalogue mapping a name to a [`DynamicFunction`].
#[derive(Default)]
pub struct FunctionRegistry {
    functions: HashMap<String, DynamicFunction>,
}

impl FunctionRegistry {
    /// Build an empty function registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `function` under `name`, erasing it with [`IntoFunction`].
    ///
    /// The name is also stamped onto the stored [`DynamicFunction`] (so arity
    /// errors can report it). Re-registering a name replaces the previous
    /// entry. Returns `&mut Self` for chaining.
    pub fn register<Marker>(
        &mut self,
        name: impl Into<String>,
        function: impl IntoFunction<Marker>,
    ) -> &mut Self {
        let name = name.into();
        let function = function.into_function().with_name(name.clone());
        self.functions.insert(name, function);
        self
    }

    /// Alias for [`register`](Self::register) matching the design's
    /// `register_function` wording (design §12).
    pub fn register_function<Marker>(
        &mut self,
        name: impl Into<String>,
        function: impl IntoFunction<Marker>,
    ) -> &mut Self {
        self.register(name, function)
    }

    /// Borrow a registered function by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DynamicFunction> {
        self.functions.get(name)
    }

    /// Whether a function with the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// The number of registered functions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Whether no functions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Iterate over the registered `(name, function)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &DynamicFunction)> {
        self.functions
            .iter()
            .map(|(name, function)| (name.as_str(), function))
    }

    /// Call the named function with `args`.
    ///
    /// # Errors
    /// Returns [`FunctionError::UnknownFunction`] when no function matches
    /// `name`, or propagates the [`FunctionError`] from
    /// [`DynamicFunction::call`] (arity or argument-type mismatch).
    pub fn call(&self, name: &str, args: ArgList) -> Result<Box<dyn Reflect>, FunctionError> {
        match self.functions.get(name) {
            Some(function) => function.call(&args),
            None => Err(FunctionError::UnknownFunction {
                name: name.into(),
            }),
        }
    }
}
