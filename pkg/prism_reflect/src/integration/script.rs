//! Script bridge: a safe facade for reading and writing reflected values by
//! string path and calling reflected functions by name.
//!
//! Scripting runtimes (Lua, WASM, …) do not know Rust types at compile time;
//! they address data by path (`"transform.translation.x"`) and behaviour by
//! function name. This bridge routes those requests through the reflection
//! layer — [`ParsedPath`](crate::ParsedPath) navigation plus the
//! [`FunctionRegistry`] — so a script can touch engine state without any
//! hand-written per-type foreign-function glue (design §24.6 "脚本桥"). All
//! argument/field type checking stays inside the reflection layer, so a bad
//! access returns a typed [`ScriptError`] rather than corrupting state.

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt;

use crate::path::ParsePathError;
use crate::{
    ApplyError, ArgList, FunctionError, FunctionRegistry, ParsedPath, Reflect, reflect_path,
    reflect_path_mut,
};

/// A reflection-backed façade exposing path access and call-by-name to a
/// scripting layer.
///
/// The bridge borrows a [`FunctionRegistry`] for [`call`](ScriptBridge::call);
/// the path accessors operate on whatever root value the caller supplies.
#[derive(Clone, Copy)]
pub struct ScriptBridge<'a> {
    functions: &'a FunctionRegistry,
}

impl<'a> ScriptBridge<'a> {
    /// Create a bridge over `functions`.
    #[must_use]
    pub fn new(functions: &'a FunctionRegistry) -> Self {
        Self { functions }
    }

    /// The function registry this bridge dispatches [`call`](Self::call) to.
    #[must_use]
    pub fn functions(&self) -> &'a FunctionRegistry {
        self.functions
    }

    /// Read the value at `path` within `root`.
    ///
    /// # Errors
    /// Returns [`ScriptError::Path`] if `path` does not parse, or
    /// [`ScriptError::NoSuchPath`] if it does not resolve within `root`.
    pub fn get<'r>(&self, root: &'r dyn Reflect, path: &str) -> Result<&'r dyn Reflect, ScriptError> {
        let parsed = ParsedPath::parse(path).map_err(ScriptError::Path)?;
        reflect_path(root, &parsed).ok_or_else(|| ScriptError::NoSuchPath(path.into()))
    }

    /// Read a mutable reference to the value at `path` within `root`.
    ///
    /// # Errors
    /// Returns [`ScriptError::Path`] if `path` does not parse, or
    /// [`ScriptError::NoSuchPath`] if it does not resolve within `root`.
    pub fn get_mut<'r>(
        &self,
        root: &'r mut dyn Reflect,
        path: &str,
    ) -> Result<&'r mut dyn Reflect, ScriptError> {
        let parsed = ParsedPath::parse(path).map_err(ScriptError::Path)?;
        reflect_path_mut(root, &parsed).ok_or_else(|| ScriptError::NoSuchPath(path.into()))
    }

    /// Write `value` into the slot at `path` within `root`, applying it onto
    /// the existing value in place.
    ///
    /// # Errors
    /// Returns [`ScriptError::Path`]/[`ScriptError::NoSuchPath`] if the path is
    /// invalid or unresolved, or [`ScriptError::Apply`] if `value` is not
    /// compatible with the target slot.
    pub fn set(
        &self,
        root: &mut dyn Reflect,
        path: &str,
        value: &dyn Reflect,
    ) -> Result<(), ScriptError> {
        let slot = self.get_mut(root, path)?;
        slot.apply(value).map_err(ScriptError::Apply)
    }

    /// Call the registered function `name` with `args`.
    ///
    /// # Errors
    /// Returns [`ScriptError::Function`] for an unknown name, an arity
    /// mismatch, an argument-type mismatch, or a callee-reported failure.
    pub fn call(&self, name: &str, args: ArgList) -> Result<Box<dyn Reflect>, ScriptError> {
        self.functions.call(name, args).map_err(ScriptError::Function)
    }

    /// Whether a function named `name` is registered.
    #[must_use]
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains(name)
    }
}

impl fmt::Debug for ScriptBridge<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScriptBridge")
            .field("functions", &self.functions.len())
            .finish()
    }
}

/// An error produced by a [`ScriptBridge`] access.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScriptError {
    /// The access path string failed to parse.
    Path(ParsePathError),
    /// The parsed path did not resolve to a value in the root.
    NoSuchPath(String),
    /// Writing the supplied value into the target slot failed.
    Apply(ApplyError),
    /// A call-by-name dispatch failed.
    Function(FunctionError),
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScriptError::Path(err) => write!(f, "invalid access path: {}", err.message()),
            ScriptError::NoSuchPath(path) => write!(f, "path `{path}` did not resolve"),
            ScriptError::Apply(err) => write!(f, "assignment failed: {err}"),
            ScriptError::Function(err) => write!(f, "function call failed: {err}"),
        }
    }
}

impl core::error::Error for ScriptError {}
