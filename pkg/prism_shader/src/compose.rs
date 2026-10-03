//! The shader composer: a module registry that resolves imports and emits one
//! final, preprocessed source string.
//!
//! [`ShaderComposer`] holds a set of named [`ShaderModule`]s. [`compose`] walks
//! the import graph from a root module in depth-first post-order, so every
//! dependency's body appears before the module that imports it, concatenates the
//! bodies (deduplicating shared dependencies and detecting cycles), and finally
//! runs [`crate::preprocess`] over the combined source against the given
//! [`ShaderDefs`]. The traversal order is deterministic: it follows the order in
//! which imports are declared in each module's source, so the same registry and
//! root always produce byte-identical output.
//!
//! [`compose`]: ShaderComposer::compose
//! [`ShaderModule`]: crate::module::ShaderModule
//! [`ShaderDefs`]: crate::def::ShaderDefs

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::def::ShaderDefs;
use crate::module::ShaderModule;
use crate::preprocess::{self, PreprocessError};

/// Something that went wrong while composing a shader.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ComposeError {
    /// A module with this name was added more than once.
    DuplicateModule(String),
    /// A referenced module (the root, or an import) was not registered.
    UnknownModule(String),
    /// The import graph contains a cycle, given as the path of module names
    /// that closes the loop (the last name repeats the first).
    ImportCycle(Vec<String>),
    /// Preprocessing the composed source failed.
    Preprocess(PreprocessError),
}

/// A registry of named shader modules that composes them into final source.
#[derive(Clone, Default, Debug)]
pub struct ShaderComposer {
    /// Registered modules keyed by name for deterministic lookup.
    modules: BTreeMap<String, ShaderModule>,
}

impl ShaderComposer {
    /// Creates an empty composer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a module.
    ///
    /// # Errors
    ///
    /// Returns [`ComposeError::DuplicateModule`] if a module with the same name
    /// is already registered.
    pub fn add_module(&mut self, module: ShaderModule) -> Result<(), ComposeError> {
        if self.modules.contains_key(module.name()) {
            return Err(ComposeError::DuplicateModule(module.name().to_string()));
        }
        self.modules.insert(module.name().to_string(), module);
        Ok(())
    }

    /// Whether a module with this name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// The number of registered modules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether no modules are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Composes `root` and its transitive imports into final preprocessed source.
    ///
    /// # Errors
    ///
    /// Returns [`ComposeError::UnknownModule`] if the root or any import is not
    /// registered, [`ComposeError::ImportCycle`] if the import graph has a cycle,
    /// or [`ComposeError::Preprocess`] if the combined source fails preprocessing.
    pub fn compose(&self, root: &str, defs: &ShaderDefs) -> Result<String, ComposeError> {
        let mut visited = BTreeSet::new();
        let mut stack: Vec<String> = Vec::new();
        let mut bodies: Vec<String> = Vec::new();
        self.visit(root, &mut visited, &mut stack, &mut bodies)?;
        let combined = bodies.join("\n");
        preprocess::preprocess(&combined, defs).map_err(ComposeError::Preprocess)
    }

    /// Depth-first post-order traversal collecting module bodies in dependency
    /// order, deduplicating via `visited` and detecting cycles via `stack`.
    fn visit(
        &self,
        name: &str,
        visited: &mut BTreeSet<String>,
        stack: &mut Vec<String>,
        bodies: &mut Vec<String>,
    ) -> Result<(), ComposeError> {
        if visited.contains(name) {
            return Ok(());
        }
        if let Some(start) = stack.iter().position(|entry| entry == name) {
            let mut cycle: Vec<String> = stack[start..].to_vec();
            cycle.push(name.to_string());
            return Err(ComposeError::ImportCycle(cycle));
        }
        let module = self
            .modules
            .get(name)
            .ok_or_else(|| ComposeError::UnknownModule(name.to_string()))?;
        stack.push(name.to_string());
        for import in module.imports() {
            self.visit(import, visited, stack, bodies)?;
        }
        stack.pop();
        visited.insert(name.to_string());
        bodies.push(module.body());
        Ok(())
    }
}
