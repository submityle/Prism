//! [`PluginGroup`]: add a related set of plugins in one call.
//!
//! A plugin group bundles several plugins (e.g. a future `PrismDefaultPlugins`)
//! so an app can install them together. M0 ships the minimal form: a group
//! declares its ordered members, and [`App::add_plugins`](crate::app::App::add_plugins)
//! adds them in that order.
//!
//! # Honestly deferred
//!
//! Group editing — `.disable::<P>()`, `.add_before::<A, B>()`,
//! `.add_after::<A, B>()` and replacement (design §6 / §24.1) — is M1. It is
//! deliberately absent here rather than stubbed, because a correct
//! implementation needs the dependency graph that M1 introduces.

use crate::plugin::Plugin;

/// An ordered builder of plugins, populated by a [`PluginGroup`].
///
/// Members are kept in insertion order; the owning [`App`](crate::app::App)
/// adds them in that order, applying the same uniqueness checks as a direct
/// [`add_plugins`](crate::app::App::add_plugins) call.
#[derive(Default)]
pub struct PluginGroupBuilder {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginGroupBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Append `plugin` to the group, in order.
    pub fn add_plugin<P: Plugin>(mut self, plugin: P) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Consume the builder, yielding its plugins in insertion order.
    pub fn into_plugins(self) -> Vec<Box<dyn Plugin>> {
        self.plugins
    }
}

/// A set of plugins added together.
///
/// Implement this for a unit type (e.g. `PrismDefaultPlugins`) and return the
/// group's ordered members from [`build`](PluginGroup::build).
pub trait PluginGroup {
    /// Produce this group's ordered members.
    fn build(self) -> PluginGroupBuilder;
}
