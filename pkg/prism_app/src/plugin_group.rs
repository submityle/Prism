//! [`PluginGroup`]: add a related set of plugins in one call, with ordered,
//! idempotent editing before assembly.
//!
//! A plugin group bundles several related plugins (e.g. a future
//! `PrismDefaultPlugins`) so an app can install them together. The group is
//! assembled through a [`PluginGroupBuilder`], which keeps its members in an
//! explicit order keyed by plugin type and lets a caller edit that set before
//! it is handed to the [`App`](crate::app::App):
//!
//! - [`add`](PluginGroupBuilder::add) — append a plugin to the end.
//! - [`add_before`](PluginGroupBuilder::add_before) /
//!   [`add_after`](PluginGroupBuilder::add_after) — insert relative to another
//!   member.
//! - [`disable`](PluginGroupBuilder::disable) /
//!   [`enable`](PluginGroupBuilder::enable) — toggle a member without removing
//!   it, so its position is preserved if it is re-enabled.
//! - [`set`](PluginGroupBuilder::set) — replace a member's instance in place.
//!
//! Each member is keyed by its concrete type, so a type appears at most once
//! and every edit is idempotent in position. When the builder is finished, its
//! enabled members are topologically ordered by their declared
//! [`dependencies`](crate::plugin::Plugin::dependencies)
//! (see [`crate::plugin_graph`]); the explicit order is the stable tie-break.
//!
//! Dependency resolution happens *at assembly time*: a dependency on a plugin
//! absent from the group, or a dependency cycle, is reported before any plugin
//! builds (design §24.1), via [`try_into_plugins`](PluginGroupBuilder::try_into_plugins)
//! or as a panic from [`into_plugins`](PluginGroupBuilder::into_plugins).

use core::any::TypeId;
use std::collections::HashMap;

use crate::plugin::Plugin;
use crate::plugin_graph::{Node, PluginGraphError, topological_order};

/// One member of a [`PluginGroupBuilder`]: a boxed plugin plus whether it is
/// currently enabled.
struct PluginEntry {
    plugin: Box<dyn Plugin>,
    name: &'static str,
    dependencies: Vec<crate::plugin::PluginDependency>,
    enabled: bool,
}

/// An ordered, type-keyed builder of plugins, populated and edited by a
/// [`PluginGroup`].
///
/// Members are kept in an explicit order (insertion plus `add_before` /
/// `add_after`), keyed by concrete plugin type so each type appears once.
/// Finishing the builder yields the enabled members, topologically ordered by
/// their declared dependencies with the explicit order as the stable tie-break.
#[derive(Default)]
pub struct PluginGroupBuilder {
    /// Explicit order of member type ids.
    order: Vec<TypeId>,
    /// Member entries keyed by concrete plugin type.
    plugins: HashMap<TypeId, PluginEntry>,
}

impl PluginGroupBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            plugins: HashMap::new(),
        }
    }

    /// The explicit position of `type_id` in [`order`](Self::order), or `None`
    /// if it is not a member.
    fn position(&self, type_id: TypeId) -> Option<usize> {
        self.order.iter().position(|id| *id == type_id)
    }

    /// Insert `plugin` at explicit index `at`.
    ///
    /// # Panics
    ///
    /// Panics if a plugin of the same type is already a member.
    fn insert_at<P: Plugin>(&mut self, at: usize, plugin: P) {
        let type_id = TypeId::of::<P>();
        if self.plugins.contains_key(&type_id) {
            panic!(
                "plugin {:?} is already a member of this group; use `set` to replace it, or \
                 `enable`/`disable` to toggle it",
                plugin.name()
            );
        }
        let entry = PluginEntry {
            name: core::any::type_name::<P>(),
            dependencies: plugin.dependencies(),
            plugin: Box::new(plugin),
            enabled: true,
        };
        self.order.insert(at, type_id);
        self.plugins.insert(type_id, entry);
    }

    /// Append `plugin` to the end of the group.
    ///
    /// # Panics
    ///
    /// Panics if a plugin of the same type is already a member.
    // Named `add` for parity with `bevy_app`'s PluginGroupBuilder; it is a group
    // editor, not an arithmetic op, so the `std::ops::Add` confusion lint does
    // not apply.
    #[allow(clippy::should_implement_trait)]
    pub fn add<P: Plugin>(mut self, plugin: P) -> Self {
        let at = self.order.len();
        self.insert_at(at, plugin);
        self
    }

    /// Insert `plugin` immediately before member `Target`.
    ///
    /// # Panics
    ///
    /// Panics if `Target` is not a member, or if `plugin`'s type is already a
    /// member.
    pub fn add_before<Target: Plugin, P: Plugin>(mut self, plugin: P) -> Self {
        let at = self.position(TypeId::of::<Target>()).unwrap_or_else(|| {
            panic!(
                "cannot add before {:?}: it is not a member of this group",
                core::any::type_name::<Target>()
            )
        });
        self.insert_at(at, plugin);
        self
    }

    /// Insert `plugin` immediately after member `Target`.
    ///
    /// # Panics
    ///
    /// Panics if `Target` is not a member, or if `plugin`'s type is already a
    /// member.
    pub fn add_after<Target: Plugin, P: Plugin>(mut self, plugin: P) -> Self {
        let at = self.position(TypeId::of::<Target>()).unwrap_or_else(|| {
            panic!(
                "cannot add after {:?}: it is not a member of this group",
                core::any::type_name::<Target>()
            )
        });
        self.insert_at(at + 1, plugin);
        self
    }

    /// Replace member `P`'s instance in place, keeping its position and enabled
    /// state.
    ///
    /// # Panics
    ///
    /// Panics if `P` is not a member.
    pub fn set<P: Plugin>(mut self, plugin: P) -> Self {
        let type_id = TypeId::of::<P>();
        let entry = self.plugins.get_mut(&type_id).unwrap_or_else(|| {
            panic!(
                "cannot set {:?}: it is not a member of this group",
                plugin.name()
            )
        });
        entry.name = core::any::type_name::<P>();
        entry.dependencies = plugin.dependencies();
        entry.plugin = Box::new(plugin);
        self
    }

    /// Disable member `P` without removing it; its explicit position is kept so
    /// a later [`enable`](Self::enable) restores it in place.
    ///
    /// # Panics
    ///
    /// Panics if `P` is not a member.
    pub fn disable<P: Plugin>(mut self) -> Self {
        self.set_enabled::<P>(false, "disable");
        self
    }

    /// Re-enable a previously [`disable`](Self::disable)d member `P`.
    ///
    /// # Panics
    ///
    /// Panics if `P` is not a member.
    pub fn enable<P: Plugin>(mut self) -> Self {
        self.set_enabled::<P>(true, "enable");
        self
    }

    fn set_enabled<P: Plugin>(&mut self, enabled: bool, verb: &str) {
        let type_id = TypeId::of::<P>();
        let entry = self.plugins.get_mut(&type_id).unwrap_or_else(|| {
            panic!(
                "cannot {verb} {:?}: it is not a member of this group",
                core::any::type_name::<P>()
            )
        });
        entry.enabled = enabled;
    }

    /// Whether member `P` is currently enabled. Returns `false` if `P` is not a
    /// member.
    pub fn is_enabled<P: Plugin>(&self) -> bool {
        self.plugins
            .get(&TypeId::of::<P>())
            .is_some_and(|e| e.enabled)
    }

    /// Resolve the enabled members into their final build order, or report the
    /// first dependency-resolution error.
    ///
    /// The enabled members are topologically ordered by their declared
    /// [`dependencies`](crate::plugin::Plugin::dependencies); the explicit
    /// order is the stable tie-break. A dependency on a plugin that is not an
    /// enabled member yields [`PluginGraphError::MissingDependency`], and a
    /// dependency cycle yields [`PluginGraphError::Cycle`] — both at assembly
    /// time, before any plugin builds.
    pub fn try_into_plugins(mut self) -> Result<Vec<Box<dyn Plugin>>, PluginGraphError> {
        // Collect enabled entries in explicit order.
        let enabled: Vec<TypeId> = self
            .order
            .iter()
            .copied()
            .filter(|id| self.plugins[id].enabled)
            .collect();

        let nodes: Vec<Node> = enabled
            .iter()
            .map(|id| {
                let entry = &self.plugins[id];
                Node {
                    type_id: *id,
                    name: entry.name,
                    dependencies: entry.dependencies.clone(),
                }
            })
            .collect();

        let order = topological_order(&nodes)?;

        let mut plugins = Vec::with_capacity(order.len());
        for idx in order {
            let type_id = enabled[idx];
            let entry = self
                .plugins
                .remove(&type_id)
                .expect("enabled member was just resolved");
            plugins.push(entry.plugin);
        }
        Ok(plugins)
    }

    /// Resolve the enabled members into their final build order.
    ///
    /// Shorthand for [`try_into_plugins`](Self::try_into_plugins) that panics on
    /// a dependency-resolution error, matching the loud assembly-time failure
    /// of adding a duplicate plugin.
    ///
    /// # Panics
    ///
    /// Panics with a [`PluginGraphError`] message if a declared dependency is
    /// missing from the group or the dependency edges contain a cycle.
    pub fn into_plugins(self) -> Vec<Box<dyn Plugin>> {
        self.try_into_plugins()
            .unwrap_or_else(|err| panic!("plugin group assembly failed: {err}"))
    }
}

/// A set of plugins added together.
///
/// Implement this for a unit type (e.g. `PrismDefaultPlugins`) and return the
/// group's ordered members from [`build`](PluginGroup::build). A caller can
/// edit the returned [`PluginGroupBuilder`] (disable/replace/reorder members)
/// before passing it to [`App::add_plugins`](crate::app::App::add_plugins).
pub trait PluginGroup {
    /// Produce this group's ordered members.
    fn build(self) -> PluginGroupBuilder;
}

/// A [`PluginGroupBuilder`] is itself a [`PluginGroup`] (the identity group),
/// so an edited builder can be passed straight to
/// [`App::add_plugins`](crate::app::App::add_plugins).
impl PluginGroup for PluginGroupBuilder {
    fn build(self) -> PluginGroupBuilder {
        self
    }
}
