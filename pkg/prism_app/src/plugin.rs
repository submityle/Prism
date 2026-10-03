//! The [`Plugin`] trait — the single unit of engine assembly.
//!
//! Every engine capability (windowing, input, rendering, gameplay systems, …)
//! is installed by adding a plugin to an [`App`]. A plugin's
//! [`build`](Plugin::build) runs immediately when it is added; the optional
//! two-phase [`ready`](Plugin::ready) / [`finish`](Plugin::finish) gate and the
//! [`cleanup`](Plugin::cleanup) step let a plugin wait for asynchronous
//! resources (a GPU device, an asset backend) and release build-time scratch.
//!
//! A plugin may declare that it must build *after* other plugin types via
//! [`dependencies`](Plugin::dependencies). Those declarations are resolved by a
//! topological sort when the plugin is assembled through a
//! [`PluginGroup`](crate::plugin_group::PluginGroup); missing or cyclic
//! dependencies are reported at assembly time, never deferred to run time
//! (design §23 risk #5).

use core::any::{TypeId, type_name};

use crate::app::App;

/// A composable unit of engine assembly added to an [`App`].
///
/// The lifecycle, in order:
///
/// 1. [`build`](Plugin::build) — called immediately when the plugin is added;
///    register systems, resources, schedules, and child plugins here.
/// 2. [`ready`](Plugin::ready) — polled before finishing; return `false` while
///    an asynchronous dependency (e.g. a GPU device) is still initializing.
/// 3. [`finish`](Plugin::finish) — called once every plugin is ready; grab the
///    now-available handles.
/// 4. [`cleanup`](Plugin::cleanup) — called after finishing; drop build-time
///    scratch state.
///
/// # Dependencies
///
/// Override [`dependencies`](Plugin::dependencies) to declare that this plugin
/// must build after other plugin types. When plugins are assembled through a
/// [`PluginGroup`](crate::plugin_group::PluginGroup), the group topologically
/// orders its members to honor those edges; a dependency on a plugin absent
/// from the group, or a dependency cycle, is reported at *assembly time*
/// (design §24.1).
///
/// # De-duplication
///
/// [`is_unique`](Plugin::is_unique) de-duplication is by plugin
/// [`name`](Plugin::name): adding a unique plugin whose name is already present
/// is an error.
pub trait Plugin: Send + Sync + 'static {
    /// Assemble this plugin into `app`. Runs immediately on
    /// [`App::add_plugins`](crate::app::App::add_plugins).
    fn build(&self, app: &mut App);

    /// The other plugin types this plugin must build after.
    ///
    /// Defaults to no dependencies. Declare one with
    /// [`PluginDependency::on`]:
    ///
    /// ```
    /// use prism_app::prelude::*;
    /// use prism_app::plugin::PluginDependency;
    ///
    /// struct Core;
    /// impl Plugin for Core {
    ///     fn build(&self, _app: &mut App) {}
    /// }
    ///
    /// struct Renderer;
    /// impl Plugin for Renderer {
    ///     fn build(&self, _app: &mut App) {}
    ///     fn dependencies(&self) -> Vec<PluginDependency> {
    ///         vec![PluginDependency::on::<Core>()]
    ///     }
    /// }
    /// ```
    fn dependencies(&self) -> Vec<PluginDependency> {
        Vec::new()
    }

    /// Whether this plugin has finished any asynchronous initialization and is
    /// ready for [`finish`](Plugin::finish). Defaults to `true` (ready at
    /// once).
    fn ready(&self, _app: &App) -> bool {
        true
    }

    /// Finalize once every plugin is [`ready`](Plugin::ready). Defaults to a
    /// no-op.
    fn finish(&self, _app: &mut App) {}

    /// Release build-time scratch after finishing. Defaults to a no-op.
    fn cleanup(&self, _app: &mut App) {}

    /// A stable, human-readable name. Defaults to the implementing type's
    /// fully-qualified name and is used for [`is_unique`](Plugin::is_unique)
    /// de-duplication and diagnostics.
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    /// Whether adding this plugin twice is an error. Defaults to `true`, which
    /// rejects a second plugin whose [`name`](Plugin::name) is already present.
    fn is_unique(&self) -> bool {
        true
    }
}

/// A declared dependency of one plugin on another plugin *type*.
///
/// Build one with [`PluginDependency::on::<P>()`](PluginDependency::on). The
/// dependency is identified by [`TypeId`], so it matches the exact plugin type
/// regardless of any [`Plugin::name`] override, and carries the type name for
/// human-readable assembly-time diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginDependency {
    type_id: TypeId,
    name: &'static str,
}

impl PluginDependency {
    /// Declare a dependency on plugin type `P`.
    pub fn on<P: Plugin>() -> Self {
        Self {
            type_id: TypeId::of::<P>(),
            name: type_name::<P>(),
        }
    }

    /// The [`TypeId`] of the depended-on plugin type, used to match it within a
    /// group.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The fully-qualified name of the depended-on plugin type, used for
    /// diagnostics.
    pub fn name(&self) -> &'static str {
        self.name
    }
}
