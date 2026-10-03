//! The [`Plugin`] trait — the single unit of engine assembly.
//!
//! Every engine capability (windowing, input, rendering, gameplay systems, …)
//! is installed by adding a plugin to an [`App`]. A plugin's
//! [`build`](Plugin::build) runs immediately when it is added; the optional
//! two-phase [`ready`](Plugin::ready) / [`finish`](Plugin::finish) gate and the
//! [`cleanup`](Plugin::cleanup) step let a plugin wait for asynchronous
//! resources (a GPU device, an asset backend) and release build-time scratch.

use core::any::type_name;

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
/// # Honestly deferred
///
/// Inter-plugin dependency declaration and topological ordering (design §6 /
/// §24.1) are M1. In M0, plugins build in the order they are added, and
/// [`is_unique`](Plugin::is_unique) de-duplication is by plugin
/// [`name`](Plugin::name).
pub trait Plugin: Send + Sync + 'static {
    /// Assemble this plugin into `app`. Runs immediately on
    /// [`App::add_plugins`](crate::app::App::add_plugins).
    fn build(&self, app: &mut App);

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
